"""
fractal_brain/checkpoint.py
Model serialization: save/load a full FractalBrain (weights + training state) to/from
JSON, with no external dependencies.

Covers the "Checkpoint save/load" and (the weights/training-state half of) "Model
serialization" items from To-Do.md.

Design: rather than hand-writing a bespoke serializer for every one of the ~15 classes
that make up a FractalBrain's state -- a large, brittle surface that would drift out of
sync the moment an attribute is added, renamed, or removed anywhere in the architecture
-- this walks the object graph generically:
  - Matrix / Vector are special-cased (their .data is already JSON-native).
  - list / tuple / dict / plain values (int, float, str, bool, None) pass through,
    recursing into their contents.
  - Any other object is serialized as {"__type__": ClassName, "__state__": {...}} from
    its __dict__, and reconstructed by creating a bare instance of the registered class
    (bypassing __init__ entirely, via object.__new__) and restoring that __dict__
    directly. This only requires every relevant *class* to be registered by name below
    -- not every attribute to be hand-enumerated -- so it keeps working automatically as
    classes gain, lose, or rename attributes.

    object.__new__ + __dict__ restoration is safe here specifically because none of this
    project's classes do anything in __init__ beyond constructing sub-objects and
    assigning attributes (no global registration, no I/O, no threads) -- verified by
    reading every __init__ in the package, not assumed.

Known, deliberate limitation: FractalBrain.teacher is NOT saved -- it's an arbitrary,
externally-injected object (e.g. orchestrator.FrozenTeacherExpert isn't even part of
this package), so there's no generally-correct way to reconstruct it. save_checkpoint()
warns if one is attached; reattach it manually after loading (`brain.teacher = ...`).

Worth knowing (found while testing this module, not specific to it): BootstrapGate
draws from Python's *global* random module rather than a per-instance RNG, so two live
FractalBrain instances in the same process share one random stream. Interleaving calls
to two instances (e.g. comparing an original against a freshly-loaded reload) is
order-dependent -- whichever instance you call first advances the shared state before
the other runs -- so it's not a fair way to check "did loading preserve everything".
Compare static attributes instead (weights, PID gains, step_count, ...), or do the
comparison across two separate processes if you want to include the stochastic parts;
see persistence_demo.py and this module's own test coverage for both.
"""
import json
import random
import warnings

from .math_utils import Matrix, Vector
from .pid import PIDController
from .markov import FractalMarkovNode, BootstrapGate
from .tentacles import LassoTentacles
from .attention import MultiHeadAttention, TransformerEncoderLayer
from .moe import GatedMoE, TransformerExpert
from .rag import VectorStore, StateRAGFusion
from .synaptic import BCMPlasticity
from .wormhole import Wormhole
from .jepa import JEPA
from .signal import DelayLine
from .core import FractalBrain

FORMAT_VERSION = 1

# Classes reconstructable by name. Extend via register_checkpoint_class() if you add
# your own stateful classes that can end up as an attribute somewhere inside a
# FractalBrain (subclasses, custom modules, etc).
_REGISTRY = {cls.__name__: cls for cls in [
    FractalBrain, PIDController, FractalMarkovNode, BootstrapGate, LassoTentacles,
    GatedMoE, TransformerExpert, TransformerEncoderLayer, MultiHeadAttention,
    VectorStore, StateRAGFusion, BCMPlasticity, Wormhole, JEPA, DelayLine,
]}


def register_checkpoint_class(cls):
    """Register an additional class so instances of it can be saved inside a
    checkpoint and reconstructed when loading. Needed only for classes not already
    built into fractal_brain (e.g. a custom teacher, or a subclass of FractalBrain)."""
    _REGISTRY[cls.__name__] = cls


def _to_jsonable(obj):
    if isinstance(obj, Matrix):
        return {"__type__": "Matrix", "data": obj.data}
    if isinstance(obj, Vector):
        return {"__type__": "Vector", "data": obj.data}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        # list-of-pairs rather than a plain JSON object, so non-string keys survive
        # the round trip intact instead of being silently stringified
        return {"__dict_items__": [[_to_jsonable(k), _to_jsonable(v)] for k, v in obj.items()]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "__dict__"):
        type_name = type(obj).__name__
        if type_name not in _REGISTRY:
            raise TypeError(
                f"Cannot checkpoint an object of type {type_name!r}: not registered. "
                f"Call fractal_brain.checkpoint.register_checkpoint_class({type_name}) "
                f"first, or remove/replace this attribute before saving."
            )
        return {"__type__": type_name, "__state__": {k: _to_jsonable(v) for k, v in obj.__dict__.items()}}
    raise TypeError(f"Cannot checkpoint a value of type {type(obj)}: {obj!r}")


def _from_jsonable(data):
    if isinstance(data, list):
        return [_from_jsonable(x) for x in data]
    if isinstance(data, dict):
        if "__dict_items__" in data:
            return {_from_jsonable(k): _from_jsonable(v) for k, v in data["__dict_items__"]}
        type_name = data.get("__type__")
        if type_name == "Matrix":
            return Matrix(data["data"])
        if type_name == "Vector":
            return Vector(data["data"])
        if type_name is not None:
            cls = _REGISTRY.get(type_name)
            if cls is None:
                raise TypeError(
                    f"Cannot restore an object of type {type_name!r}: not registered. "
                    f"Call fractal_brain.checkpoint.register_checkpoint_class({type_name}) "
                    f"with the same class before loading."
                )
            instance = object.__new__(cls)
            instance.__dict__.update({k: _from_jsonable(v) for k, v in data["__state__"].items()})
            return instance
        return {k: _from_jsonable(v) for k, v in data.items()}   # defensive fallback
    return data   # int / float / str / bool / None pass through unchanged


def serialize_brain(brain):
    """FractalBrain -> a JSON-serializable dict. Low-level building block used by
    save_checkpoint() and by storage.Storage's checkpoints table.

    Also captures Python's *global* random module state: BootstrapGate (used by the
    fractal Markov chain) draws from `random.random()` directly rather than from an
    RNG owned by the brain itself, so reproducing "what would have happened if training
    had just continued uninterrupted" after a reload requires restoring that global
    state too, not just the brain's own attributes. See deserialize_brain()'s
    restore_rng_state parameter.
    """
    if brain.teacher is not None:
        warnings.warn(
            "brain.teacher is set but will not be included in the checkpoint (it's an "
            "arbitrary externally-injected object with no generally-correct way to "
            "reconstruct it). Reattach it manually after loading: `brain.teacher = ...`.",
            stacklevel=2,
        )
    teacher_backup = brain.teacher
    brain.teacher = None
    try:
        state = _to_jsonable(brain)
    finally:
        brain.teacher = teacher_backup
    version, internal_state, gauss_next = random.getstate()
    rng_state = [version, list(internal_state), gauss_next]
    return {"format_version": FORMAT_VERSION, "state": state, "rng_state": rng_state}


def deserialize_brain(payload, restore_rng_state=True):
    """The inverse of serialize_brain(). `brain.teacher` comes back as None regardless
    of what it was at save time; reattach one manually if needed.

    restore_rng_state: if True (the default), also restores Python's *global* random
    module state to exactly what it was at save time -- needed for a reloaded brain to
    continue training identically to how an uninterrupted run would have. Set to False
    if you don't want loading a checkpoint to reset unrelated code's random draws
    elsewhere in the same process (e.g. you're managing multiple brains, or other
    random-dependent logic, and only want this brain's own weights/state back).
    """
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format_version {payload.get('format_version')!r} is not "
            f"supported by this version of fractal_brain (expected {FORMAT_VERSION})"
        )
    if restore_rng_state and "rng_state" in payload:
        version, internal_state, gauss_next = payload["rng_state"]
        random.setstate((version, tuple(internal_state), gauss_next))
    return _from_jsonable(payload["state"])


def save_checkpoint(brain, path):
    """Save a FractalBrain's full weights + training state to a JSON file at `path`."""
    payload = serialize_brain(brain)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_checkpoint(path, restore_rng_state=True):
    """Load a FractalBrain previously saved with save_checkpoint(). See
    deserialize_brain() for what restore_rng_state controls."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return deserialize_brain(payload, restore_rng_state=restore_rng_state)
