"""
fractal_brain – A self‑contained, zero‑dependency implementation of the
hybrid PID + fractal Markov chain + MoE + RAG + BCM plasticity architecture,
now including autograd, signal processing, distillation, quantization, PCA, wormholes,
logic folding, fractal matrices, and JEPA.
"""
from .math_utils import (
    Vector, Matrix, softmax, softmax_rows,
    kl_divergence, sample_multinomial, set_seed
)
from .pid import PIDController
from .markov import BootstrapGate, FractalMarkovNode, build_fractal_chain
from .tentacles import LassoTentacles
from .attention import MultiHeadAttention, TransformerEncoderLayer, gelu, layer_norm
from .moe import TransformerExpert, GatedMoE
from .rag import VectorStore, StateRAGFusion
from .synaptic import BCMPlasticity
from .core import FractalBrain
from .turbo_quant import TurboQuant
from .dim_reduction import PCA, power_iteration, truncated_svd
from .wormhole import Wormhole
from .logic_folding import LogicFolder, fold_states, fuzzy_and, fuzzy_or, fuzzy_not
from .recursive_matrices import FractalMatrix
from .jepa import JEPA
from .autograd import Value
from .signal import DelayLine, convolve1d
from .distillation import distillation_loss
from .tokenizer import BPETokenizer, normalize_text
from .dataset import TextDataset, DatasetView
from .checkpoint import (
    save_checkpoint, load_checkpoint, serialize_brain, deserialize_brain,
    register_checkpoint_class,
)
from .storage import Storage
from .interfaces import Encoder, Decoder
from .encoder_decoder import NativeEncoder, NativeAutoregressiveDecoder, RobertaAdapter, GPTAdapter, FusionModel
from .version import __version__

__all__ = [
    # math_utils
    "Vector", "Matrix", "softmax", "softmax_rows",
    "kl_divergence", "sample_multinomial", "set_seed",
    # pid
    "PIDController",
    # markov
    "BootstrapGate", "FractalMarkovNode", "build_fractal_chain",
    # tentacles
    "LassoTentacles",
    # attention
    "MultiHeadAttention", "TransformerEncoderLayer", "gelu", "layer_norm",
    # moe
    "TransformerExpert", "GatedMoE",
    # rag
    "VectorStore", "StateRAGFusion",
    # synaptic
    "BCMPlasticity",
    # core
    "FractalBrain",
    # turbo quant
    "TurboQuant",
    # dim reduction
    "PCA", "power_iteration", "truncated_svd",
    # wormhole
    "Wormhole",
    # logic folding
    "LogicFolder", "fold_states", "fuzzy_and", "fuzzy_or", "fuzzy_not",
    # recursive matrices
    "FractalMatrix",
    # jepa
    "JEPA",
    # autograd
    "Value",
    # signal
    "DelayLine", "convolve1d",
    # distillation
    "distillation_loss",
    # tokenizer
    "BPETokenizer", "normalize_text",
    # dataset
    "TextDataset", "DatasetView",
    # checkpoint
    "save_checkpoint", "load_checkpoint", "serialize_brain", "deserialize_brain",
    "register_checkpoint_class",
    # storage
    "Storage",
    # encoder-decoder fusion
    "Encoder", "Decoder",
    "NativeEncoder", "NativeAutoregressiveDecoder", "RobertaAdapter", "GPTAdapter", "FusionModel",
    "__version__",
]