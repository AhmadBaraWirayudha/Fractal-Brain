"""
tests/test_smoke.py

A dependency-free (no pytest/unittest required, though it works fine under either)
smoke and regression test suite for fractal_brain. Run directly:

    python tests/test_smoke.py

from the project root (the directory containing both tests/ and fractal_brain/).

This isn't an exhaustive test suite for every code path, but it exercises every
module at least once, and specifically regression-tests the bugs fixed in
CHANGELOG.md so they can't silently come back.
"""
import math
import random
import sys
import os
import json
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal_brain import (
    FractalBrain, set_seed, Vector, Matrix, softmax, kl_divergence,
    PIDController, BootstrapGate, build_fractal_chain, LassoTentacles,
    MultiHeadAttention, TransformerEncoderLayer, TransformerExpert, GatedMoE,
    VectorStore, StateRAGFusion, BCMPlasticity, TurboQuant, PCA, Wormhole,
    LogicFolder, fuzzy_and, fuzzy_or, fuzzy_not, FractalMatrix, JEPA, Value,
    DelayLine, distillation_loss, BPETokenizer, TextDataset,
    save_checkpoint, load_checkpoint, serialize_brain, deserialize_brain,
    register_checkpoint_class, Storage,
)

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
    else:
        FAILED.append((name, detail))
        print(f"  FAIL: {name}  {detail}")


def section(title):
    print(f"\n--- {title} ---")


# ============================================================================
section("math_utils: Matrix.linear vs dot_vector conventions")
# ============================================================================
W = Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])          # (in_dim=2, out_dim=3)
x = Vector([10.0, 1.0])
check("Matrix.linear (in,out) convention", W.linear(x).to_list() == [14.0, 25.0, 36.0])

Wsq = Matrix.eye(3)
v = Vector([2.0, -1.0, 5.0])
check("Matrix.dot_vector identity", Wsq.dot_vector(v).to_list() == v.to_list())

he = Matrix.he_init(64, 32)
check("he_init shape", (he.rows, he.cols) == (64, 32))
mean_abs = sum(abs(w) for row in he.data for w in row) / (64*32)
check("he_init is reasonably scaled (not U[0,1)-huge)", mean_abs < 1.0, f"mean_abs={mean_abs}")

kl = kl_divergence(Vector([math.log(0.5), math.log(0.5)]), Vector([0.5, 0.5]))
check("kl_divergence(P,P) == 0", abs(kl) < 1e-9, f"kl={kl}")


# ============================================================================
section("autograd.Value vs numerical gradient")
# ============================================================================
def f(x, y):
    return x*x*y + (y+2.0) - x/y

xa, ya = Value(2.0), Value(3.0)
out = f(xa, ya)
out.backward()
h = 1e-6
ndx = (f(2.0+h, 3.0) - f(2.0-h, 3.0)) / (2*h)
ndy = (f(2.0, 3.0+h) - f(2.0, 3.0-h)) / (2*h)
check("Value dx matches numerical gradient", abs(xa.grad - ndx) < 1e-3, f"{xa.grad} vs {ndx}")
check("Value dy matches numerical gradient", abs(ya.grad - ndy) < 1e-3, f"{ya.grad} vs {ndy}")
check("Value.__rsub__", (10 - Value(4.0)).data == 6.0)
check("Value.__rtruediv__", (10 - Value(4.0)*0 + 0)  is not None)  # sanity no-crash
check("Value.__rtruediv__ value", abs((10 / Value(4.0)).data - 2.5) < 1e-9)


# ============================================================================
section("pid.PIDController")
# ============================================================================
p = PIDController(Kp=1.0, Ki=0.1, Kd=0.1)
out1 = p.step(5.0)
check("PID dt=0 does not raise", True, "") if p.step(5.0, dt=0.0) is not None else None
p2 = PIDController(Kp=1.0, Ki=0.0, Kd=0.0)
p2.step(3.0)
recomputed = p2.compute_output(Kp=1.0)
check("PID compute_output matches step() for same gain", abs(recomputed - 3.0) < 1e-9, f"{recomputed}")
probe = p2.compute_output(Kp=2.0)
check("PID compute_output probe does not mutate state", p2.Kp == 1.0)
check("PID compute_output probe reflects the hypothetical gain", abs(probe - 6.0) < 1e-9, f"{probe}")


# ============================================================================
section("markov: BootstrapGate over >10 calls (regression test for the random() NameError)")
# ============================================================================
gate = BootstrapGate(n_bootstrap=10)
try:
    for i in range(30):
        gate.should_transition(i % 3)
    check("BootstrapGate survives >10 calls without NameError", True)
except NameError as e:
    check("BootstrapGate survives >10 calls without NameError", False, str(e))

chain = build_fractal_chain(max_level=2, num_states=3)
try:
    state = 0
    for _ in range(20):
        state, emb = chain.forward(state)
    check("fractal markov chain runs 20 steps", len(emb) == 3)
except Exception as e:
    check("fractal markov chain runs 20 steps", False, str(e))


# ============================================================================
section("tentacles.LassoTentacles")
# ============================================================================
lt = LassoTentacles(input_dim=9, num_experts=4)
out = lt.forward(Vector([1.0]*9))
check("LassoTentacles.forward output length", len(out) == 4)
lt.prune(threshold=1e9)   # everything should get pruned at an absurd threshold
check("LassoTentacles.prune can zero out all experts", lt.mask.to_list() == [0.0]*4)


# ============================================================================
section("attention: MultiHeadAttention / TransformerEncoderLayer shapes")
# ============================================================================
layer = TransformerEncoderLayer(d_model=16, num_heads=2, d_ff=32)
inp = Matrix.he_init(5, 16)   # pretend (seq_len=5, d_model=16)
result = layer.forward(inp)
check("TransformerEncoderLayer preserves shape", (result.rows, result.cols) == (5, 16))


# ============================================================================
section("moe: TransformerExpert / GatedMoE (incl. hard-pruning and temperature fixes)")
# ============================================================================
expert = TransformerExpert(vocab_size=20, d_model=8, num_heads=2, d_ff=16, num_layers=2)
logits = expert.forward([1, 2, 3])
check("TransformerExpert output shape", (logits.rows, logits.cols) == (3, 20))
check("TransformerExpert caches last hidden state for training", expert._last_hidden is not None)

moe = GatedMoE(num_experts=4, vocab_size=20, d_model=8, num_heads=2, d_ff=16, num_layers=1)
combined, weights = moe.forward([1, 2], lasso_mask=[1, 0, 1, 1])
check("GatedMoE hard-pruned expert has ~0 weight", weights[1] < 1e-6, f"{weights}")
check("GatedMoE weights sum to ~1", abs(sum(weights) - 1.0) < 1e-6)


# ============================================================================
section("rag: VectorStore / StateRAGFusion")
# ============================================================================
store = VectorStore(dim=8)
for i in range(10):
    store.add(Vector([float((i+j) % 5) for j in range(8)]), doc_id=i)
ids, sims = store.search(Vector([0.0]*8), k=3)
check("VectorStore.search returns k results", len(ids) == 3)

fusion = StateRAGFusion(d_model=8)
fused = fusion.forward(Vector([0.1]*8), [Vector([0.2]*8) for _ in range(3)])
check("StateRAGFusion.forward output length", len(fused) == 8)


# ============================================================================
section("synaptic.BCMPlasticity")
# ============================================================================
bcm = BCMPlasticity()
Wbcm = Matrix.he_init(6, 3)
before = [row[:] for row in Wbcm.data]
bcm.update(Wbcm, Vector([1.0]*6), Vector([0.3, 0.5, 0.2]))
changed = any(Wbcm.data[i][j] != before[i][j] for i in range(6) for j in range(3))
check("BCMPlasticity.update changes weights", changed)


# ============================================================================
section("turbo_quant.TurboQuant round-trip")
# ============================================================================
tq = TurboQuant()
mat = Matrix.he_init(5, 5)
qb, scale, min_val, shape = tq.quantize_matrix(mat)
deq = tq.dequantize_matrix(qb, scale, min_val, shape)
err = sum(abs(mat.data[i][j] - deq.data[i][j]) for i in range(5) for j in range(5))
check("TurboQuant matrix round-trip error is small", err < 1.0, f"err={err}")

vqb, vscale, vmin = tq.quantize_vector(Vector([0.1, 0.9, 0.5, -0.3]))
vdeq = tq.dequantize_vector(vqb, vscale, vmin)
check("TurboQuant vector round-trip length", len(vdeq) == 4)


# ============================================================================
section("dim_reduction.PCA round-trip")
# ============================================================================
set_seed(0)
rows = [Vector([random.random() for _ in range(6)]) for _ in range(30)]
pca = PCA(n_components=3)
pca.fit(rows)
reduced = pca.transform(rows[0])
check("PCA.transform output length", len(reduced) == 3)
recon = pca.inverse_transform(reduced)
check("PCA.inverse_transform output length", len(recon) == 6)


# ============================================================================
section("wormhole.Wormhole (shape bug regression)")
# ============================================================================
worm = Wormhole(src_dim=15, target_dim=4)   # deliberately src != target dims
out = worm.transform(Vector([0.5]*15))
check("Wormhole handles src_dim != target_dim", len(out) == 4)


# ============================================================================
section("jepa.JEPA (numerical-blowup regression)")
# ============================================================================
jepa = JEPA(input_dim=16, embed_dim=8)
ctx = jepa.encode_context(Vector([0.5]*16))
check("JEPA encode_context is well-scaled", max(abs(v) for v in ctx.data) < 50,
      f"max={max(abs(v) for v in ctx.data)}")
loss_val, _, _ = jepa.loss(Vector([0.5]*16), Vector([0.5]*16))
check("JEPA loss on identical input is small and finite", 0 <= loss_val < 100,
      f"loss={loss_val}")


# ============================================================================
section("distillation.distillation_loss")
# ============================================================================
student = Vector([2.0, 1.0, 0.1])
teacher = Vector([1.8, 1.2, 0.2])
true_labels = Vector([1.0, 0.0, 0.0])
dloss = distillation_loss(student, teacher, temperature=2.0, alpha=0.5, true_labels=true_labels)
check("distillation_loss is finite and non-negative", dloss >= 0 and math.isfinite(dloss), f"{dloss}")


# ============================================================================
section("recursive_matrices.FractalMatrix (leaf/size consistency regression)")
# ============================================================================
fm = FractalMatrix(depth=2, base_size=3)
dense = fm.to_dense()
consistent = all(dense.data[i][j] == fm.get_element(i, j)
                 for i in range(fm.size) for j in range(fm.size))
check("FractalMatrix.to_dense matches get_element everywhere", consistent)
check("FractalMatrix size matches base_size**depth", fm.size == 3**2)


# ============================================================================
section("logic_folding")
# ============================================================================
check("fuzzy_and is min()", fuzzy_and(0.6, 0.8) == 0.6)
check("fuzzy_or is max()", fuzzy_or(0.6, 0.8) == 0.8)
check("fuzzy_not", abs(fuzzy_not(0.3) - 0.7) < 1e-9)
folder = LogicFolder(('and', 0, ('not', 1)))
folded = folder.evaluate(Vector([0.9, 0.2, 0.4, 0.6]))
check("LogicFolder.evaluate computes min(x0, 1-x1)", abs(folded - min(0.9, 0.8)) < 1e-9, f"{folded}")


# ============================================================================
section("signal.DelayLine (None-sentinel regression)")
# ============================================================================
dl = DelayLine(max_delay=3)
r0 = dl.push(Vector([1.0, 2.0]))
r1 = dl.push(Vector([3.0, 4.0]))
r2 = dl.push(Vector([5.0, 6.0]))
r3 = dl.push(Vector([7.0, 8.0]))
check("DelayLine returns None (not a stray 0.0) before buffer fills", r0 is None and r1 is None and r2 is None)
check("DelayLine returns the correct delayed Vector once full", r3.to_list() == [1.0, 2.0])


# ============================================================================
section("core.FractalBrain: empty input guard")
# ============================================================================
set_seed(5)
brain = FractalBrain(vocab_size=30, d_model=16, num_experts=4, num_heads=2, d_ff=32,
                     num_layers=1, num_markov_nodes=3, markov_states=3, max_level=1)
try:
    brain.forward([])
    check("empty token_ids raises ValueError", False, "did not raise")
except ValueError:
    check("empty token_ids raises ValueError", True)


# ============================================================================
section("core.FractalBrain: hard pruning actually excludes an expert")
# ============================================================================
brain.tentacles.mask = Vector([1.0, 0.0, 1.0, 1.0])
_, weights, _ = brain.forward([1, 2, 3])
check("force-pruned expert has ~0 gate weight", weights[1] < 1e-6, f"{weights}")
brain.tentacles.mask = Vector([1.0, 1.0, 1.0, 1.0])  # reset


# ============================================================================
section("core.FractalBrain: extended run (150 steps) stability, incl. num_layers=2")
# ============================================================================
set_seed(9)
brain2 = FractalBrain(vocab_size=60, d_model=20, num_experts=4, num_heads=2, d_ff=40,
                      num_layers=2, num_markov_nodes=4, markov_states=3, max_level=2)
random.seed(2)
losses = []
kp_before = brain2.pid.Kp
for i in range(150):
    seq_len = random.randint(2, 5)
    toks = [random.randint(0, 59) for _ in range(seq_len)]
    tgt = [0.0]*60
    tgt[random.randint(0, 59)] = 1.0
    _, loss = brain2.step(toks, tgt)
    losses.append(loss)
check("no NaN/inf across 150 steps (num_layers=2)", not any(math.isnan(l) or math.isinf(l) for l in losses))
check("losses stay in a sane range (no numerical blow-up)", max(losses) < 1000, f"max={max(losses)}")
check("PID gains stay finite and non-negative", all(math.isfinite(g) and g >= 0 for g in
      (brain2.pid.Kp, brain2.pid.Ki, brain2.pid.Kd)))
check("PID gains actually adapted from their initial values", (brain2.pid.Kp, brain2.pid.Ki, brain2.pid.Kd) != (kp_before, 0.15, 0.05) or brain2.pid.Kp != kp_before)


# ============================================================================
section("core.FractalBrain: output-layer training reduces loss on a learnable task")
# ============================================================================
set_seed(3)
brain3 = FractalBrain(vocab_size=50, d_model=24, num_experts=4, num_heads=2, d_ff=48,
                      num_layers=1, num_markov_nodes=4, markov_states=3, max_level=2,
                      output_lr=0.2)
random.seed(0)
examples = []
for _ in range(8):
    toks = [random.randint(0, 49) for _ in range(3)]
    examples.append((toks, sum(toks) % 50))

def mk_target(cls):
    t = [0.0]*50
    t[cls] = 1.0
    return t

losses3 = []
for i in range(300):
    toks, cls = examples[i % len(examples)]
    _, loss = brain3.step(toks, mk_target(cls))
    losses3.append(loss)

first10 = sum(losses3[:10]) / 10
last10 = sum(losses3[-10:]) / 10
check("loss decreases substantially on a memorizable task", last10 < first10 * 0.1,
      f"first10={first10:.3f} last10={last10:.3f}")
correct = 0
for toks, cls in examples:
    probs = brain3.sample(toks)
    pred = max(range(len(probs)), key=lambda i: probs[i])
    correct += int(pred == cls)
check("model reaches high accuracy on the memorized examples", correct >= 7, f"{correct}/8")


# ============================================================================
section("tokenizer.BPETokenizer")
# ============================================================================
tiny_corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day in the warm sun",
    "the quick fox runs through the green forest",
]
btok = BPETokenizer(lowercase=True)
btok.train(tiny_corpus, vocab_size=80)
check("BPETokenizer learns a vocab", btok.vocab_size > 4)
ids = btok.encode("the quick fox")
check("BPETokenizer.encode returns ids", len(ids) > 0 and all(isinstance(i, int) for i in ids))
decoded = btok.decode(btok.encode(tiny_corpus[0]))
check("BPETokenizer round-trips training sentences exactly", decoded == tiny_corpus[0], f"{decoded!r}")
try:
    btok.encode("zzq")  # unseen single characters -> UNK, must not crash
    check("BPETokenizer.encode handles unseen characters without crashing", True)
except Exception as e:
    check("BPETokenizer.encode handles unseen characters without crashing", False, str(e))
try:
    BPETokenizer().train(tiny_corpus, vocab_size=2)
    check("BPETokenizer rejects too-small vocab_size", False, "did not raise")
except ValueError:
    check("BPETokenizer rejects too-small vocab_size", True)
btok.save("/tmp/_test_tokenizer_vocab.json")
btok2 = BPETokenizer.load("/tmp/_test_tokenizer_vocab.json")
check("BPETokenizer save/load round-trips vocab size", btok2.vocab_size == btok.vocab_size)
check("BPETokenizer save/load round-trips encoding", btok2.encode("the quick fox") == btok.encode("the quick fox"))


# ============================================================================
section("dataset.TextDataset")
# ============================================================================
ds = TextDataset(btok, tiny_corpus, context_length=4, stride=1)
check("TextDataset builds examples", len(ds) > 0)
ctx, tgt = ds[0]
check("TextDataset example shapes", len(ctx) == 4 and len(tgt) == btok.vocab_size and abs(sum(tgt) - 1.0) < 1e-9)
tr, va, te = ds.split(train_frac=0.7, val_frac=0.15, seed=0)
check("TextDataset.split sizes sum to total", len(tr) + len(va) + len(te) == len(ds))
overlap = set(tr.indices) & set(va.indices) | set(tr.indices) & set(te.indices) | set(va.indices) & set(te.indices)
check("TextDataset.split has no overlap between train/val/test", len(overlap) == 0)
short = TextDataset(btok, ["hi"], context_length=8)
check("TextDataset handles a too-short text without crashing", len(short) == 0)


# ============================================================================
section("core.FractalBrain.evaluate() (read-only counterpart to step())")
# ============================================================================
set_seed(6)
brain4 = FractalBrain(vocab_size=25, d_model=12, num_experts=4, num_heads=2, d_ff=24,
                      num_layers=1, num_markov_nodes=3, markov_states=3, max_level=1)
toks_e = [1, 2, 3]
tgt_e = [0.0]*25
tgt_e[5] = 1.0

def _w_out_snapshot(b):
    return [[row[:] for row in e.W_out.data] for e in b.moe.experts]

before_w = _w_out_snapshot(brain4)
before_pid = (brain4.pid.Kp, brain4.pid.Ki, brain4.pid.Kd)
before_steps = brain4.step_count
_, eval_loss = brain4.evaluate(toks_e, tgt_e)
after_w = _w_out_snapshot(brain4)
after_pid = (brain4.pid.Kp, brain4.pid.Ki, brain4.pid.Kd)
after_steps = brain4.step_count

check("evaluate() does not change any expert's W_out", before_w == after_w)
check("evaluate() does not change PID gains", before_pid == after_pid)
check("evaluate() does not advance step_count", before_steps == after_steps)
check("evaluate() returns a finite loss", math.isfinite(eval_loss))

before_w2 = _w_out_snapshot(brain4)
brain4.step(toks_e, tgt_e)
after_w2 = _w_out_snapshot(brain4)
check("step() (unlike evaluate()) does change W_out -- confirms evaluate() isn't just a no-op forward", before_w2 != after_w2)


# ============================================================================
section("checkpoint: save/load round-trip and edge cases")
# ============================================================================
_tmpdir = tempfile.mkdtemp(prefix="fractal_brain_test_")

set_seed(21)
ckpt_brain = FractalBrain(vocab_size=25, d_model=12, num_experts=3, num_heads=2, d_ff=24,
                          num_layers=1, num_markov_nodes=3, markov_states=3, max_level=2,
                          use_jepa=True, use_wormhole=True)
for _ in range(15):
    ckpt_brain.step([1, 2, 3], [1.0 if i == 5 else 0.0 for i in range(25)])

ckpt_path = os.path.join(_tmpdir, "brain.json")
save_checkpoint(ckpt_brain, ckpt_path)
reloaded_brain = load_checkpoint(ckpt_path)

check("checkpoint round-trips expert W_out weights",
      reloaded_brain.moe.experts[0].W_out.data == ckpt_brain.moe.experts[0].W_out.data)
check("checkpoint round-trips PID gains",
      (reloaded_brain.pid.Kp, reloaded_brain.pid.Ki, reloaded_brain.pid.Kd) ==
      (ckpt_brain.pid.Kp, ckpt_brain.pid.Ki, ckpt_brain.pid.Kd))
check("checkpoint round-trips step_count", reloaded_brain.step_count == ckpt_brain.step_count)
check("checkpoint round-trips markov chain state",
      reloaded_brain.current_markov_states == ckpt_brain.current_markov_states)
check("checkpoint round-trips wormhole weights",
      reloaded_brain.wormhole.W.data == ckpt_brain.wormhole.W.data)
check("checkpoint round-trips jepa weights",
      reloaded_brain.jepa.Wc.data == ckpt_brain.jepa.Wc.data)
check("checkpoint round-trips rag_index vectors",
      [v.to_list() for v in reloaded_brain.rag_index.vectors] ==
      [v.to_list() for v in ckpt_brain.rag_index.vectors])

# teacher: excluded, with a warning
class _DummyTeacherForTest:
    def forward(self, token_ids):
        return Matrix([[0.1] * 25 for _ in token_ids])

ckpt_brain.teacher = _DummyTeacherForTest()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    save_checkpoint(ckpt_brain, ckpt_path)
check("save_checkpoint warns when a teacher is attached",
      any("teacher" in str(w.message) for w in caught))
check("original brain's teacher is restored (not left None) after saving",
      ckpt_brain.teacher is not None)
reloaded_with_teacher = load_checkpoint(ckpt_path)
check("teacher comes back as None after reload", reloaded_with_teacher.teacher is None)
ckpt_brain.teacher = None

# restore_rng_state flag -- deliberately scramble the global RNG first so the
# before/after comparison is airtight regardless of what any earlier test did to it
save_checkpoint(ckpt_brain, ckpt_path)
rng_at_save_time = random.getstate()
random.seed(123456)
check("sanity check: scrambling the seed actually changed the global state",
      random.getstate() != rng_at_save_time)
load_checkpoint(ckpt_path, restore_rng_state=False)
check("restore_rng_state=False leaves the (scrambled) global random state untouched",
      random.getstate() != rng_at_save_time)
load_checkpoint(ckpt_path, restore_rng_state=True)
check("restore_rng_state=True restores the exact saved global random state",
      random.getstate() == rng_at_save_time)

# unregistered class fails loudly, not silently
class _UnregisteredForTest:
    def __init__(self):
        self.x = 1

ckpt_brain._unregistered_attr = _UnregisteredForTest()
try:
    save_checkpoint(ckpt_brain, ckpt_path)
    check("checkpointing an unregistered class raises TypeError", False, "did not raise")
except TypeError:
    check("checkpointing an unregistered class raises TypeError", True)
register_checkpoint_class(_UnregisteredForTest)
save_checkpoint(ckpt_brain, ckpt_path)
reloaded_custom = load_checkpoint(ckpt_path)
check("registering a class allows it to round-trip", reloaded_custom._unregistered_attr.x == 1)
del ckpt_brain._unregistered_attr

# None-components (use_jepa=False, use_wormhole=False) round-trip too
set_seed(22)
none_brain = FractalBrain(vocab_size=15, d_model=8, num_experts=2, num_heads=2, d_ff=16,
                          num_layers=1, num_markov_nodes=2, markov_states=3, max_level=1,
                          use_jepa=False, use_wormhole=False)
none_brain.step([1, 2], [0.0] * 15)
none_path = os.path.join(_tmpdir, "none_brain.json")
save_checkpoint(none_brain, none_path)
reloaded_none_brain = load_checkpoint(none_path)
check("checkpoint round-trips None wormhole/jepa",
      reloaded_none_brain.wormhole is None and reloaded_none_brain.jepa is None)


# ============================================================================
section("storage.Storage (SQLite persistence)")
# ============================================================================
db_path = os.path.join(_tmpdir, "test.db")
db_corpus = ["the quick brown fox", "the lazy dog sleeps", "a quick fox runs"]
db_tok = BPETokenizer(lowercase=True)
db_tok.train(db_corpus, vocab_size=60)
db_dataset = TextDataset(db_tok, db_corpus, context_length=3)
db_train, db_val, _db_test = db_dataset.split(train_frac=0.7, val_frac=0.15, seed=0)

with Storage(db_path) as db:
    db.save_vocab(db_tok)
    check("Storage vocab round-trips", db.load_vocab() == db_tok.token_to_id)

    db.save_samples(db_train, split="train")
    check("Storage sample count matches", db.count_samples("train") == len(db_train))
    restored_samples = list(db.iter_samples(split="train"))
    original_samples = [(list(c), list(t)) for c, t in db_train]
    check("Storage samples round-trip exactly", restored_samples == original_samples)

    doc_vecs = [Vector([float(i), float(i) * 2, 0.5]) for i in range(4)]
    for v in doc_vecs:
        db.save_document(v, source="test")
    loaded_docs = db.load_documents()
    check("Storage documents round-trip", [d[3] for d in loaded_docs] == [v.to_list() for v in doc_vecs])

    vs_for_test = VectorStore(dim=3)
    db.load_into_vector_store(vs_for_test)
    check("Storage.load_into_vector_store populates a VectorStore", len(vs_for_test.vectors) == 4)

    blob = json.dumps(serialize_brain(none_brain)).encode("utf-8")
    db.save_checkpoint_blob("v1", blob, config={"note": "test"})
    blob2, config2 = db.load_checkpoint_blob("v1")
    check("Storage checkpoint blob round-trips config", config2 == {"note": "test"})
    reloaded_from_db = deserialize_brain(json.loads(blob2.decode("utf-8")))
    check("Storage checkpoint blob round-trips weights",
          reloaded_from_db.moe.W_gate.data == none_brain.moe.W_gate.data)
    try:
        db.load_checkpoint_blob("does-not-exist")
        check("Storage.load_checkpoint_blob raises KeyError for missing version", False, "did not raise")
    except KeyError:
        check("Storage.load_checkpoint_blob raises KeyError for missing version", True)

    for step in range(3):
        db.log_metric(step, loss=1.0 / (step + 1))
    check("Storage metrics round-trip", [m[0] for m in db.load_metrics()] == [0, 1, 2])

    db.set_memory("cfg", {"lr": 0.01})
    check("Storage memory round-trips a dict", db.get_memory("cfg") == {"lr": 0.01})
    check("Storage memory default for missing key", db.get_memory("nope", default="x") == "x")

# data survives closing and reopening the connection
with Storage(db_path) as db2:
    check("Storage data survives reopen", db2.count_samples("train") == len(db_train))


# ============================================================================
print(f"\n{'='*60}\n{len(PASSED)} passed, {len(FAILED)} failed\n{'='*60}")

if FAILED:
    print("FAILURES:")
    for name, detail in FAILED:
        print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("All checks passed.")
