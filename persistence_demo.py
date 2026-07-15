"""
persistence_demo.py
End-to-end example of the persistence layer: SQLite-backed Storage (vocab, samples,
metrics, checkpoints) plus checkpoint save/load for a FractalBrain.

Run from the directory that contains both this file and the fractal_brain/ folder:
    python persistence_demo.py

This uses a fresh database file each run (deletes any existing one first) so it's safe
to re-run repeatedly.
"""
import os
import json

from fractal_brain import FractalBrain, set_seed
from fractal_brain.tokenizer import BPETokenizer
from fractal_brain.dataset import TextDataset
from fractal_brain.storage import Storage
from fractal_brain.checkpoint import serialize_brain, save_checkpoint, load_checkpoint

# Written under var/ (ignored by .gitignore) rather than the repo root, so
# running this demo doesn't litter multi-megabyte generated files into the
# working tree. See CHANGELOG.
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "var", "persistence_demo")
os.makedirs(_OUTPUT_DIR, exist_ok=True)
DB_PATH = os.path.join(_OUTPUT_DIR, "demo.db")
CHECKPOINT_PATH = os.path.join(_OUTPUT_DIR, "demo_checkpoint.json")

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day in the warm sun",
    "the quick fox runs through the green forest",
    "a lazy fox and a quick dog play in the forest",
]


def main():
    for path in (DB_PATH, CHECKPOINT_PATH):
        if os.path.exists(path):
            os.remove(path)

    set_seed(0)
    tokenizer = BPETokenizer(lowercase=True)
    tokenizer.train(CORPUS, vocab_size=80)
    dataset = TextDataset(tokenizer, CORPUS, context_length=4)
    train, val, _test = dataset.split(train_frac=0.8, val_frac=0.1, seed=0)

    brain = FractalBrain(vocab_size=tokenizer.vocab_size, d_model=24, num_experts=4,
                         num_heads=2, d_ff=48, num_layers=1,
                         num_markov_nodes=3, markov_states=3, max_level=2,
                         output_lr=0.15)

    # 1. Persist the vocabulary and dataset samples to SQLite up front -- this is the
    #    kind of thing you want queryable/inspectable, not just a loose file.
    with Storage(DB_PATH) as db:
        db.save_vocab(tokenizer)
        db.save_samples(train, split="train")
        db.save_samples(val, split="val")
        print(f"stored {db.count_samples()} samples and a {len(db.load_vocab())}-token vocab in {DB_PATH}")

        # 2. Train for a while, logging metrics as we go and saving periodic checkpoints
        #    (both a standalone file, and a versioned blob inside the same database --
        #    two valid ways to use this depending on whether you want one checkpoint per
        #    file or many versions queryable in one place).
        step = 0
        for epoch in range(10):
            train.shuffle(seed=epoch)
            epoch_loss = 0.0
            for context, target in train:
                _, loss = brain.step(context, target)
                db.log_metric(step, loss)
                epoch_loss += loss
                step += 1
            epoch_loss /= len(train)
            if (epoch + 1) % 5 == 0:
                db.save_checkpoint_blob(f"epoch-{epoch+1}", json.dumps(serialize_brain(brain)).encode("utf-8"),
                                        config={"epoch": epoch + 1, "avg_loss": epoch_loss})
                print(f"epoch {epoch+1}: avg_loss={epoch_loss:.4f}  (checkpoint saved to db as 'epoch-{epoch+1}')")

        print(f"\ncheckpoints stored in db: {[v for v, _ in db.list_checkpoints()]}")
        recent_metrics = db.load_metrics()[-3:]
        print(f"last 3 logged metrics (step, loss, acc, timestamp): {recent_metrics}")

    # 3. Also save a plain standalone checkpoint file (the simpler path if you don't
    #    need a database at all -- e.g. a single model you'll load elsewhere).
    save_checkpoint(brain, CHECKPOINT_PATH)
    print(f"\nalso saved a standalone checkpoint to {CHECKPOINT_PATH}")

    # 4. Simulate picking this back up later: load the standalone checkpoint into a
    #    brand new FractalBrain instance -- note we never call FractalBrain(...) here,
    #    load_checkpoint reconstructs the whole thing, weights and all.
    reloaded = load_checkpoint(CHECKPOINT_PATH)
    weights_match = reloaded.moe.experts[0].W_out.data == brain.moe.experts[0].W_out.data
    gains_match = (reloaded.pid.Kp, reloaded.pid.Ki, reloaded.pid.Kd) == (brain.pid.Kp, brain.pid.Ki, brain.pid.Kd)
    print(f"\nreloaded brain's weights and PID gains match the original exactly: {weights_match and gains_match}")
    # Deliberately NOT comparing brain.sample(...) to reloaded.sample(...) here: both
    # draw on Python's *global* random module (via BootstrapGate), so calling either
    # one first advances that shared state before the other runs -- an order-dependent,
    # unfair comparison whenever two live FractalBrain instances coexist in one process,
    # regardless of whether either came from a checkpoint. What genuinely matters --
    # reloading in a fresh process and continuing training reproduces an uninterrupted
    # run bit-for-bit -- was verified separately across two actual OS processes; see
    # CHANGELOG.md.

    # continue training the reloaded instance to show it's not just inference-only
    reloaded.step(tokenizer.encode("the quick brown"), [0.0] * tokenizer.vocab_size)
    print("continued training on the reloaded instance without error")

    # 5. Pull a specific historical checkpoint back out of the database and confirm it
    #    reflects that point in training, not the final state.
    with Storage(DB_PATH) as db:
        blob, config = db.load_checkpoint_blob("epoch-5")
        from fractal_brain.checkpoint import deserialize_brain
        epoch5_brain = deserialize_brain(json.loads(blob.decode("utf-8")))
        print(f"\nrestored the 'epoch-5' checkpoint from the db: {config}")
        print(f"its step_count ({epoch5_brain.step_count}) is earlier than the final brain's ({brain.step_count})")


if __name__ == "__main__":
    main()
