"""
train_on_text.py
End-to-end example: raw text -> BPE tokenizer -> dataset (train/val/test split) ->
FractalBrain training (in batches, via a real optimizer) -> greedy generation.

This is the thing orchestrator.py (synthetic random tokens, for exercising
distillation/TurboQuant/PCA) deliberately doesn't do: train on real text through a real
tokenizer, with an honest train/val split (evaluate() is used for val/test specifically
so it doesn't secretly train on them -- see core.FractalBrain.evaluate()'s docstring).

Run from the directory that contains both this file and the fractal_brain/ folder:
    python train_on_text.py
"""
from fractal_brain import FractalBrain, set_seed
from fractal_brain.tokenizer import BPETokenizer
from fractal_brain.dataset import TextDataset
from fractal_brain.optimizer import Adam
from fractal_brain.checkpoint import serialize_brain, deserialize_brain

# A small, repetitive corpus on purpose: with no real dataset/tokenizer/GPU behind it,
# this is meant to demonstrate the pipeline works end-to-end and that the model can
# pick up on *some* structure in a short demo, not to produce fluent text.
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day in the warm sun",
    "the quick fox runs through the green forest",
    "a lazy fox and a quick dog play in the forest",
    "the dog and the fox are quick and lazy friends",
    "the brown fox and the lazy dog like the warm sun",
    "a quick dog jumps over the lazy fox in the forest",
    "the green forest is home to the fox and the dog",
]


def main():
    set_seed(0)

    # 1. Tokenizer: train a small BPE vocabulary directly on the corpus
    tokenizer = BPETokenizer(lowercase=True)
    tokenizer.train(CORPUS, vocab_size=100)
    print(f"tokenizer: {tokenizer.vocab_size} tokens, {len(tokenizer.merges)} merges learned")

    # 2. Dataset: sliding-window next-token examples, split into train/val/test
    dataset = TextDataset(tokenizer, CORPUS, context_length=4, stride=1)
    train, val, test = dataset.split(train_frac=0.75, val_frac=0.15, seed=0)
    print(f"dataset: {len(dataset)} examples -> {len(train)} train / {len(val)} val / {len(test)} test")

    # 3. A small FractalBrain sized to the tiny vocabulary. Passing an Adam optimizer
    # here is optional -- omit output_optimizer entirely for the original plain-SGD
    # behavior, or pass output_lr_scheduler=... / grad_clip_norm=... for more control;
    # see optimizer.py.
    brain = FractalBrain(vocab_size=tokenizer.vocab_size, d_model=32, num_experts=4,
                         num_heads=2, d_ff=64, num_layers=1,
                         num_markov_nodes=4, markov_states=3, max_level=2,
                         output_optimizer=Adam(lr=0.01))

    # 4. Train in batches (gradients averaged over each batch before one optimizer
    # step, rather than reacting to every single example -- see train_batch()'s
    # docstring), checking validation loss each epoch with evaluate() (read-only --
    # does NOT train on the validation examples; see FractalBrain.evaluate())
    #
    # A corpus this small (8 sentences) overfits fast: train_loss keeps dropping
    # while val_loss starts rising after the first couple of epochs. That's real
    # and worth seeing, so the full curve is still printed below -- but training
    # to the last epoch and generating from *that* state was needlessly using
    # the most-overfit weights available. Instead, keep an in-memory snapshot
    # (via serialize_brain(), no disk I/O) whenever val_loss improves, and
    # restore the best one afterwards. See CHANGELOG.
    epochs = 25
    batch_size = 8
    best_val_loss = float('inf')
    best_epoch = 0
    best_snapshot = None
    for epoch in range(epochs):
        train.shuffle(seed=epoch)
        train_loss = 0.0
        n_batches = 0
        for batch in train.batches(batch_size=batch_size):
            _, loss = brain.train_batch(batch)
            train_loss += loss
            n_batches += 1
        train_loss /= n_batches

        val_loss = 0.0
        for context, target in val:
            _, loss = brain.evaluate(context, target)
            val_loss += loss
        val_loss /= max(1, len(val))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_snapshot = serialize_brain(brain)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:2d}/{epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    print(f"\nbest val_loss={best_val_loss:.4f} at epoch {best_epoch} "
          f"(final epoch val_loss={val_loss:.4f}) -- restoring that checkpoint")
    if best_snapshot is not None:
        brain = deserialize_brain(best_snapshot)

    # 5. Held-out test loss (also via evaluate(), also not trained on) -- now
    # computed from the best-validation checkpoint rather than the final,
    # more-overfit one.
    test_loss = sum(brain.evaluate(c, t)[1] for c, t in test) / max(1, len(test))
    print(f"final test_loss={test_loss:.4f}")

    # 6. Greedy generation from a short prompt
    prompt_text = "the quick brown"
    context = tokenizer.encode(prompt_text)[-4:]
    generated = list(context)
    for _ in range(8):
        probs = brain.sample(context, temperature=0.7)
        next_id = max(range(len(probs)), key=lambda i: probs[i])
        generated.append(next_id)
        context = generated[-4:]
    print(f"\nprompt: {prompt_text!r}")
    print(f"generated (greedy, temp=0.7): {tokenizer.decode(generated)!r}")


if __name__ == "__main__":
    main()
