"""
train_on_text.py
End-to-end example: raw text -> BPE tokenizer -> dataset (train/val/test split) ->
FractalBrain training -> greedy generation.

This is the thing orchestrator.py (synthetic random tokens, for exercising
distillation/TurboQuant/PCA) deliberately doesn't do: train on real text through a real
tokenizer, with an honest train/val split (evaluate() is used for val/test specifically
so it doesn't secretly train on them -- see core.FractalBrain.evaluate()'s docstring).

Run from the directory that contains both this file and the fractal_brain/ folder:
    python train_on_text.py
"""
import random
from fractal_brain import FractalBrain, set_seed
from fractal_brain.tokenizer import BPETokenizer
from fractal_brain.dataset import TextDataset

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

    # 3. A small FractalBrain sized to the tiny vocabulary
    brain = FractalBrain(vocab_size=tokenizer.vocab_size, d_model=32, num_experts=4,
                         num_heads=2, d_ff=64, num_layers=1,
                         num_markov_nodes=4, markov_states=3, max_level=2,
                         output_lr=0.15)

    # 4. Train, checking validation loss each epoch with evaluate() (read-only --
    # does NOT train on the validation examples; see FractalBrain.evaluate())
    epochs = 25
    for epoch in range(epochs):
        train.shuffle(seed=epoch)
        train_loss = 0.0
        for context, target in train:
            _, loss = brain.step(context, target)
            train_loss += loss
        train_loss /= len(train)

        val_loss = 0.0
        for context, target in val:
            _, loss = brain.evaluate(context, target)
            val_loss += loss
        val_loss /= max(1, len(val))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:2d}/{epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # 5. Held-out test loss (also via evaluate(), also not trained on)
    test_loss = sum(brain.evaluate(c, t)[1] for c, t in test) / max(1, len(test))
    print(f"\nfinal test_loss={test_loss:.4f}")

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
