"""
Makemore Part 1: Bigram Language Model

A character-level bigram language model trained on names using a simple neural network.
This implementation uses a single linear layer with softmax to predict the next character
given the current character.

Based on Andrej Karpathy's "Neural Networks: Zero to Hero" lecture series.
"""

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_words(filepath: str) -> list[str]:
    """
    Load words from a text file, one word per line.

    Args:
        filepath: Path to the text file containing words.

    Returns:
        List of words (strings) from the file.
    """
    return Path(filepath).read_text().splitlines()


# -----------------------------------------------------------------------------
# Character Mappings
# -----------------------------------------------------------------------------

def build_char_mappings(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build character-to-index and index-to-character mappings.

    Uses '.' as the special start/end token (index 0), and assigns
    indices 1-26 to letters a-z.

    Args:
        words: List of words to extract characters from.

    Returns:
        Tuple of (stoi, itos) where:
            - stoi: dict mapping characters to indices
            - itos: dict mapping indices to characters
    """
    chars = sorted(set(''.join(words)))
    stoi = {char: idx + 1 for idx, char in enumerate(chars)}
    stoi['.'] = 0  # Special start/end token
    itos = {idx: char for char, idx in stoi.items()}
    return stoi, itos


# -----------------------------------------------------------------------------
# Dataset Creation
# -----------------------------------------------------------------------------

def create_bigram_dataset(
    words: list[str],
    stoi: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create training dataset of bigrams from words.

    For each word, creates pairs of (current_char, next_char) where the
    word is padded with '.' at start and end.

    Example: "emma" -> [('.', 'e'), ('e', 'm'), ('m', 'm'), ('m', 'a'), ('a', '.')]

    Args:
        words: List of words to create bigrams from.
        stoi: Character to index mapping.

    Returns:
        Tuple of (xs, ys) tensors where:
            - xs: Input character indices
            - ys: Target (next) character indices
    """
    xs, ys = [], []

    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:], strict=True):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    return torch.tensor(xs), torch.tensor(ys)


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

def initialize_weights(
    num_classes: int = 27,
    seed: int = 2147483647
) -> torch.Tensor:
    """
    Initialize the weight matrix for the bigram model.

    Args:
        num_classes: Number of unique characters (default 27 for a-z + '.').
        seed: Random seed for reproducibility.

    Returns:
        Weight matrix of shape (num_classes, num_classes) with requires_grad=True.
    """
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn((num_classes, num_classes), generator=generator, requires_grad=True)
    return W


# -----------------------------------------------------------------------------
# Forward Pass
# -----------------------------------------------------------------------------

def forward(
    xs: torch.Tensor,
    W: torch.Tensor,
    num_classes: int = 27
) -> torch.Tensor:
    """
    Perform forward pass through the model.

    Computes probabilities for next character using:
    1. One-hot encoding of inputs
    2. Matrix multiplication with weights (logits)
    3. Softmax to get probabilities

    Args:
        xs: Input character indices tensor.
        W: Weight matrix.
        num_classes: Number of unique characters.

    Returns:
        Probability distribution over next characters, shape (batch_size, num_classes).
    """
    xenc = F.one_hot(xs, num_classes=num_classes).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)
    return probs


# -----------------------------------------------------------------------------
# Loss Computation
# -----------------------------------------------------------------------------

def compute_loss(
    probs: torch.Tensor,
    ys: torch.Tensor,
    W: torch.Tensor,
    regularization: float = 0.01
) -> torch.Tensor:
    """
    Compute negative log-likelihood loss with L2 regularization.

    Args:
        probs: Predicted probabilities, shape (batch_size, num_classes).
        ys: Target indices tensor.
        W: Weight matrix (for regularization).
        regularization: L2 regularization strength.

    Returns:
        Scalar loss value.
    """
    num_examples = ys.shape[0]
    log_probs = probs[torch.arange(num_examples), ys].log()
    nll = -log_probs.mean()
    reg_loss = regularization * (W ** 2).mean()
    return nll + reg_loss


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    xs: torch.Tensor,
    ys: torch.Tensor,
    W: torch.Tensor,
    num_iterations: int = 100,
    learning_rate: float = 50.0,
    regularization: float = 0.01,
    print_every: int = 10
) -> torch.Tensor:
    """
    Train the bigram model using gradient descent.

    Args:
        xs: Input character indices.
        ys: Target character indices.
        W: Weight matrix to optimize.
        num_iterations: Number of training iterations.
        learning_rate: Learning rate for gradient descent.
        regularization: L2 regularization strength.
        print_every: Print loss every N iterations (0 to disable).

    Returns:
        The trained weight matrix.
    """
    for iteration in range(num_iterations):
        # Forward pass
        probs = forward(xs, W)
        loss = compute_loss(probs, ys, W, regularization)

        if print_every > 0 and iteration % print_every == 0:
            print(f"Iteration {iteration:4d}: loss = {loss.item():.4f}")

        # Backward pass
        W.grad = None
        loss.backward()

        # Update weights
        if W.grad is not None:
            W.data -= learning_rate * W.grad

    # Print final loss
    if print_every > 0:
        probs = forward(xs, W)
        final_loss = compute_loss(probs, ys, W, regularization)
        print(f"Final loss: {final_loss.item():.4f}")

    return W


# -----------------------------------------------------------------------------
# Name Generation
# -----------------------------------------------------------------------------

def generate_name(
    W: torch.Tensor,
    itos: dict[int, str],
    seed: int | None = None,
    num_classes: int = 27
) -> str:
    """
    Generate a single name by sampling from the trained model.

    Args:
        W: Trained weight matrix.
        itos: Index to character mapping.
        seed: Optional random seed for reproducibility.
        num_classes: Number of unique characters.

    Returns:
        Generated name string (without the ending '.').
    """
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    out = []
    ix = 0  # Start with '.'

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=num_classes).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)

        ix = int(torch.multinomial(
            probs,
            num_samples=1,
            replacement=True,
            generator=generator
        ).item())

        if ix == 0:  # End token
            break
        out.append(itos[ix])

    return ''.join(out)


def generate_names(
    W: torch.Tensor,
    itos: dict[int, str],
    num_names: int = 5,
    seed: int = 2147483647
) -> list[str]:
    """
    Generate multiple names from the trained model.

    Args:
        W: Trained weight matrix.
        itos: Index to character mapping.
        num_names: Number of names to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of generated name strings.
    """
    generator = torch.Generator().manual_seed(seed)
    names = []

    for _ in range(num_names):
        out = []
        ix = 0

        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)

            ix = int(torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=generator
            ).item())

            if ix == 0:
                break
            out.append(itos[ix])

        names.append(''.join(out))

    return names


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Main function to run the bigram model training and generation."""

    # Configuration
    data_file = "names.txt"
    num_iterations = 200
    learning_rate = 50.0
    regularization = 0.01
    num_names_to_generate = 10
    seed = 2147483647

    print("=" * 60)
    print("Makemore Part 1: Bigram Language Model")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from '{data_file}'...")
    words = load_words(data_file)
    print(f"Loaded {len(words)} words")
    print(f"Sample words: {words[:5]}")

    # Build character mappings
    stoi, itos = build_char_mappings(words)
    num_classes = len(stoi)
    print(f"\nVocabulary size: {num_classes} characters")

    # Create dataset
    print("\nCreating bigram dataset...")
    xs, ys = create_bigram_dataset(words, stoi)
    print(f"Number of bigram examples: {xs.shape[0]}")

    # Initialize model
    print("\nInitializing model...")
    W = initialize_weights(num_classes=num_classes, seed=seed)

    # Train
    print(f"\nTraining for {num_iterations} iterations...")
    print("-" * 40)
    W = train(
        xs, ys, W,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        print_every=20
    )

    # Generate names
    print("\n" + "=" * 60)
    print(f"Generated Names ({num_names_to_generate}):")
    print("=" * 60)
    names = generate_names(W, itos, num_names=num_names_to_generate, seed=seed)
    for i, name in enumerate(names, 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
