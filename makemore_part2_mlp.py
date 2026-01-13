"""
Makemore Part 2: Multi-Layer Perceptron (MLP) Language Model

A character-level language model using an MLP architecture based on the
Bengio et al. 2003 paper "A Neural Probabilistic Language Model".

The model uses:
- Character embeddings (lookup table)
- A hidden layer with tanh activation
- Softmax output layer for next character prediction

Given a context of N previous characters, the model predicts the probability
distribution over the next character.

Based on Andrej Karpathy's "Neural Networks: Zero to Hero" lecture series.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class MLPConfig:
    """Configuration for the MLP language model."""

    block_size: int = 3  # Context length (number of characters to predict from)
    embed_dim: int = 10  # Embedding dimension for each character
    hidden_dim: int = 200  # Hidden layer dimension
    vocab_size: int = 27  # Number of characters (a-z + '.')

    # Training hyperparameters
    batch_size: int = 32
    num_iterations: int = 200000
    learning_rate_high: float = 0.1  # Initial learning rate
    learning_rate_low: float = 0.01  # Learning rate after decay
    lr_decay_step: int = 100000  # Step to decay learning rate

    # Random seeds
    seed: int = 2147483647
    data_seed: int = 42


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
    chars = sorted(set("".join(words)))
    stoi = {char: idx + 1 for idx, char in enumerate(chars)}
    stoi["."] = 0  # Special start/end token
    itos = {idx: char for char, idx in stoi.items()}
    return stoi, itos


# -----------------------------------------------------------------------------
# Dataset Creation
# -----------------------------------------------------------------------------


def build_dataset(
    words: list[str],
    stoi: dict[str, int],
    block_size: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a dataset of context windows and target characters.

    For each word, creates (context, target) pairs where context is the
    previous block_size characters and target is the next character.

    Example with block_size=3:
        "emma" -> [
            (['.', '.', '.'], 'e'),
            (['.', '.', 'e'], 'm'),
            (['.', 'e', 'm'], 'm'),
            (['e', 'm', 'm'], 'a'),
            (['m', 'm', 'a'], '.'),
        ]

    Args:
        words: List of words to create dataset from.
        stoi: Character to index mapping.
        block_size: Number of context characters.

    Returns:
        Tuple of (X, Y) tensors where:
            - X: Input context indices, shape (num_examples, block_size)
            - Y: Target character indices, shape (num_examples,)
    """
    X, Y = [], []

    for word in words:
        context = [0] * block_size  # Initialize with padding tokens
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # Slide window

    return torch.tensor(X), torch.tensor(Y)


def split_data(
    words: list[str],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split words into train, dev, and test sets.

    Args:
        words: List of words to split.
        train_ratio: Fraction of data for training.
        dev_ratio: Fraction of data for development/validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_words, dev_words, test_words).
    """
    words_copy = words.copy()
    random.seed(seed)
    random.shuffle(words_copy)

    n1 = int(train_ratio * len(words_copy))
    n2 = int((train_ratio + dev_ratio) * len(words_copy))

    return words_copy[:n1], words_copy[n1:n2], words_copy[n2:]


def create_datasets(
    words: list[str],
    stoi: dict[str, int],
    config: MLPConfig,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """
    Create train, dev, and test datasets from words.

    Args:
        words: List of words.
        stoi: Character to index mapping.
        config: Model configuration.

    Returns:
        Tuple of ((Xtr, Ytr), (Xdev, Ydev), (Xte, Yte)) tensor pairs.
    """
    train_words, dev_words, test_words = split_data(words, seed=config.data_seed)

    Xtr, Ytr = build_dataset(train_words, stoi, config.block_size)
    Xdev, Ydev = build_dataset(dev_words, stoi, config.block_size)
    Xte, Yte = build_dataset(test_words, stoi, config.block_size)

    return (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte)


# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------


@dataclass
class MLPModel:
    """Container for MLP model parameters."""

    C: torch.Tensor  # Character embeddings (vocab_size, embed_dim)
    W1: torch.Tensor  # Hidden layer weights (block_size * embed_dim, hidden_dim)
    b1: torch.Tensor  # Hidden layer bias (hidden_dim,)
    W2: torch.Tensor  # Output layer weights (hidden_dim, vocab_size)
    b2: torch.Tensor  # Output layer bias (vocab_size,)

    def parameters(self) -> list[torch.Tensor]:
        """Return list of all model parameters."""
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def num_parameters(self) -> int:
        """Return total number of model parameters."""
        return sum(p.nelement() for p in self.parameters())


def initialize_model(config: MLPConfig) -> MLPModel:
    """
    Initialize the MLP model parameters.

    The model consists of:
    - C: Character embedding matrix (vocab_size, embed_dim)
    - W1, b1: Hidden layer (input_dim, hidden_dim)
    - W2, b2: Output layer (hidden_dim, vocab_size)

    Args:
        config: Model configuration.

    Returns:
        Initialized MLPModel with requires_grad=True on all parameters.
    """
    g = torch.Generator().manual_seed(config.seed)

    input_dim = config.block_size * config.embed_dim

    C = torch.randn((config.vocab_size, config.embed_dim), generator=g)
    W1 = torch.randn((input_dim, config.hidden_dim), generator=g)
    b1 = torch.randn(config.hidden_dim, generator=g)
    W2 = torch.randn((config.hidden_dim, config.vocab_size), generator=g)
    b2 = torch.randn(config.vocab_size, generator=g)

    model = MLPModel(C=C, W1=W1, b1=b1, W2=W2, b2=b2)

    # Enable gradients for all parameters
    for p in model.parameters():
        p.requires_grad = True

    return model


# -----------------------------------------------------------------------------
# Forward Pass
# -----------------------------------------------------------------------------


def forward(
    model: MLPModel,
    X: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Perform forward pass through the MLP.

    The forward pass:
    1. Look up character embeddings from C
    2. Flatten embeddings for each context window
    3. Apply hidden layer: tanh(x @ W1 + b1)
    4. Apply output layer: h @ W2 + b2 (logits)

    Args:
        model: The MLP model parameters.
        X: Input context indices, shape (batch_size, block_size).
        block_size: Context length.

    Returns:
        Logits tensor, shape (batch_size, vocab_size).
    """
    # Embed characters: (batch_size, block_size) -> (batch_size, block_size, embed_dim)
    emb = model.C[X]

    # Flatten: (batch_size, block_size, embed_dim) -> (batch_size, block_size * embed_dim)
    emb_flat = emb.view(emb.shape[0], -1)

    # Hidden layer with tanh activation
    h = torch.tanh(emb_flat @ model.W1 + model.b1)

    # Output logits
    logits = h @ model.W2 + model.b2

    return logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        logits: Model output logits, shape (batch_size, vocab_size).
        targets: Target indices, shape (batch_size,).

    Returns:
        Scalar loss value.
    """
    return F.cross_entropy(logits, targets)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def get_learning_rate(step: int, config: MLPConfig) -> float:
    """
    Get learning rate for current training step.

    Uses a simple step decay schedule.

    Args:
        step: Current training step.
        config: Model configuration.

    Returns:
        Learning rate for this step.
    """
    if step < config.lr_decay_step:
        return config.learning_rate_high
    return config.learning_rate_low


def train_step(
    model: MLPModel,
    X: torch.Tensor,
    Y: torch.Tensor,
    learning_rate: float,
    block_size: int,
) -> float:
    """
    Perform a single training step.

    Args:
        model: The MLP model.
        X: Batch of input contexts.
        Y: Batch of target characters.
        learning_rate: Current learning rate.
        block_size: Context length.

    Returns:
        Loss value for this step.
    """
    # Forward pass
    logits = forward(model, X, block_size)
    loss = compute_loss(logits, Y)

    # Backward pass
    for p in model.parameters():
        p.grad = None
    loss.backward()

    # Update parameters
    for p in model.parameters():
        if p.grad is not None:
            p.data -= learning_rate * p.grad

    return loss.item()


def train(
    model: MLPModel,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    config: MLPConfig,
    print_every: int = 10000,
) -> list[float]:
    """
    Train the MLP model.

    Args:
        model: The MLP model to train.
        Xtr: Training inputs.
        Ytr: Training targets.
        config: Model configuration.
        print_every: Print loss every N iterations (0 to disable).

    Returns:
        List of loss values during training.
    """
    losses = []

    for step in range(config.num_iterations):
        # Sample minibatch
        ix = torch.randint(0, Xtr.shape[0], (config.batch_size,))
        X_batch = Xtr[ix]
        Y_batch = Ytr[ix]

        # Get learning rate for this step
        lr = get_learning_rate(step, config)

        # Training step
        loss = train_step(model, X_batch, Y_batch, lr, config.block_size)
        losses.append(loss)

        if print_every > 0 and step % print_every == 0:
            print(f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr}")

    return losses


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate(
    model: MLPModel,
    X: torch.Tensor,
    Y: torch.Tensor,
    block_size: int,
) -> float:
    """
    Evaluate the model on a dataset.

    Args:
        model: The trained MLP model.
        X: Input contexts.
        Y: Target characters.
        block_size: Context length.

    Returns:
        Average cross-entropy loss.
    """
    with torch.no_grad():
        logits = forward(model, X, block_size)
        loss = compute_loss(logits, Y)
    return loss.item()


# -----------------------------------------------------------------------------
# Name Generation
# -----------------------------------------------------------------------------


def sample_name(
    model: MLPModel,
    itos: dict[int, str],
    block_size: int,
    generator: torch.Generator | None = None,
) -> str:
    """
    Generate a single name by sampling from the model.

    Args:
        model: Trained MLP model.
        itos: Index to character mapping.
        block_size: Context length.
        generator: Optional random generator for reproducibility.

    Returns:
        Generated name string (without the ending '.').
    """
    out = []
    context = [0] * block_size  # Start with padding tokens

    while True:
        # Forward pass
        X = torch.tensor([context])
        logits = forward(model, X, block_size)
        probs = F.softmax(logits, dim=1)

        # Sample next character using weighted random sampling.
        # torch.multinomial selects an index proportionally to the probabilities:
        # - Instead of always picking the highest-probability character (deterministic),
        #   it randomly samples based on the probability distribution.
        # - E.g., if probs = [0.1, 0.7, 0.2], index 1 has 70% chance of being selected.
        # - This introduces controlled randomness, making generated names varied
        #   while still respecting the learned distribution.
        ix = int(torch.multinomial(probs, num_samples=1, generator=generator).item())

        # Update context
        context = context[1:] + [ix]

        if ix == 0:  # End token
            break
        out.append(itos[ix])

    return "".join(out)


def generate_names(
    model: MLPModel,
    itos: dict[int, str],
    block_size: int,
    num_names: int = 20,
    seed: int = 2147483647,
) -> list[str]:
    """
    Generate multiple names from the trained model.

    Args:
        model: Trained MLP model.
        itos: Index to character mapping.
        block_size: Context length.
        num_names: Number of names to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of generated name strings.
    """
    g = torch.Generator().manual_seed(seed + 10)  # Offset seed as in original

    names = []
    for _ in range(num_names):
        name = sample_name(model, itos, block_size, generator=g)
        names.append(name)

    return names


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    """Main function to run the MLP model training and generation."""

    # Configuration
    config = MLPConfig()
    data_file = "names.txt"

    print("=" * 70)
    print("Makemore Part 2: Multi-Layer Perceptron (MLP) Language Model")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from '{data_file}'...")
    words = load_words(data_file)
    print(f"Loaded {len(words)} words")
    print(f"Sample words: {words[:5]}")

    # Build character mappings
    stoi, itos = build_char_mappings(words)
    config.vocab_size = len(stoi)
    print(f"\nVocabulary size: {config.vocab_size} characters")

    # Create datasets
    print("\nCreating datasets...")
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = create_datasets(words, stoi, config)
    print(f"Training set:   {Xtr.shape[0]:,} examples")
    print(f"Development set: {Xdev.shape[0]:,} examples")
    print(f"Test set:       {Xte.shape[0]:,} examples")

    # Initialize model
    print("\nInitializing model...")
    model = initialize_model(config)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"  - Embeddings C:  {model.C.shape}")
    print(f"  - Hidden W1:     {model.W1.shape}")
    print(f"  - Hidden b1:     {model.b1.shape}")
    print(f"  - Output W2:     {model.W2.shape}")
    print(f"  - Output b2:     {model.b2.shape}")

    # Train
    print(f"\nTraining for {config.num_iterations:,} iterations...")
    print("-" * 70)
    train(model, Xtr, Ytr, config, print_every=20000)

    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluation:")
    train_loss = evaluate(model, Xtr, Ytr, config.block_size)
    dev_loss = evaluate(model, Xdev, Ydev, config.block_size)
    test_loss = evaluate(model, Xte, Yte, config.block_size)
    print(f"  Training loss:    {train_loss:.4f}")
    print(f"  Development loss: {dev_loss:.4f}")
    print(f"  Test loss:        {test_loss:.4f}")

    # Generate names
    print("\n" + "=" * 70)
    print("Generated Names:")
    print("=" * 70)
    names = generate_names(model, itos, config.block_size, num_names=20)
    for i, name in enumerate(names, 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
