"""
Makemore Part 3: Multi-Layer Perceptron with Batch Normalization

A character-level language model using an MLP architecture with Batch Normalization,
following the techniques from the Ioffe & Szegedy 2015 paper "Batch Normalization:
Accelerating Deep Network Training by Reducing Internal Covariate Shift".

Key concepts demonstrated:
- Batch normalization for stabilizing training of deep networks
- Custom PyTorch-like layer classes (Linear, BatchNorm1d, Tanh)
- Kaiming initialization for proper weight scaling
- Deeper networks (5 hidden layers) with improved training dynamics

Based on Andrej Karpathy's "Neural Networks: Zero to Hero" lecture series.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class BatchNormMLPConfig:
    """Configuration for the BatchNorm MLP language model."""

    block_size: int = 3  # Context length (number of characters to predict from)
    embed_dim: int = 10  # Embedding dimension for each character
    hidden_dim: int = 100  # Hidden layer dimension
    num_hidden_layers: int = 5  # Number of hidden layers
    vocab_size: int = 27  # Number of characters (a-z + '.')

    # Training hyperparameters
    batch_size: int = 32
    num_iterations: int = 200000
    learning_rate_high: float = 0.1  # Initial learning rate
    learning_rate_low: float = 0.01  # Learning rate after decay
    lr_decay_step: int = 150000  # Step to decay learning rate

    # Random seeds
    seed: int = 2147483647
    data_seed: int = 42


# -----------------------------------------------------------------------------
# Custom Layer Classes (PyTorch-like API)
# -----------------------------------------------------------------------------


class Linear:
    """
    Linear (fully connected) layer implementing y = x @ W + b.

    Uses Kaiming initialization (1/sqrt(fan_in)) for weights, which helps
    maintain stable activation magnitudes through the network.

    Attributes:
        weight: Weight matrix of shape (fan_in, fan_out).
        bias: Bias vector of shape (fan_out,), or None if bias=False.
        out: Cached output from the last forward pass.
    """

    def __init__(self, fan_in: int, fan_out: int, bias: bool = True, generator: torch.Generator | None = None):
        """
        Initialize the linear layer.

        Args:
            fan_in: Number of input features.
            fan_out: Number of output features.
            bias: Whether to include a bias term.
            generator: Optional random generator for reproducibility.
        """
        # Kaiming initialization: scale by 1/sqrt(fan_in)
        self.weight = torch.randn((fan_in, fan_out), generator=generator) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = x @ W + b."""
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        """Return list of trainable parameters."""
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """
    Batch Normalization layer for 1D inputs.

    Normalizes activations to have zero mean and unit variance within each
    mini-batch during training. Uses running statistics for inference.

    The transformation is:
        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Where gamma (scale) and beta (shift) are learnable parameters.

    Attributes:
        gamma: Learnable scale parameter, shape (dim,).
        beta: Learnable shift parameter, shape (dim,).
        running_mean: Running mean for inference, shape (dim,).
        running_var: Running variance for inference, shape (dim,).
        training: Whether the layer is in training mode.
        out: Cached output from the last forward pass.
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize the BatchNorm1d layer.

        Args:
            dim: Number of features (input dimension).
            eps: Small constant for numerical stability.
            momentum: Momentum for running statistics update.
        """
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Running statistics (updated with momentum during training)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

        self.out: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batch normalization.

        During training: normalizes using batch statistics and updates running stats.
        During inference: normalizes using running statistics.

        Args:
            x: Input tensor of shape (batch_size, dim).

        Returns:
            Normalized tensor of the same shape.
        """
        if self.training:
            xmean = x.mean(0, keepdim=True)  # Batch mean
            xvar = x.var(0, keepdim=True)  # Batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # Normalize to unit variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)

        # Scale and shift
        self.out = self.gamma * xhat + self.beta

        # Update running statistics during training
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self) -> list[torch.Tensor]:
        """Return list of trainable parameters (gamma and beta)."""
        return [self.gamma, self.beta]


class Tanh:
    """
    Tanh activation layer.

    Applies element-wise tanh: y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Attributes:
        out: Cached output from the last forward pass.
    """

    def __init__(self):
        """Initialize the Tanh layer."""
        self.out: torch.Tensor | None = None
        self.training = True  # For API compatibility

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: applies tanh activation."""
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        """Return empty list (no trainable parameters)."""
        return []


# -----------------------------------------------------------------------------
# Model Container
# -----------------------------------------------------------------------------


@dataclass
class BatchNormMLP:
    """
    Container for the BatchNorm MLP model.

    The model architecture consists of:
    - Character embeddings (C)
    - Multiple hidden layers, each with: Linear -> BatchNorm1d -> Tanh
    - Output layer: Linear -> BatchNorm1d (no activation)

    Attributes:
        C: Character embedding matrix of shape (vocab_size, embed_dim).
        layers: List of layer objects (Linear, BatchNorm1d, Tanh).
        config: Model configuration.
    """

    C: torch.Tensor
    layers: list = field(default_factory=list)
    config: BatchNormMLPConfig = field(default_factory=BatchNormMLPConfig)

    def parameters(self) -> list[torch.Tensor]:
        """Return list of all trainable parameters."""
        return [self.C] + [p for layer in self.layers for p in layer.parameters()]

    def num_parameters(self) -> int:
        """Return total number of model parameters."""
        return sum(p.nelement() for p in self.parameters())

    def train(self):
        """Set model to training mode."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        """Set model to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False


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
    config: BatchNormMLPConfig,
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


def build_layers(config: BatchNormMLPConfig, generator: torch.Generator) -> list:
    """
    Build the network layers.

    Creates a deep network with the following structure:
    - Input: Linear(embed_dim * block_size, hidden_dim) -> BatchNorm1d -> Tanh
    - Hidden layers: Linear(hidden_dim, hidden_dim) -> BatchNorm1d -> Tanh (repeated)
    - Output: Linear(hidden_dim, vocab_size) -> BatchNorm1d (no activation)

    Args:
        config: Model configuration.
        generator: Random generator for reproducibility.

    Returns:
        List of layer objects.
    """
    input_dim = config.embed_dim * config.block_size
    layers = []

    # First hidden layer (from embeddings)
    layers.extend([
        Linear(input_dim, config.hidden_dim, bias=False, generator=generator),
        BatchNorm1d(config.hidden_dim),
        Tanh(),
    ])

    # Additional hidden layers
    for _ in range(config.num_hidden_layers - 1):
        layers.extend([
            Linear(config.hidden_dim, config.hidden_dim, bias=False, generator=generator),
            BatchNorm1d(config.hidden_dim),
            Tanh(),
        ])

    # Output layer (no Tanh activation - logits go to softmax)
    layers.extend([
        Linear(config.hidden_dim, config.vocab_size, bias=False, generator=generator),
        BatchNorm1d(config.vocab_size),
    ])

    return layers


def initialize_model(config: BatchNormMLPConfig) -> BatchNormMLP:
    """
    Initialize the BatchNorm MLP model.

    Applies special initialization:
    - Last layer gamma scaled by 0.1 to make initial outputs less confident
    - All parameters have requires_grad=True

    Args:
        config: Model configuration.

    Returns:
        Initialized BatchNormMLP model.
    """
    g = torch.Generator().manual_seed(config.seed)

    # Character embeddings
    C = torch.randn((config.vocab_size, config.embed_dim), generator=g)

    # Build layers
    layers = build_layers(config, g)

    # Make last layer output less confident
    with torch.no_grad():
        # Last layer is BatchNorm1d - scale its gamma down
        layers[-1].gamma *= 0.1

    model = BatchNormMLP(C=C, layers=layers, config=config)

    # Enable gradients for all parameters
    for p in model.parameters():
        p.requires_grad = True

    return model


# -----------------------------------------------------------------------------
# Forward Pass
# -----------------------------------------------------------------------------


def forward(model: BatchNormMLP, X: torch.Tensor) -> torch.Tensor:
    """
    Perform forward pass through the network.

    The forward pass:
    1. Look up character embeddings from C
    2. Flatten embeddings for each context window
    3. Pass through all layers sequentially

    Args:
        model: The BatchNorm MLP model.
        X: Input context indices, shape (batch_size, block_size).

    Returns:
        Logits tensor, shape (batch_size, vocab_size).
    """
    # Embed characters
    emb = model.C[X]  # (batch_size, block_size, embed_dim)

    # Flatten embeddings
    x = emb.view(emb.shape[0], -1)  # (batch_size, block_size * embed_dim)

    # Pass through all layers
    for layer in model.layers:
        x = layer(x)

    return x


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


def get_learning_rate(step: int, config: BatchNormMLPConfig) -> float:
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
    model: BatchNormMLP,
    X: torch.Tensor,
    Y: torch.Tensor,
    learning_rate: float,
) -> float:
    """
    Perform a single training step.

    Args:
        model: The BatchNorm MLP model.
        X: Batch of input contexts.
        Y: Batch of target characters.
        learning_rate: Current learning rate.

    Returns:
        Loss value for this step.
    """
    # Forward pass
    logits = forward(model, X)
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
    model: BatchNormMLP,
    Xtr: torch.Tensor,
    Ytr: torch.Tensor,
    config: BatchNormMLPConfig,
    generator: torch.Generator | None = None,
    print_every: int = 10000,
) -> list[float]:
    """
    Train the BatchNorm MLP model.

    Args:
        model: The model to train.
        Xtr: Training inputs.
        Ytr: Training targets.
        config: Model configuration.
        generator: Optional random generator for batch sampling.
        print_every: Print loss every N iterations (0 to disable).

    Returns:
        List of loss values during training.
    """
    model.train()
    losses = []

    for step in range(config.num_iterations):
        # Sample minibatch
        ix = torch.randint(0, Xtr.shape[0], (config.batch_size,), generator=generator)
        X_batch = Xtr[ix]
        Y_batch = Ytr[ix]

        # Get learning rate for this step
        lr = get_learning_rate(step, config)

        # Training step
        loss = train_step(model, X_batch, Y_batch, lr)
        losses.append(loss)

        if print_every > 0 and step % print_every == 0:
            print(f"Step {step:7d}/{config.num_iterations:7d} | Loss: {loss:.4f} | LR: {lr}")

    return losses


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: BatchNormMLP, X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Evaluate the model on a dataset.

    Uses running statistics for BatchNorm (eval mode).

    Args:
        model: The trained BatchNorm MLP model.
        X: Input contexts.
        Y: Target characters.

    Returns:
        Average cross-entropy loss.
    """
    model.eval()
    logits = forward(model, X)
    loss = compute_loss(logits, Y)
    return loss.item()


# -----------------------------------------------------------------------------
# Name Generation
# -----------------------------------------------------------------------------


@torch.no_grad()
def sample_name(
    model: BatchNormMLP,
    itos: dict[int, str],
    generator: torch.Generator | None = None,
) -> str:
    """
    Generate a single name by sampling from the model.

    Args:
        model: Trained BatchNorm MLP model (should be in eval mode).
        itos: Index to character mapping.
        generator: Optional random generator for reproducibility.

    Returns:
        Generated name string (without the ending '.').
    """
    model.eval()
    block_size = model.config.block_size

    out = []
    context = [0] * block_size  # Start with padding tokens

    while True:
        # Forward pass
        X = torch.tensor([context])
        logits = forward(model, X)
        probs = F.softmax(logits, dim=1)

        # Sample next character
        ix = int(torch.multinomial(probs, num_samples=1, generator=generator).item())

        # Update context
        context = context[1:] + [ix]

        if ix == 0:  # End token
            break
        out.append(itos[ix])

    return "".join(out)


def generate_names(
    model: BatchNormMLP,
    itos: dict[int, str],
    num_names: int = 20,
    seed: int = 2147483647,
) -> list[str]:
    """
    Generate multiple names from the trained model.

    Args:
        model: Trained BatchNorm MLP model.
        itos: Index to character mapping.
        num_names: Number of names to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of generated name strings.
    """
    g = torch.Generator().manual_seed(seed + 10)  # Offset seed as in original

    names = []
    for _ in range(num_names):
        name = sample_name(model, itos, generator=g)
        names.append(name)

    return names


# -----------------------------------------------------------------------------
# Diagnostics (for debugging training)
# -----------------------------------------------------------------------------


def print_layer_statistics(model: BatchNormMLP):
    """
    Print activation statistics for each layer.

    Useful for debugging training dynamics and checking for
    vanishing/exploding activations or gradients.

    Args:
        model: The model after a forward pass (layers have cached outputs).
    """
    print("\nActivation Statistics:")
    print("-" * 60)
    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, Tanh) and layer.out is not None:
            t = layer.out
            saturated = (t.abs() > 0.97).float().mean() * 100
            print(
                f"Layer {i:2d} ({layer.__class__.__name__:10s}): "
                f"mean {t.mean():+.4f}, std {t.std():.4f}, "
                f"saturated: {saturated:.2f}%"
            )


def print_gradient_statistics(model: BatchNormMLP):
    """
    Print gradient statistics for model parameters.

    The grad:data ratio should be around 1e-3 for healthy training.

    Args:
        model: The model after a backward pass.
    """
    print("\nGradient Statistics:")
    print("-" * 70)
    for i, p in enumerate(model.parameters()):
        if p.grad is not None and p.ndim == 2:
            t = p.grad
            ratio = t.std() / p.std()
            print(
                f"Param {i:2d} {str(tuple(p.shape)):15s} | "
                f"grad mean {t.mean():+.6f} | "
                f"grad std {t.std():.2e} | "
                f"grad:data ratio {ratio:.2e}"
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    """Main function to run the BatchNorm MLP model training and generation."""

    # Configuration
    config = BatchNormMLPConfig()
    data_file = "names.txt"

    print("=" * 70)
    print("Makemore Part 3: MLP with Batch Normalization")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from '{data_file}'...")
    words = load_words(data_file)
    print(f"Loaded {len(words):,} words")
    print(f"Sample words: {words[:5]}")

    # Build character mappings
    stoi, itos = build_char_mappings(words)
    config.vocab_size = len(stoi)
    print(f"\nVocabulary size: {config.vocab_size} characters")

    # Create datasets
    print("\nCreating datasets...")
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = create_datasets(words, stoi, config)
    print(f"Training set:    {Xtr.shape[0]:,} examples")
    print(f"Development set: {Xdev.shape[0]:,} examples")
    print(f"Test set:        {Xte.shape[0]:,} examples")

    # Initialize model
    print("\nInitializing model...")
    model = initialize_model(config)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"  - Embeddings C: {model.C.shape}")
    print(f"  - Hidden layers: {config.num_hidden_layers}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Total layers: {len(model.layers)}")

    # Print layer structure
    print("\nLayer structure:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Linear):
            print(f"  {i}: Linear{tuple(layer.weight.shape)}")
        elif isinstance(layer, BatchNorm1d):
            print(f"  {i}: BatchNorm1d({len(layer.gamma)})")
        elif isinstance(layer, Tanh):
            print(f"  {i}: Tanh()")

    # Create generator for reproducible batch sampling
    g = torch.Generator().manual_seed(config.seed)

    # Train
    print(f"\nTraining for {config.num_iterations:,} iterations...")
    print("-" * 70)
    train(model, Xtr, Ytr, config, generator=g, print_every=20000)

    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluation:")
    train_loss = evaluate(model, Xtr, Ytr)
    dev_loss = evaluate(model, Xdev, Ydev)
    test_loss = evaluate(model, Xte, Yte)
    print(f"  Training loss:    {train_loss:.4f}")
    print(f"  Development loss: {dev_loss:.4f}")
    print(f"  Test loss:        {test_loss:.4f}")

    # Generate names
    print("\n" + "=" * 70)
    print("Generated Names:")
    print("=" * 70)
    names = generate_names(model, itos, num_names=20)
    for i, name in enumerate(names, 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
