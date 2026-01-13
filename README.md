
# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

### Setup with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Sync with dev dependencies (for linting, testing, etc.)
uv sync --group dev
```

Current implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499) (in progress...)
- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included `names.txt` dataset, as an example, has the most common 32K names takes from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018. It looks like:

```
emma
olivia
ava
isabella
sophia
charlotte
...
```

Let's point the script at it:

```bash
uv run makemore.py -i names.txt -o names
```

Training progress and logs and model will all be saved to the working directory `names`. The default model is a super tiny 200K param transformer; Many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, it runs on my Macbook Air and will run on anything else, but if you have a GPU then training will fly faster. As training progresses the script will print some samples throughout. However, if you'd like to sample manually, you can use the `--sample-only` flag, e.g. in a separate terminal do:

```bash
uv run makemore.py -i names.txt -o names --sample-only
```

This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test logprob of ~1.92, though much lower logprobs are achievable with some hyperparameter tuning):

```
dontell
khylum
camatena
aeriline
najlah
sherrith
ryel
irmi
taislee
mortaz
akarli
maxfelynn
biolett
zendy
laisa
halliliana
goralynn
brodynn
romima
chiyomin
loghlyn
melichae
mahmed
irot
helicha
besdy
ebokun
lucianno
```

Have fun!

### License

MIT

___

## [Video 1 - The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=5s)

- 1:05 - Create the Training Set of all the bigrams (x, y)
  - `create_datasets`
- 1:11 - one-hot encoding
  - _not used in makemore.py_
  - `torch.nn.functional.one_hot` is a PyTorch function used to perform one-hot encoding on a tensor of class indices.
  - [torch.nn.functional.one_hot](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html)
    - Takes LongTensor with index values of shape (*) and returns a tensor of shape (*, num_classes) that have zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1.
    - [One-hot on Wikipedia](https://en.wikipedia.org/wiki/One-hot)
    - Parameters
      - tensor (LongTensor) – class values of any shape.
      - num_classes (int, optional) – Total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor. Default: -1
    - Returns
      - LongTensor that has one more dimension with 1 values at the index of last dimension indicated by the input, and 0 everywhere else.
  - Examples:

```python
>>> F.one_hot(torch.arange(0, 5) % 3)
tensor([[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]])

>>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)
tensor([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]])

>>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)
tensor([[[1, 0, 0],
         [0, 1, 0]],
        [[0, 0, 1],
         [1, 0, 0]],
        [[0, 1, 0],
         [0, 0, 1]]])
```

- 1:14 - Initialize Weights for Neuron
  - torch.randn
    - torch.randn is a PyTorch function that returns a tensor filled with random numbers drawn from a standard normal distribution (Gaussian distribution) with a mean of 0 and a variance of 1. 
  - [torch.randn](https://docs.pytorch.org/docs/stable/generated/torch.randn.html)
    - `torch.randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) → Tensor`
    - Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

- 1:23 Logits - Log Counts
  - Exponentiation gives the Counts
  - Probabilities will be Counts, normalized

- 1:27 Randomly Initialize Weights
  - torch.manual_seed(args.seed)

### 00:50:14 loss function (the negative log likelihood of the data under our model)

- Evaluate the quality of the model
- Training Loss - quality in a single number
- Maximum likelihood estimation
- Likelihood = product of probailities of all the bigrams
  - Probability of the entire dataset, assigned by our trained model
  - The product of probabilites will be a very small number since each of the individual probs is less than 1
  - Log at 1 is 0 and then as probability gets smaller, closer to zero, the log woll approach negative infinity
  - log(a * b * c) = log(a) + log(b) + log(c)
  - so the Log Likelihood can start at zero and then accumulate (sum) the individual log of probabilities
  - Likelihood = product of probailities means
    - Log Likelihood = sum of individual log probailities
    - Each of the individual log probailities is a negative number
    - so their sum is also a negative number (typically larger as it is sum of individual -ve numbers)
  - The negative of Log Likelihood will be a +ve number giving us the "lower is better" semantics
  - A good normalization of the negative of Log Likelihood is an average, so divide it with the total number of individual log likelihoods added to get the Negative Log Likelihood
    - This makes a good Loss Function

- Summary
  - GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
  - equivalent to maximizing the log likelihood (because log is monotonic)
  - equivalent to minimizing the negative log likelihood
  - equivalent to minimizing the average negative log likelihood

___

## [Video 2 - The spelled-out intro to language modeling: building makemore Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3)

This video implements a Multi-Layer Perceptron (MLP) character-level language model based on [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

### Running the MLP model

```bash
uv run makemore_part2_mlp.py
```

- This will:
  1. Load the `names.txt` dataset
  2. Train an MLP model for 200,000 iterations
  3. Display training progress every 20,000 steps
  4. Evaluate on train/dev/test sets
  5. Generate 20 sample names

### Model Architecture

- **Character Embeddings**: 27 characters mapped to 2-dimensional vectors
- **Context Window**: 3 previous characters used to predict the next
- **Hidden Layer**: 200 neurons with tanh activation
- **Output Layer**: Softmax over 27 characters

Total parameters: ~11,897

### Notes

#### Data Preparation

- `words` - list of words from file

```python
words[:8] = ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']
```

- `stoi` + `itos` - character to index mappings

- `itos`

```python
{1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}
```

- Build dataset with
  - `block_size = 3` (3 chars to predict 4th char). 
    - context length: how many characters do we take to predict the next one?
  - `X = []` - input (i/p) to NN
  - `Y = []` - labels
- First 5 words produce 32 context examples, finally we will use all the words in the dataset
  - `X.shape` → `[32, 3]` (num_examples, block_size)
  - `Y.shape` → `[32]`
- NN takes X as input and predicts Y

- X

```python
tensor([[ 0,  0,  0],
        [ 0,  0,  5],
        [ 0,  5, 13],
        ...,
        [26, 26, 25],
        [26, 25, 26],
        [25, 26, 24]])
```

#### Embedding Lookup Table (C)

- 27 possible characters embedded in a 2-dimensional space
- `C = [27, 2]` - initialized randomly to begin
  - Each of the 27 characters will have a 2-dimensional embedding
- Embed all integers in input X
  - Example: to embed 5th char simply → `C[5]` → `1x2`
  - How to embed `[32, 3]` integers stored in X using lookup table C → `C[X]` → `[32 x 3 x 2]`
  - Each of the characters in X is converted to a 2D embedding per mapping in C
- One-hot encoding of char `[1, 27]` dot multiplied with `C[27, 2]` → `[1, 2]`
  - `C[x]` is our embedding

- Ecample `C`

```python
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 2), generator=g)

tensor([[ 1.5674, -0.2373],  # Row 1
        [-0.0274, -1.1008],  # Row 2
        [ 0.2859, -0.0296],  # Row 3
        [-1.5471,  0.6049],
        [ 0.0791,  0.9046],
        [-0.4713,  0.7868],
        [-0.3284, -0.4330],
        [ 1.3729,  2.9334],
        [ 1.5618, -1.6261],
        [ 0.6772, -0.8404],
        [ 0.9849, -0.1484],
        [-1.4795,  0.4483],
        [-0.0707,  2.4968],
        [ 2.4448, -0.6701],
        [-1.2199,  0.3031],
        [-1.0725,  0.7276],
        [ 0.0511,  1.3095],
        [-0.8022, -0.8504],
        [-1.8068,  1.2523],
        [ 0.1476, -1.0006],
        [-0.5030, -1.0660],
        [ 0.8480,  2.0275],
        [-0.1158, -1.2078],
        [-1.0406, -1.5367],
        [-0.5132,  0.2961],
        [-1.4904, -0.2838],
        [ 0.2569,  0.2130]])  # Row 27
```

#### Hidden Layer

- `embedding = C[X]` → `[32, 3, 2]`
  - Row 1: `[[0,0], [0,0], [0,0]]`
  - Row 2: `[[0,0], [0,0], [0,0]]`
  - ...
  - Row 32: `[[0,0], [0,0], [0,0]]`

```python
# C[X]
tensor([[[ 1.5674, -0.2373], [ 1.5674, -0.2373], [ 1.5674, -0.2373]], # Row 1 -> [3, 2]

        [[ 1.5674, -0.2373], [ 1.5674, -0.2373], [-0.4713,  0.7868]], # Row 2 -> [3, 2]

        [[ 1.5674, -0.2373], [-0.4713,  0.7868], [ 2.4448, -0.6701]], # Row 3 -> [3, 2]

        [[-0.4713,  0.7868], [ 2.4448, -0.6701], [ 2.4448, -0.6701]],

        [[ 2.4448, -0.6701], [ 2.4448, -0.6701], [-0.0274, -1.1008]],

        [[ 1.5674, -0.2373], [ 1.5674, -0.2373], [ 1.5674, -0.2373]],
        ...,

        [[ 0.2569,  0.2130], [ 0.2569,  0.2130], [-1.4904, -0.2838]],

        [[ 0.2569,  0.2130], [-1.4904, -0.2838], [ 0.2569,  0.2130]],

        [[-1.4904, -0.2838], [ 0.2569,  0.2130], [-0.5132,  0.2961]]]) # Row 32 -> [3, 2]
```

- Hidden layer weights = `[6 x 100]`
  - 6 = number of inputs from first layer (block_size × embed_dim = 3 × 2)
  - 100 = randomly chosen hidden dimension
- Hidden layer bias = `[100]`

```python
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
```

- Concatenate embeddings: `emb[:,0,:] + emb[:,1,:] + emb[:,2,:]`
  - Results in `[32 x 6]`:
    - Row 1: `[0,0,0,0,0,0]`
    - Row 2: `[0,0,0,0,0,0]`
    - ...
    - Row 32: `[0,0,0,0,0,0]`

```python
# emb[:,0,:] + emb[:,1,:] + emb[:,2,:]
# emb.view(32, 6)
tensor([[ 1.5674, -0.2373,  1.5674, -0.2373,  1.5674, -0.2373],
        [ 1.5674, -0.2373,  1.5674, -0.2373, -0.4713,  0.7868],
        [ 1.5674, -0.2373, -0.4713,  0.7868,  2.4448, -0.6701],
        [-0.4713,  0.7868,  2.4448, -0.6701,  2.4448, -0.6701],
        [ 2.4448, -0.6701,  2.4448, -0.6701, -0.0274, -1.1008],
        [ 1.5674, -0.2373,  1.5674, -0.2373,  1.5674, -0.2373],
        [ 1.5674, -0.2373,  1.5674, -0.2373, -1.0725,  0.7276],
        [ 1.5674, -0.2373, -1.0725,  0.7276, -0.0707,  2.4968],
        [-1.0725,  0.7276, -0.0707,  2.4968,  0.6772, -0.8404],
        [-0.0707,  2.4968,  0.6772, -0.8404, -0.1158, -1.2078],
        [ 0.6772, -0.8404, -0.1158, -1.2078,  0.6772, -0.8404],
        [-0.1158, -1.2078,  0.6772, -0.8404, -0.0274, -1.1008],
        [ 1.5674, -0.2373,  1.5674, -0.2373,  1.5674, -0.2373],
        [ 1.5674, -0.2373,  1.5674, -0.2373, -0.0274, -1.1008],
        [ 1.5674, -0.2373, -0.0274, -1.1008, -0.1158, -1.2078],
        [-0.0274, -1.1008, -0.1158, -1.2078, -0.0274, -1.1008],
        [ 1.5674, -0.2373,  1.5674, -0.2373,  1.5674, -0.2373],
        [ 1.5674, -0.2373,  1.5674, -0.2373,  0.6772, -0.8404],
        [ 1.5674, -0.2373,  0.6772, -0.8404,  0.1476, -1.0006],
        [ 0.6772, -0.8404,  0.1476, -1.0006, -0.0274, -1.1008],
        [ 0.1476, -1.0006, -0.0274, -1.1008,  0.2859, -0.0296],
        [-0.0274, -1.1008,  0.2859, -0.0296, -0.4713,  0.7868],
        [ 0.2859, -0.0296, -0.4713,  0.7868, -0.0707,  2.4968],
        [-0.4713,  0.7868, -0.0707,  2.4968, -0.0707,  2.4968],
        [-0.0707,  2.4968, -0.0707,  2.4968, -0.0274, -1.1008],
...
        [ 1.5674, -0.2373,  0.1476, -1.0006, -1.0725,  0.7276],
        [ 0.1476, -1.0006, -1.0725,  0.7276,  0.0511,  1.3095],
        [-1.0725,  0.7276,  0.0511,  1.3095,  1.5618, -1.6261],
        [ 0.0511,  1.3095,  1.5618, -1.6261,  0.6772, -0.8404],
        [ 1.5618, -1.6261,  0.6772, -0.8404, -0.0274, -1.1008]])```

- `[batch_size, block_size, emb_dim]` → `[batch_size, block_size * emb_dim]`
- `emb.view(32, 6)` - with PyTorch (`-1` to make it generic)
- Hidden layer: `h = tanh(emb.view(32,6) @ W1 + b1)`
  - Same bias vector will be added to each of the 32 rows of `emb.view(32,6) @ W1`
  - h.shape = [32, 100]

#### Output Layer
- `W2 = [100, 27]` → output 27 possible characters
- `b2 = [27]` → vocab size

```python
W2 = torch.randn((100, 27))
b2 = torch.randn(27)
```

- `logits = h @ W2 + b2`
  - Hidden layer `[32 x 100]` @ W2 → `[32 x 27]`
  - logits.shape = [32, 27]

#### Loss Calculation

- `counts = logits.exp()`
- `probs = counts / counts.sum(1, keepdims=True)` → `[32 x 27]`
- `probs[torch.arange(32), Y]` = probability of each of the 32 characters
- Negative log likelihood loss:
  - `loss = -probs[torch.arange(32), Y].log().mean()`
- Or simply: `F.cross_entropy(logits, Y)`

___

- What does the below expression do?

```python
probs[torch.arange(32), Y]
# or
probs[torch.arange(Y.shape[0]), Y]
```

- This is advanced PyTorch indexing that extracts the probability of the correct character for each training example.

- prob is a 2D tensor of shape (batch_size, 27)
  - each row contains the predicted probabilities for all 27 characters

- Y is a 1D tensor of shape (batch_size,)
  - contains the target character indices (the correct answers)

- torch.arange(Y.shape[0]) creates [0, 1, 2, ..., batch_size-1] — the row indices
What it does:

```python
prob[torch.arange(Y.shape[0]), Y]
#    ↑ row indices              ↑ column indices
```

- For each row i, it selects the column Y[i]. 
  - This gives you the predicted probability of the correct character for each example.

Concrete example:

```python
prob = tensor([[0.1, 0.7, 0.2],   # row 0
               [0.3, 0.4, 0.3],   # row 1
               [0.8, 0.1, 0.1]])  # row 2

torch.arange(3) = tensor([0, 1, 2])
Y =               tensor([1, 2, 0])  # correct answers: col 1, col 2, col 0

prob[torch.arange(3), Y]
# → tensor([0.7, 0.3, 0.8])
#           ↑     ↑     ↑
#        prob[0,1] prob[1,2] prob[2,0]
```

- Why it matters:
  - This is used to compute the negative log-likelihood loss
  - You want these probabilities to be high (close to 1)
  - The loss is typically -log(prob[..., Y])
