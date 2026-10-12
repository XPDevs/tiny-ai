# TinyAI

**A tiny, educational AI framework written from scratch in pure JavaScript, by XPDevs.**

No TensorFlow. No PyTorch. No ONNX. No external AI libraries. Every matrix
multiply, every layer, the tokenizer, the optimizer and the gradients are
implemented by hand, so that you can read the whole thing and understand exactly
how a Transformer works.

TinyAI is **not** trying to compete with GPT, Gemini, Claude or Llama. It is a
small, honest, fully understandable model you can learn from, modify, train and
deploy on a laptop, a Raspberry Pi or in a web browser. It is built first as an
**educational** project, but it is a real model that can genuinely be used for
real tasks: the larger you make the model and the more data you train it on, the
more capable it becomes. Train it small to learn, train it bigger to use it.

```
  ___________ _        _    ___
 |_   _|_ _\ \      / \  |_ _|
   | |  | | \ \ /\ / / _ \  | |
   | |  | |  \ V  V / ___ \ | |
   |_| |___|  \_/\_/_/   \_\___|
   tiny transformer, written from scratch in JavaScript
```

> **TinyAI is a project by XPDevs.**

---

## Table of contents

- [Why TinyAI exists](#why-tinyai-exists)
- [Features](#features)
- [The architecture](#the-architecture)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [How it works (a guided tour)](#how-it-works-a-guided-tour)
- [The command line tools](#the-command-line-tools)
- [Configuration reference](#configuration-reference)
- [Tokenizers](#tokenizers)
- [The `.GenAI` model format](#the-genai-model-format)
- [Knowledge plugins](#knowledge-plugins)
- [The web UI](#the-web-ui)
- [Growing TinyAI into something usable](#growing-tinyai-into-something-usable)
- [Performance expectations](#performance-expectations)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Extending TinyAI (roadmap)](#extending-tinyai-roadmap)
- [FAQ](#faq)
- [Credits](#credits)
- [License](#license)

---

## Why TinyAI exists

Most AI code today is a thin layer of Python on top of enormous C++ or CUDA
libraries. That is the right way to build production systems, but it hides the
ideas. TinyAI takes the opposite approach:

> **Readability is the feature.** The code prefers clarity over speed. Every
> non obvious line is explained, and every derivative is written out in the
> comments, so you can check the mathematics yourself.

The target reader is someone who has **never built an AI before**: a student, a
hobby programmer, an open source developer, or anyone who wants to see the whole
stack.

TinyAI is educational, but it is not a toy. The architecture is a genuine
decoder only Transformer, the same family of design used by modern language
models, just scaled down. That means the same code keeps working as you make the
model bigger. Grow the embedding size, add more layers, train on more text, and
TinyAI becomes a real, usable model. The only real limits are your patience and
your hardware, because the arithmetic runs in plain JavaScript rather than on a
GPU.

## Features

- **Decoder only Transformer** (a "mini GPT"): token and positional embeddings,
  multi head self attention, feed forward layers, residual connections, layer
  normalization, causal masking, softmax output.
- **Automatic differentiation from scratch**: a small tensor library with
  reverse mode autograd. `loss.backward()` and the chain rule do the rest.
- **Three tokenizers** written by hand: character, word and BPE (byte pair
  encoding).
- **Training system**: batching, cross entropy loss, Adam (and SGD) optimizers,
  gradient clipping, warmup plus cosine learning rate schedule, validation loss,
  checkpoints and resume.
- **Custom `.GenAI` model format**: a human readable header followed by base64
  encoded weights.
- **Inference**: greedy, temperature, top K and top P (nucleus) sampling, plus
  streaming output.
- **Knowledge plugins**: inject facts from JSON files without retraining.
- **Terminal chatbot** and a **vanilla web UI** (no frameworks).
- **Runs in Node.js and in the browser** from the exact same source files.
- Default model is about **1.3 million parameters**, **under 10 MB** on disk.

## The architecture

```
        token ids  [B, T]
             |
   token embedding lookup            ->  [B, T, D]
 + positional embedding              ->  [B, T, D]
             |
   +---------+----------+
   |  Transformer block  |  x nLayers
   |   +- LayerNorm      |
   |   +- Self-Attention |  + residual
   |   +- LayerNorm      |
   |   +- Feed-Forward   |  + residual
   +---------+----------+
             |
        final LayerNorm
             |
   LM head (tied to token embedding)
             |
        logits  [B, T, V]   ->  softmax -> sample the next token
```

The default configuration:

| Setting            | Value      |
|--------------------|------------|
| Vocabulary size    | 4,096      |
| Embedding size (D) | 128        |
| Attention heads    | 4          |
| Transformer layers | 4          |
| Context length     | 128        |
| Parameters         | ~1,334,000 |
| On disk size       | ~7 MB      |

## Quick start

You only need **Node.js** (version 14 or newer). There are **no dependencies to
install**.

```bash
# 1. Check everything works (fast)
node test.js

# 2. Train a small model quickly (a few minutes), good for a first run
node train.js --config training/examples/tiny.json

# 3. Chat with it in the terminal
node chat.js --model models/tinyai-tiny.genai
```

Then open `index.html` in a browser, click **Load model**, and select
`models/tinyai-tiny.genai`.

To train the full default model (much slower, needs patience):

```bash
node train.js
```

## Project structure

```
tiny-ai/
|
+-- index.html           the whole web UI in one self contained file
|                        (markup + inline CSS + inline JavaScript)
|
+-- src/
|   +-- tensor.js        Tensor class + autograd engine (the foundation)
|   +-- tokenizer.js     character / word / BPE tokenizers
|   +-- attention.js     multi head causal self attention
|   +-- transformer.js   layer norm, feed forward, transformer block
|   +-- optimizer.js     SGD and Adam optimizers
|   +-- model.js         the model, generation, and the .GenAI format
|   +-- trainer.js       the training loop, checkpoints, validation
|   +-- knowledge.js     external knowledge plugins
|
+-- training/
|   +-- datasets/        sample.txt, chat.txt
|   +-- checkpoints/     checkpoints are written here
|   +-- examples/        tiny.json, default.json, bpe.json
|
+-- knowledge/           animals.json, science.json, history.json, ...
+-- models/              trained .genai models live here
|
+-- train.js             training entry point
+-- chat.js              terminal chatbot
+-- benchmark.js         speed and size measurements
+-- test.js              test suite (including gradient checks)
|
+-- package.json
+-- README.md
```

Everything is organised into four clear places: the framework core in `src/`,
the data and configuration in `training/`, the facts in `knowledge/`, and the
trained models in `models/`. The command line tools and the single file web UI
sit at the top level so they are easy to find.

## How it works (a guided tour)

If you read the files in this order, each one builds on the last.

### 1. `src/tensor.js`, tensors and autograd

A **tensor** is just a flat array of numbers plus a shape. This file implements
`add`, `sub`, `mul`, `div`, `matmul`, `transpose`, `reshape`, `softmax`,
`layernorm` helpers, `embedding`, cross entropy, and more, each with an
explicitly derived backward pass.

It also implements `backward()`: it walks the computation graph in reverse and
applies the chain rule. This is the entire secret of deep learning.

### 2. `src/tokenizer.js`, text to numbers

Characters, words, or BPE sub words. Each tokenizer can `encode`, `decode`,
`save` and `load`.

### 3. `src/attention.js`, self attention

```
Attention(Q, K, V) = softmax( Q * K^T / sqrt(d_k) ) * V
```

Every token forms a query, a key and a value, and attends to every other token.
A causal mask hides the future, so the model can only look backwards.

### 4. `src/transformer.js`, the block

LayerNorm, then attention, then residual. LayerNorm, then feed forward, then
residual. Stack this `nLayers` times.

### 5. `src/optimizer.js`, how learning happens

Adam keeps running averages of the gradient and of the squared gradient, and
takes a well scaled step for every parameter.

### 6. `src/model.js`, putting it together

Builds the model, runs the forward pass, generates text, and reads and writes
`.GenAI` files.

### 7. `src/trainer.js`, the loop

Shows the model batches of text, measures cross entropy, back propagates, and
steps the optimizer, over and over, until the loss goes down.

## The command line tools

### Training

```bash
node train.js                                  # the default model
node train.js --config training/examples/tiny.json
node train.js --steps 500 --dmodel 64 --layers 2 --batch 8 --lr 0.001
node train.js --tokenizer bpe --vocab 512
node train.js --resume                         # continue from a checkpoint
node train.js --data training/datasets/chat.txt --out models/mychat.genai
```

| Flag           | Meaning                                    |
|----------------|--------------------------------------------|
| `--config`     | JSON config file (model and training)      |
| `--data`       | Text file to train on                      |
| `--tokenizer`  | `char` (default), `word` or `bpe`          |
| `--vocab`      | Target vocabulary size (BPE)               |
| `--steps`      | Number of training steps                   |
| `--batch`      | Batch size                                 |
| `--block`      | Context length                             |
| `--dmodel`     | Embedding dimension                        |
| `--heads`      | Attention heads                            |
| `--layers`     | Transformer layers                         |
| `--lr`         | Learning rate                              |
| `--out`        | Output `.genai` path                       |
| `--resume`     | Resume from `training/checkpoints`         |

### Chatting

```bash
node chat.js
node chat.js --model models/tinyai.genai --temperature 0.7 --topk 40 --topp 0.95
```

Inside the chat: `/help`, `/reset`, `/temp`, `/topk`, `/topp`, `/tokens`,
`/settings`, `/quit`.

### Benchmarks and tests

```bash
node benchmark.js                 # speed and model size
node test.js                      # correctness and gradient checks
```

## Configuration reference

An example config (`training/examples/default.json`):

```json
{
  "model": {
    "dModel": 128,
    "nHeads": 4,
    "nLayers": 4,
    "blockSize": 128,
    "tieWeights": true
  },
  "tokenizer": { "type": "char", "vocabSize": 4096 },
  "training": {
    "batchSize": 8,
    "lr": 0.0003,
    "minLr": 0.00003,
    "weightDecay": 0.01,
    "gradClip": 1.0,
    "warmup": 100,
    "maxSteps": 1000,
    "logEvery": 10,
    "evalInterval": 100,
    "checkpointInterval": 250,
    "valSplit": 0.1
  },
  "data": "training/datasets/sample.txt",
  "out": "models/tinyai.genai"
}
```

- `tieWeights`: share the token embedding matrix with the output head (fewer
  parameters).
- `warmup` plus cosine decay: the learning rate ramps up, then decays smoothly.
- `gradClip`: caps the gradient norm so a bad batch cannot blow up training.
- `valSplit`: fraction of the text held out to measure generalization.

## Tokenizers

```js
const { createTokenizer } = require('./src/tokenizer.js');

const charTok = createTokenizer('char', text);
const wordTok = createTokenizer('word', text);
const bpeTok  = createTokenizer('bpe', text, 512); // target vocab size

charTok.encode('hello');  // -> [ids]
charTok.decode([ids]);    // -> 'hello'
charTok.save();           // -> plain object (JSON)
```

BPE learns merges like `t + h -> th`, `th + e -> the`, growing a vocabulary of
useful sub word pieces, the same idea real models use.

## The `.GenAI` model format

A `.GenAI` file is text. The header is human readable, and the weights are
base64 encoded so the file stays small:

```
TINYAI-GENAI
version: 1
generator: TinyAI
architecture: transformer-decoder
layers: 4
heads: 4
dModel: 128
blockSize: 128
vocabSize: 4096
parameters: 1334016
tokenizer: char
---CONFIG---
{"vocabSize":4096,"dModel":128,...}
---TOKENIZER---
{"type":"char","tokens":["<pad>","<unk>",...]}
---WEIGHTS---
tokEmb 4096,128 <base64 float32>
...
---END---
```

```js
const model = TinyModel.loadFile('models/tinyai.genai');
model.saveFile('models/copy.genai');
model.generateText('Once upon a time', { maxNewTokens: 40 });
```

## Knowledge plugins

A tiny model does not have to know everything. Drop JSON files into
`knowledge/`, and TinyAI will search them **before** calling the model, so no
retraining is required:

```json
{
  "topic": "animals",
  "entries": [
    { "keywords": ["cat", "cats"], "answer": "Cats are small domesticated mammals..." }
  ]
}
```

Add your own facts by creating another file such as `knowledge/space.json`.
This is a very small version of "retrieval augmented generation" (RAG).

## The web UI

The web UI is a **single self contained file**, `index.html`. It has the
markup, the CSS and the front end JavaScript all inlined, so there are no
`app.js` or `style.css` files to manage. It is plain HTML, CSS and JavaScript,
with no frameworks and no build step.

- Chat interface with **streaming** responses (token by token)
- **Load model** button and drag and drop for `.genai` files
- **Save** and **export** conversations (localStorage plus a `.txt` download)
- **Dark mode** toggle (remembered between visits)
- Responsive layout for mobile

Open `index.html` directly, or serve the folder to enable the full knowledge
plugins (browsers block `fetch` on `file://`):

```bash
# any static server works, for example:
npx serve .
# or
python -m http.server
```

The browser runs the **same** `src/*.js` files as Node, loaded via `<script>`
tags in dependency order.

## Growing TinyAI into something usable

TinyAI starts small on purpose, so it is cheap to train while you learn. The
same code scales. To turn it into a genuinely useful assistant:

1. **Give it more data.** Train on a larger text file with
   `node train.js --data your-big-file.txt`. Loss falls as the data grows.
2. **Give it a bigger model.** Increase `--dmodel`, `--layers` and `--block`.
   More parameters means more capacity to learn.
3. **Give it a better vocabulary.** BPE with a few thousand tokens packs more
   meaning into each position than single characters.
4. **Give it knowledge.** Add facts in `knowledge/` for anything the model
   cannot reasonably memorize.

Remember the trade off: bigger models are more capable but slower to train in
pure JavaScript. Experiment small, then commit to a bigger run once you are
happy with the setup.

## Performance expectations

TinyAI is written for humans, not for GPUs. The arithmetic is naive JavaScript
loops, so it is **much** slower than a library that calls optimized BLAS or CUDA
kernels. That is a deliberate trade off. Rough numbers on a normal laptop:

| Model                       | Forward (1x128) | Train step (B=8) | Generation |
|-----------------------------|-----------------|------------------|------------|
| tiny (D=64, L=2, V~53)      | ~50 ms          | ~0.5 s           | ~65 tok/s  |
| default (D=128, L=4, V~53)  | ~0.4 s          | ~12 s            | ~3 tok/s   |

Practical advice:

- Start with `training/examples/tiny.json`.
- Use a **character or small BPE** vocabulary for quick experiments.
- Generation cost grows with context length, so keep prompts reasonably short.
- For a faster machine, it is worth trying a smaller `dModel` and `blockSize`
  before adding layers.

The good news: **inference is fast enough for a real chatbot** with the tiny
model, and the full model still fits comfortably under 10 MB.

## Testing and benchmarking

```bash
node test.js
```

The suite checks hand calculations (for example `matmul`), tokenizer round
trips, model save and load fidelity, knowledge search, and, most importantly,
compares **every backward pass against a numerical (finite difference)
gradient**. If autograd were wrong, these tests would fail.

```bash
node benchmark.js --dmodel 64 --layers 2 --block 64
```

## Extending TinyAI (roadmap)

The architecture was designed so these can be added without a rewrite:

- **Quantization:** store weights as 8 bit integers to shrink the model further.
  The `.GenAI` header already carries a version field.
- **LoRA and adapters:** freeze the base weights and train small low rank
  matrices. The parameter list is a flat `{name, tensor}` array.
- **Multi modal and image embeddings:** add a second embedding path that
  projects patches into the same `D` dimensional space.
- **Voice input and output:** pair with the Web Speech API in the browser.
- **Function calling and tool use:** intercept special tokens in `generate()`
  and call a JavaScript function.
- **Internet search and long term memory:** extend `KnowledgeBase` to query
  online sources or a vector store.

Because everything is plain JavaScript with no build step, each of these is a
small, self contained change.

## FAQ

**Does it use any libraries?**
No. Only the Node.js standard library (`fs`, `path`, `readline`) for file and
terminal access. The math is entirely bespoke.

**Can it run in a browser?**
Yes. The `src/` files work both as Node modules and as browser scripts, and the
single file web UI loads the same `.genai` models you train on the command line.

**Why is training slow?**
Because the operations are simple, readable JavaScript loops rather than
optimized native kernels. Use `tiny.json`, a small vocabulary, and a short
context while experimenting.

**Is the default model really about 1M parameters?**
With the four special tokens the default character vocabulary is small, so the
default model lands around 0.8M parameters. With a 4,096 token vocabulary (for
example BPE) it is exactly 1,334,016 parameters, within the 500k to 2M target,
and about 7 MB on disk.

**Where do the gradients come from?**
Every operation records a small backward function. `loss.backward()` walks the
graph in reverse applying the chain rule. `node test.js` verifies each one
against a numerical gradient.

**Can I actually use this, or is it only for learning?**
Both. It is designed to be read and learned from, but it is a real Transformer.
As you increase the model size and the amount of training data, it becomes a
genuinely usable text generator.

## Credits

TinyAI is created and maintained by **XPDevs**. It is an open, educational
project, and contributions, ideas and improvements are welcome.

## License

MIT. Take it, read it, break it, improve it, share it.
