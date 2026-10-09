# tiny-ai

A word-level **seq2seq GRU** with attention and a copy mechanism, written from
scratch in plain JavaScript. No libraries, no bundler, no pretrained weights, no
network calls. It runs in Node and in the browser, learns a set of small
input→output tasks, and ships as a single binary `.GenAI` file.

The point of the project is to be *readable and honestly evaluated*: every
gradient is derived by hand, and the generalization claims are backed by
held-out sets (in-distribution, compositional, systematic).

The model itself owns **no data**. Training data lives in `data/` as
procedurally generated JSON, and you hand it to the model:

```js
TinyAI.init({ dataset: examples });        // every example has { src, tgt, split }
```

```
tiny-ai/
├── tiny-ai.js         the model: GRU, attention, copy gate, BPTT, Adam, decoding, .GenAI
├── data/
│   ├── gen_common.js      shared RNG / tokenizer / split helpers
│   ├── gen_<task>.js      one procedural generator per task (seeded, deterministic)
│   ├── generate_all.js    runs every generator, writes data/<task>.json + samples
│   ├── loader.js          reads the JSON files (Node only) and injects `task`
│   └── samples/           a small committed sample of every task
├── train.js           loads data/, trains, evaluates, writes model.GenAI
├── train.sh           thin wrapper around train.js
├── chat.js / chat.sh  terminal REPL that loads a .GenAI file and chats
├── gradcheck.js       finite-difference gradient check (developer tool)
├── bench.js           throughput / accuracy benchmark
├── test.js            quick smoke test
├── index.html         browser chat UI that loads a .GenAI file
└── model.GenAI        a pre-trained model (loaded by index.html)
```

## Quick start

```bash
cd tiny-ai
node data/generate_all.js  # writes data/*.json from the generators
./train.sh                 # trains and writes model.GenAI (~5–10 min)
./chat.sh                  # chat in the terminal
node test.js               # quick smoke test, ~30 s
node gradcheck.js          # verify the gradients, ~10 s
```

Then open `index.html` in a browser and drop in `model.GenAI`.

The full datasets are git-ignored; the generators and a ~50-example sample per
task (`data/samples/`) are committed, so a fresh checkout is reproducible with
`node data/generate_all.js`.

Training options can be overridden from the environment:

```bash
STEPS=3000 node train.js my_model.GenAI
H=48 BATCH=8 STEPS=2000 node bench.js
```

## What the model actually does

Per decode step `t`:

```
encoder (per layer, forward)   f_i = GRU(x_i, f_{i-1})
encoder (per layer, backward)  b_i = GRU(x_i, b_{i+1})
enc_i = [ f_i ; b_i ]                       (2H when bidirectional)
query q = W_q · s_t                          (s_t = top decoder state)
scores_i = (q · enc_i) / sqrt(E) + w_cov · tanh(cov_i)
alpha = softmax(scores)
ctx = Σ_i alpha_i · enc_i
decoder layer 0 input: [ E[prev_t] ; ctx ]
decoder layer l input:  h_{l-1,t}
logits = W_out · top + b_out
p_gen  = sigmoid(W_gen · [top ; ctx ; E[prev_t]])
copy   = Σ_{i : src_i = w} alphaCopy_i
p(w)   = p_gen · softmax(logits)[w] + (1 − p_gen) · copy
```

The **copy gate** is what lets it reproduce a name/number it has never seen in
the target before: a pointer distribution over the source tokens is mixed with
the generative vocabulary distribution.

Training is Adam on a masked cross-entropy loss built with per-example
gradients (mini-batch = gradient accumulation), plus:

- label smoothing,
- word dropout, embedding dropout and hidden-state dropout,
- LR warmup + cosine decay,
- gradient clipping,
- a validation split with early stopping.

### Notebook of design choices (all explainable)

| Piece | Why |
| --- | --- |
| GRU cells | gated recurrences, fewer gates than LSTM, stable on tiny data |
| Bidirectional encoder | each source position sees left *and* right context |
| Stacked layers | more nonlinearity; later layers compose earlier features |
| Learned query projection + `1/√E` scaling | attention scores no longer blow up with dimension |
| Coverage (`w_cov`) | penalizes re-attending to the same source token; reduces repeats |
| Separate copy attention head | the pointer decision and the generation decision can differ |
| Label smoothing | stops the softmax from becoming over-confident |
| Mini-batch accumulation | smoother gradients; no artificial padding |
| Masked loss/attention | correct if padded batches are ever used |

## Data and evaluation splits

Each example is `{ src, tgt, split, tags[] }`. Every generator tags its own
examples, so the split rules live right next to the data they describe. There
are four splits:

| split | meaning |
| --- | --- |
| `train` | the model is allowed to fit these |
| `val_indist` | random holdout from the same distribution as training |
| `val_compositional` | new *combinations* of pieces each seen during training |
| `val_systematic` | a value that never appears during training at all |

A **compositional** example uses pieces each seen in training but never in this
combination. A **systematic** example contains a token or operation *never seen
during training at all* — it can only be solved by learning the rule (and, for
copy/reverse/sort/qa_copy, by using the copy mechanism to emit a token that was
never in a training target).

Task generators and their held-out splits:

| Task | what it is | compositional | systematic |
| --- | --- | --- | --- |
| `conversations` | intent → canonical reply, many paraphrases (the model knows its name is **tiny-ai**) | unseen phrasing wrapper | 10 whole unseen intents |
| `booking` | table booking slot filling | sizes 11/12 (unseen with a name) | 3 unseen names |
| `multi_slot` | 2–3 slot requests, 5 templates | held-out colours `gold`/`teal` | unseen rooms `attic`/`cellar` |
| `copy` | repeat the words | length 5–6 sequences | unseen words `fox`/`owl` |
| `reverse` | reverse the words | length 5–6 sequences | unseen words `fox`/`owl` |
| `sort` | ascending numeric sort | lengths 4–5 | numbers `80..99` |
| `arithmetic` | `add`/`plus`/`sum` phrasings | both operands in `21..30` | operands `42..45` |
| `qa_copy` | country → capital lookup | unseen phrasing prefix | 6 unseen countries |
| `successor` | `next 5` → `6`, `previous 40` → `39` | double step `next next 5` | numbers `90..99` |
| `scan` | command expansion | `jump thrice` | unseen primitive `look` |
| `cogs` | `subject verb object` → logical form | `(girl, boat)` pair | unseen verb `chases` |

Target sizes are set per task (`TARGET_TOKENS`) so the whole set is roughly
350k tokens; the train split is trimmed deterministically to hit the target
while the held-out splits are kept complete (capped at 400 examples each so
the evaluation stays fast).

## Where it generalizes — and where it only memorizes

This is the honest part, and it is worth being explicit. "Generalization" here
does not mean the model understands language; it means the *best* tasks are
solvable by a function the network can actually learn, and the held-out splits
measure exactly that.

**It generalizes (holds out a value/combination and still scores above chance):**

- `successor` — the cleanest case. The answer is `n ± 1`; the systematic split
  uses numbers `90..99` that never appear in training, so a lookup table cannot
  answer them. A model that learned the *ordering rule* gets them right.
- `copy` / `reverse` — the copy gate lets the decoder emit a token straight from
  the source. That is what carries the systematic split (words `fox`/`owl` never
  appear in a training target) and the compositional split (longer orderings).
- `booking` / `multi_slot` / `qa_copy` — rigid templates plus the copy gate. The
  model learns the template and copies the slots, so unseen names/sizes/rooms
  and unseen phrasings still work.
- `arithmetic` / `sort` / `scan` / `cogs` — compositional splits test *unseen
  combinations* of seen pieces (e.g. adding two operands it never saw together);
  the rule transfers even though the exact pair did not.

**It only memorizes (and that is expected):**

- `conversations` is a lookup table by construction: one canonical reply per
  intent, many paraphrases. The compositional split (a held-out *phrasing
  wrapper*) is the one place it can show paraphrase-level generalization; the
  systematic split (ten brand-new intents) is deliberately unanswerable from the
  input and should be ~0%.
- The name questions (`what is your name` → `my name is tiny-ai`) are also
  memorized, but because they are phrased many ways and the response is fixed,
  the model does generalize across the *phrasings* it saw.

The point of reporting `val_indist` / `val_compositional` / `val_systematic`
separately is to make this distinction visible instead of hiding it behind one
blended accuracy number.


## What to expect

These are exact-match numbers from a real run of the default architecture
(`H=64, E=32, layers=2, bidir=true, coverage=true, batch=6`). Exact values vary
with seed and step count.

```
                 in-distribution   compositional   systematic
greedy           ...
beam-4           ...
```

Per task (greedy / beam, held-out sets):

```
task            in-dist            compositional      systematic
...
```

Observations worth expecting:

- **Chat and structured tasks behave differently.** The template tasks
  (booking, multi_slot, copy, reverse, sort) have rigid structure and are
  learned well; the conversational set is mostly memorization.
- **Copy-based tasks generalize better than arithmetic.** copy/reverse/qa_copy
  systematic accuracy is driven by the pointer: it can copy a token straight
  from the source even though it never appeared in a training target.
  Systematic arithmetic must be learned as an actual rule, so it is harder.
- **Compositional means unseen combinations, not unseen values.** Arithmetic
  held-out operands, for example, are still in `0..41`; the model must apply
  `add` to a pair it never saw.
- **On a tiny model, greedy can beat beam.** Beam search maximizes joint
  sequence log-probability, and a small model can be miscalibrated at the
  sequence level: it may assign higher total likelihood to a degenerate
  sequence. This is an honest property of an undertrained seq2seq model — beam
  is only as good as the model's joint distribution.
- **Validation loss plateaus early.** Early stopping watches validation loss
  while the headline metric is exact match.

### It is still a tiny model

- It will not write grammatical English outside the trained templates.
- Arithmetic is limited to the ranges seen in training (plus the compositional
  region); there is no learned notion of "carry" beyond the data.
- Beam search only re-ranks; it cannot invent a token that greedy would never
  assign nonzero probability to early.

## Gradient check

Because the backward pass is hand-written, `gradcheck.js` builds a tiny model
with dropout disabled and compares analytic gradients against central finite
differences on a random sample of parameters:

```bash
node gradcheck.js          # all features on
LAYERS=1 BIDIR=0 HEADS=0 COV=0 node gradcheck.js   # minimal model
```

All configurations should print `GRADCHECK PASSED`.

## `.GenAI` file format

```
[0..5]   magic "GenAIx" (v2 writes "GenAI2")
[6..9]   uint32 little-endian header length
[10..]   UTF-8 JSON header (version, vocab, dims, tensor manifest)
[pad]    zero padding to an 8-byte boundary
[...]    Float64 tensor data in manifest order
```

Version 2 adds layers / bidir / attention-heads / coverage to the header and
names tensors `enc_fwd_<l>_*`, `enc_bwd_<l>_*`, `dec_<l>_*`, `E`, `W_out`,
`b_out`, `W_gen`, `b_gen`, `W_q_gen`, `W_q_copy`, `w_cov`. Old version-1 files
still load: missing tensors are re-derived (the attention query projection
becomes the identity), so a v1 model reproduces its original behaviour.

## Public API

```js
TinyAI.init({ dataset, splits, /* overrides */ }); // returns status
TinyAI.train(budgetMs);             // trains for ~budgetMs, returns status
TinyAI.respond(text, temperature);  // sampled reply
TinyAI.greedy(text);                // greedy reply
TinyAI.beamSearch(text, beamSize);  // beam reply
TinyAI.accuracy(examples, opts);    // exact-match over [{src,tgt}, ...]
TinyAI.splits();                    // { train, inDist, compositional, systematic }
TinyAI.status();                    // ready, steps, loss, valLoss, bestValLoss, params, ...
TinyAI.save();                      // -> ArrayBuffer (.GenAI v2)
TinyAI.load(bufferOrArray);         // -> status
TinyAI.vocabulary();                // string[]
TinyAI.setTarget(steps);            // extend the training horizon
TinyAI.isLoaded();
TinyAI.model();                     // the raw Model (advanced)
```

`init` accepts the data directly:

- `dataset`: an array of `{ src, tgt, split }`. If no `splits` are given, the
  examples are grouped by their own `.split` field. Split names are accepted in
  both the new (`val_indist` / `val_compositional` / `val_systematic`) and old
  (`in` / `comp` / `sys`) spellings.
- `splits`: an explicit `{ train, inDist, compositional, systematic }` (or the
  `val_*` spellings) object, overriding `.split`.

The original API surface (`init`, `train`, `respond`, `status`, `setTarget`,
`vocabulary`, `save`, `load`, `isLoaded`) is unchanged, so existing callers keep
working. The old in-browser default dataset (`tasks.js`) is gone — the browser
page only loads a `.GenAI` file, while training happens in Node from `data/`.

## Knobs

All defaults live in `DEFAULTS` at the top of `tiny-ai.js`:

| option | default | notes |
| --- | --- | --- |
| `hidden` / `embed` | 64 / 32 | model width |
| `layers` / `bidir` | 2 / true | encoder & decoder depth; encoder direction |
| `batchSize` | 6 | examples accumulated per update |
| `targetSteps` | 10000 | training horizon |
| `lr` / `minLr` | 0.004 / 0.0002 | cosine decay bounds |
| `warmupSteps` / `lrDecaySteps` | 300 / 10000 | schedule shape |
| `labelSmoothing` | 0.05 | target distribution smoothing |
| `wordDropout` | 0.1 | replace input token with `<unk>` |
| `embDropout` / `hiddenDropout` | 0.1 / 0.1 | regularization |
| `coverage` / `coverageWeight` | true / 0.5 | anti-repetition term |
| `attnHeads` | false | separate attention head for the copy distribution |
| `attnScale` | true | divide attention scores by `sqrt(E)` |
| `seed` | 1337 | RNG seed for init, shuffling, dropout |

Generators are deterministic too: `node data/generate_all.js` twice produces
byte-identical files.
