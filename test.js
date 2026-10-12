/**
 * ============================================================================
 * TinyAI - test.js
 * ============================================================================
 *
 * A self-contained test suite for the whole framework. Run it with:
 *
 *     node test.js
 *
 * It checks three kinds of things:
 *
 *   1. Correctness of the math   - e.g. does matmul match a hand calculation?
 *   2. Correctness of gradients  - every autograd operation is compared against
 *      a numerical (finite-difference) gradient. If they disagree, backprop is
 *      wrong and training would silently fail.
 *   3. Correctness of the pieces - tokenizers, save/load, knowledge search and
 *      a short training run that must reduce the loss.
 *
 * Exit code is 0 if everything passes, 1 otherwise.
 * ============================================================================
 */

'use strict';

const TNS = require('./src/tensor.js');
const {
  Tensor,
  tensor,
  zeros,
  ones,
  randn,
  add,
  sub,
  mul,
  div,
  matmul,
  reshape,
  transpose,
  softmax,
  meanLast,
  gelu,
  crossEntropy,
  sumAll,
  fromIds,
} = TNS;

const { layerNorm } = require('./src/transformer.js');
const { selfAttention } = require('./src/attention.js');
const { createTokenizer } = require('./src/tokenizer.js');
const { BaseTokenizer } = require('./src/tokenizer.js');
const { TinyModel } = require('./src/model.js');
const { KnowledgeBase } = require('./src/knowledge.js');
const { Trainer } = require('./src/trainer.js');

/* --------------------------------------------------------------------------
 * Tiny test harness
 * ------------------------------------------------------------------------ */
let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log('  PASS  ' + name);
  } catch (err) {
    failed++;
    console.log('  FAIL  ' + name);
    console.log('        ' + err.message);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}

function almost(a, b, eps = 1e-6) {
  return Math.abs(a - b) <= eps;
}

function rand(shape) {
  const t = randn(shape, true, 1.0);
  return t;
}

/* --------------------------------------------------------------------------
 * Numerical gradient checker
 *
 * Build a scalar loss from some inputs, then compare the gradient computed by
 * backpropagation to the gradient estimated by nudging each input slightly:
 *
 *     f'(x) ~= (f(x + eps) - f(x - eps)) / (2 eps)
 * ------------------------------------------------------------------------ */
function gradCheck(build, inputs, eps = 1e-3) {
  const loss = build();
  loss.backward();
  const analytic = inputs.map((t) => (t.grad ? Float64Array.from(t.grad) : null));

  let maxRelErr = 0;
  for (let k = 0; k < inputs.length; k++) {
    const t = inputs[k];
    const num = new Float64Array(t.size);
    for (let i = 0; i < t.size; i++) {
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const lp = build().data[0];
      t.data[i] = orig - eps;
      const lm = build().data[0];
      t.data[i] = orig;
      num[i] = (lp - lm) / (2 * eps);
    }
    const a = analytic[k];
    assert(a, 'no analytic gradient for input ' + k);
    for (let i = 0; i < t.size; i++) {
      const denom = Math.max(1, Math.abs(num[i]), Math.abs(a[i]));
      const relErr = Math.abs(a[i] - num[i]) / denom;
      if (relErr > maxRelErr) maxRelErr = relErr;
    }
  }
  return maxRelErr;
}

console.log('');
console.log('TinyAI test suite, a project by XPDevs');
console.log('======================================');
console.log('');

/* ==========================================================================
 * 1. Tensor basics
 * ========================================================================== */
console.log('Tensor basics');
test('tensor shape and item', () => {
  const t = tensor([1, 2, 3, 4, 5, 6], false, [2, 3]);
  assert(t.shape.length === 2 && t.shape[0] === 2 && t.shape[1] === 3, 'wrong shape');
  assert(t.at(1, 2) === 6, 'wrong element');
  assert(tensor(5).item() === 5, 'scalar item');
});

test('nested array shape inference', () => {
  const t = tensor([[1, 2, 3], [4, 5, 6]]);
  assert(t.shape[0] === 2 && t.shape[1] === 3, 'shape should be [2,3]');
  assert(t.data[5] === 6, 'flat data wrong');
});

test('element-wise add with broadcasting', () => {
  const a = tensor([[1, 2, 3], [4, 5, 6]], false, [2, 3]);
  const b = tensor([10, 20, 30], false, [3]);
  const c = add(a, b);
  assert(c.shape[0] === 2 && c.shape[1] === 3, 'bad broadcast shape');
  assert(c.at(1, 2) === 36, 'broadcast add wrong: ' + c.at(1, 2));
});

test('element-wise mul and div', () => {
  const a = tensor([2, 4, 6], false, [3]);
  const b = tensor([1, 2, 3], false, [3]);
  assert(mul(a, b).at(2) === 18, 'mul wrong');
  assert(div(a, b).at(2) === 2, 'div wrong');
});

/* ==========================================================================
 * 2. Matrix operations
 * ========================================================================== */
console.log('');
console.log('Matrix operations');
test('matmul matches hand calculation', () => {
  const a = tensor([[1, 2], [3, 4]], false, [2, 2]);
  const b = tensor([[5, 6], [7, 8]], false, [2, 2]);
  const c = matmul(a, b);
  assert(c.at(0, 0) === 19, 'c00=' + c.at(0, 0));
  assert(c.at(0, 1) === 22, 'c01=' + c.at(0, 1));
  assert(c.at(1, 0) === 43, 'c10=' + c.at(1, 0));
  assert(c.at(1, 1) === 50, 'c11=' + c.at(1, 1));
});

test('batched matmul', () => {
  const a = tensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]], false, [2, 2, 2]);
  const b = tensor([[2, 3], [4, 5]], false, [2, 2]);
  const c = matmul(a, b); // [2,2,2]
  // batch 0: identity @ b = b; batch 1: all-ones @ b = [[6,8],[6,8]]
  assert(c.at(0, 0, 0) === 2 && c.at(1, 1, 1) === 8, 'batched matmul wrong');
});

test('transpose swaps axes', () => {
  const a = tensor([[1, 2, 3], [4, 5, 6]], false, [2, 3]);
  const t = transpose(a);
  assert(t.shape[0] === 3 && t.shape[1] === 2, 'bad transpose shape');
  assert(t.at(0, 0) === 1 && t.at(0, 1) === 4 && t.at(2, 0) === 3, 'transpose values wrong');
});

test('reshape preserves element order', () => {
  const a = tensor([1, 2, 3, 4, 5, 6], false, [2, 3]);
  const r = reshape(a, [3, 2]);
  assert(r.shape[0] === 3 && r.shape[1] === 2, 'bad reshape shape');
  assert(r.at(1, 0) === 3 && r.at(2, 1) === 6, 'reshape order wrong');
});

test('softmax rows sum to 1', () => {
  const a = tensor([[1, 2, 3], [0, 0, 0]], false, [2, 3]);
  const s = softmax(a);
  const r0 = s.at(0, 0) + s.at(0, 1) + s.at(0, 2);
  assert(almost(r0, 1, 1e-9), 'row 0 sums to ' + r0);
  assert(almost(s.at(1, 0), 1 / 3, 1e-9), 'uniform row wrong');
});

test('softmax obeys causal mask', () => {
  const a = tensor([[1, 1, 1], [1, 1, 1]], false, [2, 3]);
  const mask = new Uint8Array([0, 1, 1, 0, 0, 1]); // token 0 sees only itself
  const s = softmax(a, mask);
  assert(almost(s.at(0, 0), 1, 1e-9), 'masked softmax should be 1 at [0,0]');
  assert(s.at(0, 1) === 0 && s.at(0, 2) === 0, 'future positions should be 0');
  assert(almost(s.at(1, 0) + s.at(1, 1), 1, 1e-9), 'row 1 should normalize over first two');
});

/* ==========================================================================
 * 3. Gradient checks
 * ========================================================================== */
console.log('');
console.log('Gradient checks (backprop vs numerical)');

test('grad: add', () => {
  const a = rand([3, 4]);
  const b = rand([4]);
  const err = gradCheck(() => sumAll(add(a, b)), [a, b]);
  assert(err < 1e-5, 'relative error ' + err);
});

test('grad: mul', () => {
  const a = rand([3, 4]);
  const b = rand([4]);
  const err = gradCheck(() => sumAll(mul(a, b)), [a, b]);
  assert(err < 1e-5, 'relative error ' + err);
});

test('grad: matmul', () => {
  const a = rand([2, 3]);
  const b = rand([3, 4]);
  const err = gradCheck(() => sumAll(matmul(a, b)), [a, b]);
  assert(err < 1e-4, 'relative error ' + err);
});

test('grad: softmax (with weights)', () => {
  const a = rand([3, 4]);
  const w = rand([3, 4]);
  const err = gradCheck(() => sumAll(mul(softmax(a), w)), [a]);
  assert(err < 1e-4, 'relative error ' + err);
});

test('grad: gelu', () => {
  const a = rand([5]);
  const err = gradCheck(() => sumAll(gelu(a)), [a]);
  assert(err < 1e-4, 'relative error ' + err);
});

test('grad: layerNorm', () => {
  const x = rand([3, 4]);
  const g = rand([4]);
  const b = rand([4]);
  const w = rand([3, 4]);
  const err = gradCheck(() => sumAll(mul(layerNorm(x, g, b), w)), [x, g, b]);
  assert(err < 1e-3, 'relative error ' + err);
});

test('grad: cross entropy', () => {
  const logits = rand([4, 5]);
  const targets = tensor([0, 3, 2, 4], false, [4]);
  const err = gradCheck(() => crossEntropy(logits, targets), [logits]);
  assert(err < 1e-4, 'relative error ' + err);
});

test('grad: self-attention', () => {
  const B = 2;
  const Tt = 3;
  const D = 4;
  const H = 2;
  const x = rand([B, Tt, D]);
  const p = {
    wq: rand([D, D]), bq: zeros([D], true),
    wk: rand([D, D]), bk: zeros([D], true),
    wv: rand([D, D]), bv: zeros([D], true),
    wo: rand([D, D]), bo: zeros([D], true),
  };
  const mask = new Uint8Array(Tt * Tt);
  for (let i = 0; i < Tt; i++) for (let j = 0; j < Tt; j++) if (j > i) mask[i * Tt + j] = 1;
  const cfg = { dModel: D, nHeads: H };
  const err = gradCheck(() => sumAll(selfAttention(x, p, cfg, mask)), [x, p.wq, p.wk, p.wv, p.wo]);
  assert(err < 1e-3, 'relative error ' + err);
});

/* ==========================================================================
 * 4. Tokenizers
 * ========================================================================== */
console.log('');
console.log('Tokenizers');
const CORPUS = 'the cat sat on the mat. the dog sat on the log. hello world!';

test('char tokenizer round-trips', () => {
  const tok = createTokenizer('char', CORPUS);
  const ids = tok.encode('the cat');
  assert(tok.decode(ids) === 'the cat', 'char round-trip failed: ' + tok.decode(ids));
  assert(tok.vocabSize >= 4, 'vocab too small');
});

test('word tokenizer encodes and decodes', () => {
  const tok = createTokenizer('word', CORPUS);
  const ids = tok.encode('the cat sat');
  assert(ids.length === 3, 'expected 3 word tokens, got ' + ids.length);
  const text = tok.decode(ids);
  assert(text.indexOf('cat') !== -1, 'decode lost a word: ' + text);
});

test('bpe tokenizer learns merges and round-trips', () => {
  const tok = createTokenizer('bpe', CORPUS, 80);
  assert(tok.merges.length > 0, 'BPE learned no merges');
  const sample = 'the cat sat';
  const ids = tok.encode(sample);
  assert(tok.decode(ids) === sample, 'BPE round-trip failed: ' + tok.decode(ids));
});

test('tokenizer save/load round-trips', () => {
  for (const type of ['char', 'word', 'bpe']) {
    const tok = createTokenizer(type, CORPUS, 80);
    const restored = BaseTokenizer.load(tok.save());
    assert(restored.vocabSize === tok.vocabSize, type + ': vocab size changed');
    assert(
      JSON.stringify(restored.encode('the cat')) === JSON.stringify(tok.encode('the cat')),
      type + ': encoding changed after save/load'
    );
  }
});

/* ==========================================================================
 * 5. Model
 * ========================================================================== */
console.log('');
console.log('Model');
function tinyModel() {
  const tok = createTokenizer('char', CORPUS);
  const model = new TinyModel({ dModel: 16, nHeads: 2, nLayers: 1, blockSize: 8, tieWeights: true }, tok);
  return { model, tok };
}

test('model forward produces correct logit shape', () => {
  const { model } = tinyModel();
  const ids = [[1, 2, 3, 4]];
  const logits = model.forwardIds(ids);
  assert(logits.shape.length === 3, 'logits should be rank 3');
  assert(logits.shape[0] === 1 && logits.shape[1] === 4, 'wrong batch/time dims');
  assert(logits.shape[2] === model.config.vocabSize, 'wrong vocab dim');
  for (let i = 0; i < logits.size; i++) assert(Number.isFinite(logits.data[i]), 'non-finite logit');
});

test('model save/load preserves weights', () => {
  const { model, tok } = tinyModel();
  const text = model.saveModel();
  const restored = TinyModel.loadModel(text);
  assert(restored.config.nLayers === model.config.nLayers, 'config changed');
  assert(restored.parameterCount() === model.parameterCount(), 'parameter count changed');
  const a = model.parameters();
  const b = restored.parameters();
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    assert(a[i].name === b[i].name, 'parameter name mismatch');
    for (let k = 0; k < a[i].tensor.size; k++) {
      maxDiff = Math.max(maxDiff, Math.abs(a[i].tensor.data[k] - b[i].tensor.data[k]));
    }
  }
  assert(maxDiff < 1e-6, 'weights changed after save/load: max diff ' + maxDiff);
});

test('untied model save/load preserves weights', () => {
  const tok = createTokenizer('char', CORPUS);
  const model = new TinyModel({ dModel: 16, nHeads: 2, nLayers: 1, blockSize: 8, tieWeights: false }, tok);
  const restored = TinyModel.loadModel(model.saveModel());
  assert(!restored.config.tieWeights, 'tieWeights flag lost');
  let maxDiff = 0;
  for (let k = 0; k < model.lmHeadW.size; k++) {
    maxDiff = Math.max(maxDiff, Math.abs(model.lmHeadW.data[k] - restored.lmHeadW.data[k]));
  }
  assert(maxDiff < 1e-6, 'untied head changed');
});

test('generate returns the requested number of tokens', () => {
  const { model } = tinyModel();
  const out = model.generate([1, 2, 3], { maxNewTokens: 5, greedy: true, eosId: -1 });
  assert(out.length === 5, 'expected 5 tokens, got ' + out.length);
});

/* ==========================================================================
 * 6. Training actually learns
 * ========================================================================== */
console.log('');
console.log('Training');
test('loss decreases over a short training run', () => {
  const tok = createTokenizer('char', 'the cat sat on the mat. the dog sat on the log. '.repeat(20));
  const model = new TinyModel({ dModel: 16, nHeads: 2, nLayers: 1, blockSize: 8, tieWeights: true }, tok);
  const trainer = new Trainer({
    model,
    tokenizer: tok,
    corpus: 'the cat sat on the mat. the dog sat on the log. '.repeat(20),
    config: { batchSize: 4, blockSize: 8, lr: 0.01, warmup: 5, maxSteps: 60, evalInterval: 0, checkpointInterval: 0 },
  });
  const first = trainer.trainStep();
  let last = first;
  for (let i = 0; i < 59; i++) last = trainer.trainStep();
  assert(last < first, 'loss did not decrease: first=' + first.toFixed(4) + ' last=' + last.toFixed(4));
});

/* ==========================================================================
 * 7. Knowledge plugins
 * ========================================================================== */
console.log('');
console.log('Knowledge');
test('knowledge search finds an answer', () => {
  const kb = new KnowledgeBase();
  kb.addPlugin({
    topic: 'test',
    entries: [{ keywords: ['cat', 'cats'], answer: 'Cats are great.' }],
  });
  const hit = kb.search('tell me about cats please');
  assert(hit && hit.answer === 'Cats are great.', 'no knowledge match');
  assert(kb.search('quantum chromodynamics') === null, 'should not match unrelated query');
});

/* ==========================================================================
 * Summary
 * ========================================================================== */
console.log('');
console.log('=================');
console.log('  ' + passed + ' passed, ' + failed + ' failed');
console.log('');
process.exit(failed === 0 ? 0 : 1);
