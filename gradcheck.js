#!/usr/bin/env node
/*
 * gradcheck.js — finite-difference gradient check for tiny-ai.js
 *
 * Builds a deliberately tiny model (no dropout), then compares the analytic
 * gradients produced by _backwardExample against central finite differences on
 * a sample of parameters. This is the safety net for the multi-layer,
 * bidirectional, coverage-enabled backward pass.
 *
 * Usage:  node gradcheck.js [seed] [checks]
 */
'use strict';

const AI = require('./tiny-ai.js');

const seed = parseInt(process.argv[2] || '1337', 10);
const CHECKS = parseInt(process.argv[3] || '40', 10);

// A handful of examples purely so the model builds a meaningful vocabulary.
const SAMPLE = [
  { src: 'book a table for 2 at noon for alice', tgt: 'booked a table for 2 at noon under alice' },
  { src: 'hi', tgt: 'hello there' },
  { src: 'copy red green blue', tgt: 'red green blue' },
  { src: 'sort 3 1 2', tgt: '1 2 3' }
];

const E = process.env;
AI.init({
  seed: seed,
  hidden: 5,
  embed: 4,
  layers: parseInt(E.LAYERS || '2', 10),
  bidir: (E.BIDIR || '1') !== '0',
  attnHeads: (E.HEADS || '1') !== '0',
  coverage: (E.COV || '1') !== '0',
  labelSmoothing: parseFloat(E.LS || '0.05'),
  wordDropout: 0,
  embDropout: 0,
  hiddenDropout: 0,
  targetSteps: 1,
  dataset: SAMPLE
});

const m = AI.model();

// A couple of short examples that exercise the copy mechanism and punctuation.
const ex = m._encodePair({ src: 'book a table for 2 at noon for alice', tgt: 'booked a table for 2 at noon under alice' });
const ex2 = m._encodePair({ src: 'hi', tgt: 'hello there' });

function lossOf(e) {
  return m._forwardExample(e.src, e.tgt, false).loss;
}

m._zeroGrads();
const gl = m._forwardExample(ex.src, ex.tgt, false);
m._backwardExample(gl, 1);
const gl2 = m._forwardExample(ex2.src, ex2.tgt, false);
m._backwardExample(gl2, 1);

const params = m._allParams();
const eps = 1e-5;

let worst = 0, worstName = '';
let checked = 0, passed = 0;

// indices of params with nonzero gradient (sampled)
const candidates = [];
for (let pi = 0; pi < params.length; pi++) {
  const [p, g] = params[pi];
  for (let j = 0; j < p.length; j++) {
    if (Math.abs(g[j]) > 1e-7) candidates.push([pi, j]);
  }
}
// deterministic sample
let s = seed >>> 0;
function rnd() { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }
for (let i = candidates.length - 1; i > 0; i--) {
  const j = (rnd() * (i + 1)) | 0;
  const t = candidates[i]; candidates[i] = candidates[j]; candidates[j] = t;
}

const N = Math.min(CHECKS, candidates.length);
console.log('vocab', m.V, 'params', m.paramCount(), 'nonzero-grad entries', candidates.length, 'checking', N);

for (let c = 0; c < N; c++) {
  const [pi, j] = candidates[c];
  const [p, g] = params[pi];
  const orig = p[j];
  p[j] = orig + eps;
  const lp = lossOf(ex) + lossOf(ex2);
  p[j] = orig - eps;
  const lm = lossOf(ex) + lossOf(ex2);
  p[j] = orig;
  const numeric = (lp - lm) / (2 * eps);
  const analytic = g[j];
  const denom = Math.max(1e-6, Math.abs(numeric) + Math.abs(analytic));
  const absdiff = Math.abs(numeric - analytic);
  const rel = absdiff / denom;
  checked++;
  if (rel < 1e-3 || absdiff < 1e-6) passed++;
  if (rel > worst) { worst = rel; worstName = 'param[' + pi + '][' + j + ']'; }
  if (rel >= 1e-3 && absdiff >= 1e-6) {
    console.log('  MISMATCH', 'p' + pi, 'j' + j,
      'analytic', analytic.toExponential(6),
      'numeric', numeric.toExponential(6),
      'rel', rel.toExponential(3));
  }
}

console.log('checked', checked, 'passed', passed, 'worst rel err', worst.toExponential(3), '(' + worstName + ')');
if (passed !== checked) {
  console.log('GRADCHECK FAILED');
  process.exit(1);
}
console.log('GRADCHECK PASSED');
