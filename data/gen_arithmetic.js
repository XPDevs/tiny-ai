/*
 * data/gen_arithmetic.js — small-number addition with several phrasings
 * ---------------------------------------------------------------------------
 *   "add 12 34" / "what is 12 plus 34" / "sum 12 and 34"  ->  "46"
 *
 *   val_systematic     an operand from a range never seen (42..45): unseen
 *                      operand values, so the rule has to generalise.
 *   val_compositional  both operands inside 21..30: seen values, but that
 *                      pair-region is held out of training.
 *   val_indist         random holdout
 *   train              operands 0..41 outside the compositional region
 *
 * Results stay small (<= 90) so the output vocabulary is tiny.
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'arithmetic';
const SEED = 404;
const TARGET_TOKENS = 50000;

const MAX = 45;
const SYS_MIN = 42;       // 42..45 never appear in training at all
const COMP_LO = 21, COMP_HI = 30;

const PHRASINGS = [
  (a, b) => 'add ' + a + ' ' + b,
  (a, b) => 'what is ' + a + ' plus ' + b,
  (a, b) => 'sum ' + a + ' and ' + b,
  (a, b) => 'compute ' + a + ' plus ' + b,
  (a, b) => 'what is the sum of ' + a + ' and ' + b
];

function isSys(a, b) { return a >= SYS_MIN || b >= SYS_MIN; }
function isComp(a, b) {
  return a >= COMP_LO && a <= COMP_HI && b >= COMP_LO && b <= COMP_HI;
}

function emit(list, a, b, split) {
  for (const p of PHRASINGS) {
    list.push(G.ex(p(a, b), String(a + b), split, ['a:' + a, 'b:' + b, 'sum:' + (a + b)]));
  }
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  for (let a = 0; a <= MAX; a++) for (let b = 0; b <= MAX; b++) if (isSys(a, b)) emit(sys, a, b, 'val_systematic');

  const comp = [];
  for (let a = COMP_LO; a <= COMP_HI; a++) for (let b = COMP_LO; b <= COMP_HI; b++) emit(comp, a, b, 'val_compositional');

  const trainPool = [];
  for (let a = 0; a <= MAX; a++) {
    for (let b = 0; b <= MAX; b++) {
      if (isSys(a, b) || isComp(a, b)) continue;
      emit(trainPool, a, b, 'train');
    }
  }
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
