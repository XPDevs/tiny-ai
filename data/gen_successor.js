/*
 * data/gen_successor.js — the simplest "learn the rule, not the table" task
 * ---------------------------------------------------------------------------
 *   "next 5"        -> "6"
 *   "previous 40"   -> "39"
 *   "next next 5"   -> "7"
 *
 * This task exists specifically to show genuine generalization rather than
 * memorization: a single target whose value is `n +/- 1` can be *computed*
 * from the input. A model that only memorised (input -> output) pairs will fail
 * on the systematic split, whereas one that learned the ordering of numbers
 * will generalize.
 *
 *   val_systematic      n in 90..99, a range never seen in training. The answer
 *                       cannot be copied (it is not in the source), so it must
 *                       be produced by the rule.
 *   val_compositional   two steps ("next next n", "n plus two") built from the
 *                       seen single-step pieces.
 *   val_indist          random holdout
 *   train               single-step next/previous for n in 3..89
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'successor';
const SEED = 1010;
const TARGET_TOKENS = 12000;

const SYS_LO = 90, SYS_HI = 99;   // never seen in training
const TRAIN_LO = 3, TRAIN_HI = 89;

const NEXT = [
  (n) => 'next ' + n,
  (n) => 'the number after ' + n,
  (n) => n + ' plus one',
  (n) => 'what comes after ' + n,
  (n) => 'increment ' + n,
  (n) => 'the successor of ' + n,
  (n) => 'add one to ' + n,
  (n) => 'count up from ' + n
];

const PREV = [
  (n) => 'previous ' + n,
  (n) => 'the number before ' + n,
  (n) => n + ' minus one',
  (n) => 'what comes before ' + n,
  (n) => 'decrement ' + n,
  (n) => 'the predecessor of ' + n,
  (n) => 'subtract one from ' + n,
  (n) => 'count down from ' + n
];

function single(list, n, split) {
  for (const p of NEXT) list.push(G.ex(p(n), String(n + 1), split, ['op:next', 'n:' + n]));
  for (const p of PREV) list.push(G.ex(p(n), String(n - 1), split, ['op:prev', 'n:' + n]));
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  for (let n = SYS_LO; n <= SYS_HI; n++) single(sys, n, 'val_systematic');

  const comp = [];
  for (let n = TRAIN_LO + 2; n <= TRAIN_HI; n++) {
    comp.push(G.ex('next next ' + n, String(n + 2), 'val_compositional', ['op:next2', 'n:' + n]));
    comp.push(G.ex(n + ' plus two', String(n + 2), 'val_compositional', ['op:next2', 'n:' + n]));
    comp.push(G.ex('previous previous ' + n, String(n - 2), 'val_compositional', ['op:prev2', 'n:' + n]));
    comp.push(G.ex(n + ' minus two', String(n - 2), 'val_compositional', ['op:prev2', 'n:' + n]));
  }

  const trainPool = [];
  for (let n = TRAIN_LO; n <= TRAIN_HI; n++) single(trainPool, n, 'train');
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
