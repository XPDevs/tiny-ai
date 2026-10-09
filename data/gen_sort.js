/*
 * data/gen_sort.js — "sort 3 12 7" -> "3 7 12"   (ascending)
 * ---------------------------------------------------------------------------
 *   val_systematic     the sequence contains a number from a range never seen
 *                      in training (80..99). Pure ordering of unseen values;
 *                      the model must copy the tokens and order them by rule.
 *   val_compositional  sequences of length 4-5 built from seen numbers
 *                      (training only ever uses lengths 2-3): unseen orderings
 *                      and unseen lengths.
 *   val_indist         random holdout
 *   train              lengths 2-3, numbers 1..79
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'sort';
const SEED = 303;
const TARGET_TOKENS = 50000;

const SEEN = [];
for (let n = 1; n <= 79; n++) SEEN.push(String(n));
const SYS = [];
for (let n = 80; n <= 99; n++) SYS.push(String(n));

function seq(rng, pool, len) {
  const chosen = Object.create(null);
  const out = [];
  while (out.length < len) {
    const v = G.pick(rng, pool);
    if (chosen[v]) continue;
    chosen[v] = 1;
    out.push(v);
  }
  return out;
}

function numeric(a, b) { return Number(a) - Number(b); }

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  // systematic: contains at least one number from the unseen range
  const sys = [];
  G.fill(sys, 1000, 80000, (r, push) => {
    const len = G.int(r, 2, 4);
    const s = seq(r, SEEN, len);
    s[G.int(r, 0, len - 1)] = G.pick(r, SYS);
    push(G.ex('sort ' + s.join(' '), s.slice().sort(numeric).join(' '), 'val_systematic', ['len:' + len]));
  }, rng);

  // compositional: longer sequences, seen numbers only
  const comp = [];
  G.fill(comp, 900, 80000, (r, push) => {
    const len = G.int(r, 4, 5);
    const s = seq(r, SEEN, len);
    push(G.ex('sort ' + s.join(' '), s.slice().sort(numeric).join(' '), 'val_compositional', ['len:' + len]));
  }, rng);

  const trainPool = [];
  G.fill(trainPool, 9000, 300000, (r, push) => {
    const len = G.int(r, 2, 3);
    const s = seq(r, SEEN, len);
    push(G.ex('sort ' + s.join(' '), s.slice().sort(numeric).join(' '), 'train', ['len:' + len]));
  }, rng);
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
