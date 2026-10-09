/*
 * data/gen_copy.js — "copy a b c d" -> "a b c d"
 * ---------------------------------------------------------------------------
 * Splits
 *   val_systematic      sequences containing a token never seen in training
 *                       ("fox", "owl"), so it can only be solved by the pointer.
 *   val_compositional   longer sequences (5-6 words) built only from seen words:
 *                       an unseen *combination* of seen pieces.
 *   val_indist          random holdout from the training distribution.
 *   train               the rest (lengths 2-4, seen words only).
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'copy';
const SEED = 101;
const TARGET_TOKENS = 30000;

const WORDS = [
  'red', 'green', 'blue', 'black', 'white', 'gold', 'silver', 'pink',
  'cat', 'dog', 'bear', 'wolf', 'deer', 'moth', 'frog',
  'moon', 'star', 'rain', 'snow', 'fire', 'wind', 'leaf', 'oak', 'pine'
];
const SYS_WORDS = ['fox', 'owl'];

function seq(rng, pool, len) {
  const out = [];
  for (let i = 0; i < len; i++) out.push(G.pick(rng, pool));
  return out;
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  // systematic: sequences that contain at least one unseen word
  const sys = [];
  G.fill(sys, 900, 60000, (r, push) => {
    const len = G.int(r, 2, 4);
    const s = seq(r, WORDS, len);
    s[G.int(r, 0, len - 1)] = G.pick(r, SYS_WORDS);
    push(G.ex('copy ' + s.join(' '), s.join(' '), 'val_systematic', ['len:' + len]));
  }, rng);

  // compositional: long sequences of seen words only (unseen combinations)
  const comp = [];
  G.fill(comp, 800, 60000, (r, push) => {
    const len = G.int(r, 5, 6);
    const s = seq(r, WORDS, len);
    push(G.ex('copy ' + s.join(' '), s.join(' '), 'val_compositional', ['len:' + len]));
  }, rng);

  // train pool (lengths 2-4, seen words) + random in-distribution holdout
  const trainPool = [];
  G.fill(trainPool, 6000, 200000, (r, push) => {
    const len = G.int(r, 2, 4);
    const s = seq(r, WORDS, len);
    push(G.ex('copy ' + s.join(' '), s.join(' '), 'train', ['len:' + len]));
  }, rng);
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
