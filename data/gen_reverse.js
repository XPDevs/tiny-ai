/*
 * data/gen_reverse.js — "reverse a b c d" -> "d c b a"
 * ---------------------------------------------------------------------------
 * Same split philosophy as copy, but the target is a reordering rather than an
 * identity map, so val_compositional also tests *ordering* of seen pieces.
 *   val_systematic      sequence contains an unseen word ("fox"/"owl")
 *   val_compositional   length 5-6 sequences of seen words
 *   val_indist          random holdout
 *   train               lengths 2-4, seen words
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'reverse';
const SEED = 202;
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

  const sys = [];
  G.fill(sys, 900, 60000, (r, push) => {
    const len = G.int(r, 2, 4);
    const s = seq(r, WORDS, len);
    s[G.int(r, 0, len - 1)] = G.pick(r, SYS_WORDS);
    push(G.ex('reverse ' + s.join(' '), s.slice().reverse().join(' '), 'val_systematic', ['len:' + len]));
  }, rng);

  const comp = [];
  G.fill(comp, 800, 60000, (r, push) => {
    const len = G.int(r, 5, 6);
    const s = seq(r, WORDS, len);
    push(G.ex('reverse ' + s.join(' '), s.slice().reverse().join(' '), 'val_compositional', ['len:' + len]));
  }, rng);

  const trainPool = [];
  G.fill(trainPool, 6000, 200000, (r, push) => {
    const len = G.int(r, 2, 4);
    const s = seq(r, WORDS, len);
    push(G.ex('reverse ' + s.join(' '), s.slice().reverse().join(' '), 'train', ['len:' + len]));
  }, rng);
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
