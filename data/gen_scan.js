/*
 * data/gen_scan.js — SCAN-style command expansion
 * ---------------------------------------------------------------------------
 *   "walk twice"                 -> "walk walk"
 *   "jump and run thrice"        -> "jump run run run"
 *
 *   val_systematic      the primitive "look" is never seen in training, so any
 *                       command mentioning it is a systematic test.
 *   val_compositional   the primitive "jump" with the modifier "thrice" is
 *                       never seen together in training (unseen combination).
 *   val_indist          random holdout
 *   train               the rest
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'scan';
const SEED = 707;
const TARGET_TOKENS = 20000;

const SEEN_PRIMS = ['jump', 'walk', 'run', 'spin', 'sit', 'hop', 'clap'];
const SYS_PRIM = 'look';
const ALL_PRIMS = SEEN_PRIMS.concat([SYS_PRIM]);
const MODS = { once: 1, twice: 2, thrice: 3 };
const CONNS = ['and', 'after', 'then', 'before'];

function text(p, m) { return p + ' ' + m; }
function body(p, m) { return new Array(MODS[m] + 1).join(p + ' ').trim(); }
function isComp(p, m) { return p === 'jump' && m === 'thrice'; }

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  const comp = [];
  const trainPool = [];

  function put(src, tgt, split, tags) {
    const e = G.ex(src, tgt, split, tags);
    if (split === 'val_systematic') sys.push(e);
    else if (split === 'val_compositional') comp.push(e);
    else trainPool.push(e);
  }

  for (const p of ALL_PRIMS) {
    for (const m of Object.keys(MODS)) {
      const tags = ['prim:' + p, 'mod:' + m];
      const split = p === SYS_PRIM ? 'val_systematic' : (isComp(p, m) ? 'val_compositional' : 'train');
      put(text(p, m), body(p, m), split, tags);
    }
  }

  for (const p of ALL_PRIMS) {
    for (const m of Object.keys(MODS)) {
      for (const p2 of ALL_PRIMS) {
        for (const m2 of Object.keys(MODS)) {
          const src = text(p, m);
          const tgt = body(p, m) + ' ' + body(p2, m2);
          const tags = ['prim:' + p, 'mod:' + m, 'prim2:' + p2, 'mod2:' + m2];
          for (const conn of CONNS) {
            const split = (p === SYS_PRIM || p2 === SYS_PRIM) ? 'val_systematic'
              : (isComp(p, m) || isComp(p2, m2)) ? 'val_compositional' : 'train';
            put(src + ' ' + conn + ' ' + text(p2, m2), tgt, split, tags.concat(['conn:' + conn]));
          }
        }
      }
    }
  }

  const withHoldout = G.withIndist(trainPool, 0.11, rng);
  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
