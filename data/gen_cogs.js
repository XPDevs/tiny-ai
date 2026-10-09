/*
 * data/gen_cogs.js — COGS-style meaning-to-form mapping
 * ---------------------------------------------------------------------------
 *   "the cat likes the ball"        -> "likes ( cat , ball )"
 *
 *   val_systematic      the verb "chases" never appears in training, so any
 *                       sentence using it is a systematic test.
 *   val_compositional   the subject/object pair (girl, boat) is held out of
 *                       training (unseen combination of seen words).
 *   val_indist          random holdout
 *   train               the rest
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'cogs';
const SEED = 808;
const TARGET_TOKENS = 15000;

const SUBJECTS = ['cat', 'dog', 'boy', 'girl', 'man', 'woman', 'robot'];
const VERBS = ['likes', 'sees', 'chases', 'hates', 'holds', 'wants'];
const OBJECTS = ['ball', 'mouse', 'cake', 'boat', 'book', 'key', 'cup', 'hat'];

const SYS_VERB = 'chases';

const PHRASINGS = [
  (s, v, o) => 'the ' + s + ' ' + v + ' the ' + o,
  (s, v, o) => s + ' ' + v + ' the ' + o,
  (s, v, o) => 'the ' + s + ' that ' + v + ' the ' + o,
  (s, v, o) => 'i think the ' + s + ' ' + v + ' the ' + o
];

function target(v, s, o) { return v + ' ( ' + s + ' , ' + o + ' )'; }
function isComp(s, o) { return s === 'girl' && o === 'boat'; }

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  const comp = [];
  const trainPool = [];

  for (const s of SUBJECTS) {
    for (const v of VERBS) {
      for (const o of OBJECTS) {
        const split = v === SYS_VERB ? 'val_systematic'
          : isComp(s, o) ? 'val_compositional' : 'train';
        const tags = ['subj:' + s, 'verb:' + v, 'obj:' + o];
        const tgt = target(v, s, o);
        for (const ph of PHRASINGS) {
          const e = G.ex(ph(s, v, o), tgt, split, tags.slice());
          if (split === 'val_systematic') sys.push(e);
          else if (split === 'val_compositional') comp.push(e);
          else trainPool.push(e);
        }
      }
    }
  }

  const withHoldout = G.withIndist(trainPool, 0.11, rng);
  return sys.concat(comp, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
