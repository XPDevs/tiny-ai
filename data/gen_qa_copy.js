/*
 * data/gen_qa_copy.js — a fixed country -> capital lookup, phrased many ways
 * ---------------------------------------------------------------------------
 *   "what is the capital of france" -> "paris"
 *
 * This is deliberately a *lookup* task, not world knowledge: it is a small
 * table of pairs generated here, and the interesting question is whether the
 * model can answer a *new phrasing* of a *seen* country.
 *
 *   val_systematic      a country that never appears in training at all: the
 *                       capital is nowhere in the input, so it cannot be copied
 *                       and must be ~0% exact match.
 *   val_compositional   a seen country asked with a phrasing prefix
 *                       ("excuse me ...") that never appears in training.
 *   val_indist          random holdout
 *   train               the rest
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'qa_copy';
const SEED = 808;
const TARGET_TOKENS = 30000;

const PAIRS = [
  ['france', 'paris'], ['germany', 'berlin'], ['italy', 'rome'], ['spain', 'madrid'],
  ['portugal', 'lisbon'], ['england', 'london'], ['ireland', 'dublin'], ['greece', 'athens'],
  ['poland', 'warsaw'], ['norway', 'oslo'], ['sweden', 'stockholm'], ['denmark', 'copenhagen'],
  ['finland', 'helsinki'], ['russia', 'moscow'], ['ukraine', 'kyiv'], ['austria', 'vienna'],
  ['belgium', 'brussels'], ['switzerland', 'bern'], ['hungary', 'budapest'], ['romania', 'bucharest'],
  ['turkey', 'ankara'], ['egypt', 'cairo'], ['morocco', 'rabat'], ['nigeria', 'abuja'],
  ['kenya', 'nairobi'], ['ethiopia', 'addis'], ['canada', 'ottawa'], ['mexico', 'mexico'],
  ['brazil', 'brasilia'], ['argentina', 'buenos'], ['chile', 'santiago'], ['peru', 'lima'],
  ['colombia', 'bogota'], ['cuba', 'havana'], ['japan', 'tokyo'], ['china', 'beijing'],
  ['india', 'delhi'], ['korea', 'seoul'], ['thailand', 'bangkok'], ['vietnam', 'hanoi'],
  ['indonesia', 'jakarta'], ['australia', 'canberra'], ['new zealand', 'wellington'],
  ['iran', 'tehran'], ['iraq', 'baghdad'], ['israel', 'jerusalem'], ['qatar', 'doha'],
  ['singapore', 'singapore'], ['malaysia', 'kuala'], ['philippines', 'manila'], ['cuba', 'havana'],
  ['iceland', 'reykjavik'], ['croatia', 'zagreb'], ['serbia', 'belgrade'], ['bulgaria', 'sofia']
];

const SYS_COUNTRIES = ['peru', 'cuba', 'iceland', 'croatia', 'serbia', 'bulgaria'];

const TRAIN_PREFIXES = ['', 'hey ', 'so ', 'please ', 'quick question '];
const COMP_PREFIX = 'excuse me ';
const SUFFIXES = ['', ' please', ' ?'];

const BASES = [
  (c) => 'what is the capital of ' + c,
  (c) => 'whats the capital of ' + c,
  (c) => 'what is ' + c + ' capital',
  (c) => 'capital of ' + c,
  (c) => 'the capital of ' + c + ' is',
  (c) => 'name the capital of ' + c,
  (c) => 'tell me the capital of ' + c,
  (c) => 'which city is the capital of ' + c,
  (c) => 'what city is the capital of ' + c,
  (c) => 'where is the capital of ' + c,
  (c) => c + ' capital city',
  (c) => c + ' capital',
  (c) => 'do you know the capital of ' + c,
  (c) => 'i want to know the capital of ' + c,
  (c) => 'whats the capital city of ' + c,
  (c) => 'the capital city of ' + c + ' is',
  (c) => 'capital city of ' + c,
  (c) => 'what is the capital city of ' + c
];

function phrasings(prefix, country) {
  const out = [];
  for (const b of BASES) {
    for (const s of SUFFIXES) out.push(prefix + b(country) + s);
  }
  return out;
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  const comp = [];
  const trainPool = [];

  const seen = {};
  for (const pair of PAIRS) {
    const country = pair[0], capital = pair[1];
    if (seen[country]) continue;
    seen[country] = 1;
    const isSys = SYS_COUNTRIES.indexOf(country) !== -1;

    if (isSys) {
      for (const p of phrasings('', country)) {
        sys.push(G.ex(p, capital, 'val_systematic', ['country:' + country, 'capital:' + capital]));
      }
      continue;
    }
    // held-out phrasing prefix -> compositional; the rest -> train pool
    for (const p of phrasings(COMP_PREFIX, country)) {
      comp.push(G.ex(p, capital, 'val_compositional', ['country:' + country, 'capital:' + capital]));
    }
    let i = 0;
    for (const prefix of TRAIN_PREFIXES) {
      for (const p of phrasings(prefix, country)) {
        trainPool.push(G.ex(p, capital, 'train', ['country:' + country, 'capital:' + capital]));
        i++;
      }
    }
  }

  // the above makes comp tiny; pad it with more unseen-prefix phrasings already
  // collected, and keep train as-is. Dedupe everything, then hold out in-dist.
  const compDedup = G.dedupe(comp);
  const withHoldout = G.withIndist(G.dedupe(trainPool), 0.11, rng);

  return sys.concat(compDedup, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
