/*
 * data/gen_booking.js — slot-filling that stresses the copy gate
 * ---------------------------------------------------------------------------
 *   "book a table for 4 at 8pm for alice"
 *       -> "booked a table for 4 at 8pm under alice"
 *
 * Slots: name, size, time. Several phrasings so the template itself varies.
 *
 *   val_systematic      the name is one never seen in training at all.
 *   val_compositional   a seen name with a size that never appears with a name
 *                       in training (sizes 11/12 are held out of train).
 *   val_indist          random holdout
 *   train               the rest
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'booking';
const SEED = 505;
const TARGET_TOKENS = 50000;

const NAMES = [
  'alice', 'bob', 'carol', 'dave', 'erin', 'frank', 'grace', 'heidi',
  'ivan', 'judy', 'karl', 'laura', 'mallory', 'nina', 'oscar', 'peggy',
  'rupert', 'sybil', 'trent', 'uma', 'victor', 'wendy', 'xavier', 'dana'
];
const SYS_NAMES = ['quinn', 'xena', 'yusuf'];

const SIZES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'];
const COMP_SIZES = ['11', '12'];

const TIMES = ['noon', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', 'morning', 'evening'];

const TEMPLATES = [
  (s, t, n) => ['book a table for ' + s + ' at ' + t + ' for ' + n, 'booked a table for ' + s + ' at ' + t + ' under ' + n],
  (s, t, n) => ['reserve a table for ' + s + ' at ' + t + ' under ' + n, 'reserved a table for ' + s + ' at ' + t + ' under ' + n],
  (s, t, n) => ['book ' + s + ' seats at ' + t + ' for ' + n, 'booked ' + s + ' seats at ' + t + ' under ' + n],
  (s, t, n) => ['table for ' + s + ' at ' + t + ' for ' + n, 'table for ' + s + ' at ' + t + ' under ' + n],
  (s, t, n) => ['can i book a table for ' + s + ' at ' + t + ' for ' + n, 'booked a table for ' + s + ' at ' + t + ' under ' + n],
  (s, t, n) => ['booking for ' + n + ' , ' + s + ' people at ' + t, 'booked a table for ' + s + ' at ' + t + ' under ' + n]
];

function emit(list, name, size, time, split) {
  const tags = ['name:' + name, 'size:' + size, 'time:' + time];
  for (const tpl of TEMPLATES) {
    const p = tpl(size, time, name);
    list.push(G.ex(p[0], p[1], split, tags.slice()));
  }
}

function capDeterministic(list, cap, rng) {
  if (list.length <= cap) return list;
  G.shuffle(list, rng);
  return list.slice(0, cap);
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = [];
  for (const n of SYS_NAMES) for (const s of SIZES) for (const t of TIMES) emit(sys, n, s, t, 'val_systematic');

  const comp = [];
  for (const n of NAMES) for (const s of COMP_SIZES) for (const t of TIMES) emit(comp, n, s, t, 'val_compositional');

  const trainPool = [];
  for (const n of NAMES) {
    for (const s of SIZES) {
      if (COMP_SIZES.indexOf(s) !== -1) continue;
      for (const t of TIMES) emit(trainPool, n, s, t, 'train');
    }
  }
  const withHoldout = G.withIndist(trainPool, 0.11, rng);

  return capDeterministic(sys, 1200, rng)
    .concat(capDeterministic(comp, 1000, rng), withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
