/*
 * data/gen_multi_slot.js — requests with 2-3 slots, several templates
 * ---------------------------------------------------------------------------
 *   "set the kitchen to red and the office to blue"
 *   "put the mug in the kitchen at noon"
 *   "remind alice to bring the cake to the office"
 *
 *   val_systematic      a room never seen in training (attic, cellar) is forced
 *                       into the room slot.
 *   val_compositional   a seen room with a colour that never appears in
 *                       training (gold/teal are held out of train).
 *   val_indist          random holdout
 *   train               the rest
 * ---------------------------------------------------------------------------
 */
'use strict';
const G = require('./gen_common.js');

const TASK = 'multi_slot';
const SEED = 606;
const TARGET_TOKENS = 40000;

const ROOMS = ['kitchen', 'office', 'bedroom', 'garden', 'garage', 'hall', 'porch', 'studio'];
const SYS_ROOMS = ['attic', 'cellar'];

const COLORS = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'grey'];
const COMP_COLORS = ['gold', 'teal'];

const ITEMS = ['mug', 'cake', 'book', 'lamp', 'chair', 'plant', 'radio', 'basket', 'kettle', 'rug', 'clock', 'vase'];
const LISTS = ['shopping', 'guest', 'todo', 'travel', 'party'];
const TIMES = ['noon', '3pm', '5pm', '6pm', '7pm', '8pm', 'morning', 'evening'];
const NAMES = ['alice', 'bob', 'carol', 'dave', 'erin', 'frank', 'grace', 'heidi', 'ivan', 'judy'];

// Build one example into `sink` (deduped). `kinds` restricts which templates
// may be used: 0=colors, 1=move, 2=list, 3=put(time), 4=remind(name).
function one(rng, rooms, colors, split, kinds, sink) {
  const kind = kinds[(rng() * kinds.length) | 0];
  const room = G.pick(rng, rooms);
  const room2 = G.pick(rng, rooms);
  const color = G.pick(rng, colors);
  const color2 = G.pick(rng, colors);
  const item = G.pick(rng, ITEMS);
  const item2 = G.pick(rng, ITEMS);
  const list = G.pick(rng, LISTS);
  const time = G.pick(rng, TIMES);
  const name = G.pick(rng, NAMES);

  let e;
  if (kind === 0) {
    e = G.ex('set the ' + room + ' to ' + color + ' and the ' + room2 + ' to ' + color2,
      'set ' + room + ' to ' + color + ' and ' + room2 + ' to ' + color2, split,
      ['room:' + room, 'room2:' + room2, 'color:' + color, 'color2:' + color2]);
  } else if (kind === 1) {
    e = G.ex('move the ' + item + ' from the ' + room + ' to the ' + room2,
      'moved ' + item + ' from ' + room + ' to ' + room2, split,
      ['item:' + item, 'room:' + room, 'room2:' + room2]);
  } else if (kind === 2) {
    e = G.ex('add ' + item + ' and ' + item2 + ' to the ' + list + ' list',
      'added ' + item + ' and ' + item2 + ' to the ' + list + ' list', split,
      ['item:' + item, 'item2:' + item2, 'list:' + list]);
  } else if (kind === 3) {
    e = G.ex('put the ' + item + ' in the ' + room + ' at ' + time,
      'put ' + item + ' in ' + room + ' at ' + time, split,
      ['item:' + item, 'room:' + room, 'time:' + time]);
  } else {
    e = G.ex('remind ' + name + ' to bring the ' + item + ' to the ' + room,
      'reminded ' + name + ' to bring ' + item + ' to ' + room, split,
      ['name:' + name, 'item:' + item, 'room:' + room]);
  }
  const k = e.src + '\u0000' + e.tgt;
  if (sink.seen[k]) return;
  sink.seen[k] = 1;
  sink.list.push(e);
}

function build(rng, n, rooms, colors, split, kinds, sink) {
  let tries = 0;
  while (sink.list.length < n && tries < n * 4) { tries++; one(rng, rooms, colors, split, kinds, sink); }
}

function generate(opts) {
  opts = opts || {};
  const rng = G.makeRng(opts.seed || SEED);

  const sys = { list: [], seen: {} };
  const comp = { list: [], seen: {} };
  const train = { list: [], seen: {} };

  // systematic: force room-slot templates to draw from the unseen rooms
  build(rng, 1000, SYS_ROOMS, COLORS, 'val_systematic', [0, 1, 3, 4], sys);
  // compositional: colour template with held-out colours
  build(rng, 900, ROOMS, COMP_COLORS, 'val_compositional', [0], comp);
  // training distribution (no held-out rooms/colours/templates)
  build(rng, 7000, ROOMS, COLORS, 'train', [0, 1, 2, 3, 4], train);

  const withHoldout = G.withIndist(train.list, 0.11, rng);
  return sys.list.concat(comp.list, withHoldout);
}

module.exports = { TASK: TASK, SEED: SEED, TARGET_TOKENS: TARGET_TOKENS, generate: generate };
