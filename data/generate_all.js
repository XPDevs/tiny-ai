/*
 * data/generate_all.js — run every generator and write the dataset files
 * ---------------------------------------------------------------------------
 *   node data/generate_all.js
 *
 * Produces, for each task:
 *   data/<task>.json           the full dataset (git-ignored)
 *   data/samples/<task>.json   a small committed sample for review
 *
 * The training split is trimmed (deterministically) so that the *total* token
 * count lands near the task's TARGET_TOKENS; held-out splits are never trimmed,
 * so the evaluation sets are always complete. Running this twice produces
 * byte-identical files (same generators, same seeds, same RNG).
 * ---------------------------------------------------------------------------
 */
'use strict';
const fs = require('fs');
const path = require('path');
const G = require('./gen_common.js');

const TASKS = [
  'conversations',
  'booking',
  'multi_slot',
  'copy',
  'reverse',
  'sort',
  'arithmetic',
  'qa_copy',
  'successor',
  'scan',
  'cogs'
];

const DATA_DIR = __dirname;
const SAMPLE_DIR = path.join(__dirname, 'samples');
const SAMPLE_PER_SLIT = 13;
const HELD_CAP = 400;

function capSplit(list, cap, seed) {
  if (list.length <= cap) return list;
  const copy = list.slice();
  G.shuffle(copy, G.makeRng(seed));
  return copy.slice(0, cap);
}

function trimTrain(list, targetTokens, seed) {
  const rng = G.makeRng(seed + 1);
  const held = [];
  for (const split of ['val_indist', 'val_compositional', 'val_systematic']) {
    held.push.apply(held, capSplit(list.filter((e) => e.split === split), HELD_CAP, seed + split.length));
  }
  const train = list.filter((e) => e.split === 'train');
  G.shuffle(train, rng);

  let heldTokens = 0;
  for (const e of held) heldTokens += G.tokenCountEx(e);
  const budget = Math.max(0, targetTokens - heldTokens);

  const kept = [];
  let used = 0;
  for (const e of train) {
    const c = G.tokenCountEx(e);
    if (kept.length > 0 && used + c > budget) continue;
    kept.push(e);
    used += c;
    if (used >= budget) break;
  }

  const out = held.concat(kept);
  out.sort(function (a, b) {
    if (a.split !== b.split) return a.split < b.split ? -1 : 1;
    return G.sortBySrc(a, b);
  });
  return out;
}

function sampleOf(list) {
  const bySplit = { train: [], val_indist: [], val_compositional: [], val_systematic: [] };
  for (const e of list) bySplit[e.split].push(e);
  const out = [];
  for (const split of G.SPLITS) {
    for (const e of bySplit[split].slice(0, SAMPLE_PER_SLIT)) out.push(e);
  }
  return out;
}

function loadGenerators() {
  const gens = [];
  for (const task of TASKS) {
    const mod = require('./gen_' + task + '.js');
    gens.push(mod);
  }
  return gens;
}

function main() {
  if (!fs.existsSync(SAMPLE_DIR)) fs.mkdirSync(SAMPLE_DIR, { recursive: true });

  let grandTokens = 0;
  let grandExamples = 0;
  console.log('task            examples   train   indist    comp     sys     tokens');
  console.log('-----------------------------------------------------------------------');

  for (const gen of loadGenerators()) {
    const all = gen.generate({ seed: gen.SEED });
    const trimmed = trimTrain(all, gen.TARGET_TOKENS, gen.SEED);

    fs.writeFileSync(path.join(DATA_DIR, gen.TASK + '.json'), JSON.stringify(trimmed));
    fs.writeFileSync(path.join(SAMPLE_DIR, gen.TASK + '.json'), JSON.stringify(sampleOf(trimmed), null, 2) + '\n');

    const c = G.countsBySplit(trimmed);
    const t = G.tokensBySplit(trimmed);
    const total = t.train + t.val_indist + t.val_compositional + t.val_systematic;
    grandTokens += total;
    grandExamples += trimmed.length;

    console.log(
      pad(gen.TASK, 15) +
      pad(String(trimmed.length), 10) +
      pad(String(c.train), 8) +
      pad(String(c.val_indist), 9) +
      pad(String(c.val_compositional), 8) +
      pad(String(c.val_systematic), 8) +
      pad(String(total), 10)
    );
  }

  console.log('-----------------------------------------------------------------------');
  console.log(pad('TOTAL', 15) + pad(String(grandExamples), 10) + pad('', 33) + pad(String(grandTokens), 10));
}

function pad(s, n) {
  s = String(s);
  while (s.length < n) s = ' ' + s;
  return s;
}

if (require.main === module) main();

module.exports = { TASKS: TASKS, trimTrain: trimTrain };
