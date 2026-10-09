#!/usr/bin/env node
/*
 * train.js — train the from-scratch seq2seq GRU, evaluate it several ways, and
 * write a binary .GenAI model file that index.html can load.
 *
 * The data is loaded from the generated files in data/ (run
 * `node data/generate_all.js` first). Every example is tagged with one of:
 *
 *   train             the model is allowed to fit
 *   val_indist        random holdout from the same distribution as training
 *   val_compositional new combinations of pieces each seen during training
 *   val_systematic    a value never seen during training at all
 *
 * We report exact-match accuracy over each held-out split, for greedy and
 * beam-search decoding (always over the *same* examples so the comparison is
 * fair), plus a per-task matrix so it is obvious which generalization fails.
 *
 * Usage:  ./train.sh [output.GenAI]
 *         node train.js model.GenAI
 */
'use strict';

const fs = require('fs');
const path = require('path');
const AI = require('./tiny-ai.js');
const L = require('./data/loader.js');

const OUT = process.argv[2] || path.join(__dirname, 'model.GenAI');

function line(s) { console.log(s); }
function pad(s, n) { return String(s) + ' '.repeat(Math.max(0, (n || 30) - String(s).length)); }
function pct(x) { return (100 * x).toFixed(1) + '%'; }

/* ================================================================== *
 * 1. data + setup
 * ================================================================== */
const all = L.load();
const STEPS = process.env.STEPS ? parseInt(process.env.STEPS, 10) : null;
AI.init(STEPS ? { dataset: all, targetSteps: STEPS } : { dataset: all });

let st = AI.status();
line('== setup ==');
line(`  dataset ${all.length} examples from data/*.json`);
line(`  vocab ${st.vocab} tokens · ${st.params} parameters · hidden=${st.hidden} embed=${st.embed}`);
line(`  layers=${st.layers} bidir=${st.bidir} coverage=${st.coverage} batch=${st.batchSize}`);
line(`  train ${st.trainExamples} · val ${st.valExamples} · target ${st.targetSteps} steps`);

/* ================================================================== *
 * 2. train
 * ================================================================== */
const t0 = Date.now();
let last = 0;
process.stdout.write('training');
while (!st.ready) {
  st = AI.train(1000);
  if (Date.now() - last > 1000) {
    last = Date.now();
    process.stdout.write('\r  step ' + st.steps +
      '  loss ' + (st.loss || 0).toFixed(3) +
      '  val ' + (st.valLoss == null ? ' -  ' : st.valLoss.toFixed(3)) +
      '  best ' + (st.bestValLoss == null ? ' -  ' : st.bestValLoss.toFixed(3)) +
      '  lr ' + st.lr.toExponential(1) + '   ');
  }
}
const secs = ((Date.now() - t0) / 1000).toFixed(1);
process.stdout.write('\r' + ' '.repeat(78) + '\r');

line('\n== training ==');
line(`  trained ${st.steps} steps in ${secs}s` +
  (st.stopped ? ' (early-stopped on validation)' : '') +
  `  final loss ${(st.loss || 0).toFixed(3)}` +
  `  val ${st.valLoss == null ? '-' : st.valLoss.toFixed(3)}` +
  `  best val ${st.bestValLoss == null ? '-' : st.bestValLoss.toFixed(3)}`);

/* ================================================================== *
 * 3. smoke tests (chat + booking, greedy)
 * ================================================================== */
line('\n== smoke tests (greedy) ==');
for (const m of ['hi', 'hello', 'how are you', 'what is your name', 'tell me a joke',
                 'do you like math', 'what is ai', 'bye', 'thanks', 'are you real']) {
  line('  chat  ' + pad(m, 24) + ' -> ' + AI.respond(m, 0));
}
for (const m of ['book a table for 4 at 8pm for alice',
                 'reserve a table for 2 at noon under bob',
                 'put the mug in the kitchen at 8pm']) {
  line('  task  ' + pad(m, 42) + ' -> ' + AI.respond(m, 0));
}

/* ================================================================== *
 * 4. exact-match evaluation (greedy + beam)
 * ================================================================== */
const S = AI.splits();

// Greedy and beam are always run over the SAME examples (the first EVAL_CAP of
// each set), otherwise the comparison is meaningless. Beam is much slower than
// greedy, and the held-out sets here number in the thousands, so we cap.
const EVAL_CAP = parseInt(process.env.EVAL_CAP || '200', 10);

function greedy(exs) { return exs.length ? AI.accuracy(exs, { strategy: 'greedy', maxExamples: EVAL_CAP }) : null; }
function beam(exs) { return exs.length ? AI.accuracy(exs, { strategy: 'beam', beamSize: 4, maxExamples: EVAL_CAP }) : null; }

line('\n== exact-match accuracy (held-out, capped at ' + EVAL_CAP + '/set) ==');
for (const [label, exs] of [['in-dist', S.inDist], ['compositional', S.compositional], ['systematic', S.systematic]]) {
  if (!exs.length) continue;
  const n = Math.min(EVAL_CAP, exs.length);
  line('  ' + pad(label, 16) + ' n=' + String(n).padEnd(5) +
    ' greedy ' + pad(pct(greedy(exs)), 8) + ' beam ' + pct(beam(exs)));
}

/* ---- per-task matrix --------------------------------------------- */
const splitsByTask = [{ key: 'inDist', name: 'indist' }, { key: 'compositional', name: 'comp' }, { key: 'systematic', name: 'sys' }];
const tasks = [];
for (const s of splitsByTask) for (const ex of S[s.key]) if (tasks.indexOf(ex.task) === -1) tasks.push(ex.task);
tasks.sort();

function matrix(title, fn) {
  line('\n  -- ' + title + ' exact match by task --');
  line('     ' + pad('task', 14) + pad('indist', 9) + pad('comp', 9) + 'sys');
  for (const t of tasks) {
    let row = '     ' + pad(t, 14);
    for (const s of splitsByTask) {
      const exs = S[s.key].filter((e) => e.task === t);
      row += pad(exs.length ? fn(exs) : '-', 9);
    }
    line(row);
  }
}
matrix('greedy', (exs) => pct(greedy(exs)));
matrix('beam', (exs) => pct(beam(exs)));

/* ================================================================== *
 * 5. qualitative held-out examples (greedy)
 * ================================================================== */
line('\n== held-out examples (greedy) ==');
const showcases = [
  S.systematic.find(e => e.task === 'copy'),
  S.systematic.find(e => e.task === 'sort'),
  S.systematic.find(e => e.task === 'arithmetic'),
  S.compositional.find(e => e.task === 'booking'),
  S.compositional.find(e => e.task === 'scan'),
  S.compositional.find(e => e.task === 'cogs')
].filter(Boolean);
for (const ex of showcases) {
  const got = AI.greedy(ex.src);
  const ok = got === ex.tgt;
  line('  ' + (ok ? 'PASS ' : 'FAIL ') + pad(ex.task, 11) +
    pad(ex.src, 34) + ' -> ' + got + (ok ? '' : '   (want ' + ex.tgt + ')'));
}

/* ================================================================== *
 * 6. save .GenAI + reload round-trip
 * ================================================================== */
const preG = greedy(S.inDist);
const buf = AI.save();
fs.writeFileSync(OUT, Buffer.from(new Uint8Array(buf)));
line('\n== file ==');
line('  wrote ' + OUT + '  (' + buf.byteLength + ' bytes)');

const st2 = AI.load(buf);
line('  reloaded: vocab ' + st2.vocab + ' · params ' + st2.params +
  ' · hidden ' + st2.hidden + ' · embed ' + st2.embed +
  ' · layers ' + st2.layers + ' · bidir ' + st2.bidir);
const reG = greedy(S.inDist);
line('  reloaded in-dist greedy ' + pct(reG) + ' (pre-save ' + pct(preG) + ')');
line('\ndone.');
