/*
 * data/loader.js — read the generated dataset files (Node only)
 * ---------------------------------------------------------------------------
 *   const L = require('./data/loader.js');
 *   const all = L.load();                 // every example, each tagged `.task`
 *   const split = L.groupBySplit(all);    // { train, val_indist, ... }
 *   const tasks = L.groupByTask(all);     // { conversations: [...], ... }
 *   const grid  = L.byTaskSplit(all);     // { task: { split: [...] } }
 *
 * The loader does not modify the JSON files. It injects a `task` field taken
 * from the file name, so a generator never has to repeat it on every example.
 * If the files are missing it throws with a reminder to run generate_all.js.
 * ---------------------------------------------------------------------------
 */
'use strict';
const fs = require('fs');
const path = require('path');

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

const SPLITS = ['train', 'val_indist', 'val_compositional', 'val_systematic'];

function load(dataDir) {
  const dir = dataDir || __dirname;
  const missing = [];
  const out = [];
  for (const task of TASKS) {
    const file = path.join(dir, task + '.json');
    if (!fs.existsSync(file)) { missing.push(file); continue; }
    const rows = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (const row of rows) {
      row.task = task;
      out.push(row);
    }
  }
  if (missing.length) {
    throw new Error('missing dataset files:\n  ' + missing.join('\n  ') +
      '\nrun: node data/generate_all.js');
  }
  return out;
}

function groupBySplit(list) {
  const g = {};
  for (const s of SPLITS) g[s] = [];
  for (const e of list) (g[e.split] || (g[e.split] = [])).push(e);
  return g;
}

function groupByTask(list) {
  const g = {};
  for (const e of list) (g[e.task] || (g[e.task] = [])).push(e);
  return g;
}

function byTaskSplit(list) {
  const g = {};
  for (const e of list) {
    const t = (g[e.task] || (g[e.task] = {}));
    (t[e.split] || (t[e.split] = [])).push(e);
  }
  return g;
}

function summary(list) {
  const grid = byTaskSplit(list);
  const rows = [];
  for (const task of TASKS) {
    if (!grid[task]) continue;
    const c = { total: 0 };
    for (const s of SPLITS) c[s] = (grid[task][s] || []).length;
    c.total = (grid[task].train || []).length + (grid[task].val_indist || []).length +
      (grid[task].val_compositional || []).length + (grid[task].val_systematic || []).length;
    rows.push({ task: task, split: c, total: c.total });
  }
  return rows;
}

module.exports = {
  TASKS: TASKS,
  SPLITS: SPLITS,
  load: load,
  groupBySplit: groupBySplit,
  groupByTask: groupByTask,
  byTaskSplit: byTaskSplit,
  summary: summary
};
