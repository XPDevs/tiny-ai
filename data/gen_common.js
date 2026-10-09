/*
 * data/gen_common.js — shared helpers for the procedural dataset generators.
 * ---------------------------------------------------------------------------
 * Node-only (the browser never runs generators; it just loads a .GenAI file).
 *
 * Everything here is deterministic: given the same seed and the same generator
 * code, every generator produces byte-identical JSON. That is what makes the
 * splits reproducible and the evaluation numbers comparable across runs.
 *
 * Example shape produced by every generator (the single source of truth):
 *
 *   {
 *     "src":   "book a table for 4 at 8pm for alice",
 *     "tgt":   "booked a table for 4 at 8pm under alice",
 *     "split": "train",                       // or a val_* value
 *     "tags":  ["size:4", "time:8pm", "name:alice"]
 *   }
 *
 * The loader adds a "task" field derived from the file name; generators may
 * also set it (the loader overwrites it either way).
 * ---------------------------------------------------------------------------
 */
'use strict';

const SPLITS = ['train', 'val_indist', 'val_compositional', 'val_systematic'];

// Same RNG as tiny-ai.js so a seed means the same thing everywhere.
function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return function () {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// Tokenizer must match tiny-ai.js exactly: [a-z0-9']+ runs plus single
// punctuation marks. Token counts are computed with this so the "target tokens"
// in the plan match what the model actually sees.
function words(text) {
  return String(text || '').toLowerCase().match(/[a-z0-9'-]+|[?!.,:;]/g) || [];
}

function tokenCount(src, tgt) {
  return words(src).length + words(tgt).length;
}

function tokenCountEx(ex) {
  return tokenCount(ex.src, ex.tgt);
}

function shuffle(arr, rng) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (rng() * (i + 1)) | 0;
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}

// rng integer in [min, max]
function int(rng, min, max) {
  return min + ((rng() * (max - min + 1)) | 0);
}

function pick(rng, arr) {
  return arr[(rng() * arr.length) | 0];
}

// Build one example in the canonical shape.
function ex(src, tgt, split, tags) {
  return { src: String(src), tgt: String(tgt), split: split || 'train', tags: tags || [] };
}

// Turn `frac` of the examples currently tagged "train" into "val_indist".
// Deterministic because it consumes the shared rng in list order.
function markIndist(list, frac, rng) {
  for (const e of list) {
    if (e.split === 'train' && rng() < frac) e.split = 'val_indist';
  }
  return list;
}

// Dedupe by "src\u0000tgt"; keeps deterministic first-seen order.
function dedupe(list) {
  const seen = Object.create(null);
  const out = [];
  for (const e of list) {
    const k = e.src + '\u0000' + e.tgt;
    if (seen[k]) continue;
    seen[k] = 1;
    out.push(e);
  }
  return out;
}

// Add examples until `list` holds `cap` unique items or `tries` attempts run out.
// `make` is called with (rng, push) and should produce 0..n examples per call.
function fill(list, cap, tries, make, rng) {
  const seen = Object.create(null);
  for (const e of list) seen[e.src + '\u0000' + e.tgt] = 1;
  let attempts = 0;
  while (list.length < cap && attempts < tries) {
    attempts++;
    make(rng, (e) => {
      const k = e.src + '\u0000' + e.tgt;
      if (seen[k]) return;
      seen[k] = 1;
      list.push(e);
    });
  }
  return list;
}

// Split a pool into train + val_indist with the given indist fraction.
function withIndist(pool, indistFrac, rng) {
  const arr = dedupe(pool);
  shuffle(arr, rng);
  const out = [];
  for (const e of arr) {
    e.split = 'train';
    out.push(e);
  }
  markIndist(out, indistFrac, rng);
  return out;
}

function sortBySrc(a, b) {
  if (a.src < b.src) return -1;
  if (a.src > b.src) return 1;
  if (a.tgt < b.tgt) return -1;
  if (a.tgt > b.tgt) return 1;
  return 0;
}

function countsBySplit(list) {
  const c = { train: 0, val_indist: 0, val_compositional: 0, val_systematic: 0 };
  for (const e of list) c[e.split] = (c[e.split] || 0) + 1;
  return c;
}

function tokensBySplit(list) {
  const c = { train: 0, val_indist: 0, val_compositional: 0, val_systematic: 0 };
  for (const e of list) c[e.split] = (c[e.split] || 0) + tokenCountEx(e);
  return c;
}

module.exports = {
  SPLITS: SPLITS,
  makeRng: makeRng,
  words: words,
  tokenCount: tokenCount,
  tokenCountEx: tokenCountEx,
  shuffle: shuffle,
  int: int,
  pick: pick,
  ex: ex,
  markIndist: markIndist,
  dedupe: dedupe,
  fill: fill,
  withIndist: withIndist,
  sortBySrc: sortBySrc,
  countsBySplit: countsBySplit,
  tokensBySplit: tokensBySplit
};
