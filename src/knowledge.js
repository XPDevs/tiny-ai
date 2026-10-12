/**
 * ============================================================================
 * TinyAI - src/knowledge.js
 * ============================================================================
 *
 * KNOWLEDGE PLUGINS - MAKING A TINY MODEL USEFUL
 * ----------------------------------------------
 * A one-million-parameter model cannot memorise the world. But it does not
 * have to: we can bolt on external knowledge that is searched *before* the
 * model generates an answer. This is the same idea behind "retrieval-augmented
 * generation" (RAG) in large systems, just much simpler.
 *
 * A knowledge plugin is a small JSON file:
 *
 *   {
 *     "topic": "animals",
 *     "entries": [
 *       { "keywords": ["cat", "cats", "kitten"], "answer": "Cats are ..." },
 *       ...
 *     ]
 *   }
 *
 * At chat time we look for the entry whose keywords best match the user's
 * question. If we find one we return its answer directly. If not, we let the
 * Transformer continue the text. Knowledge can be added or edited at any time
 * WITHOUT retraining the model.
 * ============================================================================
 */

'use strict';

class KnowledgeBase {
  constructor() {
    this.entries = [];
    this.plugins = [];
  }

  /** Add one parsed plugin object. */
  addPlugin(plugin) {
    if (!plugin || !Array.isArray(plugin.entries)) return this;
    this.plugins.push(plugin.topic || 'unknown');
    for (const e of plugin.entries) {
      if (e && Array.isArray(e.keywords) && typeof e.answer === 'string') {
        this.entries.push({ keywords: e.keywords, answer: e.answer, topic: plugin.topic });
      }
    }
    return this;
  }

  /** Add many plugins at once. */
  addPlugins(plugins) {
    for (const p of plugins) this.addPlugin(p);
    return this;
  }

  get size() {
    return this.entries.length;
  }

  /**
   * Score how well an entry matches the query.
   * Every keyword that appears in the query as a whole word adds points equal
   * to its length, so more specific (longer) keywords win over generic ones.
   */
  _score(entry, queryLower) {
    let score = 0;
    for (const kw of entry.keywords) {
      const k = kw.toLowerCase();
      // Word-boundary match for normal words; substring for punctuation.
      const isWord = /^[a-z0-9']+$/.test(k);
      const hit = isWord
        ? new RegExp('(^|[^a-z0-9])' + k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '([^a-z0-9]|$)').test(queryLower)
        : queryLower.indexOf(k) !== -1;
      if (hit) score += k.length;
    }
    return score;
  }

  /**
   * Return the best matching answer, or null if nothing matches.
   * @param {string} query
   * @param {number} [minScore] minimum score required to accept a match
   */
  search(query, minScore = 1) {
    const q = String(query).toLowerCase();
    let best = null;
    let bestScore = minScore - 1e-9;
    for (const entry of this.entries) {
      const s = this._score(entry, q);
      if (s > bestScore) {
        bestScore = s;
        best = entry;
      }
    }
    return best ? { answer: best.answer, topic: best.topic, score: bestScore } : null;
  }

  /** Node helper: load and merge every *.json file in a directory. */
  static loadFromDir(dir) {
    if (typeof require !== 'function') throw new Error('loadFromDir() is Node-only');
    const fs = require('fs');
    const path = require('path');
    const kb = new KnowledgeBase();
    if (!fs.existsSync(dir)) return kb;
    for (const file of fs.readdirSync(dir)) {
      if (!file.toLowerCase().endsWith('.json')) continue;
      const full = path.join(dir, file);
      try {
        kb.addPlugin(JSON.parse(fs.readFileSync(full, 'utf8')));
      } catch (err) {
        // A broken plugin should never stop the whole app.
        console.warn('Knowledge: could not read ' + file + ': ' + err.message);
      }
    }
    return kb;
  }
}

const TinyKnowledge = { KnowledgeBase };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyKnowledge;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyKnowledge);
}
