/**
 * ============================================================================
 * TinyAI - src/tokenizer.js
 * ============================================================================
 *
 * WHAT IS A TOKENIZER?
 * --------------------
 * Neural networks do not understand text - they understand numbers. A
 * tokenizer is the bridge. It turns a string like "hello" into a list of
 * integer ids like [8, 5, 12, 12, 15], and it can turn those ids back into
 * text. Every language model starts and ends with a tokenizer.
 *
 * TinyAI ships THREE tokenizers, all written from scratch:
 *
 *   1. CharacterTokenizer - one token per character.  Simplest possible.
 *        "cat" -> ['c','a','t']
 *
 *   2. WordTokenizer - one token per word / punctuation mark.
 *        "the cat sat" -> ['the','cat','sat']
 *
 *   3. BPETokenizer - "Byte Pair Encoding". Starts from characters and
 *      repeatedly glues together the most common neighbouring pair. This is
 *      how real models (GPT etc.) get a vocabulary of thousands of pieces.
 *        "the" -> ['the']    (if 'the' became a learned merge)
 *
 * All three share the same interface:
 *
 *    .encode(text)  -> Array<number>
 *    .decode(ids)   -> string
 *    .save()        -> plain object (JSON-friendly)
 *    .vocabSize     -> number
 *
 * and there is a factory `Tokenizer.load(obj)` to rebuild one from `save()`.
 * ============================================================================
 */

'use strict';

/* Special tokens occupy the first ids so the mapping is stable. */
const SPECIALS = ['<pad>', '<unk>', '<bos>', '<eos>'];
const PAD = 0;
const UNK = 1;
const BOS = 2;
const EOS = 3;

/* ==========================================================================
 * Base class - shared bookkeeping
 * ========================================================================== */
class BaseTokenizer {
  constructor() {
    this.idToToken = [];
    this.tokenToId = new Map();
    this.type = 'base';
  }

  /** Populate the id<->token maps from a list of tokens (specials first). */
  _setVocabulary(tokens) {
    this.idToToken = tokens.slice();
    this.tokenToId = new Map();
    for (let i = 0; i < this.idToToken.length; i++) {
      this.tokenToId.set(this.idToToken[i], i);
    }
  }

  get vocabSize() {
    return this.idToToken.length;
  }

  /** Encode to ids. Subclasses must override. */
  encode() {
    throw new Error('encode() not implemented');
  }

  /** Decode from ids. Subclasses should override `_joinToken`. */
  decode(ids) {
    const parts = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id === PAD || id === BOS || id === EOS) continue;
      const tok = this.idToToken[id];
      if (tok === undefined) continue;
      parts.push(tok === '<unk>' ? '\uFFFD' : tok);
    }
    return this._joinToken(parts);
  }

  _joinToken(parts) {
    return parts.join('');
  }

  /* Serialization is shared: subclasses add extra fields. */
  save() {
    return {
      type: this.type,
      tokens: this.idToToken.slice(),
      extra: this._saveExtra(),
    };
  }

  _saveExtra() {
    return {};
  }

  _loadExtra() {
    // Subclasses override this to restore extra fields saved by _saveExtra().
  }

  static load(obj) {
    let tok;
    if (obj.type === 'char') tok = new CharTokenizer();
    else if (obj.type === 'word') tok = new WordTokenizer();
    else if (obj.type === 'bpe') tok = new BpeTokenizer();
    else throw new Error('Unknown tokenizer type: ' + obj.type);
    tok._setVocabulary(obj.tokens);
    tok._loadExtra(obj.extra || {});
    return tok;
  }
}

/* ==========================================================================
 * 1. Character tokenizer
 * ========================================================================== */
class CharTokenizer extends BaseTokenizer {
  constructor() {
    super();
    this.type = 'char';
    this._setVocabulary(SPECIALS.slice());
  }

  /** Learn the set of characters that appear in `text`. */
  train(text) {
    const seen = new Set();
    const ordered = [];
    for (const ch of text) {
      if (!seen.has(ch)) {
        seen.add(ch);
        ordered.push(ch);
      }
    }
    this._setVocabulary(SPECIALS.concat(ordered));
    return this;
  }

  encode(text) {
    const ids = [];
    for (const ch of text) {
      const id = this.tokenToId.get(ch);
      ids.push(id === undefined ? UNK : id);
    }
    return ids;
  }

  _joinToken(parts) {
    return parts.join('');
  }
}

/* ==========================================================================
 * 2. Word tokenizer
 * ========================================================================== */

/**
 * Split a string into words and punctuation. We keep punctuation as separate
 * tokens so that decoding can put the spacing back sensibly.
 */
function wordSplit(text) {
  const matches = text.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9']/g);
  return matches || [];
}

class WordTokenizer extends BaseTokenizer {
  constructor() {
    super();
    this.type = 'word';
    this._setVocabulary(SPECIALS.slice());
  }

  train(text) {
    const counts = new Map();
    const order = [];
    for (const w of wordSplit(text)) {
      if (!counts.has(w)) {
        counts.set(w, 0);
        order.push(w);
      }
      counts.set(w, counts.get(w) + 1);
    }
    this._setVocabulary(SPECIALS.concat(order));
    return this;
  }

  encode(text) {
    const ids = [];
    for (const w of wordSplit(text)) {
      const id = this.tokenToId.get(w);
      ids.push(id === undefined ? UNK : id);
    }
    return ids;
  }

  /** Join tokens, removing the space that would precede punctuation. */
  _joinToken(parts) {
    return parts.join(' ').replace(/ ([.,!?;:'")\]}])/g, '$1').replace(/([(\[{]) /g, '$1');
  }
}

/* ==========================================================================
 * 3. Byte Pair Encoding (BPE) tokenizer
 * ========================================================================== */

/**
 * BPE learns a vocabulary by starting with single characters and repeatedly
 * merging the most frequent adjacent pair of symbols.
 *
 * Example, on the tiny corpus "aaabdaaabac":
 *
 *   initial: a a a b d a a a b a c
 *   pair "aa" is the most common -> merge it into the symbol "aa"
 *   now:    aa a b d aa a b a c
 *   pair "aa" and "a"... etc.
 *
 * The list of merges is stored in order; encoding applies them in that order.
 * This is the algorithm behind GPT-2/3/4 tokenization (in spirit).
 */
class BpeTokenizer extends BaseTokenizer {
  constructor() {
    super();
    this.type = 'bpe';
    this.merges = []; // ordered list of [left, right] string pairs
    this._setVocabulary(SPECIALS.slice());
  }

  /**
   * @param {string} text       training corpus
   * @param {number} vocabSize  stop once we have this many tokens
   */
  train(text, vocabSize = 512) {
    // 1. Start from the individual characters.
    const baseChars = [];
    const seen = new Set();
    for (const ch of text) {
      if (!seen.has(ch)) {
        seen.add(ch);
        baseChars.push(ch);
      }
    }
    const vocab = SPECIALS.concat(baseChars);
    const merges = [];

    // 2. Work on the whole corpus as a list of symbols.
    let symbols = Array.from(text); // each element is one character

    while (vocab.length < vocabSize) {
      // Count adjacent pairs.
      const pairs = new Map();
      for (let i = 0; i + 1 < symbols.length; i++) {
        const key = symbols[i] + '\u0000' + symbols[i + 1]; // separator can't appear in text
        pairs.set(key, (pairs.get(key) || 0) + 1);
      }
      if (pairs.size === 0) break;

      // Find the most frequent pair (ties broken by insertion order).
      let bestKey = null;
      let bestCount = 1; // only merge pairs seen more than once
      for (const [key, count] of pairs) {
        if (count > bestCount) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (bestKey === null) break;

      const [left, right] = bestKey.split('\u0000');
      merges.push([left, right]);
      vocab.push(left + right);

      // Apply the merge to the symbol list.
      const merged = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i + 1 < symbols.length && symbols[i] === left && symbols[i + 1] === right) {
          merged.push(left + right);
          i++;
        } else {
          merged.push(symbols[i]);
        }
      }
      symbols = merged;
    }

    this.merges = merges;
    this._setVocabulary(vocab);
    return this;
  }

  /** Apply the learned merges in order, then map symbols to ids. */
  encode(text) {
    let symbols = Array.from(text);
    for (let m = 0; m < this.merges.length; m++) {
      const left = this.merges[m][0];
      const right = this.merges[m][1];
      if (symbols.length < 2) break;
      const merged = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i + 1 < symbols.length && symbols[i] === left && symbols[i + 1] === right) {
          merged.push(left + right);
          i++;
        } else {
          merged.push(symbols[i]);
        }
      }
      symbols = merged;
    }
    const ids = [];
    for (const sym of symbols) {
      const id = this.tokenToId.get(sym);
      ids.push(id === undefined ? UNK : id);
    }
    return ids;
  }

  _saveExtra() {
    return { merges: this.merges };
  }

  _loadExtra(extra) {
    this.merges = extra.merges || [];
  }
}

/* ==========================================================================
 * Factory
 * ========================================================================== */

/**
 * Create a tokenizer of the requested kind and (optionally) train it.
 * @param {'char'|'word'|'bpe'} type
 * @param {string} [text]         training corpus
 * @param {number} [vocabSize]    target vocabulary (BPE only)
 */
function createTokenizer(type, text, vocabSize) {
  if (type === 'char') {
    const t = new CharTokenizer();
    if (text !== undefined) t.train(text);
    return t;
  }
  if (type === 'word') {
    const t = new WordTokenizer();
    if (text !== undefined) t.train(text);
    return t;
  }
  if (type === 'bpe') {
    const t = new BpeTokenizer();
    if (text !== undefined) t.train(text, vocabSize || 512);
    return t;
  }
  throw new Error('Unknown tokenizer type: ' + type);
}

const TinyTokenizer = {
  BaseTokenizer,
  CharTokenizer,
  WordTokenizer,
  BpeTokenizer,
  createTokenizer,
  wordSplit,
  SPECIALS,
  PAD,
  UNK,
  BOS,
  EOS,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyTokenizer;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyTokenizer);
}
