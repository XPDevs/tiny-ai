/**
 * ============================================================================
 * TinyAI - src/model.js
 * ============================================================================
 *
 * THE MODEL
 * ---------
 * This file assembles a complete decoder-only Transformer (a "mini-GPT") out
 * of the pieces built in the other files, and gives it three important
 * abilities:
 *
 *   1. forward(ids)      - turn token ids into next-token scores (logits)
 *   2. generate(...)     - sample new tokens, with greedy / top-k / top-p /
 *                          temperature decoding and streaming
 *   3. saveModel()/loadModel() - read and write the `.GenAI` file format
 *
 * THE ARCHITECTURE, TOP TO BOTTOM
 * -------------------------------
 *
 *      token ids [B, T]
 *          |
 *      token embedding lookup            -> [B, T, D]
 *   +  positional embedding              -> [B, T, D]
 *          |
 *      [ Transformer block ] x nLayers
 *          |   (attention + feed-forward, each with residual + layernorm)
 *      final layernorm
 *          |
 *      language-model head (weights tied to the token embedding)
 *          |
 *      logits [B, T, V]      (a score for every possible next token)
 *
 * The default configuration is about 1.3 million parameters and compresses to
 * well under 10 MB in the .GenAI format (weights are stored as 32-bit floats).
 * ============================================================================
 */

'use strict';

const _deps = typeof require === 'function' ? require('./tensor.js') : globalThis.TinyAI;
const {
  Tensor,
  tensor,
  zeros,
  ones,
  randn,
  fromIds,
  add,
  matmul,
  embedding,
  sliceRows,
  transpose,
  argmax,
  noGrad,
} = _deps;

const _transformer =
  typeof require === 'function' ? require('./transformer.js') : globalThis.TinyAI;
const { transformerBlock, layerNorm } = _transformer;

const _tokenizer =
  typeof require === 'function' ? require('./tokenizer.js') : globalThis.TinyAI;

/* --------------------------------------------------------------------------
 * Default configuration - see README.md for what each field means.
 * ------------------------------------------------------------------------ */
const DEFAULT_CONFIG = {
  vocabSize: 4096, // how many distinct tokens the model knows
  dModel: 128,     // width of the model (size of each token vector)
  nHeads: 4,       // number of attention heads (dModel must divide evenly)
  nLayers: 4,      // how many transformer blocks
  blockSize: 128,  // maximum context length (how far back it can look)
  tieWeights: true, // share the embedding matrix with the output head
};

/* ==========================================================================
 * Base64 <-> Float32 byte helpers (work in Node and the browser)
 * ========================================================================== */

function bytesToBase64(bytes) {
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function base64ToBytes(b64) {
  if (typeof Buffer !== 'undefined') {
    const b = Buffer.from(b64, 'base64');
    const out = new Uint8Array(b.length);
    out.set(b);
    return out;
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Encode a Float64Array as base64 of its 32-bit representation. */
function encodeWeights(f64) {
  const f32 = new Float32Array(f64.length);
  for (let i = 0; i < f64.length; i++) f32[i] = f64[i];
  return bytesToBase64(new Uint8Array(f32.buffer));
}

/** Decode base64 back into a Float64Array of the given length. */
function decodeWeights(b64, length) {
  const bytes = base64ToBytes(b64);
  const count = Math.floor(bytes.byteLength / 4);
  const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, count);
  const out = new Float64Array(length);
  for (let i = 0; i < length; i++) out[i] = f32[i];
  return out;
}

/* ==========================================================================
 * The TinyModel class
 * ========================================================================== */
class TinyModel {
  /**
   * @param {object} [config]     partial config, merged with defaults
   * @param {object} [tokenizer]  a tokenizer (its vocab size wins if present)
   */
  constructor(config = {}, tokenizer = null) {
    this.config = Object.assign({}, DEFAULT_CONFIG, config);
    this.tokenizer = tokenizer;
    if (tokenizer && tokenizer.vocabSize) {
      // If the caller did not explicitly override vocabSize, follow the
      // tokenizer so the ids line up.
      if (config.vocabSize === undefined) this.config.vocabSize = tokenizer.vocabSize;
    }
    this._maskCache = {};
    this._build();
  }

  /* ----------------------------------------------------------------------
   * Parameter initialization
   * -------------------------------------------------------------------- */
  _build() {
    const c = this.config;
    const V = c.vocabSize;
    const D = c.dModel;
    const H = c.nHeads;
    const L = c.nLayers;
    const B = c.blockSize;

    if (D % H !== 0) throw new Error('dModel must be divisible by nHeads');

    const std = 0.02; // small random weights, the GPT-2 initialization

    // Token and positional embeddings.
    this.tokEmb = randn([V, D], true, std);
    this.posEmb = randn([B, D], true, std);

    // Transformer blocks.
    this.blocks = [];
    for (let l = 0; l < L; l++) {
      this.blocks.push({
        ln1g: ones([D], true),
        ln1b: zeros([D], true),
        wq: randn([D, D], true, std),
        bq: zeros([D], true),
        wk: randn([D, D], true, std),
        bk: zeros([D], true),
        wv: randn([D, D], true, std),
        bv: zeros([D], true),
        wo: randn([D, D], true, std),
        bo: zeros([D], true),
        ln2g: ones([D], true),
        ln2b: zeros([D], true),
        fcw: randn([D, 4 * D], true, std),
        fcb: zeros([4 * D], true),
        fpw: randn([4 * D, D], true, std),
        fpb: zeros([D], true),
      });
    }

    // Final layer norm.
    this.lnfg = ones([D], true);
    this.lnfb = zeros([D], true);

    // Optional untied output head.
    if (!c.tieWeights) {
      this.lmHeadW = randn([D, V], true, std);
      this.lmHeadB = zeros([V], true);
    }
  }

  /** A flat list of every trainable parameter, with names (for the optimizer). */
  parameters() {
    const list = [];
    const add = (name, t) => list.push({ name, tensor: t });
    add('tokEmb', this.tokEmb);
    add('posEmb', this.posEmb);
    this.blocks.forEach((b, l) => {
      for (const k of Object.keys(b)) add('blocks.' + l + '.' + k, b[k]);
    });
    add('lnfg', this.lnfg);
    add('lnfb', this.lnfb);
    if (!this.config.tieWeights) {
      add('lmHeadW', this.lmHeadW);
      add('lmHeadB', this.lmHeadB);
    }
    return list;
  }

  /** Just the tensors (convenience for the optimizer). */
  parameterTensors() {
    return this.parameters().map((p) => p.tensor);
  }

  /** Total number of learnable parameters. */
  parameterCount() {
    return this.parameterTensors().reduce((s, t) => s + t.size, 0);
  }

  /** Human-readable architecture summary. */
  summary() {
    const c = this.config;
    return {
      architecture: 'transformer-decoder',
      layers: c.nLayers,
      heads: c.nHeads,
      dModel: c.dModel,
      blockSize: c.blockSize,
      vocabSize: c.vocabSize,
      tieWeights: c.tieWeights,
      parameters: this.parameterCount(),
    };
  }

  /* ----------------------------------------------------------------------
   * Causal mask
   * -------------------------------------------------------------------- */
  _causalMask(T) {
    if (this._maskCache[T]) return this._maskCache[T];
    const m = new Uint8Array(T * T);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) if (j > i) m[i * T + j] = 1; // block the future
    }
    this._maskCache[T] = m;
    return m;
  }

  /* ----------------------------------------------------------------------
   * Forward pass
   * -------------------------------------------------------------------- */
  /**
   * @param {Tensor} ids  integer tensor of shape [B, T]
   * @returns {Tensor}    logits of shape [B, T, V]
   */
  forward(ids) {
    const c = this.config;
    const D = c.dModel;
    const B = ids.shape[0];
    const T = ids.shape[1];
    if (T > c.blockSize) {
      throw new Error('Input length ' + T + ' exceeds blockSize ' + c.blockSize);
    }

    // 1. Embed tokens and add position information.
    let x = embedding(this.tokEmb, ids);            // [B, T, D]
    x = add(x, sliceRows(this.posEmb, 0, T));       // [B, T, D]

    // 2. Run the transformer blocks.
    const mask = this._causalMask(T);
    for (let l = 0; l < this.blocks.length; l++) {
      x = transformerBlock(x, this.blocks[l], c, mask);
    }

    // 3. Final normalization.
    x = layerNorm(x, this.lnfg, this.lnfb);

    // 4. Project to vocabulary scores.
    if (c.tieWeights) {
      return matmul(x, transpose(this.tokEmb, 0, 1)); // [B, T, V]
    }
    return add(matmul(x, this.lmHeadW), this.lmHeadB);
  }

  /** Convenience: run forward on a plain 2D array of ids. */
  forwardIds(ids2d) {
    return this.forward(fromIds(ids2d));
  }

  /* ----------------------------------------------------------------------
   * Decoding / sampling
   * -------------------------------------------------------------------- */

  /**
   * Turn one row of logits into a chosen token id.
   * Supports greedy, temperature, top-k and top-p (nucleus) sampling, plus an
   * optional repetition penalty.
   */
  _sampleRow(logits, opts) {
    const temperature = opts.temperature !== undefined ? opts.temperature : 1.0;
    const topK = opts.topK || 0;
    const topP = opts.topP !== undefined ? opts.topP : 1.0;
    const greedy = !!opts.greedy || temperature === 0;
    const V = logits.length;

    const scores = new Float64Array(V);
    for (let i = 0; i < V; i++) scores[i] = logits[i] / (temperature || 1);

    // Optional repetition penalty: discourage tokens we have already produced.
    if (opts.repetitionPenalty && opts.history) {
      const pen = opts.repetitionPenalty;
      const seen = new Set(opts.history);
      for (const id of seen) {
        if (id >= 0 && id < V) scores[id] = scores[id] > 0 ? scores[id] / pen : scores[id] * pen;
      }
    }

    if (greedy) {
      // Greedy decoding: just take the single most likely token.
      return argmax(scores);
    }

    // Rank all tokens by score (descending).
    const order = new Array(V);
    for (let i = 0; i < V; i++) order[i] = i;
    order.sort((a, b) => scores[b] - scores[a]);

    // top-k: keep only the k best candidates.
    let candidates = topK > 0 ? order.slice(0, topK) : order;

    // Softmax over the candidates.
    let maxv = -Infinity;
    for (const id of candidates) if (scores[id] > maxv) maxv = scores[id];
    const probs = [];
    let sum = 0;
    for (const id of candidates) {
      const e = Math.exp(scores[id] - maxv);
      probs.push(e);
      sum += e;
    }
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;

    // top-p (nucleus): keep the smallest set whose total probability >= topP.
    if (topP < 1.0) {
      let cumulative = 0;
      let cut = probs.length;
      for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (cumulative >= topP) {
          cut = i + 1;
          break;
        }
      }
      candidates = candidates.slice(0, cut);
      probs.length = cut;
      let norm = 0;
      for (const p of probs) norm += p;
      for (let i = 0; i < probs.length; i++) probs[i] /= norm;
    }

    // Sample from the (re-normalized) distribution.
    let r = Math.random();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return candidates[i];
    }
    return candidates[candidates.length - 1];
  }

  /**
   * Produce a single next token given the current sequence of ids.
   * This is the building block of `generate()` and is also used by the web UI
   * to stream tokens to the screen one at a time.
   */
  nextToken(ids, opts = {}) {
    const blockSize = this.config.blockSize;
    const V = this.config.vocabSize;
    const ctx = ids.slice(Math.max(0, ids.length - blockSize));
    const T = ctx.length;
    const logits = noGrad(() => this.forward(fromIds([ctx])));
    const base = (T - 1) * V;
    const row = logits.data.subarray(base, base + V);
    return this._sampleRow(row, Object.assign({}, opts, { history: ids }));
  }

  /**
   * Generate a continuation from a prompt given as token ids.
   *
   * @param {number[]} promptIds
   * @param {object} [opts]
   *   maxNewTokens, temperature, topK, topP, greedy, repetitionPenalty,
   *   eosId, onToken(id) callback for streaming.
   * @returns {number[]} the newly generated ids (not including the prompt)
   */
  generate(promptIds, opts = {}) {
    const maxNew = opts.maxNewTokens !== undefined ? opts.maxNewTokens : 64;
    const eosId = opts.eosId !== undefined ? opts.eosId : 3;

    let ids = promptIds.slice();
    const produced = [];

    for (let step = 0; step < maxNew; step++) {
      const next = this.nextToken(ids, opts);
      ids = ids.concat(next);
      produced.push(next);
      if (opts.onToken) opts.onToken(next);
      if (next === eosId) break;
    }

    return produced;
  }

  /**
   * Generate text from a prompt string, using the attached tokenizer.
   * `opts.onText` receives each newly decoded fragment for streaming UIs.
   */
  generateText(prompt, opts = {}) {
    if (!this.tokenizer) throw new Error('generateText() needs a tokenizer');
    const promptIds = this.tokenizer.encode(prompt);
    const full = promptIds.slice();
    let streamed = this.tokenizer.decode(promptIds).length;
    const onToken = (id) => {
      full.push(id);
      if (opts.onText) {
        const text = this.tokenizer.decode(full);
        const delta = text.slice(streamed);
        streamed = text.length;
        if (delta) opts.onText(delta);
      }
    };
    const produced = this.generate(promptIds, Object.assign({}, opts, { onToken }));
    return this.tokenizer.decode(produced);
  }

  /* ----------------------------------------------------------------------
   * The .GenAI file format
   * -------------------------------------------------------------------- */
  /**
   * Serialize the model to a human-readable text format:
   *
   *   TINYAI-GENAI
   *   version: 1
   *   ...
   *   ---CONFIG---
   *   { ... }
   *   ---TOKENIZER---
   *   { ... }
   *   ---WEIGHTS---
   *   tokEmb 4096,128 <base64>
   *   ...
   *   ---END---
   *
   * The header is plain text so you can inspect a model in any text editor.
   * Weight arrays are base64-encoded 32-bit floats to keep the file small.
   */
  saveModel() {
    const lines = [];
    const s = this.summary();
    lines.push('TINYAI-GENAI');
    lines.push('version: 1');
    lines.push('generator: TinyAI');
    lines.push('created: ' + new Date().toISOString());
    lines.push('architecture: ' + s.architecture);
    lines.push('layers: ' + s.layers);
    lines.push('heads: ' + s.heads);
    lines.push('dModel: ' + s.dModel);
    lines.push('blockSize: ' + s.blockSize);
    lines.push('vocabSize: ' + s.vocabSize);
    lines.push('parameters: ' + s.parameters);
    lines.push('tieWeights: ' + s.tieWeights);
    lines.push('tokenizer: ' + (this.tokenizer ? this.tokenizer.type : 'none'));
    lines.push('---CONFIG---');
    lines.push(JSON.stringify(this.config));
    lines.push('---TOKENIZER---');
    lines.push(JSON.stringify(this.tokenizer ? this.tokenizer.save() : null));
    lines.push('---WEIGHTS---');
    for (const p of this.parameters()) {
      lines.push(p.name + ' ' + p.tensor.shape.join(',') + ' ' + encodeWeights(p.tensor.data));
    }
    lines.push('---END---');
    return lines.join('\n');
  }

  /** Rebuild a model from text produced by saveModel(). */
  static loadModel(text) {
    const lines = text.split(/\r?\n/);
    if (lines[0].trim() !== 'TINYAI-GENAI') {
      throw new Error('Not a TinyAI .GenAI file');
    }

    let config = null;
    let tokenizer = null;
    let i = 1;
    // Skip the human-readable header until we reach ---CONFIG---.
    for (; i < lines.length; i++) {
      if (lines[i] === '---CONFIG---') {
        config = JSON.parse(lines[i + 1]);
        i += 2;
        break;
      }
    }
    if (lines[i] === '---TOKENIZER---') {
      const tokObj = JSON.parse(lines[i + 1]);
      tokenizer = tokObj ? _tokenizer.BaseTokenizer.load(tokObj) : null;
      i += 2;
    }
    if (lines[i] !== '---WEIGHTS---') throw new Error('Malformed .GenAI file');

    const model = new TinyModel(config, tokenizer);

    // Build a name -> tensor lookup.
    const map = new Map();
    for (const p of model.parameters()) map.set(p.name, p.tensor);

    i += 1;
    for (; i < lines.length; i++) {
      const line = lines[i];
      if (!line || line === '---END---') break;
      const firstSpace = line.indexOf(' ');
      const secondSpace = line.indexOf(' ', firstSpace + 1);
      const name = line.slice(0, firstSpace);
      const data = line.slice(secondSpace + 1);
      const target = map.get(name);
      if (!target) continue;
      const decoded = decodeWeights(data, target.size);
      target.data.set(decoded);
    }
    return model;
  }

  /** Node convenience: read and load a .GenAI file from disk. */
  static loadFile(path) {
    if (typeof require !== 'function') throw new Error('loadFile() is Node-only');
    const fs = require('fs');
    return TinyModel.loadModel(fs.readFileSync(path, 'utf8'));
  }

  /** Node convenience: write the model to disk as a .GenAI file. */
  saveFile(path) {
    if (typeof require !== 'function') throw new Error('saveFile() is Node-only');
    const fs = require('fs');
    const text = this.saveModel();
    fs.writeFileSync(path, text);
    return Buffer.byteLength(text);
  }

  /** Approximate on-disk size in MB. */
  sizeMB() {
    const text = this.saveModel();
    let bytes;
    if (typeof Buffer !== 'undefined') bytes = Buffer.byteLength(text);
    else bytes = new TextEncoder().encode(text).length;
    return bytes / (1024 * 1024);
  }
}

const TinyModelNS = {
  TinyModel,
  DEFAULT_CONFIG,
  encodeWeights,
  decodeWeights,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyModelNS;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyModelNS);
}
