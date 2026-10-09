/*
 * tiny-ai.js — v2
 * ---------------------------------------------------------------------------
 * From-scratch, no libraries, no pretrained weights, no API calls.
 *
 * A word-level seq2seq GRU with multi-layer (optionally bidirectional) encoder,
 * multi-layer decoder, dot-product attention with masking, a coverage term, and
 * a copy mechanism. Everything here is plain JavaScript.
 *
 * Math, per decode step t:
 *
 *   encoder (per layer, forward dir):   f_i = GRU(x_i, f_{i-1})
 *   encoder (per layer, backward dir):  b_i = GRU(x_i, b_{i+1})
 *   enc_i = [ f_i ; b_i ]                       (2H when bidirectional)
 *   query q = W_q * s_t                          (s_t = top decoder state)
 *   scores_i = (q . enc_i)/sqrt(E) + w_cov*tanh(cov_i)
 *   alpha = softmax(scores)
 *   ctx = sum_i alpha_i enc_i
 *   decoder layer 0 input: [ E[prev_t] ; ctx ]
 *   decoder layer l input: h_{l-1, t}
 *   top = s_{L-1, t}
 *   logits = W_out * top + b_out
 *   p_gen = sigmoid(W_gen . [top ; ctx ; E[prev_t]])
 *   copy  = sum_{i: src_i = w} alphaCopy_i
 *   p(w)  = p_gen * softmax(logits)[w] + (1 - p_gen) * copy
 *
 * Training adds: mini-batch (gradient accumulation), label smoothing, word
 * dropout, embedding/hidden dropout, LR warmup + cosine decay, gradient
 * clipping, validation split with early stopping, greedy / beam decoding and
 * an exact-match metric.
 *
 * WHAT CHANGED vs. the original single-example model
 *   - encoder/decoder are now stacks of GRU layers (cfg.layers)
 *   - encoder can run bidirectionally (cfg.bidir)
 *   - attention uses a learned query projection, 1/sqrt(E) scaling and a mask
 *   - optional separate attention head for the copy distribution (cfg.attnHeads)
 *   - optional coverage mechanism (cfg.coverage) with a coverage loss
 *   - mini-batch = average of per-example gradients; attention/loss support
 *     masking so padded positions would get zero weight (no padding is used
 *     by default because each sequence keeps its true length)
 *   - label smoothing, word dropout, embedding + hidden dropout
 *   - warmup + cosine LR schedule
 *   - greedy and beam-search decoding, exact-match accuracy
 *   - .GenAI format bumped to v2; v1 files still load (migration path)
 *   - the model is data-free: it trains on whatever dataset / splits you pass
 *     to init(), e.g. TinyAI.init({ dataset: [...], splits: {...} })
 * ---------------------------------------------------------------------------
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.TinyAI = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ============================================================
   * 1. Utilities
   * ============================================================ */

  function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  function gaussian(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randArray(n, scale, rng) {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = gaussian(rng) * scale;
    return a;
  }

  function zeros(n) { return new Float64Array(n); }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function now() {
    return (typeof performance !== 'undefined') ? performance.now() : Date.now();
  }

  function softmaxArr(logits, out, n) {
    let max = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
    if (max === -Infinity) max = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(logits[i] - max);
      out[i] = e;
      sum += e;
    }
    const inv = 1 / (sum + 1e-12);
    for (let i = 0; i < n; i++) out[i] *= inv;
  }

  // Elementwise dropout. The returned mask already contains the 1/(1-p)
  // rescaling, so backward simply multiplies the input gradient by mask.
  function dropVec(src, n, rate, rng) {
    const v = new Float64Array(n);
    const mask = new Float64Array(n);
    const keep = 1 - rate;
    for (let i = 0; i < n; i++) {
      if (rng() < keep) {
        const k = 1 / keep;
        mask[i] = k;
        v[i] = src[i] * k;
      } else {
        mask[i] = 0;
        v[i] = 0;
      }
    }
    return { v: v, mask: mask };
  }

  function wordDropOnce(src, unkId, rate, rng) {
    const out = src.slice();
    for (let i = 0; i < out.length; i++) if (rng() < rate) out[i] = unkId;
    return out;
  }

  // Warmup (linear) then cosine decay to cfg.minLr.
  function lrAt(step, cfg) {
    const w = Math.max(1, cfg.warmupSteps | 0);
    const d = Math.max(w + 1, cfg.lrDecaySteps | 0);
    const base = cfg.lr, min = cfg.minLr;
    if (step < w) return base * (step + 1) / w;
    const p = Math.min(1, (step - w) / (d - w));
    return min + 0.5 * (base - min) * (1 + Math.cos(Math.PI * p));
  }

  function seqEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }

  function initHidden(H, L) {
    const a = new Array(L);
    for (let l = 0; l < L; l++) a[l] = zeros(H);
    return a;
  }

  /* ============================================================
   * 2. Tokenizer
   * ============================================================ */

  function tokenize(text) {
    return String(text || '').toLowerCase()
      .match(/[a-z0-9'-]+|[?!.,:;]/g) || [];
  }

  function detokenize(tokens) {
    let s = tokens.join(' ');
    return s.replace(/ ([?!.,:;])/g, '$1');
  }

  function encodeTokens(tokens, tokToId) {
    const unk = tokToId['<unk>'];
    return tokens.map(t => tokToId[t] !== undefined ? tokToId[t] : unk);
  }

  function buildVocab(dataset, minCount) {
    const counts = Object.create(null);
    for (const ex of dataset) {
      for (const tok of tokenize(ex.src)) counts[tok] = (counts[tok] || 0) + 1;
      for (const tok of tokenize(ex.tgt)) counts[tok] = (counts[tok] || 0) + 1;
    }
    const vocab = ['<pad>', '<bos>', '<eos>', '<unk>'];
    const sorted = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    for (const tok of sorted) if (counts[tok] >= minCount) vocab.push(tok);
    const tokToId = Object.create(null);
    vocab.forEach((t, i) => { tokToId[t] = i; });
    return { vocab, tokToId };
  }

  /* ============================================================
   * 2b. Data / splits
   * ============================================================
   * The model owns no data. init() accepts `dataset` (an array of
   * { src, tgt, split }) and/or an explicit `splits` object. Split names are
   * accepted in both the new ("val_indist"/"val_compositional"/
   * "val_systematic") and legacy ("in"/"comp"/"sys") spellings so old callers
   * keep working. If no splits are given, examples are grouped by their own
   * `.split` field.
   */
  const SPLIT_ALIASES = {
    train: 'train',
    in: 'inDist', indist: 'inDist', inDist: 'inDist', val_indist: 'inDist',
    comp: 'compositional', compositional: 'compositional', val_compositional: 'compositional',
    sys: 'systematic', systematic: 'systematic', val_systematic: 'systematic'
  };

  function emptySplits() {
    return { train: [], inDist: [], compositional: [], systematic: [], all: [] };
  }

  function groupBySplit(dataset) {
    const s = emptySplits();
    for (const ex of dataset) {
      const key = SPLIT_ALIASES[ex.split] || 'train';
      s[key].push(ex);
    }
    s.all = dataset.slice();
    return s;
  }

  function normalizeSplits(splits) {
    const s = emptySplits();
    for (const k of Object.keys(splits)) {
      const key = SPLIT_ALIASES[k];
      if (key) s[key] = s[key].concat(splits[k] || []);
    }
    s.all = s.train.concat(s.inDist, s.compositional, s.systematic);
    return s;
  }

  function resolveData(opts) {
    if (opts.splits) return normalizeSplits(opts.splits);
    if (opts.dataset && opts.dataset.length) return groupBySplit(opts.dataset);
    return emptySplits();
  }

  /* ============================================================
   * 3. GRU cell (forward + backward)
   * ============================================================
   * Same math as the original implementation:
   *   z = sigmoid(Wz x + Uz h + bz)
   *   r = sigmoid(Wr x + Ur h + br)
   *   n = tanh(Wh x + Uh (r*h) + bh)
   *   h' = (1-z)*h + z*n
   */

  function createGRU(H, D, rng) {
    const scale = 1 / Math.sqrt(H);
    return {
      H, D,
      Wz: randArray(H * D, scale, rng), Uz: randArray(H * H, scale, rng), bz: zeros(H),
      Wr: randArray(H * D, scale, rng), Ur: randArray(H * H, scale, rng), br: zeros(H),
      Wh: randArray(H * D, scale, rng), Uh: randArray(H * H, scale, rng), bh: zeros(H),
      dWz: zeros(H * D), dUz: zeros(H * H), dbz: zeros(H),
      dWr: zeros(H * D), dUr: zeros(H * H), dbr: zeros(H),
      dWh: zeros(H * D), dUh: zeros(H * H), dbh: zeros(H)
    };
  }

  function newGruCache(H) {
    return {
      z: zeros(H), r: zeros(H), n: zeros(H), nPre: zeros(H),
      _dz: zeros(H), _dn: zeros(H), _dr: zeros(H), _drh: zeros(H)
    };
  }

  function gruForward(cell, x, hPrev, hOut, cache) {
    const H = cell.H, D = cell.D;
    const { Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh } = cell;
    const z = cache.z, r = cache.r, n = cache.n, nPre = cache.nPre;

    for (let i = 0; i < H; i++) {
      let sz = bz[i], sr = br[i], sn = bh[i];
      const rowD = i * D, rowH = i * H;
      for (let k = 0; k < D; k++) {
        const xk = x[k];
        sz += Wz[rowD + k] * xk;
        sr += Wr[rowD + k] * xk;
        sn += Wh[rowD + k] * xk;
      }
      for (let j = 0; j < H; j++) {
        const hj = hPrev[j];
        sz += Uz[rowH + j] * hj;
        sr += Ur[rowH + j] * hj;
      }
      z[i] = sigmoid(sz);
      r[i] = sigmoid(sr);
      nPre[i] = sn;
    }
    for (let i = 0; i < H; i++) {
      let sn = nPre[i];
      const rowH = i * H;
      for (let j = 0; j < H; j++) sn += Uh[rowH + j] * r[j] * hPrev[j];
      n[i] = Math.tanh(sn);
      hOut[i] = (1 - z[i]) * hPrev[i] + z[i] * n[i];
    }
  }

  function gruBackward(cell, x, hPrev, dH, cache, dX, dHprev) {
    const H = cell.H, D = cell.D;
    const { Wz, Uz, Wr, Ur, Wh, Uh } = cell;
    const z = cache.z, r = cache.r, n = cache.n;
    const dz = cache._dz, dn = cache._dn, dr = cache._dr, drh = cache._drh;

    for (let i = 0; i < H; i++) {
      const dh = dH[i];
      dz[i] = dh * (n[i] - hPrev[i]) * z[i] * (1 - z[i]);
      dn[i] = dh * z[i] * (1 - n[i] * n[i]);
      dHprev[i] += dh * (1 - z[i]);
    }
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let i = 0; i < H; i++) s += Uh[i * H + j] * dn[i];
      drh[j] = s;
    }
    for (let j = 0; j < H; j++) {
      dr[j] = drh[j] * hPrev[j] * r[j] * (1 - r[j]);
      dHprev[j] += drh[j] * r[j];
    }
    for (let i = 0; i < H; i++) {
      const rowD = i * D, rowH = i * H;
      for (let k = 0; k < D; k++) {
        const xk = x[k];
        cell.dWz[rowD + k] += dz[i] * xk;
        cell.dWr[rowD + k] += dr[i] * xk;
        cell.dWh[rowD + k] += dn[i] * xk;
      }
      for (let j = 0; j < H; j++) {
        const hj = hPrev[j];
        cell.dUz[rowH + j] += dz[i] * hj;
        cell.dUr[rowH + j] += dr[i] * hj;
        cell.dUh[rowH + j] += dn[i] * r[j] * hj;
      }
      cell.dbz[i] += dz[i];
      cell.dbr[i] += dr[i];
      cell.dbh[i] += dn[i];
    }
    for (let j = 0; j < H; j++) {
      let s = 0;
      for (let i = 0; i < H; i++) {
        s += Uz[i * H + j] * dz[i];
        s += Ur[i * H + j] * dr[i];
      }
      dHprev[j] += s;
    }
    if (dX) {
      for (let k = 0; k < D; k++) {
        let s = 0;
        for (let i = 0; i < H; i++) {
          const rowD = i * D;
          s += Wz[rowD + k] * dz[i];
          s += Wr[rowD + k] * dr[i];
          s += Wh[rowD + k] * dn[i];
        }
        dX[k] += s;
      }
    }
  }

  function pushCell(out, c) {
    out.push([c.Wz, c.dWz], [c.Uz, c.dUz], [c.bz, c.dbz],
             [c.Wr, c.dWr], [c.Ur, c.dUr], [c.br, c.dbr],
             [c.Wh, c.dWh], [c.Uh, c.dUh], [c.bh, c.dbh]);
  }

  /* ============================================================
   * 4. Configuration
   * ============================================================ */

  const DEFAULTS = {
    hidden: 64,
    embed: 32,
    layers: 2,
    bidir: true,

    targetSteps: 10000,
    batchSize: 6,
    lr: 0.004,
    minLr: 0.0002,
    warmupSteps: 300,
    lrDecaySteps: 10000,

    maxGen: 24,
    seed: 1337,
    valEvery: 500,
    patience: 8,

    labelSmoothing: 0.05,
    wordDropout: 0.1,
    embDropout: 0.1,
    hiddenDropout: 0.1,

    coverage: true,
    coverageWeight: 0.5,
    attnHeads: false,
    attnScale: true,

    minCount: 1
  };

  /* ============================================================
   * 5. Model
   * ============================================================ */

  function Model(opts) {
    const rng = makeRng(opts.seed);
    this.rng = rng;
    this.cfg = opts;

    // Data + splits come from init(); the model itself owns none.
    const split = resolveData(opts);
    this.splitSets = split;
    this.dataset = split.all;

    let vocab, tokToId;
    if (opts.vocab && opts.vocab.length) {
      vocab = opts.vocab.slice();
      tokToId = Object.create(null);
      vocab.forEach((t, i) => { tokToId[t] = i; });
    } else {
      const r = buildVocab(this.dataset, opts.minCount);
      vocab = r.vocab;
      tokToId = r.tokToId;
    }
    this.vocab = vocab;
    this.tokToId = tokToId;
    this.V = vocab.length;
    this.padId = tokToId['<pad>'];
    this.bosId = tokToId['<bos>'];
    this.eosId = tokToId['<eos>'];
    this.unkId = tokToId['<unk>'];

    const encode = (ex) => this._encodePair(ex);
    this.trainData = split.train.map(encode);
    this.valData = split.inDist.map(encode);
    this.inDistData = split.inDist.map(encode);
    this.compData = split.compositional.map(encode);
    this.sysData = split.systematic.map(encode);

    const V = this.V, H = opts.hidden, D = opts.embed, L = opts.layers;
    this.H = H; this.D = D; this.layers = L; this.bidir = !!opts.bidir;
    this.encOutDim = (this.bidir ? 2 : 1) * H;

    // Shared embedding
    this.E = randArray(V * D, 1 / Math.sqrt(D), rng);
    this.dE = zeros(V * D);

    // Encoder stack: one {fwd, bwd} pair per layer.
    this.encCells = [];
    for (let l = 0; l < L; l++) {
      const inDim = (l === 0) ? D : this.encOutDim;
      this.encCells.push({
        fwd: createGRU(H, inDim, rng),
        bwd: this.bidir ? createGRU(H, inDim, rng) : null
      });
    }

    // Decoder stack: layer 0 reads [embedding ; context], others read hidden.
    this.decCells = [];
    for (let l = 0; l < L; l++) {
      const inDim = (l === 0) ? (D + this.encOutDim) : H;
      this.decCells.push(createGRU(H, inDim, rng));
    }

    // Output projection
    this.W_out = randArray(V * H, 1 / Math.sqrt(H), rng);
    this.b_out = zeros(V);
    this.dW_out = zeros(V * H);
    this.db_out = zeros(V);

    // Copy gate input [top ; ctx ; prev_embedding]
    const genLen = H + this.encOutDim + D;
    this.W_gen = randArray(genLen, 1 / Math.sqrt(H), rng);
    this.b_gen = zeros(1);
    this.dW_gen = zeros(genLen);
    this.db_gen = zeros(1);

    // Attention query projection(s): H -> encOutDim
    this.W_q_gen = randArray(this.encOutDim * H, 1 / Math.sqrt(H), rng);
    this.dW_q_gen = zeros(this.encOutDim * H);
    if (opts.attnHeads) {
      this.W_q_copy = randArray(this.encOutDim * H, 1 / Math.sqrt(H), rng);
      this.dW_q_copy = zeros(this.encOutDim * H);
    } else {
      this.W_q_copy = null; this.dW_q_copy = null;
    }

    // Coverage scalar
    this.w_cov = zeros(1);
    this.dw_cov = zeros(1);

    // Adam state (lazy)
    this.adam = null;

    // Training bookkeeping
    this.steps = 0;
    this.loss = null;
    this.valLoss = null;
    this.bestValLoss = Infinity;
    this.badEvals = 0;
    this.stopped = false;
    this.lastLr = opts.lr;
    this._order = null;
    this._orderPos = 0;
  }

  Model.prototype._encodePair = function (ex) {
    const src = encodeTokens(tokenize(ex.src), this.tokToId);
    const tgt = encodeTokens(tokenize(ex.tgt), this.tokToId);
    if (src.length === 0) src.push(this.unkId);
    if (tgt.length === 0) tgt.push(this.unkId);
    return { src, tgt };
  };

  /* -------- encoder forward (fills g) -------- */
  Model.prototype._runEncoder = function (g, training) {
    const H = this.H, D = this.D, L = this.layers, E = this.encOutDim, bidir = this.bidir;
    const S = g.S, src = g.src;

    g.encOut = new Array(S);
    g.encIn = [];        // encIn[l][i] = input vector fed to layer l
    g.encInMasks = [];   // encInMasks[l][i] = dropout mask on that input
    g.encStatesF = [];
    g.encStatesB = [];
    g.encCachesF = [];
    g.encCachesB = [];

    // Layer 0 input = embedding (optionally embedding-dropped).
    let layerIn = new Array(S);
    const embMasks = new Array(S);
    for (let i = 0; i < S; i++) {
      const e = this.E.subarray(src[i] * D, (src[i] + 1) * D);
      if (training && this.cfg.embDropout > 0) {
        const d = dropVec(e, D, this.cfg.embDropout, this.rng);
        layerIn[i] = d.v; embMasks[i] = d.mask;
      } else {
        layerIn[i] = e.slice(); embMasks[i] = null;
      }
    }
    g.encIn.push(layerIn);
    g.encInMasks.push(embMasks);

    for (let l = 0; l < L; l++) {
      const fwd = this.encCells[l].fwd, bwd = this.encCells[l].bwd;

      const sf = new Array(S + 1); sf[0] = zeros(H);
      const cf = new Array(S);
      for (let i = 0; i < S; i++) {
        const h = zeros(H), c = newGruCache(H);
        gruForward(fwd, layerIn[i], sf[i], h, c);
        sf[i + 1] = h; cf[i] = c;
      }

      let sb = null, cb = null;
      if (bidir) {
        sb = new Array(S + 1); sb[S] = zeros(H); cb = new Array(S);
        for (let i = S - 1; i >= 0; i--) {
          const h = zeros(H), c = newGruCache(H);
          gruForward(bwd, layerIn[i], sb[i + 1], h, c);
          sb[i] = h; cb[i] = c;
        }
      }

      const outLayer = new Array(S);
      for (let i = 0; i < S; i++) {
        const o = zeros(E);
        o.set(sf[i + 1], 0);
        if (bidir) o.set(sb[i], H);
        outLayer[i] = o;
      }
      g.encStatesF.push(sf); g.encStatesB.push(sb);
      g.encCachesF.push(cf); g.encCachesB.push(cb);
      g.encOut = outLayer;

      if (l < L - 1) {
        const nextIn = new Array(S), masks = new Array(S);
        for (let i = 0; i < S; i++) {
          if (training && this.cfg.hiddenDropout > 0) {
            const d = dropVec(outLayer[i], E, this.cfg.hiddenDropout, this.rng);
            nextIn[i] = d.v; masks[i] = d.mask;
          } else {
            nextIn[i] = outLayer[i]; masks[i] = null;
          }
        }
        layerIn = nextIn;
        g.encIn.push(layerIn);
        g.encInMasks.push(masks);
      }
    }
  };

  /* -------- attention (fills att) --------
   * srcMask (optional): 0/undefined marks a padded source position, which is
   * forced to zero attention weight. */
  Model.prototype._attention = function (g, topPrev, covBefore, att) {
    const H = this.H, E = this.encOutDim, S = g.S;
    const invScale = this.cfg.attnScale ? 1 / Math.sqrt(E) : 1;

    const q = zeros(E);
    for (let a = 0; a < E; a++) {
      let s = 0; const row = a * H;
      for (let k = 0; k < H; k++) s += this.W_q_gen[row + k] * topPrev[k];
      q[a] = s;
    }
    const sc = zeros(S);
    for (let i = 0; i < S; i++) {
      if (g.srcMask && !g.srcMask[i]) { sc[i] = -1e30; continue; }
      let s = 0; const eo = g.encOut[i];
      for (let a = 0; a < E; a++) s += q[a] * eo[a];
      s *= invScale;
      if (this.cfg.coverage) s += this.w_cov[0] * Math.tanh(covBefore[i]);
      sc[i] = s;
    }
    const alpha = zeros(S);
    softmaxArr(sc, alpha, S);
    const ctx = zeros(E);
    for (let i = 0; i < S; i++) {
      const al = alpha[i]; if (al === 0) continue;
      const eo = g.encOut[i];
      for (let a = 0; a < E; a++) ctx[a] += al * eo[a];
    }
    att.q = q; att.scoresGen = sc; att.alphaGen = alpha; att.ctx = ctx;

    if (this.cfg.attnHeads) {
      const qc = zeros(E);
      for (let a = 0; a < E; a++) {
        let s = 0; const row = a * H;
        for (let k = 0; k < H; k++) s += this.W_q_copy[row + k] * topPrev[k];
        qc[a] = s;
      }
      const sc2 = zeros(S);
      for (let i = 0; i < S; i++) {
        if (g.srcMask && !g.srcMask[i]) { sc2[i] = -1e30; continue; }
        let s = 0; const eo = g.encOut[i];
        for (let a = 0; a < E; a++) s += qc[a] * eo[a];
        sc2[i] = s * invScale;
      }
      const a2 = zeros(S);
      softmaxArr(sc2, a2, S);
      att.qCopy = qc; att.scoresCopy = sc2; att.alphaCopy = a2;
    } else {
      att.alphaCopy = alpha;
    }
  };

  /* -------- full forward over one (src, tgt) pair -------- */
  Model.prototype._forwardExample = function (src, tgt, training) {
    const H = this.H, D = this.D, V = this.V, L = this.layers, E = this.encOutDim;
    const S = src.length, T = tgt.length + 1;
    const prevTokens = [this.bosId].concat(tgt);
    const targets = tgt.concat([this.eosId]);

    const g = { S, T, src, tgt, prevTokens, targets, training: !!training };
    this._runEncoder(g, training);

    g.decStates = [];
    g.decLayerIn = []; g.decLayerMask = []; g.decCaches = [];
    for (let l = 0; l < L; l++) {
      const a = new Array(T + 1);
      for (let t = 0; t <= T; t++) a[t] = zeros(H);
      g.decStates.push(a);
      g.decLayerIn.push(new Array(T));
      g.decLayerMask.push(new Array(T));
      g.decCaches.push(new Array(T));
    }
    g.attn = new Array(T);
    g.decEmb = new Array(T);
    g.decEmbMask = new Array(T);

    const eps = this.cfg.labelSmoothing;
    const qTarget = 1 - eps + eps / V, qOther = eps / V;

    let cov = zeros(S);
    let ceSum = 0, covSum = 0;

    for (let t = 0; t < T; t++) {
      const att = {};
      att.covBefore = this.cfg.coverage ? cov.slice() : null;
      this._attention(g, g.decStates[L - 1][t], cov, att);
      const ctx = att.ctx;

      const embSrc = this.E.subarray(prevTokens[t] * D, (prevTokens[t] + 1) * D);
      let emb, embMask = null;
      if (training && this.cfg.embDropout > 0) {
        const d = dropVec(embSrc, D, this.cfg.embDropout, this.rng);
        emb = d.v; embMask = d.mask;
      } else {
        emb = embSrc.slice();
      }
      g.decEmb[t] = emb; g.decEmbMask[t] = embMask;

      const in0 = zeros(D + E);
      in0.set(emb, 0); in0.set(ctx, D);

      let prevHout = null;
      for (let l = 0; l < L; l++) {
        let xin = (l === 0) ? in0 : prevHout;
        let xmask = null;
        if (l > 0 && training && this.cfg.hiddenDropout > 0) {
          const d = dropVec(prevHout, H, this.cfg.hiddenDropout, this.rng);
          xin = d.v; xmask = d.mask;
        }
        g.decLayerIn[l][t] = xin;
        g.decLayerMask[l][t] = xmask;
        const c = newGruCache(H), hout = zeros(H);
        gruForward(this.decCells[l], xin, g.decStates[l][t], hout, c);
        g.decStates[l][t + 1] = hout;
        g.decCaches[l][t] = c;
        prevHout = hout;
      }
      const top = prevHout;

      const logits = zeros(V), pVocab = zeros(V);
      for (let w = 0; w < V; w++) {
        let s = this.b_out[w]; const row = w * H;
        for (let i = 0; i < H; i++) s += this.W_out[row + i] * top[i];
        logits[w] = s;
      }
      softmaxArr(logits, pVocab, V);

      const genLen = H + E + D;
      const gi = zeros(genLen);
      gi.set(top, 0); gi.set(ctx, H); gi.set(emb, H + E);
      let pGenRaw = this.b_gen[0];
      for (let i = 0; i < genLen; i++) pGenRaw += this.W_gen[i] * gi[i];
      const pGen = sigmoid(pGenRaw);

      const copyDist = zeros(V);
      for (let i = 0; i < S; i++) copyDist[src[i]] += att.alphaCopy[i];

      const finalProbs = zeros(V);
      for (let w = 0; w < V; w++) finalProbs[w] = pGen * pVocab[w] + (1 - pGen) * copyDist[w];

      const target = targets[t];
      let ce = 0;
      for (let w = 0; w < V; w++) {
        const q = (w === target) ? qTarget : qOther;
        ce += -q * Math.log(finalProbs[w] + 1e-12);
      }
      ceSum += ce;

      if (this.cfg.coverage) {
        if (t > 0) {
          let cl = 0;
          for (let i = 0; i < S; i++) cl += Math.min(att.alphaGen[i], att.covBefore[i]);
          covSum += cl;
        }
        for (let i = 0; i < S; i++) cov[i] += att.alphaGen[i];
      }

      att.logits = logits; att.pVocab = pVocab; att.pGen = pGen; att.pGenRaw = pGenRaw;
      att.copyDist = copyDist; att.finalProbs = finalProbs; att.target = target; att.gi = gi;
      g.attn[t] = att;
    }

    g.loss = (ceSum + (this.cfg.coverage ? this.cfg.coverageWeight * covSum : 0)) / T;
    return g;
  };

  /* -------- backward over one example, scaled by `scale` -------- */
  Model.prototype._backwardExample = function (g, scale) {
    const H = this.H, D = this.D, V = this.V, L = this.layers, E = this.encOutDim, bidir = this.bidir;
    const S = g.S, T = g.T, src = g.src, prevTokens = g.prevTokens;
    const eps = this.cfg.labelSmoothing;
    const qTarget = 1 - eps + eps / V, qOther = eps / V;

    const dDecStates = [];
    for (let l = 0; l < L; l++) {
      const a = new Array(T + 1);
      for (let t = 0; t <= T; t++) a[t] = zeros(H);
      dDecStates.push(a);
    }
    const dEncOut = new Array(S);
    for (let i = 0; i < S; i++) dEncOut[i] = zeros(E);

    const invT = scale / T;
    const invScale = this.cfg.attnScale ? 1 / Math.sqrt(E) : 1;

    for (let t = T - 1; t >= 0; t--) {
      const att = g.attn[t];
      const topPrev = g.decStates[L - 1][t];
      const topOut = g.decStates[L - 1][t + 1];

      // ---- output distribution gradient (label-smoothed CE) ----
      const dFinal = zeros(V);
      for (let w = 0; w < V; w++) {
        const q = (w === att.target) ? qTarget : qOther;
        dFinal[w] = -q / (att.finalProbs[w] + 1e-12) * invT;
      }

      const pGen = att.pGen, pVocab = att.pVocab, copyDist = att.copyDist;
      const dPvocab = zeros(V), dCopyDist = zeros(V);
      let dPgenRaw = 0;
      for (let w = 0; w < V; w++) {
        dPvocab[w] = pGen * dFinal[w];
        dPgenRaw += dFinal[w] * (pVocab[w] - copyDist[w]);
        dCopyDist[w] = (1 - pGen) * dFinal[w];
      }
      dPgenRaw *= pGen * (1 - pGen);

      // ---- copy gate ----
      const gi = att.gi, genLen = gi.length;
      for (let i = 0; i < genLen; i++) this.dW_gen[i] += dPgenRaw * gi[i];
      this.db_gen[0] += dPgenRaw;

      const dTop = zeros(H);
      for (let i = 0; i < H; i++) dTop[i] += dPgenRaw * this.W_gen[i];
      const dCtx = zeros(E);
      for (let i = 0; i < E; i++) dCtx[i] += dPgenRaw * this.W_gen[H + i];
      const dEmb = zeros(D);
      for (let i = 0; i < D; i++) dEmb[i] += dPgenRaw * this.W_gen[H + E + i];

      // ---- softmax + W_out ----
      let sumDPv = 0;
      for (let w = 0; w < V; w++) sumDPv += dPvocab[w] * pVocab[w];
      const dLogits = zeros(V);
      for (let w = 0; w < V; w++) dLogits[w] = pVocab[w] * (dPvocab[w] - sumDPv);
      for (let w = 0; w < V; w++) {
        const dl = dLogits[w]; if (dl === 0) continue;
        this.db_out[w] += dl;
        const row = w * H;
        for (let i = 0; i < H; i++) {
          this.dW_out[row + i] += dl * topOut[i];
          dTop[i] += dl * this.W_out[row + i];
        }
      }

      // ---- add top-output gradient into the decoder state ----
      for (let i = 0; i < H; i++) dDecStates[L - 1][t + 1][i] += dTop[i];

      // ---- decoder layers, top-down ----
      for (let l = L - 1; l >= 0; l--) {
        const xin = g.decLayerIn[l][t];
        const hPrev = g.decStates[l][t];
        const dH = dDecStates[l][t + 1];
        const dX = zeros(this.decCells[l].D);
        const dHprev = dDecStates[l][t];
        gruBackward(this.decCells[l], xin, hPrev, dH, g.decCaches[l][t], dX, dHprev);
        if (l > 0) {
          const mask = g.decLayerMask[l][t];
          const prevOut = dDecStates[l - 1][t + 1];
          if (mask) for (let k = 0; k < H; k++) prevOut[k] += dX[k] * mask[k];
          else for (let k = 0; k < H; k++) prevOut[k] += dX[k];
        } else {
          for (let k = 0; k < D; k++) dEmb[k] += dX[k];
          for (let a = 0; a < E; a++) dCtx[a] += dX[D + a];
        }
      }

      // ---- attention backward (generate head) ----
      const alphaGen = att.alphaGen;
      const dAlphaGen = zeros(S);
      for (let i = 0; i < S; i++) {
        const eo = g.encOut[i];
        let dot = 0;
        for (let a = 0; a < E; a++) {
          dot += dCtx[a] * eo[a];
          dEncOut[i][a] += alphaGen[i] * dCtx[a];   // direct ctx path
        }
        dAlphaGen[i] += dot;
      }
      let dAlphaCopy = null;
      if (this.cfg.attnHeads) {
        dAlphaCopy = zeros(S);
        for (let i = 0; i < S; i++) dAlphaCopy[i] += dCopyDist[src[i]];
      } else {
        for (let i = 0; i < S; i++) dAlphaGen[i] += dCopyDist[src[i]];
      }
      // coverage loss gradient (covBefore treated as a constant)
      if (this.cfg.coverage && t > 0) {
        const cw = this.cfg.coverageWeight * invT;
        for (let i = 0; i < S; i++) if (alphaGen[i] < att.covBefore[i]) dAlphaGen[i] += cw;
      }
      let sumDA = 0;
      for (let i = 0; i < S; i++) sumDA += dAlphaGen[i] * alphaGen[i];
      const dScores = zeros(S);
      for (let i = 0; i < S; i++) dScores[i] = alphaGen[i] * (dAlphaGen[i] - sumDA);
      const q = att.q;
      for (let i = 0; i < S; i++) {
        const ds = dScores[i]; if (ds === 0) continue;
        const eo = g.encOut[i];
        for (let a = 0; a < E; a++) dEncOut[i][a] += ds * q[a] * invScale;
      }
      const dq = zeros(E);
      for (let a = 0; a < E; a++) {
        let s = 0;
        for (let i = 0; i < S; i++) s += dScores[i] * g.encOut[i][a];
        dq[a] = s * invScale;
      }
      if (this.cfg.coverage) {
        let dwc = 0;
        for (let i = 0; i < S; i++) dwc += dScores[i] * Math.tanh(att.covBefore[i]);
        this.dw_cov[0] += dwc;
      }
      const dTopPrev = zeros(H);
      for (let a = 0; a < E; a++) {
        const dqa = dq[a]; if (dqa === 0) continue;
        const row = a * H;
        for (let k = 0; k < H; k++) {
          this.dW_q_gen[row + k] += dqa * topPrev[k];
          dTopPrev[k] += dqa * this.W_q_gen[row + k];
        }
      }

      // ---- separate copy head backward ----
      if (this.cfg.attnHeads) {
        let sumDA2 = 0;
        for (let i = 0; i < S; i++) sumDA2 += dAlphaCopy[i] * att.alphaCopy[i];
        const dScores2 = zeros(S);
        for (let i = 0; i < S; i++) dScores2[i] = att.alphaCopy[i] * (dAlphaCopy[i] - sumDA2);
        const qc = att.qCopy;
        for (let i = 0; i < S; i++) {
          const ds = dScores2[i]; if (ds === 0) continue;
          const eo = g.encOut[i];
          for (let a = 0; a < E; a++) dEncOut[i][a] += ds * qc[a] * invScale;
        }
        const dqc = zeros(E);
        for (let a = 0; a < E; a++) {
          let s = 0;
          for (let i = 0; i < S; i++) s += dScores2[i] * g.encOut[i][a];
          dqc[a] = s * invScale;
        }
        for (let a = 0; a < E; a++) {
          const dqa = dqc[a]; if (dqa === 0) continue;
          const row = a * H;
          for (let k = 0; k < H; k++) {
            this.dW_q_copy[row + k] += dqa * topPrev[k];
            dTopPrev[k] += dqa * this.W_q_copy[row + k];
          }
        }
      }

      // query depends on the *previous* top state
      for (let k = 0; k < H; k++) dDecStates[L - 1][t][k] += dTopPrev[k];

      // ---- embedding of the previous token ----
      const prevTok = prevTokens[t];
      const em = g.decEmbMask[t];
      for (let k = 0; k < D; k++) {
        this.dE[prevTok * D + k] += em ? dEmb[k] * em[k] : dEmb[k];
      }
    }

    // ---- encoder backward, top layer down ----
    let dLayerOut = dEncOut;
    for (let l = L - 1; l >= 0; l--) {
      const inDim = (l === 0) ? D : E;
      const dLayerIn = new Array(S);
      for (let i = 0; i < S; i++) dLayerIn[i] = zeros(inDim);

      // forward direction
      const sf = g.encStatesF[l], cf = g.encCachesF[l];
      let carry = zeros(H);
      for (let i = S - 1; i >= 0; i--) {
        const dH = zeros(H);
        for (let k = 0; k < H; k++) dH[k] = dLayerOut[i][k] + carry[k];
        const dHprev = zeros(H), dX = zeros(inDim);
        gruBackward(this.encCells[l].fwd, g.encIn[l][i], sf[i], dH, cf[i], dX, dHprev);
        for (let k = 0; k < inDim; k++) dLayerIn[i][k] += dX[k];
        carry = dHprev;
      }

      // backward direction
      if (bidir) {
        const sb = g.encStatesB[l], cb = g.encCachesB[l];
        let carryB = zeros(H);
        for (let i = 0; i < S; i++) {
          const dH = zeros(H);
          for (let k = 0; k < H; k++) dH[k] = dLayerOut[i][H + k] + carryB[k];
          const dHprev = zeros(H), dX = zeros(inDim);
          gruBackward(this.encCells[l].bwd, g.encIn[l][i], sb[i + 1], dH, cb[i], dX, dHprev);
          for (let k = 0; k < inDim; k++) dLayerIn[i][k] += dX[k];
          carryB = dHprev;
        }
      }

      if (l > 0) {
        const masks = g.encInMasks[l];
        const prevOut = new Array(S);
        for (let i = 0; i < S; i++) {
          const o = zeros(E), m = masks[i];
          if (m) for (let k = 0; k < E; k++) o[k] = dLayerIn[i][k] * m[k];
          else for (let k = 0; k < E; k++) o[k] = dLayerIn[i][k];
          prevOut[i] = o;
        }
        dLayerOut = prevOut;
      } else {
        const masks = g.encInMasks[0];
        for (let i = 0; i < S; i++) {
          const m = masks[i];
          for (let k = 0; k < D; k++) {
            this.dE[src[i] * D + k] += m ? dLayerIn[i][k] * m[k] : dLayerIn[i][k];
          }
        }
      }
    }
  };

  Model.prototype._allParams = function () {
    const out = [[this.E, this.dE]];
    for (let l = 0; l < this.layers; l++) {
      pushCell(out, this.encCells[l].fwd);
      if (this.encCells[l].bwd) pushCell(out, this.encCells[l].bwd);
    }
    for (let l = 0; l < this.layers; l++) pushCell(out, this.decCells[l]);
    out.push([this.W_out, this.dW_out], [this.b_out, this.db_out]);
    out.push([this.W_gen, this.dW_gen], [this.b_gen, this.db_gen]);
    out.push([this.W_q_gen, this.dW_q_gen]);
    if (this.cfg.attnHeads) out.push([this.W_q_copy, this.dW_q_copy]);
    if (this.cfg.coverage) out.push([this.w_cov, this.dw_cov]);
    return out;
  };

  Model.prototype._zeroGrads = function () {
    const p = this._allParams();
    for (let i = 0; i < p.length; i++) p[i][1].fill(0);
  };

  Model.prototype.update = function (lr) {
    if (!this.adam) {
      const params = this._allParams();
      this.adam = {
        t: 0,
        state: params.map(([p]) => ({ m: zeros(p.length), v: zeros(p.length) }))
      };
    }
    const a = this.adam;
    a.t++;
    const bc1 = 1 - Math.pow(0.9, a.t);
    const bc2 = 1 - Math.pow(0.999, a.t);
    const params = this._allParams();
    for (let pi = 0; pi < params.length; pi++) {
      const [p, g] = params[pi];
      const st = a.state[pi];
      const m = st.m, v = st.v;
      for (let j = 0; j < p.length; j++) {
        let gj = g[j];
        if (gj > 5) gj = 5; else if (gj < -5) gj = -5;
        const mi = 0.9 * m[j] + 0.1 * gj;
        const vi = 0.999 * v[j] + 0.001 * gj * gj;
        m[j] = mi; v[j] = vi;
        const mhat = mi / bc1;
        const vhat = vi / bc2;
        p[j] -= lr * mhat / (Math.sqrt(vhat) + 1e-8);
        g[j] = 0;
      }
    }
  };

  Model.prototype._ensureOrder = function () {
    if (!this._order || this._orderPos >= this._order.length) {
      const n = this.trainData.length;
      const order = new Array(n);
      for (let i = 0; i < n; i++) order[i] = i;
      for (let i = n - 1; i > 0; i--) {
        const j = (this.rng() * (i + 1)) | 0;
        const t = order[i]; order[i] = order[j]; order[j] = t;
      }
      this._order = order; this._orderPos = 0;
    }
  };

  Model.prototype._sampleTrain = function () {
    this._ensureOrder();
    return this.trainData[this._order[this._orderPos++]];
  };

  Model.prototype.trainStep = function () {
    if (this.stopped) return 0;
    const bs = Math.max(1, this.cfg.batchSize | 0);
    this._zeroGrads();
    let lossSum = 0;
    for (let b = 0; b < bs; b++) {
      const ex = this._sampleTrain();
      let src = ex.src;
      if (this.cfg.wordDropout > 0) src = wordDropOnce(src, this.unkId, this.cfg.wordDropout, this.rng);
      const g = this._forwardExample(src, ex.tgt, true);
      this._backwardExample(g, 1 / bs);
      lossSum += g.loss;
    }
    const lr = lrAt(this.steps, this.cfg);
    this.update(lr);
    this.steps++;
    this.lastLr = lr;
    if (this.cfg.valEvery > 0 && this.steps % this.cfg.valEvery === 0) this.checkValidation();
    return lossSum / bs;
  };

  Model.prototype.evaluate = function (data, maxExamples) {
    const n = Math.min(data.length, maxExamples || data.length);
    let total = 0;
    for (let i = 0; i < n; i++) {
      const ex = data[i];
      total += this._forwardExample(ex.src, ex.tgt, false).loss;
    }
    return n ? total / n : 0;
  };

  Model.prototype.checkValidation = function () {
    this.valLoss = this.evaluate(this.valData, 80);
    if (this.valLoss < this.bestValLoss - 1e-4) {
      this.bestValLoss = this.valLoss;
      this.badEvals = 0;
    } else {
      this.badEvals++;
      if (this.badEvals >= this.cfg.patience) this.stopped = true;
    }
    return this.valLoss;
  };

  /* ============================================================
   * 6. Decoding
   * ============================================================ */

  Model.prototype._encodeOnly = function (src) {
    const g = { S: src.length, src: src };
    this._runEncoder(g, false);
    return g;
  };

  Model.prototype._encodeMessage = function (message) {
    const src = encodeTokens(tokenize(message), this.tokToId);
    if (src.length === 0) src.push(this.unkId);
    return src;
  };

  // One decode step. Returns the (temperature-scaled) final distribution plus
  // the new hidden state and attention. Does not mutate `st`.
  Model.prototype._genStep = function (g, st, temperature) {
    const H = this.H, D = this.D, V = this.V, L = this.layers, E = this.encOutDim;
    const att = {};
    this._attention(g, st.h[L - 1], st.cov, att);
    const ctx = att.ctx;
    const emb = this.E.subarray(st.prev * D, (st.prev + 1) * D);
    const in0 = zeros(D + E);
    in0.set(emb, 0); in0.set(ctx, D);

    const newH = new Array(L);
    let x = in0;
    for (let l = 0; l < L; l++) {
      const c = newGruCache(H), hout = zeros(H);
      gruForward(this.decCells[l], (l === 0 ? in0 : x), st.h[l], hout, c);
      newH[l] = hout; x = hout;
    }
    const top = newH[L - 1];

    const logits = zeros(V), pVocab = zeros(V);
    for (let w = 0; w < V; w++) {
      let s = this.b_out[w]; const row = w * H;
      for (let i = 0; i < H; i++) s += this.W_out[row + i] * top[i];
      logits[w] = s;
    }
    if (temperature && temperature !== 1) for (let w = 0; w < V; w++) logits[w] /= temperature;
    softmaxArr(logits, pVocab, V);

    const genLen = H + E + D;
    const gi = zeros(genLen);
    gi.set(top, 0); gi.set(ctx, H); gi.set(emb, H + E);
    let pGenRaw = this.b_gen[0];
    for (let i = 0; i < genLen; i++) pGenRaw += this.W_gen[i] * gi[i];
    const pGen = sigmoid(pGenRaw);

    const copyDist = zeros(V);
    for (let i = 0; i < g.S; i++) copyDist[g.src[i]] += att.alphaCopy[i];

    const probs = zeros(V);
    for (let w = 0; w < V; w++) probs[w] = pGen * pVocab[w] + (1 - pGen) * copyDist[w];
    return { probs: probs, alphaGen: att.alphaGen, newH: newH, att: att };
  };

  Model.prototype._newDecodeState = function (S) {
    return { h: initHidden(this.H, this.layers), cov: zeros(S), prev: this.bosId };
  };

  Model.prototype._advanceCov = function (st, alphaGen) {
    if (!this.cfg.coverage) return;
    for (let i = 0; i < st.cov.length; i++) st.cov[i] += alphaGen[i];
  };

  Model.prototype.greedy = function (message) {
    const src = Array.isArray(message) ? message : this._encodeMessage(message);
    const g = this._encodeOnly(src);
    const st = this._newDecodeState(src.length);
    const out = [];
    for (let t = 0; t < this.cfg.maxGen; t++) {
      const r = this._genStep(g, st, 1);
      let best = 0, bv = -Infinity;
      for (let w = 0; w < this.V; w++) if (r.probs[w] > bv) { bv = r.probs[w]; best = w; }
      st.h = r.newH;
      this._advanceCov(st, r.alphaGen);
      if (best === this.eosId) break;
      st.prev = best;
      if (best === this.padId || best === this.bosId || best === this.unkId) continue;
      out.push(best);
      if (out.length >= this.cfg.maxGen) break;
    }
    return out;
  };

  Model.prototype.sample = function (message, temperature) {
    const src = Array.isArray(message) ? message : this._encodeMessage(message);
    const g = this._encodeOnly(src);
    const st = this._newDecodeState(src.length);
    const temp = Math.max(0.15, temperature || 0.7);
    const out = [];
    for (let t = 0; t < this.cfg.maxGen; t++) {
      const r = this._genStep(g, st, temp);
      let total = 0;
      for (let w = 0; w < this.V; w++) total += r.probs[w];
      let rr = this.rng() * total, pick = 0;
      for (let w = 0; w < this.V; w++) { rr -= r.probs[w]; if (rr <= 0) { pick = w; break; } }
      st.h = r.newH;
      this._advanceCov(st, r.alphaGen);
      if (pick === this.eosId) break;
      st.prev = pick;
      if (pick === this.padId || pick === this.bosId || pick === this.unkId) continue;
      out.push(pick);
      if (out.length >= this.cfg.maxGen) break;
    }
    return out;
  };

  // Beam search over token ids. Returns the best token-id sequence.
  Model.prototype.beam = function (message, beamSize) {
    const src = Array.isArray(message) ? message : this._encodeMessage(message);
    const g = this._encodeOnly(src);
    const B = Math.max(1, beamSize || 4);
    const S = src.length;
    let beams = [{ tokens: [], logp: 0, h: initHidden(this.H, this.layers), cov: zeros(S), prev: this.bosId }];
    const finished = [];

    for (let step = 0; step < this.cfg.maxGen; step++) {
      const cand = [];
      for (const b of beams) {
        const r = this._genStep(g, { h: b.h, cov: b.cov, prev: b.prev }, 1);
        const newCov = b.cov.slice();
        if (this.cfg.coverage) for (let i = 0; i < S; i++) newCov[i] += r.alphaGen[i];
        for (let w = 0; w < this.V; w++) {
          const p = r.probs[w];
          if (p <= 1e-12) continue;
          const logp = b.logp + Math.log(p);
          if (w === this.eosId) {
            finished.push({ tokens: b.tokens.slice(), logp: logp });
            continue;
          }
          if (w === this.padId || w === this.bosId || w === this.unkId) continue;
          cand.push({ tokens: b.tokens.concat([w]), logp: logp, h: r.newH, cov: newCov, prev: w });
        }
      }
      if (cand.length === 0) break;
      cand.sort((x, y) => y.logp - x.logp);
      beams = cand.slice(0, B);
      if (beams.length && beams[0].tokens.length >= this.cfg.maxGen) break;
    }

    const all = finished.concat(beams.map(b => ({ tokens: b.tokens, logp: b.logp })));
    if (all.length === 0) return [];
    // length-normalised score; alpha=0.7 keeps long answers from dominating
    const score = (c) => c.logp / Math.pow(c.tokens.length + 1, 0.7);
    all.sort((x, y) => score(y) - score(x));
    return all[0].tokens;
  };

  Model.prototype.respond = function (message, temperature) {
    return detokenize(this.sample(message, temperature).map(i => this.vocab[i]));
  };

  Model.prototype.replyGreedy = function (message) {
    return detokenize(this.greedy(message).map(i => this.vocab[i]));
  };

  Model.prototype.replyBeam = function (message, beamSize) {
    return detokenize(this.beam(message, beamSize).map(i => this.vocab[i]));
  };

  // Exact-match accuracy over raw examples ({src, tgt}).
  Model.prototype.accuracy = function (examples, opts) {
    opts = opts || {};
    const strategy = opts.strategy || 'greedy';
    const beamSize = opts.beamSize || 4;
    const max = opts.maxExamples || examples.length;
    const n = Math.min(examples.length, max);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const ex = examples[i];
      const src = encodeTokens(tokenize(ex.src), this.tokToId);
      if (src.length === 0) src.push(this.unkId);
      const got = strategy === 'beam' ? this.beam(src, beamSize) : this.greedy(src);
      const ref = encodeTokens(tokenize(ex.tgt), this.tokToId);
      if (seqEqual(got, ref)) correct++;
    }
    return n ? correct / n : 0;
  };

  Model.prototype.paramCount = function () {
    let n = 0;
    const p = this._allParams();
    for (let i = 0; i < p.length; i++) n += p[i][0].length;
    return n;
  };

  /* ============================================================
   * 7. .GenAI binary format (save / load)
   * ============================================================
   *   [0..5]   magic "GenAIx" (v2 writes "GenAI2")
   *   [6..9]   uint32 header length (LE)
   *   [10..]   UTF-8 JSON header (version, vocab, dims, tensor manifest)
   *   [pad]    zero padding to the next 8-byte boundary
   *   [...]    Float64 tensor data, in manifest order
   *
   * v1 files (single-layer, unidirectional, no coverage) still load: missing
   * tensors fall back to sensible defaults (W_q_gen becomes identity) so the
   * original model behaves exactly as before.
   */

  function modelTensors(m) {
    const t = {};
    t.E = m.E;
    for (let l = 0; l < m.layers; l++) {
      const c = m.encCells[l];
      t['enc_fwd_' + l + '_Wz'] = c.fwd.Wz; t['enc_fwd_' + l + '_Uz'] = c.fwd.Uz; t['enc_fwd_' + l + '_bz'] = c.fwd.bz;
      t['enc_fwd_' + l + '_Wr'] = c.fwd.Wr; t['enc_fwd_' + l + '_Ur'] = c.fwd.Ur; t['enc_fwd_' + l + '_br'] = c.fwd.br;
      t['enc_fwd_' + l + '_Wh'] = c.fwd.Wh; t['enc_fwd_' + l + '_Uh'] = c.fwd.Uh; t['enc_fwd_' + l + '_bh'] = c.fwd.bh;
      if (c.bwd) {
        t['enc_bwd_' + l + '_Wz'] = c.bwd.Wz; t['enc_bwd_' + l + '_Uz'] = c.bwd.Uz; t['enc_bwd_' + l + '_bz'] = c.bwd.bz;
        t['enc_bwd_' + l + '_Wr'] = c.bwd.Wr; t['enc_bwd_' + l + '_Ur'] = c.bwd.Ur; t['enc_bwd_' + l + '_br'] = c.bwd.br;
        t['enc_bwd_' + l + '_Wh'] = c.bwd.Wh; t['enc_bwd_' + l + '_Uh'] = c.bwd.Uh; t['enc_bwd_' + l + '_bh'] = c.bwd.bh;
      }
    }
    for (let l = 0; l < m.layers; l++) {
      const c = m.decCells[l];
      t['dec_' + l + '_Wz'] = c.Wz; t['dec_' + l + '_Uz'] = c.Uz; t['dec_' + l + '_bz'] = c.bz;
      t['dec_' + l + '_Wr'] = c.Wr; t['dec_' + l + '_Ur'] = c.Ur; t['dec_' + l + '_br'] = c.br;
      t['dec_' + l + '_Wh'] = c.Wh; t['dec_' + l + '_Uh'] = c.Uh; t['dec_' + l + '_bh'] = c.bh;
    }
    t.W_out = m.W_out; t.b_out = m.b_out;
    t.W_gen = m.W_gen; t.b_gen = m.b_gen;
    t.W_q_gen = m.W_q_gen;
    if (m.cfg.attnHeads) t.W_q_copy = m.W_q_copy;
    if (m.cfg.coverage) t.w_cov = m.w_cov;
    return t;
  }

  function padFor(offset) { return (8 - (offset % 8)) % 8; }

  function serializeBuffer(m) {
    if (!m) throw new Error('no model to save');
    const tensors = modelTensors(m);
    const names = Object.keys(tensors);
    let floatCount = 0;
    for (const n of names) floatCount += tensors[n].length;

    const header = {
      format: 'GenAI',
      version: 2,
      hidden: m.H,
      embed: m.D,
      layers: m.layers,
      bidir: m.bidir,
      attnHeads: !!m.cfg.attnHeads,
      coverage: !!m.cfg.coverage,
      attnScale: !!m.cfg.attnScale,
      maxGen: m.cfg.maxGen,
      vocab: m.vocab,
      specials: { pad: m.padId, bos: m.bosId, eos: m.eosId, unk: m.unkId },
      tensors: names.map((n) => ({ name: n, length: tensors[n].length }))
    };
    const headerBytes = new TextEncoder().encode(JSON.stringify(header));
    const pad = padFor(10 + headerBytes.length);
    const buf = new ArrayBuffer(10 + headerBytes.length + pad + floatCount * 8);
    const u8 = new Uint8Array(buf);
    const dv = new DataView(buf);
    u8.set([0x47, 0x65, 0x6e, 0x41, 0x49, 0x32], 0); // "GenAI2"
    dv.setUint32(6, headerBytes.length, true);
    u8.set(headerBytes, 10);
    let off = 10 + headerBytes.length + pad;
    for (const n of names) {
      const t = tensors[n];
      for (let i = 0; i < t.length; i++) { dv.setFloat64(off, t[i], true); off += 8; }
    }
    return buf;
  }

  function legacyTensorName(name) {
    if (name.indexOf('enc_') === 0) return 'enc_fwd_0_' + name.slice(4);
    if (name.indexOf('dec_') === 0) return 'dec_0_' + name.slice(4);
    return name;
  }

  function loadFromBuffer(input) {
    const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3], u8[4]);
    if (magic !== 'GenAI') throw new Error('not a .GenAI file (bad magic)');
    const hlen = dv.getUint32(6, true);
    const header = JSON.parse(new TextDecoder().decode(u8.subarray(10, 10 + hlen)));
    if (header.format !== 'GenAI') throw new Error('unsupported .GenAI format');

    const version = header.version || 1;
    let opts;
    if (version < 2) {
      opts = Object.assign({}, DEFAULTS, {
        hidden: header.hidden, embed: header.embed,
        maxGen: header.maxGen || DEFAULTS.maxGen, vocab: header.vocab,
        layers: 1, bidir: false, attnHeads: false, coverage: false, attnScale: false
      });
    } else {
      opts = Object.assign({}, DEFAULTS, {
        hidden: header.hidden, embed: header.embed,
        maxGen: header.maxGen || DEFAULTS.maxGen, vocab: header.vocab,
        layers: header.layers || DEFAULTS.layers,
        bidir: header.bidir !== undefined ? header.bidir : DEFAULTS.bidir,
        attnHeads: !!header.attnHeads,
        coverage: !!header.coverage,
        attnScale: header.attnScale !== undefined ? header.attnScale : true
      });
    }
    const m = new Model(opts);
    const tensors = modelTensors(m);
    const pad = padFor(10 + hlen);
    let off = 10 + hlen + pad;
    for (const spec of header.tensors) {
      const name = version < 2 ? legacyTensorName(spec.name) : spec.name;
      const t = tensors[name];
      if (!t) throw new Error('unknown tensor in file: ' + spec.name);
      const n = Math.min(spec.length, t.length);
      for (let i = 0; i < n; i++) { t[i] = dv.getFloat64(off, true); off += 8; }
    }
    if (version < 2) {
      // v1 used raw dot-product attention with no query projection.
      const H = m.H;
      m.W_q_gen.fill(0);
      for (let i = 0; i < Math.min(H, m.encOutDim); i++) m.W_q_gen[i * H + i] = 1;
    }
    if (header.specials) {
      m.padId = header.specials.pad; m.bosId = header.specials.bos;
      m.eosId = header.specials.eos; m.unkId = header.specials.unk;
    }
    m.stopped = true;
    m.steps = m.cfg.targetSteps;
    model = m;
    return status();
  }

  /* ============================================================
   * 8. Public API
   * ============================================================ */

  let model = null;

  function init(options) {
    const opts = Object.assign({}, DEFAULTS, options || {});
    model = new Model(opts);
    return status();
  }

  function train(budgetMs) {
    if (!model) init();
    const start = now();
    const deadline = start + Math.max(1, budgetMs || 10);
    let n = 0, lossSum = 0;
    do {
      const l = model.trainStep();
      lossSum += l; n++;
      if (model.stopped) break;
    } while (now() < deadline);

    const avg = n ? lossSum / n : 0;
    model.loss = model.loss === null ? avg : model.loss * 0.97 + avg * 0.03;
    return status();
  }

  function respond(message, temperature) {
    if (!model) init();
    return model.respond(message, temperature);
  }

  function setTarget(steps) {
    if (!model) init();
    model.cfg.targetSteps = Math.max(model.steps, steps);
    return status();
  }

  function accuracy(examples, opts) {
    if (!model) init();
    return model.accuracy(examples, opts);
  }

  function splits() {
    if (!model) init();
    const s = model.splitSets;
    return {
      train: s.train, inDist: s.inDist,
      compositional: s.compositional, systematic: s.systematic
    };
  }

  function status() {
    if (!model) return { ready: false };
    return {
      ready: model.stopped || model.steps >= model.cfg.targetSteps,
      stopped: model.stopped,
      steps: model.steps,
      targetSteps: model.cfg.targetSteps,
      loss: model.loss,
      valLoss: model.valLoss,
      bestValLoss: model.bestValLoss,
      bestLoss: model.bestValLoss,
      lr: model.lastLr,
      params: model.paramCount(),
      vocab: model.vocab.length,
      hidden: model.H,
      embed: model.D,
      layers: model.layers,
      bidir: model.bidir,
      batchSize: model.cfg.batchSize,
      coverage: !!model.cfg.coverage,
      trainExamples: model.trainData.length,
      valExamples: model.valData.length,
      examples: model.trainData.length + model.valData.length +
        model.compData.length + model.sysData.length
    };
  }

  function vocabulary() {
    return model ? model.vocab.slice() : [];
  }

  function save() { return serializeBuffer(model); }
  function load(input) { return loadFromBuffer(input); }
  function isLoaded() { return !!model; }

  return {
    init: init,
    train: train,
    respond: respond,
    greedy: function (m) { if (!model) init(); return model.replyGreedy(m); },
    beamSearch: function (m, b) { if (!model) init(); return model.replyBeam(m, b); },
    accuracy: accuracy,
    splits: splits,
    status: status,
    setTarget: setTarget,
    vocabulary: vocabulary,
    save: save,
    load: load,
    isLoaded: isLoaded,
    tokenize: tokenize,
    detokenize: detokenize,
    groupBySplit: groupBySplit,
    defaults: DEFAULTS,
    model: function () { return model; }
  };
});
