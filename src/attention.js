/**
 * ============================================================================
 * TinyAI - src/attention.js
 * ============================================================================
 *
 * SELF-ATTENTION - THE HEART OF A TRANSFORMER
 * -------------------------------------------
 * Attention lets every token in a sequence look at every other token and
 * decide "how much should I care about you?" when building its own meaning.
 *
 * Think of the sentence: "The animal did not cross the street because it was
 * too tired." To understand "it", the model must attend back to "animal".
 * Attention is the mechanism that learns to do this.
 *
 * THE MECHANICS (Scaled Dot-Product Attention)
 * --------------------------------------------
 * Each token produces three vectors by multiplying its input with learned
 * weight matrices:
 *
 *    Query (Q): "what am I looking for?"
 *    Key   (K): "what do I contain?"
 *    Value (V): "what do I actually offer?"
 *
 * For every token i we compare its query with every key j using a dot product.
 * A large dot product means "query i and key j are a good match". Dividing by
 * sqrt(d_k) keeps the numbers from growing with the head size (which would
 * push softmax into a region with tiny gradients). Then softmax turns the
 * scores into a probability distribution, and we take a weighted average of
 * the value vectors:
 *
 *      Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
 *
 * MULTI-HEAD ATTENTION
 * --------------------
 * Instead of doing this once with full-size vectors, we split the embedding
 * into `H` independent "heads" (each of size d_k = D / H) and run attention in
 * every head in parallel. Different heads can learn to track different kinds
 * of relationships (syntax, position, coreference, ...). The head outputs are
 * concatenated and projected back to size D.
 *
 * CAUSAL MASKING
 * --------------
 * This is a *decoder-only* model: it predicts the next token, so it must not
 * be allowed to peek at future tokens. We enforce this with a mask that sets
 * every "future" attention score to -infinity before the softmax, so those
 * positions get probability zero.
 * ============================================================================
 */

'use strict';

const _deps = typeof require === 'function' ? require('./tensor.js') : globalThis.TinyAI;
const { add, matmul, reshape, transpose, softmax, mul } = _deps;

/**
 * Multi-head causal self-attention.
 *
 * @param {Tensor} x       input activations, shape [B, T, D]
 * @param {object} p       parameter block: wq,bq,wk,bk,wv,bv,wo,bo
 * @param {object} config  { dModel, nHeads }
 * @param {Uint8Array} mask flat [T, T] causal mask (1 = blocked)
 * @returns {Tensor}       output, shape [B, T, D]
 */
function selfAttention(x, p, config, mask) {
  const B = x.shape[0];
  const T = x.shape[1];
  const D = config.dModel;
  const H = config.nHeads;
  const dh = D / H; // size of each head

  // --- 1. Project the input into queries, keys and values. -----------------
  // Each is a linear layer:  out = x @ W + b
  const q = add(matmul(x, p.wq), p.bq); // [B, T, D]
  const k = add(matmul(x, p.wk), p.bk); // [B, T, D]
  const v = add(matmul(x, p.wv), p.bv); // [B, T, D]

  // --- 2. Split the D dimension into H heads of size dh. -------------------
  // [B, T, D] -> [B, T, H, dh] -> [B, H, T, dh]
  const qh = transpose(reshape(q, [B, T, H, dh]), 1, 2);
  const kh = transpose(reshape(k, [B, T, H, dh]), 1, 2);
  const vh = transpose(reshape(v, [B, T, H, dh]), 1, 2);

  // --- 3. Scale the queries by 1/sqrt(dh). ---------------------------------
  const scale = 1 / Math.sqrt(dh);
  const qs = mul(qh, scale);

  // --- 4. Scores = Q K^T. ---------------------------------------------------
  // [B, H, T, dh] @ [B, H, dh, T] -> [B, H, T, T]
  const khT = transpose(kh, 2, 3);
  const scores = matmul(qs, khT);

  // --- 5. Softmax over the last axis, with the causal mask. ----------------
  const attn = softmax(scores, mask); // [B, H, T, T]

  // --- 6. Context = attention weights @ values. ----------------------------
  const ctx = matmul(attn, vh); // [B, H, T, dh]

  // --- 7. Merge the heads back together. -----------------------------------
  // [B, H, T, dh] -> [B, T, H, dh] -> [B, T, D]
  const ctxT = transpose(ctx, 1, 2);
  const merged = reshape(ctxT, [B, T, D]);

  // --- 8. Final output projection. -----------------------------------------
  return add(matmul(merged, p.wo), p.bo); // [B, T, D]
}

const TinyAttention = { selfAttention };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyAttention;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyAttention);
}
