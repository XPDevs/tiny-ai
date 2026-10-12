/**
 * ============================================================================
 * TinyAI - src/transformer.js
 * ============================================================================
 *
 * THE TRANSFORMER BLOCK
 * ---------------------
 * A decoder-only Transformer is just the same block stacked N times. Each block
 * has two sub-layers:
 *
 *   1. Multi-head self-attention  (tokens exchange information)
 *   2. A feed-forward network     (each token is processed on its own)
 *
 * Both sub-layers are wrapped in a *residual connection*
 *
 *      output = x + sublayer(x)
 *
 * and preceded by *layer normalization*
 *
 *      output = x + sublayer( layerNorm(x) )
 *
 * This "pre-norm" arrangement (normalize before the sub-layer) is what modern
 * GPT models use because it trains reliably without extra tricks.
 *
 * WHY RESIDUALS? They give gradients a clean, direct path back through the
 * network, which is what makes deep networks trainable at all.
 *
 * WHY LAYER NORM? It re-centers and re-scales each token's vector to have mean
 * 0 and variance 1 before the sub-layer sees it, keeping activations in a
 * healthy range. It is computed independently for every token.
 * ============================================================================
 */

'use strict';

const _deps = typeof require === 'function' ? require('./tensor.js') : globalThis.TinyAI;
const { add, sub, mul, div, sqrt, meanLast, gelu, matmul } = _deps;
const _attn =
  typeof require === 'function' ? require('./attention.js') : globalThis.TinyAI;
const { selfAttention } = _attn;

/**
 * Layer normalization.
 *
 *   mean = average of x over the last dimension (the embedding dim)
 *   var  = average of (x - mean)^2
 *   out  = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * `gamma` (scale) and `beta` (shift) are learned parameters, one per feature.
 * `eps` is a tiny constant that prevents division by zero.
 *
 * @param {Tensor} x     shape [.., D]
 * @param {Tensor} gamma shape [D]
 * @param {Tensor} beta  shape [D]
 * @param {number} eps
 */
function layerNorm(x, gamma, beta, eps = 1e-5) {
  const mean = meanLast(x);            // [.., 1]
  const centered = sub(x, mean);       // broadcasting subtracts the mean
  const variance = meanLast(mul(centered, centered)); // [.., 1]
  const std = sqrt(add(variance, eps)); // [.., 1]
  const normalized = div(centered, std); // [.., D]
  return add(mul(normalized, gamma), beta); // scale and shift per feature
}

/**
 * The position-wise feed-forward network.
 *
 * It expands the embedding to 4x its size, applies the GELU non-linearity,
 * then projects back down. This is where most of a transformer's parameters
 * live, and where a lot of the "thinking" happens.
 *
 *   ffn(x) = GELU(x @ W1 + b1) @ W2 + b2
 */
function feedForward(x, p) {
  const hidden = gelu(add(matmul(x, p.fcw), p.fcb)); // [B, T, 4D]
  return add(matmul(hidden, p.fpw), p.fpb);          // [B, T, D]
}

/**
 * One complete Transformer block.
 *
 * @param {Tensor} x       input, shape [B, T, D]
 * @param {object} p       all parameters for this block
 * @param {object} config  { dModel, nHeads }
 * @param {Uint8Array} mask causal attention mask
 */
function transformerBlock(x, p, config, mask) {
  // Sub-layer 1: attention with a residual connection.
  const h1 = layerNorm(x, p.ln1g, p.ln1b);
  const a = selfAttention(h1, p, config, mask);
  const x1 = add(x, a);

  // Sub-layer 2: feed-forward with a residual connection.
  const h2 = layerNorm(x1, p.ln2g, p.ln2b);
  const f = feedForward(h2, p);
  return add(x1, f);
}

const TinyTransformer = { layerNorm, feedForward, transformerBlock };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyTransformer;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyTransformer);
}
