/**
 * ============================================================================
 * TinyAI - src/optimizer.js
 * ============================================================================
 *
 * WHAT IS AN OPTIMIZER?
 * ---------------------
 * After `loss.backward()` every parameter has a gradient that says "which way
 * is uphill on the loss surface". The optimizer's job is to move each
 * parameter *downhill* by a small amount - that is how the model learns.
 *
 * The simplest rule is gradient descent:
 *
 *      w := w - learningRate * dLoss/dw
 *
 * But plain gradient descent is slow and unstable. TinyAI provides:
 *
 *   - SGDOptimizer  : plain (or momentum) gradient descent.
 *   - AdamOptimizer : the algorithm almost everyone actually uses. It keeps a
 *                     running average of the gradient (momentum) and of the
 *                     squared gradient, and uses them to take a
 *                     well-scaled step for every parameter individually.
 *
 *
 * ADAM, STEP BY STEP
 * ------------------
 * For each parameter w with gradient g and step number t:
 *
 *   m = beta1 * m + (1 - beta1) * g            (1st moment: mean of gradient)
 *   v = beta2 * v + (1 - beta2) * g^2          (2nd moment: mean of grad^2)
 *
 *   m_hat = m / (1 - beta1^t)                  (bias correction)
 *   v_hat = v / (1 - beta2^t)
 *
 *   w = w - lr * m_hat / (sqrt(v_hat) + eps)   (the update)
 *
 * The bias correction matters because m and v start at zero, which would
 * otherwise make the first few steps far too small.
 *
 * Defaults: lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8.
 * ============================================================================
 */

'use strict';

/* ==========================================================================
 * SGD (optionally with momentum)
 * ========================================================================== */
class SGDOptimizer {
  constructor(params, options = {}) {
    this.params = params;
    this.lr = options.lr !== undefined ? options.lr : 0.01;
    this.momentum = options.momentum !== undefined ? options.momentum : 0;
    this.weightDecay = options.weightDecay || 0;
    this.velocity = this.params.map((p) => new Float64Array(p.size));
  }

  step() {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      for (let k = 0; k < p.size; k++) {
        let g = p.grad[k];
        if (this.weightDecay) g += this.weightDecay * p.data[k];
        if (this.momentum) {
          this.velocity[i][k] = this.momentum * this.velocity[i][k] + g;
          g = this.velocity[i][k];
        }
        p.data[k] -= this.lr * g;
      }
    }
  }
}

/* ==========================================================================
 * Adam (and AdamW-style decoupled weight decay)
 * ========================================================================== */
class AdamOptimizer {
  constructor(params, options = {}) {
    this.params = params;
    this.lr = options.lr !== undefined ? options.lr : 3e-4;
    this.beta1 = options.beta1 !== undefined ? options.beta1 : 0.9;
    this.beta2 = options.beta2 !== undefined ? options.beta2 : 0.999;
    this.eps = options.eps !== undefined ? options.eps : 1e-8;
    this.weightDecay = options.weightDecay || 0;
    this.t = 0;

    // First and second moment estimates, one buffer per parameter.
    this.m = this.params.map((p) => new Float64Array(p.size));
    this.v = this.params.map((p) => new Float64Array(p.size));
  }

  step() {
    this.t += 1;
    // Bias-correction denominators (computed once per step).
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);

    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i];
      const v = this.v[i];
      for (let k = 0; k < p.size; k++) {
        let g = p.grad[k];

        // AdamW: decay weights toward zero, decoupled from the gradient.
        if (this.weightDecay) p.data[k] -= this.lr * this.weightDecay * p.data[k];

        m[k] = this.beta1 * m[k] + (1 - this.beta1) * g;
        v[k] = this.beta2 * v[k] + (1 - this.beta2) * g * g;

        const mHat = m[k] / bc1;
        const vHat = v[k] / bc2;
        p.data[k] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }

  /** Reset the moment estimates (useful when resuming training). */
  reset() {
    this.t = 0;
    for (const buf of this.m) buf.fill(0);
    for (const buf of this.v) buf.fill(0);
  }

  /** Serialize optimizer state (used by checkpoints). */
  state() {
    return {
      t: this.t,
      lr: this.lr,
      beta1: this.beta1,
      beta2: this.beta2,
      eps: this.eps,
      weightDecay: this.weightDecay,
    };
  }
}

const TinyOptimizer = { SGDOptimizer, AdamOptimizer };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyOptimizer;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyOptimizer);
}
