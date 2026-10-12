/**
 * ============================================================================
 * TinyAI - src/trainer.js
 * ============================================================================
 *
 * THE TRAINING LOOP
 * -----------------
 * Training is a simple idea repeated many times:
 *
 *   1. Show the model a batch of examples (here: chunks of text).
 *   2. Ask it to predict the next token after every position.
 *   3. Measure how wrong it was (cross-entropy loss).
 *   4. Compute the gradient of that loss w.r.t. every parameter (backprop).
 *   5. Nudge every parameter in the downhill direction (the optimizer).
 *   6. Repeat. The loss should fall.
 *
 * NEXT-TOKEN PREDICTION
 * ---------------------
 * We feed the model a sequence of tokens and ask it to predict the token that
 * comes next at *every* position:
 *
 *   input :  the  cat  sat  on  the
 *   target:  cat  sat  on   the  mat
 *
 * The loss is the average cross-entropy over all positions in the batch. This
 * single, self-supervised objective is what large language models are trained
 * with too.
 *
 * This file also handles:
 *   - splitting data into training and validation sets
 *   - gradient clipping (keeps training stable)
 *   - a learning-rate warmup + cosine decay schedule
 *   - validation loss measurement
 *   - checkpoints you can resume from
 * ============================================================================
 */

'use strict';

const _deps = typeof require === 'function' ? require('./tensor.js') : globalThis.TinyAI;
const { fromIds, tensor, reshape, crossEntropy, noGrad } = _deps;

const _opt = typeof require === 'function' ? require('./optimizer.js') : globalThis.TinyAI;
const { AdamOptimizer } = _opt;

/* Node-only filesystem access, resolved lazily so the file still parses in a
 * browser (the browser simply never constructs a Trainer). */
let fs = null;
let path = null;
if (typeof require === 'function') {
  fs = require('fs');
  path = require('path');
}

const DEFAULT_TRAIN_CONFIG = {
  batchSize: 8,
  blockSize: 128,
  lr: 3e-4,
  minLr: 3e-5,
  weightDecay: 0.01,
  beta1: 0.9,
  beta2: 0.999,
  gradClip: 1.0,
  warmup: 100,
  maxSteps: 1000,
  logEvery: 10,
  evalInterval: 100,
  evalBatches: 10,
  checkpointInterval: 200,
  checkpointDir: 'training/checkpoints',
  valSplit: 0.1,
  seed: 1234,
};

class Trainer {
  /**
   * @param {object} args
   *   model      TinyModel
   *   tokenizer  tokenizer (for decoding samples while training)
   *   corpus     string OR array of token ids
   *   config     partial training config
   */
  constructor({ model, tokenizer, corpus, config = {} }) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.config = Object.assign({}, DEFAULT_TRAIN_CONFIG, config);
    this.config.blockSize = Math.min(this.config.blockSize, model.config.blockSize);

    // Turn the corpus into a single flat array of ids.
    const ids = typeof corpus === 'string' ? tokenizer.encode(corpus) : corpus.slice();
    if (ids.length < this.config.blockSize + 2) {
      throw new Error('Corpus is too small to train on.');
    }

    // Deterministic train / validation split.
    const nVal = Math.max(this.config.blockSize + 2, Math.floor(ids.length * this.config.valSplit));
    this.valIds = ids.slice(ids.length - nVal);
    this.trainIds = ids.slice(0, ids.length - nVal);

    // Optimizer over the model's parameters.
    this.optimizer = new AdamOptimizer(model.parameterTensors(), {
      lr: this.config.lr,
      beta1: this.config.beta1,
      beta2: this.config.beta2,
      weightDecay: this.config.weightDecay,
    });

    this.step = 0;
    this.tokensSeen = 0;
    this.history = []; // { step, loss, valLoss? }
  }

  /* ----------------------------------------------------------------------
   * Data batching
   * -------------------------------------------------------------------- */
  getBatch() {
    const B = this.config.batchSize;
    const T = this.config.blockSize;
    const maxStart = this.trainIds.length - T - 1;
    const inputs = [];
    const targets = [];
    for (let b = 0; b < B; b++) {
      const start = Math.floor(Math.random() * maxStart);
      const window = this.trainIds.slice(start, start + T + 1);
      inputs.push(window.slice(0, T));
      targets.push(window.slice(1, T + 1));
    }
    return { inputs, targets };
  }

  /* ----------------------------------------------------------------------
   * Learning-rate schedule: warmup then cosine decay.
   * -------------------------------------------------------------------- */
  currentLr() {
    const c = this.config;
    if (this.step < c.warmup) {
      return (c.lr * (this.step + 1)) / c.warmup;
    }
    const progress = (this.step - c.warmup) / Math.max(1, c.maxSteps - c.warmup);
    const cosine = 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)));
    return c.minLr + (c.lr - c.minLr) * cosine;
  }

  /* ----------------------------------------------------------------------
   * A single optimization step
   * -------------------------------------------------------------------- */
  trainStep() {
    const { inputs, targets } = this.getBatch();
    const B = inputs.length;
    const T = inputs[0].length;
    const V = this.model.config.vocabSize;

    // 1. Forward: [B, T] ids -> [B, T, V] logits.
    const logits = this.model.forward(fromIds(inputs));

    // 2. Flatten [B, T, V] -> [B*T, V] and the targets -> [B*T].
    const flatLogits = reshape(logits, [B * T, V]);
    const targetTensor = tensor(targets.flat(), false, [B * T]);

    // 3. Loss + backward.
    const loss = crossEntropy(flatLogits, targetTensor);
    loss.backward();

    // 4. Gradient clipping (global L2 norm).
    if (this.config.gradClip > 0) this._clipGradients(this.config.gradClip);

    // 5. Optimizer step (with the scheduled learning rate).
    this.optimizer.lr = this.currentLr();
    this.optimizer.step();

    this.step += 1;
    this.tokensSeen += B * T;
    return loss.data[0];
  }

  /** Scale all gradients so their combined L2 norm never exceeds `maxNorm`. */
  _clipGradients(maxNorm) {
    const params = this.model.parameterTensors();
    let norm = 0;
    for (const p of params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.size; i++) norm += p.grad[i] * p.grad[i];
    }
    norm = Math.sqrt(norm);
    if (norm > maxNorm) {
      const scale = maxNorm / (norm + 1e-6);
      for (const p of params) {
        if (!p.grad) continue;
        for (let i = 0; i < p.size; i++) p.grad[i] *= scale;
      }
    }
    return norm;
  }

  /* ----------------------------------------------------------------------
   * Validation
   * -------------------------------------------------------------------- */
  evaluate(batches) {
    const n = batches || this.config.evalBatches;
    const B = this.config.batchSize;
    const T = this.config.blockSize;
    const V = this.model.config.vocabSize;
    const maxStart = this.valIds.length - T - 1;
    if (maxStart <= 0) return NaN;

    let total = 0;
    noGrad(() => {
      for (let i = 0; i < n; i++) {
        const inputs = [];
        const targets = [];
        for (let b = 0; b < B; b++) {
          const start = Math.floor(Math.random() * maxStart);
          const window = this.valIds.slice(start, start + T + 1);
          inputs.push(window.slice(0, T));
          targets.push(window.slice(1, T + 1));
        }
        const logits = this.model.forward(fromIds(inputs));
        const flatLogits = reshape(logits, [B * T, V]);
        const targetTensor = tensor(targets.flat(), false, [B * T]);
        const loss = crossEntropy(flatLogits, targetTensor);
        total += loss.data[0];
      }
    });
    return total / n;
  }

  /* ----------------------------------------------------------------------
   * The main loop
   * -------------------------------------------------------------------- */
  /**
   * @param {function} [onLog]  called with { step, loss, valLoss, lr, tokens,
   *                            tokensPerSec, elapsedMs } every logEvery steps.
   */
  train(onLog) {
    const startTime = Date.now();
    const startStep = this.step;
    const startTokens = this.tokensSeen;
    let runningLoss = 0;
    let runningCount = 0;
    let lastLogTime = startTime;

    for (; this.step < this.config.maxSteps; ) {
      const target = this.step;
      const loss = this.trainStep();
      runningLoss += loss;
      runningCount += 1;

      const atLog = this.step % this.config.logEvery === 0 || this.step === this.config.maxSteps;
      let valLoss = undefined;

      if (atLog && this.config.evalInterval > 0 && this.step % this.config.evalInterval === 0) {
        valLoss = this.evaluate();
      }

      if (this.config.checkpointInterval > 0 && this.step % this.config.checkpointInterval === 0) {
        this.saveCheckpoint();
      }

      if (atLog) {
        const now = Date.now();
        const avgLoss = runningLoss / runningCount;
        const tokens = this.tokensSeen - startTokens;
        const seconds = (now - startTime) / 1000;
        const stats = {
          step: this.step,
          loss: avgLoss,
          valLoss,
          lr: this.currentLr(),
          tokens,
          tokensPerSec: seconds > 0 ? tokens / seconds : 0,
          elapsedMs: now - startTime,
        };
        this.history.push({ step: this.step, loss: avgLoss, valLoss });
        if (onLog) onLog(stats);
        runningLoss = 0;
        runningCount = 0;
        lastLogTime = now;
      }

      if (this.step <= target) break; // safety: never spin forever
    }
    return this.history;
  }

  /* ----------------------------------------------------------------------
   * Checkpoints
   * -------------------------------------------------------------------- */
  saveCheckpoint(dir) {
    if (!fs) return;
    const outDir = dir || this.config.checkpointDir;
    fs.mkdirSync(outDir, { recursive: true });
    const modelPath = path.join(outDir, 'checkpoint.genai');
    const statePath = path.join(outDir, 'checkpoint.json');
    this.model.saveFile(modelPath);
    fs.writeFileSync(
      statePath,
      JSON.stringify(
        {
          step: this.step,
          tokensSeen: this.tokensSeen,
          trainConfig: this.config,
          optimizer: this.optimizer.state(),
          history: this.history.slice(-200),
        },
        null,
        2
      )
    );
    return { modelPath, statePath };
  }

  /**
   * Restore step count and optimizer state from a checkpoint directory.
   * The model weights themselves are loaded by the caller with loadModel().
   */
  resumeFromCheckpoint(statePath) {
    if (!fs) throw new Error('resumeFromCheckpoint() is Node-only');
    if (!fs.existsSync(statePath)) return false;
    const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    this.step = state.step || 0;
    this.tokensSeen = state.tokensSeen || 0;
    this.history = state.history || [];
    // The Adam moments are not stored (to keep checkpoints small); the reset
    // is harmless because Adam adapts within a few dozen steps.
    this.optimizer.t = 0;
    return true;
  }
}

const TinyTrainer = {
  Trainer,
  DEFAULT_TRAIN_CONFIG,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TinyTrainer;
}
if (typeof globalThis !== 'undefined') {
  globalThis.TinyAI = globalThis.TinyAI || {};
  Object.assign(globalThis.TinyAI, TinyTrainer);
}
