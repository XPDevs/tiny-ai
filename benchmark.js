/**
 * ============================================================================
 * TinyAI - benchmark.js
 * ============================================================================
 *
 * Measures how fast TinyAI runs, so you can see the cost of every operation
 * and experiment with the trade-offs between model size and speed.
 *
 *     node benchmark.js
 *     node benchmark.js --model models/tinyai.genai
 *     node benchmark.js --dmodel 64 --layers 2 --block 64
 *
 * It reports:
 *   - parameter count and on-disk size
 *   - forward-pass latency
 *   - training-step latency
 *   - text generation speed (tokens/second)
 * ============================================================================
 */

'use strict';

const fs = require('fs');
const { TinyModel } = require('./src/model.js');
const { createTokenizer } = require('./src/tokenizer.js');
const { Trainer } = require('./src/trainer.js');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) args[key] = argv[++i];
      else args[key] = true;
    }
  }
  return args;
}

function timeit(label, iterations, fn) {
  // Warm-up.
  fn();
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) fn();
  const end = process.hrtime.bigint();
  const ms = Number(end - start) / 1e6 / iterations;
  return { label, ms, perSec: 1000 / ms };
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  let model;
  let tokenizer;

  if (args.model && fs.existsSync(args.model)) {
    model = TinyModel.loadFile(args.model);
    tokenizer = model.tokenizer;
    console.log('Loaded model: ' + args.model);
  } else {
    // Build an untrained model of the requested size.
    const sample = fs.existsSync('training/datasets/sample.txt')
      ? fs.readFileSync('training/datasets/sample.txt', 'utf8')
      : 'the quick brown fox jumps over the lazy dog. ';
    tokenizer = createTokenizer('char', sample);
    model = new TinyModel(
      {
        dModel: parseInt(args.dmodel || 128, 10),
        nHeads: parseInt(args.heads || 4, 10),
        nLayers: parseInt(args.layers || 4, 10),
        blockSize: parseInt(args.block || 128, 10),
        tieWeights: true,
        vocabSize: tokenizer.vocabSize,
      },
      tokenizer
    );
    console.log('Built a fresh model (no --model given).');
  }

  const T = Math.min(model.config.blockSize, parseInt(args.block || model.config.blockSize, 10));
  const V = model.config.vocabSize;

  console.log('');
  console.log('TinyAI benchmark, a project by XPDevs');
  console.log('');
  console.log('=== Configuration ===');
  console.log('  architecture : transformer-decoder');
  console.log('  layers       : ' + model.config.nLayers);
  console.log('  heads        : ' + model.config.nHeads);
  console.log('  dModel       : ' + model.config.dModel);
  console.log('  blockSize    : ' + model.config.blockSize);
  console.log('  vocabSize    : ' + V);
  console.log('  parameters   : ' + model.parameterCount().toLocaleString());
  console.log('  size on disk : ' + model.sizeMB().toFixed(2) + ' MB');
  console.log('');

  // --- Forward pass ---------------------------------------------------------
  const ids2d = [Array.from({ length: T }, () => Math.floor(Math.random() * V))];
  const fwd = timeit('forward', 20, () => model.forwardIds(ids2d));
  console.log('=== Speed ===');
  console.log('  forward [1 x ' + T + ']      : ' + fwd.ms.toFixed(2) + ' ms  (' + fwd.perSec.toFixed(0) + '/s)');

  // --- One training step ----------------------------------------------------
  const corpus = 'the quick brown fox jumps over the lazy dog. '.repeat(200);
  const trainer = new Trainer({
    model,
    tokenizer,
    corpus,
    config: { batchSize: 4, blockSize: Math.min(T, 64), maxSteps: 1000, evalInterval: 0, checkpointInterval: 0 },
  });
  const train = timeit('trainStep', 5, () => trainer.trainStep());
  console.log('  train step (batch 4)   : ' + train.ms.toFixed(2) + ' ms  (' + train.perSec.toFixed(2) + '/s)');

  // --- Generation -----------------------------------------------------------
  const genStart = process.hrtime.bigint();
  const produced = model.generate(tokenizer.encode('the '), {
    maxNewTokens: 40,
    temperature: 0.8,
    topK: 40,
    eosId: -1,
  });
  const genEnd = process.hrtime.bigint();
  const genMs = Number(genEnd - genStart) / 1e6;
  console.log(
    '  generation             : ' +
      genMs.toFixed(1) +
      ' ms for ' +
      produced.length +
      ' tokens  (' +
      (produced.length / (genMs / 1000)).toFixed(1) +
      ' tok/s)'
  );
  console.log('');
}

main();
