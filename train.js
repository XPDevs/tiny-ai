/**
 * ============================================================================
 * TinyAI - train.js
 * ============================================================================
 *
 * The training entry point. From the project root, run:
 *
 *     node train.js
 *     node train.js --config training/examples/tiny.json
 *     node train.js --steps 500 --dmodel 64 --layers 2 --tokenizer bpe
 *
 * It will:
 *   1. read a text dataset,
 *   2. build a tokenizer from that text,
 *   3. build the Transformer,
 *   4. train it with Adam + cross-entropy + backprop,
 *   5. save a `.GenAI` model you can chat with.
 *
 * Everything here is plain JavaScript - nothing is hidden in a library.
 * ============================================================================
 */

'use strict';

const fs = require('fs');
const path = require('path');

const { TinyModel } = require('./src/model.js');
const { createTokenizer } = require('./src/tokenizer.js');
const { Trainer } = require('./src/trainer.js');

/* --------------------------------------------------------------------------
 * A tiny command-line argument parser: --key value  or  --flag
 * ------------------------------------------------------------------------ */
function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
        args[key] = argv[++i];
      } else {
        args[key] = true;
      }
    }
  }
  return args;
}

function loadJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function printBanner() {
  console.log('');
  console.log('  ___________ _        _    ___ ');
  console.log(' |_   _|_ _\\ \\      / \\  |_ _|');
  console.log('   | |  | | \\ \\ /\\ / / _ \\  | | ');
  console.log('   | |  | |  \\ V  V / ___ \\ | | ');
  console.log('   |_| |___|  \\_/\\_/_/   \\_\\___|');
  console.log('   tiny transformer, written from scratch in JavaScript');
  console.log('   a TinyAI project by XPDevs');
  console.log('');
}

function main() {
  const args = parseArgs(process.argv.slice(2));

  // --- 1. Load the config file (default: training/examples/default.json) ---
  const configPath = args.config || 'training/examples/default.json';
  const config = fs.existsSync(configPath) ? loadJson(configPath) : {};
  const modelCfg = config.model || {};
  const trainCfg = config.training || {};

  // --- 2. Where is the data? ------------------------------------------------
  const dataPath = args.data || config.data || 'training/datasets/sample.txt';
  if (!fs.existsSync(dataPath)) {
    console.error('Dataset not found: ' + dataPath);
    process.exit(1);
  }
  const text = fs.readFileSync(dataPath, 'utf8');
  console.log('Dataset : ' + dataPath + '  (' + text.length + ' characters)');

  // --- 3. Build and train the tokenizer. ------------------------------------
  const tokType = args.tokenizer || (config.tokenizer && config.tokenizer.type) || 'char';
  const vocabTarget = parseInt(args.vocab || (config.tokenizer && config.tokenizer.vocabSize) || 512, 10);
  const tokenizer = createTokenizer(tokType, text, vocabTarget);
  console.log('Tokenizer: ' + tokType + '  (vocabulary of ' + tokenizer.vocabSize + ' tokens)');

  // --- 4. Build the model. --------------------------------------------------
  // Command-line flags override the config file.
  const modelConfig = {
    dModel: parseInt(args.dmodel || modelCfg.dModel || 128, 10),
    nHeads: parseInt(args.heads || modelCfg.nHeads || 4, 10),
    nLayers: parseInt(args.layers || modelCfg.nLayers || 4, 10),
    blockSize: parseInt(args.block || modelCfg.blockSize || 128, 10),
    tieWeights: modelCfg.tieWeights !== undefined ? modelCfg.tieWeights : true,
    vocabSize: tokenizer.vocabSize,
  };

  let model;
  const checkpointState = path.join(
    args.checkpointdir || trainCfg.checkpointDir || 'training/checkpoints',
    'checkpoint.json'
  );
  const checkpointModel = path.join(
    args.checkpointdir || trainCfg.checkpointDir || 'training/checkpoints',
    'checkpoint.genai'
  );

  if (args.resume && fs.existsSync(checkpointModel)) {
    console.log('Resuming from checkpoint: ' + checkpointModel);
    model = TinyModel.loadFile(checkpointModel);
  } else {
    model = new TinyModel(modelConfig, tokenizer);
  }

  console.log('Parameters: ' + model.parameterCount().toLocaleString());
  console.log('Model size (approx, on disk): ' + model.sizeMB().toFixed(2) + ' MB');
  console.log('');

  // --- 5. Set up the trainer. ----------------------------------------------
  const overrides = Object.assign({}, trainCfg);
  if (args.steps) overrides.maxSteps = parseInt(args.steps, 10);
  if (args.batch) overrides.batchSize = parseInt(args.batch, 10);
  if (args.block) overrides.blockSize = parseInt(args.block, 10);
  if (args.lr) overrides.lr = parseFloat(args.lr);
  if (args.checkpointdir) overrides.checkpointDir = args.checkpointdir;
  if (args.checkpointinterval) overrides.checkpointInterval = parseInt(args.checkpointinterval, 10);

  const trainer = new Trainer({ model, tokenizer, corpus: text, config: overrides });
  if (args.resume) trainer.resumeFromCheckpoint(checkpointState);

  // --- 6. Train! ------------------------------------------------------------
  const startTime = Date.now();
  trainer.train((s) => {
    let line =
      'step ' +
      String(s.step).padStart(6) +
      ' | loss ' +
      s.loss.toFixed(4) +
      (s.valLoss !== undefined && !Number.isNaN(s.valLoss) ? ' | val ' + s.valLoss.toFixed(4) : '') +
      ' | lr ' +
      s.lr.toExponential(2) +
      ' | ' +
      s.tokensPerSec.toFixed(0) +
      ' tok/s';
    console.log(line);

    // Every so often, print a sample so you can watch the model improve.
    if (s.step % (overrides.evalInterval || 100) === 0) {
      const sample = model.generateText('TinyAI is', {
        maxNewTokens: 30,
        temperature: 0.8,
        topK: 40,
        eosId: -1,
      });
      console.log('   sample> TinyAI is' + sample.replace(/\n/g, ' '));
    }
  });

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

  // --- 7. Save the final model. --------------------------------------------
  const outPath = args.out || config.out || 'models/tinyai.genai';
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  const bytes = model.saveFile(outPath);

  console.log('');
  console.log('Training finished in ' + elapsed + 's (' + trainer.tokensSeen.toLocaleString() + ' tokens seen).');
  console.log('Saved model to ' + outPath + ' (' + (bytes / (1024 * 1024)).toFixed(2) + ' MB).');
  console.log('');
  console.log('Chat with it:   node chat.js --model ' + outPath);
  console.log('');
}

printBanner();
main();
