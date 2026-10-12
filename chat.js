/**
 * ============================================================================
 * TinyAI - chat.js
 * ============================================================================
 *
 * A terminal chatbot. Run:
 *
 *     node chat.js
 *     node chat.js --model models/tinyai.genai --temperature 0.8
 *
 * How a reply is produced:
 *   1. The user's message is checked against the knowledge plugins. If a
 *      strong match is found, that answer is returned immediately (no model
 *      call needed).
 *   2. Otherwise the Transformer generates a continuation, streaming tokens to
 *      the screen as they are produced.
 *
 * Type /help inside the chat to see the commands.
 * ============================================================================
 */

'use strict';

const fs = require('fs');
const readline = require('readline');
const { TinyModel } = require('./src/model.js');
const { KnowledgeBase } = require('./src/knowledge.js');

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

function main() {
  const args = parseArgs(process.argv.slice(2));
  const modelPath = args.model || 'models/tinyai.genai';
  const knowledgeDir = args.knowledge || 'knowledge';

  // --- Load the model -------------------------------------------------------
  if (!fs.existsSync(modelPath)) {
    console.error('No model found at "' + modelPath + '".');
    console.error('Train one first:  node train.js');
    process.exit(1);
  }
  const model = TinyModel.loadFile(modelPath);
  const tokenizer = model.tokenizer;

  // --- Load the knowledge plugins ------------------------------------------
  const knowledge = KnowledgeBase.loadFromDir(knowledgeDir);

  console.log('');
  console.log('TinyAI chat, a project by XPDevs  (model: ' + modelPath + ')');
  console.log(
    'Parameters: ' +
      model.parameterCount().toLocaleString() +
      ' | vocab: ' +
      model.config.vocabSize +
      ' | context: ' +
      model.config.blockSize
  );
  console.log('Knowledge entries loaded: ' + knowledge.size + ' (from ' + knowledgeDir + '/)');
  console.log("Type /help for commands, /quit to exit.");
  console.log('');

  // --- Runtime options (can be changed with slash-commands) -----------------
  const options = {
    temperature: args.temperature !== undefined ? parseFloat(args.temperature) : 0.8,
    topK: args.topk !== undefined ? parseInt(args.topk, 10) : 40,
    topP: args.topp !== undefined ? parseFloat(args.topp) : 0.95,
    maxNewTokens: args.maxtokens !== undefined ? parseInt(args.maxtokens, 10) : 60,
    repetitionPenalty: 1.2,
    eosId: 3,
  };

  // Conversation context, as a rolling list of token ids.
  let contextIds = [];
  const maxContext = model.config.blockSize;

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: 'You: ',
  });

  function handleCommand(line) {
    const parts = line.trim().split(/\s+/);
    const cmd = parts[0].toLowerCase();
    switch (cmd) {
      case '/quit':
      case '/exit':
        console.log('Bye!');
        process.exit(0);
        break;
      case '/help':
        console.log('Commands:');
        console.log('  /reset            clear the conversation context');
        console.log('  /temp <value>     set temperature (e.g. /temp 0.7)');
        console.log('  /topk <n>         set top-k sampling');
        console.log('  /topp <value>     set top-p (nucleus) sampling');
        console.log('  /tokens <n>       max new tokens per reply');
        console.log('  /settings         show current settings');
        console.log('  /quit             exit');
        break;
      case '/reset':
        contextIds = [];
        console.log('(context cleared)');
        break;
      case '/temp':
        options.temperature = parseFloat(parts[1]);
        console.log('temperature = ' + options.temperature);
        break;
      case '/topk':
        options.topK = parseInt(parts[1], 10);
        console.log('topK = ' + options.topK);
        break;
      case '/topp':
        options.topP = parseFloat(parts[1]);
        console.log('topP = ' + options.topP);
        break;
      case '/tokens':
        options.maxNewTokens = parseInt(parts[1], 10);
        console.log('maxNewTokens = ' + options.maxNewTokens);
        break;
      case '/settings':
        console.log(JSON.stringify(options, null, 2));
        break;
      default:
        console.log('Unknown command. Type /help.');
    }
  }

  function respond(userText) {
    // 1. Try the knowledge plugins first.
    const hit = knowledge.search(userText, 4);
    if (hit) {
      process.stdout.write('TinyAI: ');
      process.stdout.write(hit.answer + '\n\n');
      // Feed the exchange into the context so the model "remembers" it.
      contextIds = contextIds.concat(tokenizer.encode(userText), tokenizer.encode(hit.answer));
      return;
    }

    // 2. Otherwise, generate with the model.
    contextIds = contextIds.concat(tokenizer.encode(userText));
    if (contextIds.length > maxContext - 8) {
      contextIds = contextIds.slice(contextIds.length - (maxContext - 8));
    }

    process.stdout.write('TinyAI: ');
    let wrote = false;
    const produced = model.generate(contextIds, {
      maxNewTokens: options.maxNewTokens,
      temperature: options.temperature,
      topK: options.topK,
      topP: options.topP,
      repetitionPenalty: options.repetitionPenalty,
      eosId: options.eosId,
      onToken: (id) => {
        process.stdout.write(tokenizer.decode([id]));
        wrote = true;
      },
    });
    if (!wrote) process.stdout.write('(no output)');
    process.stdout.write('\n\n');

    // Remember what we said for the next turn.
    contextIds = contextIds.concat(produced);
  }

  rl.prompt();
  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (trimmed === '') {
      rl.prompt();
      return;
    }
    if (trimmed.startsWith('/')) {
      handleCommand(trimmed);
    } else {
      respond(trimmed);
    }
    rl.prompt();
  });
  rl.on('close', () => {
    console.log('');
    process.exit(0);
  });
}

main();
