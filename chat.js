#!/usr/bin/env node
/*
 * chat.js — talk to a trained .GenAI model in the terminal
 * ---------------------------------------------------------------------------
 *   ./chat.sh [model.GenAI]      (default model.GenAI)
 *   TEMP=0.6 ./chat.sh           (sampling temperature, default 0.3)
 *
 * Loads the model once and reads lines from stdin, printing one reply per line.
 * Type "exit" (or Ctrl-D) to quit.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const AI = require('./tiny-ai.js');

const file = process.argv[2] || path.join(__dirname, 'model.GenAI');
const temp = process.env.TEMP !== undefined ? parseFloat(process.env.TEMP) : 0.3;

if (!fs.existsSync(file)) {
  console.error('no model at ' + file + '\nrun ./train.sh first (or pass a path: ./chat.sh my.GenAI)');
  process.exit(1);
}

let st;
try {
  st = AI.load(fs.readFileSync(file));
} catch (err) {
  console.error('failed to load ' + file + ': ' + err.message);
  process.exit(1);
}

console.log('tiny-ai chat');
console.log('  model  ' + file);
console.log('  vocab  ' + st.vocab + ' words · ' + st.params + ' parameters' +
  ' · hidden ' + st.hidden + ' · embed ' + st.embed + ' · ' + st.layers + ' layers' +
  (st.bidir ? ' bidir' : '') + (st.coverage ? ' +coverage' : ''));
console.log('  temp   ' + temp + '   (set TEMP=0.6 for a different temperature)');
console.log('  type "exit" to quit, or try "what is your name"\n');

const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: 'you> ' });
rl.prompt();

rl.on('line', function (line) {
  const text = line.trim();
  if (!text) { rl.prompt(); return; }
  if (text === 'exit' || text === 'quit') { rl.close(); return; }
  let reply;
  try { reply = AI.respond(text, temp); }
  catch (err) { reply = '(error: ' + err.message + ')'; }
  if (!reply || !reply.length) reply = '(no answer — try rephrasing)';
  console.log('tiny-ai> ' + reply);
  rl.prompt();
});

rl.on('close', function () {
  console.log('\nbye.');
  process.exit(0);
});
