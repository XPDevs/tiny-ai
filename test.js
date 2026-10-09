const AI = require('./tiny-ai.js');
const L = require('./data/loader.js');
const all = L.load();

/* Quick smoke test. For the full evaluation suite use train.js. */
const STEPS = parseInt(process.env.STEPS || '1500', 10);

AI.init({ seed: 1337, targetSteps: STEPS, dataset: all });
const t0 = Date.now();
let last = 0;
while (!AI.status().ready) {
  const s = AI.train(1000);
  if (Date.now() - last > 1000) { last = Date.now(); console.log('  steps', s.steps, 'loss', s.loss && s.loss.toFixed(3)); }
}
const ms = Date.now() - t0;
const st = AI.status();
console.log('trained', st.steps, 'steps in', ms, 'ms | loss', st.loss.toFixed(3), '| val', st.valLoss == null ? '-' : st.valLoss.toFixed(3), '| params', st.params);

const prompts = [
  'hi', 'how are you', 'tell me a joke', 'bye',
  'book a table for 3 at 8pm for alice',
  'copy red green blue',
  'reverse cat dog fox',
  'sort 3 1 2',
  'add 4 5'
];
for (const p of prompts) {
  console.log('  ' + JSON.stringify(p).padEnd(40), '->', JSON.stringify(AI.greedy(p)));
}
