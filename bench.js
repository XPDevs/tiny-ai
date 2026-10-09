const AI = require('./tiny-ai.js');
const L = require('./data/loader.js');
const all = L.load();

/* Throughput / accuracy benchmark. Env: STEPS, H (hidden), BATCH, BIDIR=0, HEADS=0, COV=0. */
const STEPS = parseInt(process.env.STEPS || '4000', 10);
const H = parseInt(process.env.H || '64', 10);
const BATCH = parseInt(process.env.BATCH || '6', 10);

AI.init({
  seed: 1337,
  hidden: H,
  targetSteps: STEPS,
  batchSize: BATCH,
  bidir: process.env.BIDIR !== '0',
  attnHeads: process.env.HEADS !== '0',
  coverage: process.env.COV !== '0',
  dataset: all
});

const t0 = Date.now();
let st = AI.status();
let last = 0;
while (!st.ready) {
  st = AI.train(1000);
  if (Date.now() - last > 1000) {
    last = Date.now();
    console.log('steps', st.steps, 'loss', (st.loss || 0).toFixed(3), 'val', st.valLoss == null ? '-' : st.valLoss.toFixed(3), 'ms', Date.now() - t0);
  }
}
const ms = Date.now() - t0;
console.log('TRAIN DONE', ms, 'ms |', (st.steps / (ms / 1000)).toFixed(1), 'steps/s | bestVal', st.bestValLoss && st.bestValLoss.toFixed(4), '| params', st.params);

const S = AI.splits();
for (const [name, set] of [['in-dist', S.inDist], ['compositional', S.compositional], ['systematic', S.systematic]]) {
  const g = AI.accuracy(set, { strategy: 'greedy', maxExamples: 200 });
  const b = AI.accuracy(set, { strategy: 'beam', beamSize: 4, maxExamples: 150 });
  console.log('  ' + name.padEnd(14) + 'n=' + String(set.length).padEnd(4) + ' greedy ' + (100 * g).toFixed(1) + '%  beam ' + (100 * b).toFixed(1) + '%');
}
