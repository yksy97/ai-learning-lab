// Transformer の勾配チェックと、自己回帰モデルの学習テスト
import assert from 'node:assert/strict';
import { Transformer, softmaxRows } from '../public/js/transformer.js';
import { mulberry32 } from '../public/js/nn.js';
import { ARModel, V, BINS } from '../public/js/ar.js';

// 1) 数値微分による勾配チェック（全パラメータ）
{
  const r = mulberry32(5);
  const net = new Transformer({ vocab: 5, seqLen: 4, d: 8, heads: 2, layers: 2, ff: 12 }, r);
  // LayerNorm のゲインやバイアスも非自明な値にしておく
  for (const p of net.params) for (let i = 0; i < p.w.length; i++) p.w[i] += 0.1 * (r() - 0.5);
  const B = 3, n = B * 4;
  const inp = Int32Array.from({ length: n }, (_, i) => (i % 4 === 0 ? 5 : Math.floor(r() * 5)));
  const tgt = Int32Array.from({ length: n }, () => Math.floor(r() * 5));
  const lossOf = () => {
    const P = softmaxRows(net.forward(inp, B), n, 5);
    let L = 0;
    for (let i = 0; i < n; i++) L -= Math.log(P[i * 5 + tgt[i]]);
    return L / n;
  };
  net.zeroGrad();
  const P = softmaxRows(net.forward(inp, B), n, 5);
  const d = new Float64Array(n * 5);
  for (let i = 0; i < n; i++) for (let j = 0; j < 5; j++) d[i * 5 + j] = (P[i * 5 + j] - (j === tgt[i] ? 1 : 0)) / n;
  net.backward(d);
  const eps = 1e-6;
  let maxErr = 0, checked = 0;
  for (const p of net.params) {
    for (let i = 0; i < p.w.length; i++) {
      const w = p.w[i];
      p.w[i] = w + eps; const a = lossOf();
      p.w[i] = w - eps; const b = lossOf();
      p.w[i] = w;
      const num = (a - b) / (2 * eps);
      maxErr = Math.max(maxErr, Math.abs(num - p.g[i]) / Math.max(1, Math.abs(num)));
      checked++;
    }
  }
  console.log(`Transformer 勾配チェック: ${checked} パラメータ, 最大誤差 ${maxErr.toExponential(2)}`);
  assert.ok(maxErr < 1e-6);
}

// 2) 確率表の整合性：全マスの確率の和が 1
{
  const m = new ARModel({ seed: 3 });
  const { density } = m.computeTables();
  const s = density.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(s - 1) < 1e-9, `確率の和 ${s}`);
  // 表から求めた確率と、直接の順伝播から求めた確率が一致するか
  const tk = [2, 5, 1, 7];
  const { probs } = m.inspect(tk);
  const direct = probs.reduce((acc, p, k) => acc * p[tk[k]], 1);
  const [i1, i2] = m.binsOf(tk);
  assert.ok(Math.abs(direct - density[i1 * BINS + i2]) < 1e-12);
  assert.deepEqual(m.tokensOf(i1, i2), tk);
}

// 3) 学習テスト：NLL が下限に近づき、8 つの山を全部覆う
for (const order of ['coarse', 'axis']) {
  const m = new ARModel({ dataset: 'ring8', order, seed: 2 });
  const t0 = Date.now();
  const steps = 1500;
  let avg = 0;
  for (let i = 0; i < steps; i++) {
    const r = m.trainStep();
    if (i >= steps - 100) avg += r.nll / 100;
  }
  const ms = Date.now() - t0;
  const X = m.sample(4000, mulberry32(9));
  const hits = new Array(8).fill(0);
  let near = 0;
  for (let i = 0; i < 4000; i++) {
    const x = X[2 * i], y = X[2 * i + 1];
    for (let k = 0; k < 8; k++) {
      const t = (k / 8) * Math.PI * 2;
      if (Math.hypot(x - 1.4 * Math.cos(t), y - 1.4 * Math.sin(t)) < 0.3) { hits[k]++; near++; }
    }
  }
  console.log(`${order}: NLL ${avg.toFixed(3)} (下限 ${m.entropy.toFixed(3)}) | 山ごとのサンプル数 ${hits.join(',')} | 山の近く ${(near / 40).toFixed(1)}% | ${(steps / ms * 1000).toFixed(0)} step/s`);
  assert.ok(avg - m.entropy < 0.6, 'NLL が下限から離れすぎ');
  assert.ok(Math.min(...hits) > 200, 'モードを取りこぼしている');
}
console.log('OK');
