// 数値微分による勾配チェックと、簡単な分布での学習収束テスト
import assert from 'node:assert/strict';
import { MLP, mulberry32 } from '../public/js/nn.js';
import { GAN } from '../public/js/gan.js';

// 1) 勾配チェック（tanh で滑らかにして比較）
{
  const r = mulberry32(3);
  const net = new MLP([2, 5, 5, 1], 'tanh', 'linear', r);
  const X = Float64Array.from([0.3, -0.7, 1.1, 0.2]);
  const lossOf = () => { const o = net.forward(X, 2); return o[0] * o[0] + 0.5 * o[1]; };
  net.zeroGrad();
  const o = net.forward(X, 2);
  const dIn = net.backward(Float64Array.from([2 * o[0], 0.5]));
  const eps = 1e-6;
  let maxErr = 0;
  for (const l of net.layers) {
    for (let i = 0; i < l.W.length; i++) {
      const w = l.W[i];
      l.W[i] = w + eps; const a = lossOf();
      l.W[i] = w - eps; const b = lossOf();
      l.W[i] = w;
      maxErr = Math.max(maxErr, Math.abs((a - b) / (2 * eps) - l.gW[i]));
    }
  }
  for (let k = 0; k < X.length; k++) {
    const x = X[k];
    X[k] = x + eps; const a = lossOf();
    X[k] = x - eps; const b = lossOf();
    X[k] = x;
    maxErr = Math.max(maxErr, Math.abs((a - b) / (2 * eps) - dIn[k]));
  }
  console.log('勾配チェック 最大誤差:', maxErr.toExponential(2));
  assert.ok(maxErr < 1e-6);
}

// 2) 収束テスト：生成分布の平均と標準偏差が本物に近づくか
function stats(X, n) {
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += X[2 * i] / n; my += X[2 * i + 1] / n; }
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += (X[2 * i] - mx) ** 2 / n; sy += (X[2 * i + 1] - my) ** 2 / n; }
  return [mx, my, Math.sqrt(sx), Math.sqrt(sy)];
}
for (const ds of ['gaussian', 'ring8', 'circle']) {
  const gan = new GAN({ dataset: ds, seed: 2 });
  const t0 = Date.now();
  for (let i = 0; i < 5000; i++) gan.trainStep();
  const ms = Date.now() - t0;
  const n = 4000;
  const real = stats(gan.sampleReal(n), n);
  const fake = stats(gan.generate(gan.sampleNoise(n), n), n);
  const err = Math.max(...real.map((v, i) => Math.abs(v - fake[i])));
  console.log(ds, '本物', real.map((v) => v.toFixed(2)).join(','), '| 生成', fake.map((v) => v.toFixed(2)).join(','),
    '| D(real)', gan.last.dReal.toFixed(2), 'D(fake)', gan.last.dFake.toFixed(2), `| ${(5000 / ms * 1000).toFixed(0)} step/s`);
  assert.ok(err < 0.35, `${ds} の統計量が一致しない (誤差 ${err})`);
}
console.log('OK');
