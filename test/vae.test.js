// VAE の勾配チェックと学習テスト
import assert from 'node:assert/strict';
import { VAE } from '../public/js/vae.js';
import { mulberry32, gaussian } from '../public/js/nn.js';
import { DATASETS } from '../public/js/datasets.js';
import { precisionRecall, modesCovered } from '../public/js/metrics.js';

// 1) 勾配チェック：ε を固定した上で、損失の数値微分と一致するか
{
  const v = new VAE({ hidden: 6, depth: 2, batch: 4, seed: 3, beta: 0.7, sigmaX: 0.3 });
  const r = mulberry32(11);
  const n = 4, L = v.cfg.latentDim;
  const X = Float64Array.from({ length: n * 2 }, () => gaussian(r));
  const E = Float64Array.from({ length: n * L }, () => gaussian(r));
  const lossOf = () => {
    const enc = v.enc.forward(X, n);
    const Z = new Float64Array(n * L);
    let kl = 0;
    for (let i = 0; i < n; i++) for (let k = 0; k < L; k++) {
      const mu = enc[i * 2 * L + k], lv = enc[i * 2 * L + L + k], s = Math.exp(0.5 * lv);
      Z[i * L + k] = mu + s * E[i * L + k];
      kl += 0.5 * (mu * mu + s * s - 1 - lv) / n;
    }
    const Xh = v.dec.forward(Z, n);
    let rec = 0;
    const inv = 1 / (v.cfg.sigmaX ** 2);
    for (let i = 0; i < n * 2; i++) rec += 0.5 * inv * (Xh[i] - X[i]) ** 2 / n;
    return rec + v.cfg.beta * kl;
  };
  // 勾配を計算（Adam は使わず、勾配だけ取り出す）
  const enc0 = v.enc.layers.map((l) => l.W.slice());
  v.trainStep(X, E);
  const gradsEnc = v.enc.layers.map((l) => l.gW.slice());
  const gradsDec = v.dec.layers.map((l) => l.gW.slice());
  // Adam が重みを動かしてしまったので元に戻す
  v.enc.layers.forEach((l, i) => l.W.set(enc0[i]));
  v.dec.layers.forEach((l) => l.W.set(l.W)); // デコーダは下で個別に戻す
  const v2 = new VAE({ hidden: 6, depth: 2, batch: 4, seed: 3, beta: 0.7, sigmaX: 0.3 });
  const eps = 1e-6;
  let maxErr = 0;
  const check = (net, grads) => {
    net.layers.forEach((l, li) => {
      for (let i = 0; i < l.W.length; i += Math.max(1, Math.floor(l.W.length / 12))) {
        const w = l.W[i];
        l.W[i] = w + eps; const a = lossOf.call(null);
        l.W[i] = w - eps; const b = lossOf.call(null);
        l.W[i] = w;
        const num = (a - b) / (2 * eps);
        maxErr = Math.max(maxErr, Math.abs(num - grads[li][i]));
      }
    });
  };
  // v2 は v と同じ初期値（同じシード）なので、v2 の重みで数値微分する
  const saveEnc = v.enc, saveDec = v.dec;
  v.enc = v2.enc; v.dec = v2.dec;
  check(v2.enc, gradsEnc);
  check(v2.dec, gradsDec);
  v.enc = saveEnc; v.dec = saveDec;
  console.log('VAE 勾配チェック 最大誤差:', maxErr.toExponential(2));
  assert.ok(maxErr < 1e-6);
}

// 2) 学習テスト：8つの山をひと通り覆えるか、KL が妥当な範囲に収まるか
for (const beta of [1, 4]) {
  const v = new VAE({ dataset: 'ring8', beta, seed: 2 });
  const t0 = Date.now();
  for (let i = 0; i < 3000; i++) v.trainStep();
  const ms = Date.now() - t0;
  const { X } = v.sample(500, mulberry32(7));
  const m = modesCovered(X, 500, DATASETS.ring8.modes);
  const real = new Float64Array(500 * 2);
  const rr = mulberry32(99);
  for (let i = 0; i < 500; i++) { const [x, y] = DATASETS.ring8.sample(rr); real[2 * i] = x; real[2 * i + 1] = y; }
  const pr = precisionRecall(real, X, 500);
  console.log(`β=${beta}: 山 ${m.covered}/8 | 品質 ${pr.precision.toFixed(2)} 網羅 ${pr.recall.toFixed(2)} | KL ${v.last.kl.toFixed(2)} MSE ${v.last.mse.toFixed(3)} | ${(3000 / ms * 1000).toFixed(0)} step/s`);
  if (beta === 1) {
    assert.ok(m.covered >= 5, 'β=1 で山を取りこぼしすぎ');
    assert.ok(v.last.kl > 0.5, 'KL が小さすぎる（事後崩壊している）');
  }
}

// 3) 事後崩壊：β を極端に大きくすると、潜在変数が使われなくなる
{
  const v = new VAE({ dataset: 'ring8', beta: 150, seed: 2 });
  for (let i = 0; i < 1500; i++) v.trainStep();
  const { mu, sd } = v.encode(v.sampleReal(300), 300);
  let spread = 0, meanSd = 0;
  for (let i = 0; i < 300 * 2; i++) { spread += (mu[i] ** 2) / 600; meanSd += sd[i] / 600; }
  console.log(`β=150: μ の広がり ${Math.sqrt(spread).toFixed(3)} / σ の平均 ${meanSd.toFixed(3)} / KL ${v.last.kl.toFixed(3)}`);
  assert.ok(v.last.kl < 0.05 && meanSd > 0.9, '事後崩壊が起きていない');
}
console.log('OK');
