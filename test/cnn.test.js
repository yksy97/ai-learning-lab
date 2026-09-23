// CNN のテスト：逆伝播が数値微分と一致すること、実際に分類できるようになること
import { CNN, CLASSES, makeImage } from '../public/js/cnn.js';
import { mulberry32 } from '../public/js/nn.js';

const ok = (cond, msg) => { if (!cond) { console.error('NG:', msg); process.exit(1); } console.log('  ○', msg); };

// ---- 勾配チェック ----
{
  const net = new CNN({ filters: 3, seed: 5 });
  const r = mulberry32(99);
  const img = makeImage('v', r);
  const label = 0;

  const g = net.zeroGrad();
  net.accumulate(img, label, g);

  const lossOf = () => -Math.log(Math.max(net.forwardOne(img).probs[label], 1e-12));
  const eps = 1e-6;
  let worst = 0;
  for (const key of ['w', 'b', 'W', 'B']) {
    const p = net[key];
    for (let trial = 0; trial < 12; trial++) {
      const i = Math.floor(r() * p.length);
      const orig = p[i];
      p[i] = orig + eps; const lp = lossOf();
      p[i] = orig - eps; const lm = lossOf();
      p[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = g[key][i];
      const denom = Math.max(1e-7, Math.abs(num) + Math.abs(ana));
      worst = Math.max(worst, Math.abs(num - ana) / denom);
    }
  }
  console.log(`勾配チェック（逆伝播 vs 数値微分）最大の相対誤差 ${worst.toExponential(2)}`);
  ok(worst < 1e-5, '逆伝播が数値微分と一致する');
}

// ---- 学習できること ----
{
  const net = new CNN({ seed: 1 });
  console.log(`パラメータ数 ${net.paramCount().toLocaleString()}`);
  const before = net.evaluate(80).acc;
  const t0 = Date.now();
  for (let i = 0; i < 400; i++) net.trainStep();
  const after = net.evaluate(80).acc;
  console.log(`正解率 ${(before * 100).toFixed(0)}% → ${(after * 100).toFixed(0)}%（400 ステップ・${((Date.now() - t0) / 1000).toFixed(1)} 秒）`);
  ok(after > 0.8, `学習後の正解率が 80% を超える（${(after * 100).toFixed(0)}%）`);

  // カーネルが「何かの向き」を見るようになっているか（重みが偏る）
  let maxSpread = 0;
  for (let f = 0; f < net.F; f++) {
    const k = net.w.slice(f * 9, f * 9 + 9);
    maxSpread = Math.max(maxSpread, Math.max(...k) - Math.min(...k));
  }
  ok(maxSpread > 0.5, 'カーネルに濃淡がつく（向きを見るようになる）');

  // 位置をずらしても同じ答えになるか（畳み込みとプーリングの効果）
  const r = mulberry32(7);
  let same = 0;
  for (let i = 0; i < 40; i++) {
    const c = i % CLASSES.length;
    if (net.predict(makeImage(CLASSES[c].id, r)).pred === c) same++;
  }
  console.log(`位置も太さも違う40枚のうち ${same} 枚を正しく分類`);
  ok(same >= 30, '位置や太さが変わっても分類できる');
}

console.log('OK');
