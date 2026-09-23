// 小さな畳み込みニューラルネットワーク（CNN）。
// 16×16 の白黒画像を4種類に分ける。構成は
//   畳み込み(3×3 × F枚) → ReLU → 最大プーリング(2×2) → 全結合 → softmax
// 実際の CNN はこの段を何段も積むが、しくみは1段でも全部見える。
import { mulberry32, gaussian } from './nn.js';

export const S = 16;            // 画像の一辺
export const K = 3;             // カーネルの一辺
export const POOL = 2;
export const SP = S / POOL;     // プーリング後の一辺

export const CLASSES = [
  { id: 'v', label: 'たて棒' },
  { id: 'h', label: 'よこ棒' },
  { id: 'd', label: 'ななめ' },
  { id: 'o', label: 'しかく' },
];

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/** 1枚の画像を作る。位置・太さ・傾き・かすれを毎回すこし変える */
export function makeImage(kind, rand) {
  const img = new Float64Array(S * S);
  const put = (x, y, v = 1) => {
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= S || yi >= S) return;
    img[yi * S + xi] = Math.min(1, img[yi * S + xi] + v);
  };
  const thick = 1 + Math.floor(rand() * 2);          // 1〜2 画素ぶんの太さ
  const len = 8 + Math.floor(rand() * 5);            // 長さ
  const cx = 3 + rand() * (S - 6 - 1);
  const cy = 3 + rand() * (S - 6 - 1);

  if (kind === 'v' || kind === 'h') {
    const tilt = (rand() - 0.5) * 0.5;               // すこしだけ傾ける
    for (let i = -len / 2; i <= len / 2; i += 0.5) {
      for (let t = 0; t < thick; t++) {
        if (kind === 'v') put(cx + i * tilt + t, cy + i);
        else put(cx + i, cy + i * tilt + t);
      }
    }
  } else if (kind === 'd') {
    const dir = rand() < 0.5 ? 1 : -1;
    for (let i = -len / 2; i <= len / 2; i += 0.5) {
      for (let t = 0; t < thick; t++) put(cx + i + t, cy + dir * i);
    }
  } else {
    const r = 3 + Math.floor(rand() * 3);            // しかくの半径
    for (let i = -r; i <= r; i += 0.5) {
      for (let t = 0; t < thick; t++) {
        put(cx + i, cy - r - t); put(cx + i, cy + r + t);
        put(cx - r - t, cy + i); put(cx + r + t, cy + i);
      }
    }
  }
  // かすれ（少しだけノイズ）
  for (let i = 0; i < img.length; i++) {
    img[i] = clamp(img[i] + (rand() - 0.5) * 0.12, 0, 1);
  }
  return img;
}

export const CNN_DEFAULTS = { filters: 6, lr: 0.02, batch: 16, seed: 1 };

export class CNN {
  constructor(cfg = {}) {
    this.cfg = { ...CNN_DEFAULTS, ...cfg };
    const F = this.cfg.filters;
    this.rand = mulberry32(this.cfg.seed * 7717 + 3);
    const r = this.rand;

    this.F = F;
    this.C = CLASSES.length;
    this.flat = F * SP * SP;

    // カーネルと全結合の重み
    this.w = new Float64Array(F * K * K).map(() => gaussian(r) * 0.5);
    this.b = new Float64Array(F);
    this.W = new Float64Array(this.C * this.flat).map(() => gaussian(r) * (1 / Math.sqrt(this.flat)));
    this.B = new Float64Array(this.C);

    // Adam
    this.m = {};
    this.v = {};
    for (const k of ['w', 'b', 'W', 'B']) {
      this.m[k] = new Float64Array(this[k].length);
      this.v[k] = new Float64Array(this[k].length);
    }
    this.t = 0;
    this.step = 0;
    this.last = null;
  }

  /** 1枚ぶんの順伝播。途中の値も返す（画面に出すため） */
  forwardOne(img) {
    const F = this.F;
    const conv = new Float64Array(F * S * S);
    const relu = new Float64Array(F * S * S);
    const pool = new Float64Array(F * SP * SP);
    const arg = new Int32Array(F * SP * SP);

    for (let f = 0; f < F; f++) {
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          let s = this.b[f];
          for (let dy = 0; dy < K; dy++) {
            const iy = y + dy - 1;
            if (iy < 0 || iy >= S) continue;
            for (let dx = 0; dx < K; dx++) {
              const ix = x + dx - 1;
              if (ix < 0 || ix >= S) continue;
              s += this.w[(f * K + dy) * K + dx] * img[iy * S + ix];
            }
          }
          const o = (f * S + y) * S + x;
          conv[o] = s;
          relu[o] = s > 0 ? s : 0;
        }
      }
      // 最大プーリング
      for (let py = 0; py < SP; py++) {
        for (let px = 0; px < SP; px++) {
          let best = -Infinity, bi = 0;
          for (let dy = 0; dy < POOL; dy++) {
            for (let dx = 0; dx < POOL; dx++) {
              const o = (f * S + py * POOL + dy) * S + px * POOL + dx;
              if (relu[o] > best) { best = relu[o]; bi = o; }
            }
          }
          const po = (f * SP + py) * SP + px;
          pool[po] = best;
          arg[po] = bi;
        }
      }
    }

    const logits = new Float64Array(this.C);
    for (let c = 0; c < this.C; c++) {
      let s = this.B[c];
      const o = c * this.flat;
      for (let i = 0; i < this.flat; i++) s += this.W[o + i] * pool[i];
      logits[c] = s;
    }
    return { conv, relu, pool, arg, logits, probs: softmax(logits) };
  }

  predict(img) {
    const r = this.forwardOne(img);
    let best = 0;
    for (let c = 1; c < this.C; c++) if (r.probs[c] > r.probs[best]) best = c;
    return { ...r, pred: best };
  }

  /** 学習データを1バッチ作る */
  sampleBatch(n = this.cfg.batch) {
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const c = Math.floor(this.rand() * this.C);
      xs.push(makeImage(CLASSES[c].id, this.rand));
      ys.push(c);
    }
    return { xs, ys };
  }

  /** 勾配を足しこむ（テストから1枚ずつ呼べるように分けてある） */
  accumulate(img, label, g) {
    const F = this.F;
    const r = this.forwardOne(img);
    const loss = -Math.log(Math.max(r.probs[label], 1e-12));

    // 出力 → 全結合
    const dLogits = new Float64Array(this.C);
    for (let c = 0; c < this.C; c++) dLogits[c] = r.probs[c] - (c === label ? 1 : 0);
    const dPool = new Float64Array(this.flat);
    for (let c = 0; c < this.C; c++) {
      const o = c * this.flat, d = dLogits[c];
      g.B[c] += d;
      for (let i = 0; i < this.flat; i++) {
        g.W[o + i] += d * r.pool[i];
        dPool[i] += d * this.W[o + i];
      }
    }

    // プーリングを戻す（最大だった場所にだけ流す）→ ReLU → 畳み込み
    const dRelu = new Float64Array(F * S * S);
    for (let i = 0; i < dPool.length; i++) dRelu[r.arg[i]] += dPool[i];
    for (let f = 0; f < F; f++) {
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const o = (f * S + y) * S + x;
          if (r.conv[o] <= 0) continue;          // ReLU の微分
          const d = dRelu[o];
          if (d === 0) continue;
          g.b[f] += d;
          for (let dy = 0; dy < K; dy++) {
            const iy = y + dy - 1;
            if (iy < 0 || iy >= S) continue;
            for (let dx = 0; dx < K; dx++) {
              const ix = x + dx - 1;
              if (ix < 0 || ix >= S) continue;
              g.w[(f * K + dy) * K + dx] += d * img[iy * S + ix];
            }
          }
        }
      }
    }
    return loss;
  }

  zeroGrad() {
    return {
      w: new Float64Array(this.w.length), b: new Float64Array(this.b.length),
      W: new Float64Array(this.W.length), B: new Float64Array(this.B.length),
    };
  }

  trainStep() {
    const { xs, ys } = this.sampleBatch();
    const g = this.zeroGrad();
    let loss = 0, correct = 0;
    for (let i = 0; i < xs.length; i++) {
      loss += this.accumulate(xs[i], ys[i], g);
      const p = this.forwardOne(xs[i]).probs;
      let best = 0;
      for (let c = 1; c < this.C; c++) if (p[c] > p[best]) best = c;
      if (best === ys[i]) correct++;
    }
    const n = xs.length;
    for (const k of ['w', 'b', 'W', 'B']) for (let i = 0; i < g[k].length; i++) g[k][i] /= n;
    this.adamStep(g);
    this.step++;
    this.last = { step: this.step, loss: loss / n, acc: correct / n };
    return this.last;
  }

  adamStep(g, lr = this.cfg.lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    this.t++;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (const k of ['w', 'b', 'W', 'B']) {
      const p = this[k], m = this.m[k], v = this.v[k], gr = g[k];
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * gr[i];
        v[i] = b2 * v[i] + (1 - b2) * gr[i] * gr[i];
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }

  /** 新しく作った画像で正解率を測る */
  evaluate(n = 80) {
    const r = mulberry32(12345);
    let correct = 0;
    const conf = Array.from({ length: this.C }, () => new Array(this.C).fill(0));
    for (let i = 0; i < n; i++) {
      const c = i % this.C;
      const img = makeImage(CLASSES[c].id, r);
      const { pred } = this.predict(img);
      conf[c][pred]++;
      if (pred === c) correct++;
    }
    return { acc: correct / n, conf };
  }

  paramCount() {
    return this.w.length + this.b.length + this.W.length + this.B.length;
  }
}

export function softmax(z) {
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  const e = new Float64Array(z.length);
  let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - max); s += e[i]; }
  for (let i = 0; i < z.length; i++) e[i] /= s;
  return e;
}
