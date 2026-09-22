// 依存ライブラリなしの小さなデコーダ型（自己回帰）Transformer
//   埋め込み + 位置埋め込み → [LayerNorm → 因果的マルチヘッド自己注意 → 残差
//                              → LayerNorm → MLP(ReLU) → 残差] × layers
//   → LayerNorm → 語彙へのロジット
// 順伝播・逆伝播・Adam をすべて手書きしている。
import { gaussian } from './nn.js';

class Param {
  constructor(size, init = 0, rand = null) {
    this.w = new Float64Array(size);
    if (rand) for (let i = 0; i < size; i++) this.w[i] = gaussian(rand) * init;
    else if (init) this.w.fill(init);
    this.g = new Float64Array(size);
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
  }
}

// ---- 基本演算（行列は行優先、W は [in][out]） ----
function linear(X, n, inD, outD, W, b) {
  const Y = new Float64Array(n * outD);
  for (let i = 0; i < n; i++) {
    const xo = i * inD, yo = i * outD;
    if (b) for (let j = 0; j < outD; j++) Y[yo + j] = b[j];
    for (let k = 0; k < inD; k++) {
      const x = X[xo + k];
      if (x === 0) continue;
      const wo = k * outD;
      for (let j = 0; j < outD; j++) Y[yo + j] += x * W[wo + j];
    }
  }
  return Y;
}

function linearBack(dY, X, n, inD, outD, W, gW, gb) {
  const dX = new Float64Array(n * inD);
  for (let i = 0; i < n; i++) {
    const xo = i * inD, yo = i * outD;
    if (gb) for (let j = 0; j < outD; j++) gb[j] += dY[yo + j];
    for (let k = 0; k < inD; k++) {
      const x = X[xo + k];
      const wo = k * outD;
      let s = 0;
      for (let j = 0; j < outD; j++) {
        const d = dY[yo + j];
        s += d * W[wo + j];
        gW[wo + j] += x * d;
      }
      dX[xo + k] = s;
    }
  }
  return dX;
}

const LN_EPS = 1e-5;
function layerNorm(X, n, d, g, b) {
  const Y = new Float64Array(n * d);
  const Xhat = new Float64Array(n * d);
  const rstd = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let mu = 0;
    for (let k = 0; k < d; k++) mu += X[o + k];
    mu /= d;
    let v = 0;
    for (let k = 0; k < d; k++) v += (X[o + k] - mu) ** 2;
    const r = 1 / Math.sqrt(v / d + LN_EPS);
    rstd[i] = r;
    for (let k = 0; k < d; k++) {
      const xh = (X[o + k] - mu) * r;
      Xhat[o + k] = xh;
      Y[o + k] = xh * g[k] + b[k];
    }
  }
  return { Y, Xhat, rstd };
}

function layerNormBack(dY, cache, n, d, g, gg, gb) {
  const { Xhat, rstd } = cache;
  const dX = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const o = i * d;
    let m1 = 0, m2 = 0;
    for (let k = 0; k < d; k++) {
      const dy = dY[o + k];
      gg[k] += dy * Xhat[o + k];
      gb[k] += dy;
      const dxh = dy * g[k];
      m1 += dxh;
      m2 += dxh * Xhat[o + k];
    }
    m1 /= d; m2 /= d;
    for (let k = 0; k < d; k++) {
      const dxh = dY[o + k] * g[k];
      dX[o + k] = rstd[i] * (dxh - m1 - Xhat[o + k] * m2);
    }
  }
  return dX;
}

export class Transformer {
  /**
   * @param {{vocab:number, seqLen:number, d:number, heads:number, layers:number, ff:number}} cfg
   *   vocab: 出力語彙数。入力側にはこれに加えて BOS（= vocab 番）を使う
   */
  constructor(cfg, rand) {
    this.cfg = cfg;
    const { vocab, seqLen, d, layers, ff } = cfg;
    this.params = [];
    const P = (size, init, useRand = true) => {
      const p = new Param(size, init, useRand ? rand : null);
      this.params.push(p);
      return p;
    };
    this.tokEmb = P((vocab + 1) * d, 0.3);
    this.posEmb = P(seqLen * d, 0.3);
    this.blocks = [];
    for (let l = 0; l < layers; l++) {
      this.blocks.push({
        ln1g: P(d, 1, false), ln1b: P(d, 0, false),
        Wq: P(d * d, 1 / Math.sqrt(d)), Wk: P(d * d, 1 / Math.sqrt(d)), Wv: P(d * d, 1 / Math.sqrt(d)),
        Wo: P(d * d, 1 / Math.sqrt(d) / Math.sqrt(2 * layers)), bo: P(d, 0, false),
        ln2g: P(d, 1, false), ln2b: P(d, 0, false),
        W1: P(d * ff, Math.sqrt(2 / d)), b1: P(ff, 0, false),
        W2: P(ff * d, 1 / Math.sqrt(ff) / Math.sqrt(2 * layers)), b2: P(d, 0, false),
      });
    }
    this.lnfg = P(d, 1, false);
    this.lnfb = P(d, 0, false);
    this.Wout = P(d * vocab, 1 / Math.sqrt(d));
    this.bout = P(vocab, 0, false);
    this.t = 0;
  }

  paramCount() {
    return this.params.reduce((s, p) => s + p.w.length, 0);
  }

  /**
   * @param {Int32Array} tokens 長さ B*T の入力トークン列（先頭は BOS）
   * @param {number} B バッチ内の系列数
   * @returns {Float64Array} ロジット [B*T][vocab]
   */
  forward(tokens, B) {
    const { vocab, seqLen: T, d, heads } = this.cfg;
    const n = B * T;
    const dh = d / heads;
    const scale = 1 / Math.sqrt(dh);
    let h = new Float64Array(n * d);
    const E = this.tokEmb.w, Pe = this.posEmb.w;
    for (let i = 0; i < n; i++) {
      const tk = tokens[i], pos = i % T;
      for (let k = 0; k < d; k++) h[i * d + k] = E[tk * d + k] + Pe[pos * d + k];
    }
    const caches = [];
    for (const blk of this.blocks) {
      const c = { hIn: h };
      c.ln1 = layerNorm(h, n, d, blk.ln1g.w, blk.ln1b.w);
      const a = c.ln1.Y;
      c.q = linear(a, n, d, d, blk.Wq.w, null);
      c.k = linear(a, n, d, d, blk.Wk.w, null);
      c.v = linear(a, n, d, d, blk.Wv.w, null);
      // 因果的自己注意：位置 i は 0..i だけを見る
      const att = new Float64Array(B * heads * T * T);
      const ctx = new Float64Array(n * d);
      for (let b = 0; b < B; b++) {
        for (let hd = 0; hd < heads; hd++) {
          const ho = hd * dh;
          for (let i = 0; i < T; i++) {
            const qi = (b * T + i) * d + ho;
            const ao = ((b * heads + hd) * T + i) * T;
            let mx = -Infinity;
            for (let j = 0; j <= i; j++) {
              const kj = (b * T + j) * d + ho;
              let s = 0;
              for (let e = 0; e < dh; e++) s += c.q[qi + e] * c.k[kj + e];
              s *= scale;
              att[ao + j] = s;
              if (s > mx) mx = s;
            }
            let sum = 0;
            for (let j = 0; j <= i; j++) { att[ao + j] = Math.exp(att[ao + j] - mx); sum += att[ao + j]; }
            for (let j = 0; j <= i; j++) {
              att[ao + j] /= sum;
              const vj = (b * T + j) * d + ho;
              const w = att[ao + j];
              for (let e = 0; e < dh; e++) ctx[qi + e] += w * c.v[vj + e];
            }
          }
        }
      }
      c.att = att;
      c.ctx = ctx;
      const o = linear(ctx, n, d, d, blk.Wo.w, blk.bo.w);
      const h1 = new Float64Array(n * d);
      for (let i = 0; i < n * d; i++) h1[i] = h[i] + o[i];
      c.h1 = h1;
      c.ln2 = layerNorm(h1, n, d, blk.ln2g.w, blk.ln2b.w);
      c.u = linear(c.ln2.Y, n, d, this.cfg.ff, blk.W1.w, blk.b1.w);
      const r = new Float64Array(c.u.length);
      for (let i = 0; i < r.length; i++) r[i] = c.u[i] > 0 ? c.u[i] : 0;
      c.r = r;
      const f = linear(r, n, this.cfg.ff, d, blk.W2.w, blk.b2.w);
      const h2 = new Float64Array(n * d);
      for (let i = 0; i < n * d; i++) h2[i] = h1[i] + f[i];
      caches.push(c);
      h = h2;
    }
    this.lnf = layerNorm(h, n, d, this.lnfg.w, this.lnfb.w);
    const logits = linear(this.lnf.Y, n, d, vocab, this.Wout.w, this.bout.w);
    this.cache = { tokens, B, n, caches };
    return logits;
  }

  /** 各層・各ヘッドのアテンション重み（最後の forward の b 番目の系列） */
  attention(b = 0) {
    const { seqLen: T, heads } = this.cfg;
    return this.cache.caches.map((c) => {
      const out = [];
      for (let hd = 0; hd < heads; hd++) {
        const o = ((b * heads + hd) * T) * T;
        out.push(c.att.slice(o, o + T * T));
      }
      return out;
    });
  }

  zeroGrad() {
    for (const p of this.params) p.g.fill(0);
  }

  /** ロジットに対する勾配から全パラメータの勾配を計算 */
  backward(dLogits) {
    const { vocab, seqLen: T, d, heads, ff } = this.cfg;
    const { tokens, B, n, caches } = this.cache;
    const dh = d / heads;
    const scale = 1 / Math.sqrt(dh);
    let dh_ = linearBack(dLogits, this.lnf.Y, n, d, vocab, this.Wout.w, this.Wout.g, this.bout.g);
    dh_ = layerNormBack(dh_, this.lnf, n, d, this.lnfg.w, this.lnfg.g, this.lnfb.g);
    for (let l = this.blocks.length - 1; l >= 0; l--) {
      const blk = this.blocks[l], c = caches[l];
      // MLP
      const dr = linearBack(dh_, c.r, n, ff, d, blk.W2.w, blk.W2.g, blk.b2.g);
      for (let i = 0; i < dr.length; i++) if (c.u[i] <= 0) dr[i] = 0;
      const dm = linearBack(dr, c.ln2.Y, n, d, ff, blk.W1.w, blk.W1.g, blk.b1.g);
      const dln2 = layerNormBack(dm, c.ln2, n, d, blk.ln2g.w, blk.ln2g.g, blk.ln2b.g);
      const dh1 = new Float64Array(n * d);
      for (let i = 0; i < n * d; i++) dh1[i] = dh_[i] + dln2[i];
      // 注意機構
      const dctx = linearBack(dh1, c.ctx, n, d, d, blk.Wo.w, blk.Wo.g, blk.bo.g);
      const dq = new Float64Array(n * d), dk = new Float64Array(n * d), dv = new Float64Array(n * d);
      const dA = new Float64Array(T);
      for (let b = 0; b < B; b++) {
        for (let hd = 0; hd < heads; hd++) {
          const ho = hd * dh;
          for (let i = 0; i < T; i++) {
            const qi = (b * T + i) * d + ho;
            const ao = ((b * heads + hd) * T + i) * T;
            let dot = 0;
            for (let j = 0; j <= i; j++) {
              const vj = (b * T + j) * d + ho;
              let s = 0;
              const a = c.att[ao + j];
              for (let e = 0; e < dh; e++) {
                s += dctx[qi + e] * c.v[vj + e];
                dv[vj + e] += a * dctx[qi + e];
              }
              dA[j] = s;
              dot += a * s;
            }
            for (let j = 0; j <= i; j++) {
              const dS = c.att[ao + j] * (dA[j] - dot) * scale;
              if (dS === 0) continue;
              const kj = (b * T + j) * d + ho;
              for (let e = 0; e < dh; e++) {
                dq[qi + e] += dS * c.k[kj + e];
                dk[kj + e] += dS * c.q[qi + e];
              }
            }
          }
        }
      }
      const a = c.ln1.Y;
      const da = linearBack(dq, a, n, d, d, blk.Wq.w, blk.Wq.g, null);
      const da2 = linearBack(dk, a, n, d, d, blk.Wk.w, blk.Wk.g, null);
      const da3 = linearBack(dv, a, n, d, d, blk.Wv.w, blk.Wv.g, null);
      for (let i = 0; i < n * d; i++) da[i] += da2[i] + da3[i];
      const dln1 = layerNormBack(da, c.ln1, n, d, blk.ln1g.w, blk.ln1g.g, blk.ln1b.g);
      const dh0 = new Float64Array(n * d);
      for (let i = 0; i < n * d; i++) dh0[i] = dh1[i] + dln1[i];
      dh_ = dh0;
    }
    // 埋め込み
    const gE = this.tokEmb.g, gP = this.posEmb.g;
    for (let i = 0; i < n; i++) {
      const tk = tokens[i], pos = i % T;
      for (let k = 0; k < d; k++) {
        gE[tk * d + k] += dh_[i * d + k];
        gP[pos * d + k] += dh_[i * d + k];
      }
    }
  }

  adamStep(lr, beta1 = 0.9, beta2 = 0.99, eps = 1e-8) {
    this.t++;
    const c1 = 1 - Math.pow(beta1, this.t);
    const c2 = 1 - Math.pow(beta2, this.t);
    for (const p of this.params) {
      const { w, g, m, v } = p;
      for (let i = 0; i < w.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    }
  }
}

/** ロジット行ごとのソフトマックス */
export function softmaxRows(logits, n, V) {
  const P = new Float64Array(n * V);
  for (let i = 0; i < n; i++) {
    const o = i * V;
    let mx = -Infinity;
    for (let j = 0; j < V; j++) if (logits[o + j] > mx) mx = logits[o + j];
    let s = 0;
    for (let j = 0; j < V; j++) { P[o + j] = Math.exp(logits[o + j] - mx); s += P[o + j]; }
    for (let j = 0; j < V; j++) P[o + j] /= s;
  }
  return P;
}
