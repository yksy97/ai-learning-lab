// 依存ライブラリなしの小さな多層パーセプトロン（順伝播・逆伝播・Adam）

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand) {
  let u = 0;
  while (u === 0) u = rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const LEAK = 0.2;

const ACT_KIND = { lrelu: 0, tanh: 1, linear: 2 };

class Dense {
  constructor(inDim, outDim, act, rand) {
    this.inDim = inDim;
    this.outDim = outDim;
    this.kind = ACT_KIND[act];
    this.W = new Float64Array(outDim * inDim);
    this.b = new Float64Array(outDim);
    // He 初期化
    const scale = Math.sqrt(2 / inDim);
    for (let i = 0; i < this.W.length; i++) this.W[i] = gaussian(rand) * scale;
    this.gW = new Float64Array(this.W.length);
    this.gb = new Float64Array(this.b.length);
    // Adam のモーメント
    this.mW = new Float64Array(this.W.length);
    this.vW = new Float64Array(this.W.length);
    this.mb = new Float64Array(this.b.length);
    this.vb = new Float64Array(this.b.length);
  }

  forward(X, n) {
    const { inDim, outDim, W, b, kind } = this;
    const Z = new Float64Array(n * outDim);
    const A = new Float64Array(n * outDim);
    for (let i = 0; i < n; i++) {
      const xo = i * inDim;
      const zo = i * outDim;
      for (let j = 0; j < outDim; j++) {
        let s = b[j];
        const wo = j * inDim;
        for (let k = 0; k < inDim; k++) s += X[xo + k] * W[wo + k];
        Z[zo + j] = s;
        A[zo + j] = kind === 0 ? (s > 0 ? s : LEAK * s) : kind === 1 ? Math.tanh(s) : s;
      }
    }
    this.X = X;
    this.Z = Z;
    this.A = A;
    this.n = n;
    return A;
  }

  backward(dA, accumulate) {
    const { inDim, outDim, W, X, Z, A, n, kind, gW, gb } = this;
    const dX = new Float64Array(n * inDim);
    for (let i = 0; i < n; i++) {
      const xo = i * inDim;
      for (let j = 0; j < outDim; j++) {
        const idx = i * outDim + j;
        const deriv = kind === 0 ? (Z[idx] > 0 ? 1 : LEAK) : kind === 1 ? 1 - A[idx] * A[idx] : 1;
        const dz = dA[idx] * deriv;
        if (dz === 0) continue;
        const wo = j * inDim;
        if (accumulate) {
          gb[j] += dz;
          for (let k = 0; k < inDim; k++) {
            gW[wo + k] += dz * X[xo + k];
            dX[xo + k] += dz * W[wo + k];
          }
        } else {
          for (let k = 0; k < inDim; k++) dX[xo + k] += dz * W[wo + k];
        }
      }
    }
    return dX;
  }
}

export class MLP {
  /**
   * @param {number[]} sizes 例: [2, 32, 32, 1]
   * @param {string} hiddenAct 隠れ層の活性化関数
   * @param {string} outAct 出力層の活性化関数
   */
  constructor(sizes, hiddenAct, outAct, rand) {
    this.sizes = sizes;
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const last = i === sizes.length - 2;
      this.layers.push(new Dense(sizes[i], sizes[i + 1], last ? outAct : hiddenAct, rand));
    }
    this.t = 0;
  }

  get inDim() { return this.sizes[0]; }
  get outDim() { return this.sizes[this.sizes.length - 1]; }

  forward(X, n) {
    let a = X;
    for (const l of this.layers) a = l.forward(a, n);
    return a;
  }

  /** 出力に対する勾配 dOut を逆伝播し、入力に対する勾配を返す。
   *  accumulate=false ならパラメータ勾配は溜めない（相手ネットワーク越しに勾配だけ流したいとき用） */
  backward(dOut, accumulate = true) {
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d, accumulate);
    return d;
  }

  zeroGrad() {
    for (const l of this.layers) {
      l.gW.fill(0);
      l.gb.fill(0);
    }
  }

  adamStep(lr, beta1 = 0.5, beta2 = 0.999, eps = 1e-8) {
    this.t++;
    const c1 = 1 - Math.pow(beta1, this.t);
    const c2 = 1 - Math.pow(beta2, this.t);
    const upd = (p, g, m, v) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
    for (const l of this.layers) {
      upd(l.W, l.gW, l.mW, l.vW);
      upd(l.b, l.gb, l.mb, l.vb);
    }
  }

  paramCount() {
    return this.layers.reduce((s, l) => s + l.W.length + l.b.length, 0);
  }
}
