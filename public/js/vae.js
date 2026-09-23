// VAE（変分オートエンコーダ）
//   エンコーダ: x → (平均 μ, 対数分散 logσ²)
//   再パラメータ化: z = μ + σ·ε（ε は標準正規乱数）
//   デコーダ:   z → x̂
//   損失 = 再構成誤差 / (2σ_x²) + β × KL(q(z|x) ‖ N(0, I))
import { MLP, mulberry32, gaussian } from './nn.js';
import { DATASETS } from './datasets.js';

export const VAE_DEFAULTS = {
  dataset: 'ring8',
  latentDim: 2,
  hidden: 32,
  depth: 2,
  lr: 0.002,
  batch: 64,
  beta: 1,        // KL 項の重み（β-VAE）
  sigmaX: 0.1,    // デコーダ出力のばらつき（再構成の厳しさ）
  seed: 1,
};

export class VAE {
  constructor(cfg = {}) {
    this.cfg = { ...VAE_DEFAULTS, ...cfg };
    const c = this.cfg;
    this.rand = mulberry32(c.seed * 6271 + 17);
    const hid = Array(c.depth).fill(c.hidden);
    this.enc = new MLP([2, ...hid, 2 * c.latentDim], 'lrelu', 'linear', this.rand);
    this.dec = new MLP([c.latentDim, ...hid, 2], 'lrelu', 'linear', this.rand);
    this.dataset = DATASETS[c.dataset];
    this.step = 0;
    this.last = null;
  }

  paramCount() {
    return this.enc.paramCount() + this.dec.paramCount();
  }

  sampleReal(n) {
    const X = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = this.dataset.sample(this.rand);
      X[2 * i] = x;
      X[2 * i + 1] = y;
    }
    return X;
  }

  /** エンコード：各点の μ と σ を返す */
  encode(X, n) {
    const L = this.cfg.latentDim;
    const out = this.enc.forward(X, n);
    const mu = new Float64Array(n * L);
    const sd = new Float64Array(n * L);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < L; k++) {
        mu[i * L + k] = out[i * 2 * L + k];
        sd[i * L + k] = Math.exp(0.5 * out[i * 2 * L + L + k]);
      }
    }
    return { mu, sd };
  }

  decode(Z, n) {
    return Float64Array.from(this.dec.forward(Z, n));
  }

  /** 事前分布 N(0, I) から z を引いて新しい点を作る */
  sample(n, rand = this.rand) {
    const L = this.cfg.latentDim;
    const Z = new Float64Array(n * L);
    for (let i = 0; i < Z.length; i++) Z[i] = gaussian(rand);
    return { Z, X: this.decode(Z, n) };
  }

  /**
   * 1ステップ学習する。
   * @param {Float64Array} [xr] 本物のバッチ（省略時はデータ分布から引く）
   * @param {Float64Array} [eps] 再パラメータ化に使う乱数（テスト用に固定できる）
   */
  trainStep(xr, eps) {
    const { batch: n, lr, beta, sigmaX, latentDim: L } = this.cfg;
    const X = xr || this.sampleReal(n);
    this.enc.zeroGrad();
    this.dec.zeroGrad();

    // --- エンコード ---
    const encOut = this.enc.forward(X, n); // [μ(L), logσ²(L)]
    const E = eps || Float64Array.from({ length: n * L }, () => gaussian(this.rand));
    const Z = new Float64Array(n * L);
    const sd = new Float64Array(n * L);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < L; k++) {
        const mu = encOut[i * 2 * L + k];
        const lv = encOut[i * 2 * L + L + k];
        const s = Math.exp(0.5 * lv);
        sd[i * L + k] = s;
        Z[i * L + k] = mu + s * E[i * L + k];
      }
    }

    // --- デコード & 再構成誤差 ---
    const Xh = this.dec.forward(Z, n);
    const dXh = new Float64Array(n * 2);
    let recon = 0, mse = 0;
    const inv = 1 / (sigmaX * sigmaX);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 2; j++) {
        const d = Xh[i * 2 + j] - X[i * 2 + j];
        mse += (d * d) / n;
        recon += (0.5 * inv * d * d) / n;
        dXh[i * 2 + j] = (inv * d) / n;
      }
    }
    const dZ = this.dec.backward(dXh);

    // --- KL とエンコーダへの逆伝播 ---
    let kl = 0;
    const dEnc = new Float64Array(n * 2 * L);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < L; k++) {
        const mu = encOut[i * 2 * L + k];
        const lv = encOut[i * 2 * L + L + k];
        const s = sd[i * L + k];
        kl += (0.5 * (mu * mu + s * s - 1 - lv)) / n;
        const dz = dZ[i * L + k];
        // 再構成からの勾配
        dEnc[i * 2 * L + k] += dz;
        dEnc[i * 2 * L + L + k] += dz * 0.5 * s * E[i * L + k];
        // KL からの勾配
        dEnc[i * 2 * L + k] += (beta * mu) / n;
        dEnc[i * 2 * L + L + k] += (beta * 0.5 * (s * s - 1)) / n;
      }
    }
    this.enc.backward(dEnc);

    this.enc.adamStep(lr, 0.9, 0.999);
    this.dec.adamStep(lr, 0.9, 0.999);

    this.step++;
    this.last = { step: this.step, recon, kl, mse, loss: recon + beta * kl };
    return this.last;
  }

  /**
   * 学習した分布 p(x) の近似（デコーダから z を M 個引いて混合ガウスとみなす）。
   * @param {Float64Array} grid 評価したい点（[x,y] の並び）
   */
  density(grid, gridN, M = 160, rand = mulberry32(4242)) {
    const { latentDim: L, sigmaX } = this.cfg;
    const Z = new Float64Array(M * L);
    for (let i = 0; i < Z.length; i++) Z[i] = gaussian(rand);
    const C = this.decode(Z, M);
    const out = new Float64Array(gridN);
    const norm = 1 / (2 * Math.PI * sigmaX * sigmaX * M);
    const inv = 1 / (2 * sigmaX * sigmaX);
    for (let g = 0; g < gridN; g++) {
      const x = grid[2 * g], y = grid[2 * g + 1];
      let s = 0;
      for (let m = 0; m < M; m++) {
        const dx = x - C[2 * m], dy = y - C[2 * m + 1];
        s += Math.exp(-(dx * dx + dy * dy) * inv);
      }
      out[g] = s * norm;
    }
    return out;
  }
}
