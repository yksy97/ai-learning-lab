// GAN 本体：Generator と Discriminator の交互学習
import { MLP, mulberry32, gaussian } from './nn.js';
import { DATASETS } from './datasets.js';

const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
// log(sigmoid(z)) を数値的に安定に計算
const logSig = (z) => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)));

export const DEFAULTS = {
  dataset: 'ring8',
  noiseDim: 2,
  hidden: 32,
  depth: 3,
  lrD: 0.002,
  lrG: 0.001,
  dSteps: 1,
  batch: 64,
  loss: 'nonsat', // 'nonsat' | 'minimax'
  seed: 1,
};

export class GAN {
  constructor(cfg = {}) {
    this.cfg = { ...DEFAULTS, ...cfg };
    const c = this.cfg;
    this.rand = mulberry32(c.seed * 9973 + 7);
    const hid = Array(c.depth).fill(c.hidden);
    this.G = new MLP([c.noiseDim, ...hid, 2], 'lrelu', 'linear', this.rand);
    this.D = new MLP([2, ...hid, 1], 'lrelu', 'linear', this.rand); // 出力はロジット
    this.step = 0;
    this.history = []; // {step, lossD, lossG, dReal, dFake}
    this.last = null;
    this.dataset = DATASETS[c.dataset];
  }

  sampleReal(n) {
    const X = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = this.dataset.sample(this.rand);
      X[i * 2] = x;
      X[i * 2 + 1] = y;
    }
    return X;
  }

  sampleNoise(n) {
    const Z = new Float64Array(n * this.cfg.noiseDim);
    for (let i = 0; i < Z.length; i++) Z[i] = gaussian(this.rand);
    return Z;
  }

  generate(Z, n) {
    return Float64Array.from(this.G.forward(Z, n));
  }

  /** D(x) を確率で返す */
  discriminate(X, n) {
    const L = this.D.forward(X, n);
    const P = new Float64Array(n);
    for (let i = 0; i < n; i++) P[i] = sigmoid(L[i]);
    return P;
  }

  /** 入力 x に対する D のロジットの勾配（G が点をどちらへ動かしたいか） */
  dGradient(X, n) {
    this.D.forward(X, n);
    const d = new Float64Array(n).fill(1);
    return this.D.backward(d, false);
  }

  trainStep() {
    const { batch: n, lrD, lrG, dSteps, loss } = this.cfg;
    let lossD = 0, dReal = 0, dFake = 0;

    // ---- Discriminator の更新：本物→1、偽物→0 を当てるように ----
    for (let s = 0; s < dSteps; s++) {
      this.D.zeroGrad();
      const xr = this.sampleReal(n);
      const lr_ = this.D.forward(xr, n);
      const gr = new Float64Array(n);
      lossD = 0; dReal = 0; dFake = 0;
      for (let i = 0; i < n; i++) {
        const p = sigmoid(lr_[i]);
        gr[i] = (p - 1) / n;
        lossD -= logSig(lr_[i]) / n;
        dReal += p / n;
      }
      this.D.backward(gr);

      const xf = this.generate(this.sampleNoise(n), n);
      const lf = this.D.forward(xf, n);
      const gf = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = sigmoid(lf[i]);
        gf[i] = p / n;
        lossD -= logSig(-lf[i]) / n;
        dFake += p / n;
      }
      this.D.backward(gf);
      this.D.adamStep(lrD);
    }

    // ---- Generator の更新：D を騙すように ----
    this.G.zeroGrad();
    const z = this.sampleNoise(n);
    const xg = this.G.forward(z, n);
    const lg = this.D.forward(xg, n);
    const dl = new Float64Array(n);
    let lossG = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(lg[i]);
      if (loss === 'minimax') {
        // 元論文の min log(1 - D(G(z)))：D が強いと勾配が消えやすい
        dl[i] = p / n;
        lossG += logSig(-lg[i]) / n;
      } else {
        // 非飽和版 min -log D(G(z))：実用上の標準
        dl[i] = (p - 1) / n;
        lossG -= logSig(lg[i]) / n;
      }
    }
    const dx = this.D.backward(dl, false); // D のパラメータは更新しない
    this.G.backward(dx);
    this.G.adamStep(lrG);

    this.step++;
    this.last = { step: this.step, lossD, lossG, dReal, dFake };
    return this.last;
  }
}
