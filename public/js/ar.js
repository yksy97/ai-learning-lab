// 2次元の点を「トークン列」に変換し、自己回帰 Transformer で p(x) を学習する
import { Transformer, softmaxRows } from './transformer.js';
import { mulberry32 } from './nn.js';
import { DATASETS } from './datasets.js';

export const RANGE = 2.2;   // 表示・量子化する範囲 [-RANGE, RANGE]
export const V = 8;         // 1トークンあたりの語彙数
export const BINS = V * V;  // 1軸あたりの分割数（粗 8 × 細 8 = 64）
export const T = 4;         // 系列長
const BOS = V;
const CELL = (2 * RANGE) / BINS;

// どの位置がどの軸・どの細かさを表すか
export const ORDERS = {
  coarse: {
    label: '粗→細（x₁粗, x₂粗, x₁細, x₂細）',
    slots: [{ axis: 0, level: 0 }, { axis: 1, level: 0 }, { axis: 0, level: 1 }, { axis: 1, level: 1 }],
  },
  axis: {
    label: '軸ごと（x₁粗, x₁細, x₂粗, x₂細）',
    slots: [{ axis: 0, level: 0 }, { axis: 0, level: 1 }, { axis: 1, level: 0 }, { axis: 1, level: 1 }],
  },
};
export const slotName = (s) => `${s.axis === 0 ? 'x₁' : 'x₂'} ${s.level === 0 ? '粗' : '細'}`;

export const AR_DEFAULTS = {
  dataset: 'ring8',
  order: 'coarse',
  d: 32,
  heads: 2,
  layers: 2,
  lr: 0.003,
  batch: 64,
  seed: 1,
};

export const toBin = (v) => Math.min(BINS - 1, Math.max(0, Math.floor((v + RANGE) / CELL)));
export const binLo = (i) => -RANGE + i * CELL;
export { CELL };

export class ARModel {
  constructor(cfg = {}) {
    this.cfg = { ...AR_DEFAULTS, ...cfg };
    const c = this.cfg;
    this.rand = mulberry32(c.seed * 7919 + 13);
    this.slots = ORDERS[c.order].slots;
    this.net = new Transformer({ vocab: V, seqLen: T, d: c.d, heads: c.heads, layers: c.layers, ff: 2 * c.d }, this.rand);
    this.dataset = DATASETS[c.dataset];
    this.step = 0;
    this.last = null;
    this.tables = null;
    this.entropy = this.dataEntropy();
  }

  /** 軸ごとのビン番号 → トークン列 */
  tokensOf(i1, i2) {
    const bins = [i1, i2];
    return this.slots.map((s) => (s.level === 0 ? bins[s.axis] >> 3 : bins[s.axis] & 7));
  }

  /** トークン列 → 軸ごとのビン番号 */
  binsOf(tokens) {
    const bins = [0, 0];
    this.slots.forEach((s, k) => {
      bins[s.axis] += s.level === 0 ? tokens[k] * V : tokens[k];
    });
    return bins;
  }

  /** 量子化した本物の分布のエントロピー（NLL の理論的な下限） */
  dataEntropy(n = 200000) {
    const r = mulberry32(12345);
    const h = new Float64Array(BINS * BINS);
    for (let i = 0; i < n; i++) {
      const [x, y] = this.dataset.sample(r);
      h[toBin(x) * BINS + toBin(y)]++;
    }
    let H = 0;
    for (const c of h) if (c > 0) H -= (c / n) * Math.log(c / n);
    return H;
  }

  sampleRealPoints(n) {
    const X = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = this.dataset.sample(this.rand);
      X[2 * i] = x;
      X[2 * i + 1] = y;
    }
    return X;
  }

  trainStep() {
    const { batch: B, lr } = this.cfg;
    const inp = new Int32Array(B * T);
    const tgt = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const [x, y] = this.dataset.sample(this.rand);
      const tk = this.tokensOf(toBin(x), toBin(y));
      for (let t = 0; t < T; t++) {
        inp[b * T + t] = t === 0 ? BOS : tk[t - 1];
        tgt[b * T + t] = tk[t];
      }
    }
    const net = this.net;
    net.zeroGrad();
    const logits = net.forward(inp, B);
    const P = softmaxRows(logits, B * T, V);
    const d = new Float64Array(B * T * V);
    const perPos = new Float64Array(T);
    for (let i = 0; i < B * T; i++) {
      const o = i * V;
      perPos[i % T] -= Math.log(Math.max(P[o + tgt[i]], 1e-12)) / B;
      for (let j = 0; j < V; j++) d[o + j] = (P[o + j] - (j === tgt[i] ? 1 : 0)) / B;
    }
    net.backward(d);
    net.adamStep(lr);
    this.step++;
    const nll = perPos.reduce((a, b) => a + b, 0);
    this.last = { step: this.step, nll, perPos: Array.from(perPos) };
    this.stale = true; // 重みが変わったので確率表は古くなった
    return this.last;
  }

  /**
   * 全ての接頭辞（8³ = 512 通り）を1回で流し、各位置の条件付き分布を表にする。
   * これで 64×64 の全マスの確率 p(x) と、祖先サンプリングが計算できる。
   */
  computeTables() {
    const B = V * V * V;
    const inp = new Int32Array(B * T);
    for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) for (let c = 0; c < V; c++) {
      const s = (a * V + b) * V + c;
      inp[s * T] = BOS; inp[s * T + 1] = a; inp[s * T + 2] = b; inp[s * T + 3] = c;
    }
    const P = softmaxRows(this.net.forward(inp, B), B * T, V);
    const row = (s, t) => P.subarray((s * T + t) * V, (s * T + t + 1) * V);
    const p0 = Float64Array.from(row(0, 0));
    const p1 = [], p2 = [], p3 = [];
    for (let a = 0; a < V; a++) p1.push(Float64Array.from(row(a * V * V, 1)));
    for (let ab = 0; ab < V * V; ab++) p2.push(Float64Array.from(row(ab * V, 2)));
    for (let abc = 0; abc < V * V * V; abc++) p3.push(Float64Array.from(row(abc, 3)));
    // 各マスの確率
    const density = new Float64Array(BINS * BINS); // [i1 * BINS + i2]
    for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) for (let c = 0; c < V; c++) for (let e = 0; e < V; e++) {
      const p = p0[a] * p1[a][b] * p2[a * V + b][c] * p3[(a * V + b) * V + c][e];
      const [i1, i2] = this.binsOf([a, b, c, e]);
      density[i1 * BINS + i2] = p;
    }
    this.tables = { p0, p1, p2, p3, density };
    this.stale = false;
    return this.tables;
  }

  ensureTables() {
    return this.tables && !this.stale ? this.tables : this.computeTables();
  }

  /** 接頭辞 tokens[0..k-1] が与えられたときの位置 k の条件付き分布 */
  conditional(prefix, t = this.ensureTables()) {
    const k = prefix.length;
    if (k === 0) return t.p0;
    if (k === 1) return t.p1[prefix[0]];
    if (k === 2) return t.p2[prefix[0] * V + prefix[1]];
    return t.p3[(prefix[0] * V + prefix[1]) * V + prefix[2]];
  }

  static draw(p, u) {
    let acc = 0;
    for (let j = 0; j < p.length; j++) { acc += p[j]; if (u < acc) return j; }
    return p.length - 1;
  }

  /** 1点を祖先サンプリング：トークンを1つずつ引いていく */
  sampleTokens(rand = this.rand, t = this.ensureTables()) {
    const tk = [];
    for (let k = 0; k < T; k++) tk.push(ARModel.draw(this.conditional(tk, t), rand()));
    return tk;
  }

  /** トークン列 → マス内の一様な位置（量子化を外して連続値に戻す） */
  pointOf(tokens, rand = this.rand) {
    const [i1, i2] = this.binsOf(tokens);
    return [binLo(i1) + rand() * CELL, binLo(i2) + rand() * CELL];
  }

  sample(n, rand = this.rand) {
    const t = this.ensureTables();
    const X = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = this.pointOf(this.sampleTokens(rand, t), rand);
      X[2 * i] = x;
      X[2 * i + 1] = y;
    }
    return X;
  }

  /** 1系列ぶんの条件付き分布とアテンション（可視化用） */
  inspect(tokens) {
    const inp = new Int32Array([BOS, tokens[0], tokens[1], tokens[2]]);
    const P = softmaxRows(this.net.forward(inp, 1), T, V);
    const probs = [];
    for (let t = 0; t < T; t++) probs.push(Array.from(P.subarray(t * V, (t + 1) * V)));
    return { probs, attention: this.net.attention(0) };
  }

  /** 既知のトークンから決まる領域（ビン範囲）。未確定の軸は全範囲 */
  region(prefix) {
    const r = [[0, BINS], [0, BINS]];
    prefix.forEach((tk, k) => {
      const s = this.slots[k];
      if (s.level === 0) r[s.axis] = [tk * V, tk * V + V];
      else r[s.axis] = [r[s.axis][0] + tk, r[s.axis][0] + tk + 1];
    });
    return r;
  }

  /** 位置 k の候補 8 個それぞれの領域 */
  candidates(prefix) {
    return Array.from({ length: V }, (_, j) => this.region([...prefix, j]));
  }
}
