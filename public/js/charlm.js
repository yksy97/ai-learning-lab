// 文字単位の言語モデル。LLM そのものではないが、心臓部（次の1つを確率で予測する）は同じ。
// 語彙を「文字」にすることで、ブラウザの中で最後まで学習させられる大きさに収めている。
import { Transformer, softmaxRows } from './transformer.js';
import { mulberry32 } from './nn.js';

// 題材：短い定型文。同じ言い回しが繰り返し出てくるので、小さなモデルでも型を覚えられる。
export const CORPUS = [
  'おはようございます。',
  'おつかれさまです。',
  'よろしくおねがいします。',
  'ありがとうございます。',
  'しつれいします。',
  'かしこまりました。',
  'しょうちいたしました。',
  'ただいまかくにんいたします。',
  'すこしおまちください。',
  'おまたせいたしました。',
  'ごめいわくをおかけしました。',
  'もうしわけございません。',
  'たいへんもうしわけございません。',
  'ごかくにんおねがいします。',
  'ごれんらくありがとうございます。',
  'おせわになっております。',
  'いつもおせわになっております。',
  'ほんじつはよろしくおねがいします。',
  'あすのかいぎはじゅうじからです。',
  'かいぎしつはさんかいです。',
  'しりょうをおくりします。',
  'しりょうをかくにんしました。',
  'ないようをかくにんいたします。',
  'へんじがおそくなりました。',
  'ごかくにんいただけますでしょうか。',
  'おてすうをおかけします。',
  'おてすうですがおねがいします。',
  'あしたまでにおくります。',
  'らいしゅうまでにしあげます。',
  'すこしじかんをください。',
  'いまてがはなせません。',
  'のちほどごれんらくします。',
  'のちほどかくにんいたします。',
  'ごしつもんありがとうございます。',
  'しつもんがございます。',
  'すこしおしえてください。',
  'ここがわかりませんでした。',
  'たすかりました。ありがとうございます。',
  'ではまたあした。',
  'おさきにしつれいします。',
  'いってきます。',
  'いってらっしゃい。',
  'おきをつけてください。',
  'それではよろしくおねがいします。',
];

export const CHARLM_DEFAULTS = {
  seqLen: 16, d: 32, heads: 2, layers: 1, lr: 0.004, batch: 16, seed: 1,
};

export class CharLM {
  constructor(cfg = {}) {
    this.cfg = { ...CHARLM_DEFAULTS, ...cfg };
    const c = this.cfg;
    this.rand = mulberry32(c.seed * 104729 + 7);

    // 1文ずつ独立に学ばせる。全文をつなげてしまうと「文の並び順」まで覚えてしまい、
    // 文の始まりがいつも同じになって、温度を変えても結果が変わらなくなる。
    this.text = CORPUS.join('\n') + '\n';
    this.chars = [...new Set(this.text)].sort();
    this.stoi = new Map(this.chars.map((ch, i) => [ch, i]));
    this.V = this.chars.length;
    this.NL = this.stoi.get('\n');

    // 各文の前後を改行で埋めた列。先頭の窓も作れるように、頭に seqLen ぶんの改行を置く
    this.lines = CORPUS.map((line) =>
      Int32Array.from([...('\n'.repeat(c.seqLen) + line + '\n')].map((ch) => this.stoi.get(ch))));
    this.data = Int32Array.from([...this.text].map((ch) => this.stoi.get(ch)));
    this.net = new Transformer(
      { vocab: this.V, seqLen: c.seqLen, d: c.d, heads: c.heads, layers: c.layers, ff: 2 * c.d },
      this.rand,
    );
    this.step = 0;
    this.last = null;
    this.entropy = this.dataEntropy();
  }

  /** 文字 → 番号。知らない文字は改行あつかい（語彙にない文字は学習していない） */
  encode(s) {
    return [...s].map((ch) => (this.stoi.has(ch) ? this.stoi.get(ch) : this.stoi.get('\n')));
  }

  decode(ids) {
    return ids.map((i) => this.chars[i]).join('');
  }

  /** 題材そのものが持つばらつき。2文字前までを見て次を当てる場合の下限の目安 */
  dataEntropy(order = 2) {
    const counts = new Map();
    for (const line of this.lines) {
      for (let i = order; i < line.length; i++) {
        const key = Array.from(line.slice(i - order, i)).join(',');
        if (!counts.has(key)) counts.set(key, new Map());
        const m = counts.get(key);
        m.set(line[i], (m.get(line[i]) || 0) + 1);
      }
    }
    let total = 0, sum = 0;
    for (const m of counts.values()) {
      const n = [...m.values()].reduce((a, b) => a + b, 0);
      for (const c of m.values()) sum -= c * Math.log(c / n);
      total += n;
    }
    return sum / total;
  }

  /**
   * 学習用の窓を1つ取る（1文の中だけを見る）。
   * 最後の「。→ 改行」も必ず学習対象に入るよう、開始位置の上限に注意する。
   */
  sampleWindow(T) {
    const line = this.lines[Math.floor(this.rand() * this.lines.length)];
    const s = Math.floor(this.rand() * Math.max(1, line.length - T));
    return { line, s };
  }

  trainStep() {
    const { batch: B, seqLen: T, lr } = this.cfg;
    const V = this.V;
    const inp = new Int32Array(B * T);
    const tgt = new Int32Array(B * T);
    for (let b = 0; b < B; b++) {
      const { line, s } = this.sampleWindow(T);
      for (let t = 0; t < T; t++) {
        inp[b * T + t] = line[s + t];
        tgt[b * T + t] = line[s + t + 1];
      }
    }
    const net = this.net;
    net.zeroGrad();
    const logits = net.forward(inp, B);
    const P = softmaxRows(logits, B * T, V);
    const d = new Float64Array(B * T * V);
    let nll = 0;
    for (let i = 0; i < B * T; i++) {
      const o = i * V;
      nll -= Math.log(Math.max(P[o + tgt[i]], 1e-12)) / (B * T);
      for (let j = 0; j < V; j++) d[o + j] = (P[o + j] - (j === tgt[i] ? 1 : 0)) / (B * T);
    }
    net.backward(d);
    net.adamStep(lr);
    this.step++;
    this.last = { step: this.step, nll };
    return this.last;
  }

  /**
   * 続きの1文字の確率を返す。
   * @param {string} prefix 直前までの文字列（長いときは末尾 seqLen 文字だけ使う）
   * @returns {Float64Array} 語彙ぶんの確率
   */
  nextProbs(prefix) {
    const T = this.cfg.seqLen;
    let ids = this.encode(prefix.length ? prefix : '\n');
    if (ids.length > T) ids = ids.slice(-T);
    const pad = T - ids.length;
    // 足りないぶんは先頭を改行で埋める（文の先頭という合図になる）
    const inp = Int32Array.from([...new Array(pad).fill(this.stoi.get('\n')), ...ids]);
    const logits = this.net.forward(inp, 1);
    const P = softmaxRows(logits, T, this.V);
    const o = (T - 1) * this.V;          // 最後の位置の予測が「次の1文字」
    return P.slice(o, o + this.V);
  }

  /** 温度をかけた分布。T→0 で一番高いものだけ、T が大きいほど平らになる */
  withTemperature(probs, temp) {
    const out = new Float64Array(probs.length);
    if (temp <= 0.01) {
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      out[best] = 1;
      return out;
    }
    let max = -Infinity;
    const logp = new Float64Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      logp[i] = Math.log(Math.max(probs[i], 1e-12)) / temp;
      if (logp[i] > max) max = logp[i];
    }
    let s = 0;
    for (let i = 0; i < probs.length; i++) { out[i] = Math.exp(logp[i] - max); s += out[i]; }
    for (let i = 0; i < probs.length; i++) out[i] /= s;
    return out;
  }

  sampleFrom(probs, rand = this.rand) {
    let r = rand(), acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return i;
    }
    return probs.length - 1;
  }

  /** 1文字ずつ継ぎ足して生成する。各ステップの様子も返す */
  generate(prefix, n, temp, rand = this.rand) {
    let s = prefix;
    const trace = [];
    for (let i = 0; i < n; i++) {
      const raw = this.nextProbs(s);
      const p = this.withTemperature(raw, temp);
      const id = this.sampleFrom(p, rand);
      trace.push({ ch: this.chars[id], p: p[id], raw: raw[id] });
      s += this.chars[id];
      if (this.chars[id] === '\n') break;
    }
    return { text: s, trace };
  }

  /** 上位 k 件を [{ch, p}] で返す */
  top(probs, k = 8) {
    return [...probs]
      .map((p, i) => ({ ch: this.chars[i], p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, k);
  }
}
