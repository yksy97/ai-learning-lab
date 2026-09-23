// RAG の検索部分：日本語の文書を数字のベクトルにして、質問に近い文書を探す
//   ・分かち書きの代わりに「文字 2-gram」を使う（日本語の検索でよく使われる方法）
//   ・重み付けは TF-IDF（珍しい語ほど重くする）
//   ・類似度はコサイン類似度
//   ・2次元の地図は、ベクトルの主成分分析（べき乗法で2成分）で作る
import { CORPUS } from './rag-corpus.js';

const ASCII = /[A-Za-z0-9]+/g;

/** 文字 2-gram ＋ 英数字の単語に分解する */
export function tokenize(text) {
  const s = text.replace(/[\s、。，．,.「」『』（）()：:；;・？?！!／/]+/g, ' ');
  const out = [];
  for (const w of s.match(ASCII) || []) out.push(w.toLowerCase());
  const plain = s.replace(ASCII, ' ').replace(/\s+/g, '');
  for (let i = 0; i + 1 < plain.length; i++) out.push(plain.slice(i, i + 2));
  if (plain.length === 1) out.push(plain);
  return out;
}

/** 文書を分割する。'doc' は1件そのまま、'sentence' は文ごと */
export function chunkCorpus(mode = 'doc') {
  const chunks = [];
  for (const d of CORPUS) {
    if (mode === 'sentence') {
      const parts = d.text.split(/(?<=。)/).map((t) => t.trim()).filter(Boolean);
      parts.forEach((t, i) => chunks.push({ id: `${d.id}#${i + 1}`, title: d.title, text: t, source: d.id }));
    } else {
      chunks.push({ id: d.id, title: d.title, text: d.text, source: d.id });
    }
  }
  return chunks;
}

/**
 * 言い換えの表。
 * 本物の RAG は「意味のベクトル（埋め込み）」を使うので、言い方が違っても近いと判断できる。
 * ここではその代わりに、小さな言い換え表で「意味の近さ」をまねる。
 */
export const SYNONYMS = [
  ['生成AI', 'ChatGPT', 'チャットGPT', '生成エーアイ', 'AI', 'LLM', '大規模言語モデル'],
  ['領収書', 'レシート', '受領書'],
  ['年次有給休暇', '有給', '有休', '年休', '有給休暇'],
  ['在宅勤務', 'テレワーク', 'リモートワーク', '在宅'],
  ['宿泊費', 'ホテル代', '宿泊代', '宿泊'],
  ['交通費', '旅費', '運賃'],
  ['時間外労働', '残業', '超過勤務'],
  ['健康診断', '健診', '人間ドック'],
  ['パスワード', 'パスコード', '暗証番号'],
  ['インシデント', '事故', 'トラブル', '紛失'],
  ['所属長', '上司', '上長', '部長'],
  ['備品', '事務用品', '文房具'],
  ['配送', '運送', '輸送', '納品'],
];

/** 質問に含まれる語の「言い換え」を、重みを下げて足す */
export function expandQuery(q) {
  const hits = [];
  for (const group of SYNONYMS) {
    if (group.some((w) => q.includes(w))) {
      for (const w of group) if (!q.includes(w)) hits.push(w);
    }
  }
  return hits;
}

export class RagIndex {
  /**
   * @param {{chunkMode?: 'doc'|'sentence', weighting?: 'tfidf'|'count'}} opt
   */
  constructor(opt = {}) {
    this.opt = { chunkMode: 'doc', weighting: 'tfidf', expand: true, ...opt };
    this.chunks = chunkCorpus(this.opt.chunkMode);
    this.build();
  }

  build() {
    const df = new Map();
    this.docTf = this.chunks.map((c) => {
      const tf = new Map();
      for (const t of tokenize(`${c.title} ${c.text}`)) tf.set(t, (tf.get(t) || 0) + 1);
      for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
      return tf;
    });
    this.df = df;
    this.N = this.chunks.length;
    this.vocab = [...df.keys()];
    this.idf = new Map(this.vocab.map((t) => [t, Math.log((this.N + 1) / (df.get(t) + 1)) + 1]));
    this.vectors = this.docTf.map((tf) => this.vectorize(tf));
    this.coords = null;
  }

  /** 語の出現数 → 重み付きの正規化ベクトル（Map） */
  vectorize(tf) {
    const v = new Map();
    let norm = 0;
    for (const [t, n] of tf) {
      const idf = this.idf.get(t);
      if (idf === undefined) continue; // 資料にない語は無視される
      const w = this.opt.weighting === 'count' ? n : (1 + Math.log(n)) * idf;
      v.set(t, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [t, w] of v) v.set(t, w / norm);
    return v;
  }

  queryVector(q) {
    const tf = new Map();
    for (const t of tokenize(q)) tf.set(t, (tf.get(t) || 0) + 1);
    const added = this.opt.expand ? expandQuery(q) : [];
    for (const w of added) {
      // 言い換えは本人の言葉より軽く扱う
      for (const t of tokenize(w)) tf.set(t, (tf.get(t) || 0) + 0.6);
    }
    return { vec: this.vectorize(tf), tf, expanded: added };
  }

  static cosine(a, b) {
    let s = 0;
    const [small, big] = a.size < b.size ? [a, b] : [b, a];
    for (const [t, w] of small) {
      const w2 = big.get(t);
      if (w2 !== undefined) s += w * w2;
    }
    return s;
  }

  /**
   * 質問に近い順に並べる。
   * @returns {{chunk, score, terms: {term: string, contrib: number}[]}[]}
   */
  search(q) {
    const { vec } = this.queryVector(q);
    return this.chunks
      .map((chunk, i) => {
        const dv = this.vectors[i];
        let score = 0;
        const terms = [];
        for (const [t, w] of vec) {
          const w2 = dv.get(t);
          if (w2 === undefined) continue;
          score += w * w2;
          terms.push({ term: t, contrib: w * w2 });
        }
        terms.sort((a, b) => b.contrib - a.contrib);
        return { chunk, score, terms: terms.slice(0, 8), index: i };
      })
      .sort((a, b) => b.score - a.score);
  }

  /** 検索結果からプロンプトを組み立てる（実際に LLM へ渡す形） */
  static buildPrompt(question, picked) {
    const docs = picked
      .map((r, i) => `[${i + 1}] ${r.chunk.title}\n${r.chunk.text}`)
      .join('\n\n');
    return `あなたは社内規程に詳しいアシスタントです。以下の資料だけを根拠に、質問に日本語で答えてください。資料に書かれていないことは「資料には記載がありません」と答えてください。答えの文末には、使った資料の番号を [1] のように付けてください。

# 資料
${docs || '（該当する資料が見つかりませんでした）'}

# 質問
${question}`;
  }

  /**
   * 文書どうしの「近さ」をなるべく保つように2次元へ配置する（古典的 MDS）。
   * 類似度行列を中心化し、上位2つの固有ベクトルをべき乗法で求める。
   */
  projection() {
    if (this.coords) return this.coords;
    const n = this.N;
    const S = [];
    for (let i = 0; i < n; i++) {
      S.push(new Float64Array(n));
      for (let j = 0; j < n; j++) S[i][j] = RagIndex.cosine(this.vectors[i], this.vectors[j]);
    }
    // 二重中心化
    const rowMean = new Float64Array(n);
    let all = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += S[i][j];
      rowMean[i] = s / n;
      all += s / (n * n);
    }
    const B = [];
    for (let i = 0; i < n; i++) {
      B.push(new Float64Array(n));
      for (let j = 0; j < n; j++) B[i][j] = S[i][j] - rowMean[i] - rowMean[j] + all;
    }
    // 上位2つの固有ベクトル
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) - 0.5;
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = rnd();
      let lambda = 0;
      for (let it = 0; it < 200; it++) {
        const nv = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let j = 0; j < n; j++) s += B[i][j] * v[j];
          nv[i] = s;
        }
        for (const p of comps) {
          let d = 0;
          for (let i = 0; i < n; i++) d += nv[i] * p.v[i];
          for (let i = 0; i < n; i++) nv[i] -= d * p.v[i];
        }
        let norm = 0;
        for (let i = 0; i < n; i++) norm += nv[i] * nv[i];
        norm = Math.sqrt(norm) || 1;
        lambda = norm;
        for (let i = 0; i < n; i++) nv[i] /= norm;
        v = nv;
      }
      comps.push({ v, lambda });
    }
    const coords = [];
    for (let i = 0; i < n; i++) {
      coords.push(comps.map((c) => c.v[i] * Math.sqrt(Math.max(c.lambda, 0))));
    }
    let mx = 1e-9;
    for (const [a2, b2] of coords) mx = Math.max(mx, Math.abs(a2), Math.abs(b2));
    this.coords = coords.map(([a2, b2]) => [a2 / mx, b2 / mx]);
    return this.coords;
  }

  /**
   * 質問を同じ地図の上に置く。
   * 近い資料ほど強く引き寄せられる位置（類似度で重みづけした平均）に置く。
   */
  projectQuery(q) {
    const co = this.projection();
    const { vec } = this.queryVector(q);
    let wsum = 0, x = 0, y = 0, best = 0;
    for (let i = 0; i < this.N; i++) {
      const s = RagIndex.cosine(vec, this.vectors[i]);
      best = Math.max(best, s);
      const w = Math.pow(Math.max(s, 0), 3);
      wsum += w;
      x += w * co[i][0];
      y += w * co[i][1];
    }
    if (wsum < 1e-9) return [0, 0, best]; // どの資料にも似ていない → 地図の中央
    return [x / wsum, y / wsum, best];
  }
}
