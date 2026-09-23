// 手計算ドリル。小さい数字で1回自分で計算すると、式が「動くもの」になる。
// 問題は毎回作り直す（数字が変わるので、答えを覚えるのではなく手順を覚える）。

const r = (lo, hi, d = 1) => +(lo + Math.random() * (hi - lo)).toFixed(d);
const sum = (a) => a.reduce((x, y) => x + y, 0);
const mono = (s) => `<span class="mono">${s}</span>`;

export const DRILLS = [
  {
    id: 'softmax',
    label: 'ソフトマックス',
    intro: 'ロジット（好きな実数）を、合計 1 の確率に直す計算です。Transformer が次のトークンを選ぶときに毎回やっています。',
    make() {
      const z = [r(-1, 3), r(-1, 3), r(-1, 3)];
      const e = z.map(Math.exp);
      const s = sum(e);
      const p = e.map((v) => v / s);
      return {
        question: `ロジット z = ${mono(`[${z.join(', ')}]`)} のとき、1番目の確率 p₁ を求めてください（小数第2位まで）。`,
        inputs: [{ label: 'p₁', answer: p[0], tol: 0.02 }],
        steps: [
          `まず全部を exp する：${mono(`exp(${z[0]}) = ${e[0].toFixed(3)}, exp(${z[1]}) = ${e[1].toFixed(3)}, exp(${z[2]}) = ${e[2].toFixed(3)}`)}`,
          `合計を出す：${mono(`${e[0].toFixed(3)} + ${e[1].toFixed(3)} + ${e[2].toFixed(3)} = ${s.toFixed(3)}`)}`,
          `1番目を合計で割る：${mono(`${e[0].toFixed(3)} / ${s.toFixed(3)} = ${p[0].toFixed(3)}`)}`,
          `確かめ：3つ足すと ${mono((sum(p)).toFixed(3))} になる（必ず 1）。`,
        ],
        formula: 'softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)',
      };
    },
  },
  {
    id: 'cosine',
    label: 'コサイン類似度',
    intro: 'RAG の検索で、質問と資料の近さを測る計算です。内積を、それぞれの長さで割ります。',
    make() {
      const q = [Math.round(r(1, 5, 0)), Math.round(r(0, 5, 0))];
      const d = [Math.round(r(1, 5, 0)), Math.round(r(0, 5, 0))];
      const dot = q[0] * d[0] + q[1] * d[1];
      const nq = Math.hypot(...q), nd = Math.hypot(...d);
      return {
        question: `質問ベクトル q = ${mono(`(${q.join(', ')})`)}、資料ベクトル d = ${mono(`(${d.join(', ')})`)} のコサイン類似度は？（小数第2位まで）`,
        inputs: [{ label: 'cos(q, d)', answer: dot / (nq * nd), tol: 0.02 }],
        steps: [
          `内積：${mono(`${q[0]}×${d[0]} + ${q[1]}×${d[1]} = ${dot}`)}`,
          `q の長さ：${mono(`√(${q[0]}² + ${q[1]}²) = ${nq.toFixed(3)}`)}`,
          `d の長さ：${mono(`√(${d[0]}² + ${d[1]}²) = ${nd.toFixed(3)}`)}`,
          `割る：${mono(`${dot} / (${nq.toFixed(3)} × ${nd.toFixed(3)}) = ${(dot / (nq * nd)).toFixed(3)}`)}`,
          '向きが同じなら 1、直角なら 0。長さは効かない、というのがポイント。',
        ],
        formula: 'cos(q, d) = (q · d) / (‖q‖ ‖d‖)',
      };
    },
  },
  {
    id: 'ce',
    label: '交差エントロピー',
    intro: '「正解にどれだけ確率を割り当てられたか」の罰点です。Transformer の学習はこれを小さくしているだけです。',
    make() {
      const raw = [r(0.1, 0.6, 2), r(0.1, 0.6, 2), r(0.1, 0.6, 2)];
      const s = sum(raw);
      const p = raw.map((v) => +(v / s).toFixed(2));
      const idx = Math.floor(Math.random() * 3);
      const loss = -Math.log(p[idx]);
      return {
        question: `モデルの予測が ${mono(`p = [${p.join(', ')}]`)} で、正解が ${idx + 1} 番目のとき、この1件の損失は？（小数第2位まで）`,
        inputs: [{ label: '−log p(正解)', answer: loss, tol: 0.05 }],
        steps: [
          `正解の確率だけを見る：${mono(`p(${idx + 1}) = ${p[idx]}`)}`,
          `その log を取る：${mono(`log(${p[idx]}) = ${Math.log(p[idx]).toFixed(3)}`)}`,
          `符号を反転：${mono(`−(${Math.log(p[idx]).toFixed(3)}) = ${loss.toFixed(3)}`)}`,
          '正解の確率が 1 なら損失は 0。0 に近づくほど損失は無限に大きくなる。',
        ],
        formula: 'L = −log p(正解)',
      };
    },
  },
  {
    id: 'kl',
    label: 'KL（VAE の形）',
    intro: 'VAE の KL 項です。エンコーダの分布 N(μ, σ²) を標準正規分布 N(0, 1) にどれだけ近いかで測ります。',
    make() {
      const mu = r(-1.5, 1.5, 2);
      const sd = r(0.3, 1.6, 2);
      const kl = 0.5 * (sd * sd + mu * mu - 1 - 2 * Math.log(sd));
      return {
        question: `q = N(μ=${mono(String(mu))}, σ=${mono(String(sd))}) と p = N(0, 1) の KL は？（小数第2位まで）`,
        inputs: [{ label: 'KL(q‖p)', answer: kl, tol: 0.05 }],
        steps: [
          `式に入れる：${mono(`0.5 × (σ² + μ² − 1 − 2 log σ)`)}`,
          `σ² = ${mono((sd * sd).toFixed(3))}、μ² = ${mono((mu * mu).toFixed(3))}、log σ = ${mono(Math.log(sd).toFixed(3))}`,
          `${mono(`0.5 × (${(sd * sd).toFixed(3)} + ${(mu * mu).toFixed(3)} − 1 − ${(2 * Math.log(sd)).toFixed(3)}) = ${kl.toFixed(3)}`)}`,
          'μ=0, σ=1 のとき KL = 0。事後崩壊は、この値がほぼ 0 に張り付いた状態。',
        ],
        formula: 'KL(N(μ,σ²) ‖ N(0,1)) = ½(σ² + μ² − 1 − 2 log σ)',
      };
    },
  },
  {
    id: 'gan',
    label: 'GAN の損失',
    intro: 'D の判定から、その回の損失を出します。GAN の画面に出ている数字は、これを 64 個平均したものです。',
    make() {
      const dr = r(0.3, 0.95, 2), df = r(0.05, 0.7, 2);
      const ld = -Math.log(dr) - Math.log(1 - df);
      return {
        question: `本物への判定が ${mono(`D(x) = ${dr}`)}、偽物への判定が ${mono(`D(G(z)) = ${df}`)} のとき、D の損失は？（小数第2位まで）`,
        inputs: [{ label: 'L_D', answer: ld, tol: 0.05 }],
        steps: [
          `本物の項：${mono(`−log(${dr}) = ${(-Math.log(dr)).toFixed(3)}`)}（本物を本物と見抜けているほど小さい）`,
          `偽物の項：${mono(`−log(1 − ${df}) = ${(-Math.log(1 - df)).toFixed(3)}`)}（偽物を偽物と見抜けているほど小さい）`,
          `足す：${mono(`${(-Math.log(dr)).toFixed(3)} + ${(-Math.log(1 - df)).toFixed(3)} = ${ld.toFixed(3)}`)}`,
          `ちなみに、見分けがつかない均衡（D = 0.5）だと ${mono('2 × log 2 = 1.386')}。画面の点線がこれ。`,
        ],
        formula: 'L_D = −log D(x) − log(1 − D(G(z)))',
      };
    },
  },
];
