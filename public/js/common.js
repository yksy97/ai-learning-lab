// ページ共通：ナビゲーション、URL パラメータ、学習履歴
import { DATASETS } from './datasets.js';
import { attachGlossary } from './glossary.js';
import { GOALS } from './goals.js';

/** URL の ?dataset= を読み、なければ既定値 */
export function initialDataset(fallback) {
  const q = new URLSearchParams(location.search).get('dataset');
  return q && DATASETS[q] ? q : fallback;
}

/** ページ間を移動しても同じ分布が選ばれるように、ナビのリンクと URL を更新 */
export function syncDataset(dataset) {
  for (const a of document.querySelectorAll('.tabs a, .subtabs a')) {
    const url = new URL(a.getAttribute('href'), location.href);
    url.searchParams.set('dataset', dataset);
    a.setAttribute('href', url.pathname.split('/').pop() + url.search);
  }
  const url = new URL(location.href);
  url.searchParams.set('dataset', dataset);
  history.replaceState(null, '', url);
}

export function fillDatasetSelect(sel, value, noteEl) {
  for (const [k, v] of Object.entries(DATASETS)) sel.add(new Option(v.label, k));
  sel.value = value;
  if (noteEl) noteEl.textContent = DATASETS[value].note;
}

/** 長時間学習しても点数が増えすぎないよう、一定数を超えたら2点ずつ平均して間引く */
export class History {
  constructor(keys, max = 600) {
    this.keys = keys;
    this.max = max;
    this.points = [];
    this.every = 1;
    this.acc = null;
    this.cnt = 0;
  }
  push(r) {
    if (!this.acc) this.acc = Object.fromEntries(this.keys.map((k) => [k, 0]));
    for (const k of this.keys) this.acc[k] += r[k];
    this.cnt++;
    if (this.cnt < this.every) return;
    const p = { step: r.step };
    for (const k of this.keys) p[k] = this.acc[k] / this.cnt;
    this.points.push(p);
    this.acc = null;
    this.cnt = 0;
    if (this.points.length > this.max) {
      const merged = [];
      for (let i = 0; i + 1 < this.points.length; i += 2) {
        const a = this.points[i], b = this.points[i + 1], m = { step: b.step };
        for (const k of this.keys) m[k] = (a[k] + b[k]) / 2;
        merged.push(m);
      }
      this.points = merged;
      this.every *= 2;
    }
  }
}

// ---------------------------------------------------------------------------
// ページ共通のヘッダー（アプリ名・モデルのタブ・ガイド / 詳細の切り替え）
// ---------------------------------------------------------------------------
export const MODELS = [
  { id: 'gan', label: 'GAN', guide: 'gan-guide.html', detail: 'gan.html', kind: 'モデルを動かして学ぶ',
    tagline: '偽札職人と警察の勝負として学ぶ、敵対的生成ネットワーク' },
  { id: 'vae', label: 'VAE', guide: 'vae-guide.html', detail: 'vae.html', kind: 'モデルを動かして学ぶ',
    tagline: '記憶して描き直す模写職人。潜在空間が整っていく様子を見る' },
  { id: 'ar', label: '自己回帰 Transformer', detail: 'ar.html', kind: 'モデルを動かして学ぶ',
    action: '実験する', tagline: '20の質問のように、1トークンずつ位置を絞り込んで作る' },
  { id: 'rag', label: 'RAG', guide: 'rag-guide.html', detail: 'rag.html', kind: 'モデルを動かして学ぶ',
    tagline: '図書館の司書と作家。質問に関係する資料を探してから答える' },
  { id: 'structure', label: '共通のしくみ', detail: 'structure.html', kind: 'モデルを動かして学ぶ',
    action: '骨格を見る', tagline: '4つのモデルを、入力から学習まで同じ枠に並べて見る' },
  { id: 'compare', label: '比較', detail: 'compare.html', kind: 'モデルを動かして学ぶ',
    action: '見比べる', tagline: '同じデータを3つのモデルに学習させ、得意・不得意を見比べる' },
  { id: 'math0', label: 'AIを理解する道具', detail: 'math0.html', kind: '基礎から積む',
    action: '道具を触る', tagline: '確率・log・softmax・ベクトル。AI の式に出てくる道具を、つまみを動かして確かめる' },
  { id: 'learn', label: '学習パス', detail: 'learn.html', kind: '確かめる・調べる',
    action: '順路を見る', tagline: 'どこから始めて、次に何をするか。進み具合と復習を記録する' },
  { id: 'glossary', label: '用語と記号', detail: 'glossary.html', kind: '確かめる・調べる',
    action: '調べる', tagline: '分からない言葉を調べる。意味・たとえ・記号・式の4点セットで引ける' },
  { id: 'drills', label: 'ドリル', detail: 'drills.html', kind: '確かめる・調べる',
    action: '手を動かす', tagline: '理解を確認する。softmax・コサイン類似度・交差エントロピー・KL を手で計算する' },
];

/**
 * ヘッダーを描画する。
 * @param {{id?: string, mode?: 'guide'|'detail', sub?: string}} opt
 */
export function mountHeader(opt = {}) {
  const el = document.getElementById('nav');
  if (!el) return;
  const model = MODELS.find((m) => m.id === opt.id);
  const tabs = MODELS.map((m) => {
    const href = m.guide || m.detail;
    const cur = m.id === opt.id ? ' aria-current="page"' : '';
    return `<a href="${href}"${cur}>${m.label}</a>`;
  }).join('');
  let sub = '';
  if (model && model.guide && model.detail) {
    const a = (href, label, on) => `<a href="${href}"${on ? ' aria-current="page"' : ''}>${label}</a>`;
    sub = `<nav class="subtabs" aria-label="表示の種類">
      ${a(model.guide, 'しくみを知る', opt.mode === 'guide')}
      ${a(model.detail, '実験する', opt.mode === 'detail')}
    </nav>`;
  }
  el.innerHTML = `<div class="brand"><a href="index.html">AI Learning Lab</a>${model ? `<span class="crumb">${model.label}</span>` : ''}</div>
    <nav class="tabs" aria-label="モデル">${tabs}</nav>
    ${sub}
    ${opt.sub ? `<p class="sub">${opt.sub}</p>` : ''}`;
}


/**
 * ページの冒頭に「このページのゴール」を出す。
 * 学習目標を先に見せると、どこに注意して読めばよいかが決まる（先行オーガナイザ）。
 * ヘッダーの直後に差し込むので、ページ側の HTML を変えなくてよい。
 * @param {string} id GOALS のキー
 */
export function mountGoal(id) {
  const g = GOALS[id];
  const header = document.querySelector('header.top');
  if (!g || !header) return;
  const el = document.createElement('section');
  el.className = 'panel goal';
  const links = [
    g.prereq ? `<div><span>先に見ておくと楽</span><a href="${g.prereq.h}">${g.prereq.t}</a></div>` : '',
    g.next ? `<div><span>このあと</span><a href="${g.next.h}">${g.next.t}</a></div>` : '',
  ].join('');
  el.innerHTML = `
    <div class="goal-main">
      <h2>このページのゴール</h2>
      <ul>${g.goals.map((t) => `<li>${t}</li>`).join('')}</ul>
      ${g.pitfalls ? `<details class="pitfall">
        <summary>つまずきやすいところ（${g.pitfalls.length}）</summary>
        ${g.pitfalls.map((t) => `<p>${t}</p>`).join('')}
      </details>` : ''}
    </div>
    <div class="goal-meta">
      <div><span>目安の時間</span><b>${g.minutes} 分</b></div>
      ${links}
    </div>`;
  header.insertAdjacentElement('afterend', el);
}

/**
 * 本文中の専門用語を自動でカードにする。
 * ガイドのように中身が差し替わるページでも効くよう、変化を見て貼り直す。
 */
export function enableGlossary(selectors = ['.story-body', '.term', '.guide', '.hero', '.note', '.answer', '.gl-body', '.ex-model']) {
  const run = () => {
    for (const sel of selectors) {
      for (const el of document.querySelectorAll(sel)) {
        // 文字の長さを目印にする（リンクを貼っても本文の長さは変わらない）
        const sig = String(el.textContent.length);
        if (el.dataset.glSig === sig) continue;
        attachGlossary(el);
        el.dataset.glSig = String(el.textContent.length);
      }
    }
  };
  run();
  let timer = null;
  new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(run, 150);
  }).observe(document.body, { childList: true, subtree: true });
}
