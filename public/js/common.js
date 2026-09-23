// ページ共通：ナビゲーション、URL パラメータ、学習履歴
import { DATASETS } from './datasets.js';

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
  { id: 'gan', label: 'GAN', guide: 'gan-guide.html', detail: 'gan.html', kind: '生成モデル',
    tagline: '偽札職人と警察の勝負として学ぶ、敵対的生成ネットワーク' },
  { id: 'vae', label: 'VAE', guide: 'vae-guide.html', detail: 'vae.html', kind: '生成モデル',
    tagline: '記憶して描き直す模写職人。潜在空間が整っていく様子を見る' },
  { id: 'ar', label: '自己回帰 Transformer', detail: 'ar.html', kind: '生成モデル',
    tagline: '20の質問のように、1トークンずつ位置を絞り込んで作る' },
  { id: 'rag', label: 'RAG', guide: 'rag-guide.html', detail: 'rag.html', kind: '検索と生成',
    tagline: '図書館の司書と作家。質問に関係する資料を探してから答える' },
  { id: 'compare', label: '比較', detail: 'compare.html', kind: 'まとめ',
    tagline: '同じデータを GAN・Transformer・VAE に学習させて並べる' },
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
      ${a(model.guide, 'たとえ話ガイド', opt.mode === 'guide')}
      ${a(model.detail, '詳細', opt.mode === 'detail')}
    </nav>`;
  }
  el.innerHTML = `<div class="brand"><a href="index.html">ML Visualizer</a>${model ? `<span class="crumb">${model.label}</span>` : ''}</div>
    <nav class="tabs" aria-label="モデル">${tabs}</nav>
    ${sub}
    ${opt.sub ? `<p class="sub">${opt.sub}</p>` : ''}`;
}
