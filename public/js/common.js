// ページ共通：ナビゲーション、URL パラメータ、学習履歴
import { DATASETS } from './datasets.js';

/** URL の ?dataset= を読み、なければ既定値 */
export function initialDataset(fallback) {
  const q = new URLSearchParams(location.search).get('dataset');
  return q && DATASETS[q] ? q : fallback;
}

/** ページ間を移動しても同じ分布が選ばれるように、ナビのリンクと URL を更新 */
export function syncDataset(dataset) {
  for (const a of document.querySelectorAll('.tabs a')) {
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
