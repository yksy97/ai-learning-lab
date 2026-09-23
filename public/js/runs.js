// 実験の記録（スナップショット）。このブラウザの中だけに保存する。
// 「変える前」と「変えた後」を並べられないと、実験は考察にならない。
const KEY = 'mlviz.runs.v1';
const MAX = 6;

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

function save(all) {
  try {
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    // 容量が足りないときは古いものを捨てて1回だけ再試行する
    try {
      for (const k of Object.keys(all)) all[k] = all[k].slice(-2);
      localStorage.setItem(KEY, JSON.stringify(all));
    } catch { /* 保存できなくても実験自体は続けられる */ }
  }
}

/** @param {{label:string, img:string, stats:{k:string,v:string}[], config:Object, step:number}} run */
export function saveRun(pageId, run) {
  const all = load();
  const list = all[pageId] || [];
  list.push({ ...run, id: `r${Date.now().toString(36)}`, at: Date.now() });
  all[pageId] = list.slice(-MAX);
  save(all);
  return all[pageId];
}

export function listRuns(pageId) {
  return load()[pageId] || [];
}

export function removeRun(pageId, id) {
  const all = load();
  all[pageId] = (all[pageId] || []).filter((r) => r.id !== id);
  save(all);
  return all[pageId];
}

export function clearRuns(pageId) {
  const all = load();
  delete all[pageId];
  save(all);
}

/** 2つの実験の設定を比べて、違う項目だけを返す */
export function diffConfig(a, b) {
  const keys = [...new Set([...Object.keys(a || {}), ...Object.keys(b || {})])];
  return keys.filter((k) => String(a[k]) !== String(b[k])).map((k) => ({ k, a: a[k], b: b[k] }));
}

/** 考察（何が違った／なぜ）の保存 */
export function saveNote(pageId, pair, note) {
  const all = load();
  all.notes = all.notes || {};
  all.notes[`${pageId}:${pair}`] = { ...note, at: Date.now() };
  save(all);
}

export function getNote(pageId, pair) {
  return (load().notes || {})[`${pageId}:${pair}`] || null;
}
