// RAG ページ：質問 → 検索 → プロンプト組み立て → 回答
import { RagIndex } from './rag.js';
import { PRESET_QUESTIONS } from './rag-corpus.js';
import { RagMap } from './rag-viz.js';
import { mountHeader } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({
  id: 'rag', mode: 'detail',
  sub: '質問に関係する社内規程を探し、それをプロンプトに詰めてから答える（Retrieval-Augmented Generation）',
});

const state = {
  k: 3,
  threshold: 0.2,
  chunkMode: 'doc',
  weighting: 'tfidf',
  expand: true,
  question: PRESET_QUESTIONS[0].q,
  results: null,
  hoverIndex: null,
};

let index;
const map = new RagMap($('map'));

function rebuild() {
  index = new RagIndex({ chunkMode: state.chunkMode, weighting: state.weighting, expand: state.expand });
  index.projection();
  $('corpusNote').textContent = `${index.chunks.length} 個のチャンク、語彙 ${index.vocab.length} 種類（文字 2-gram）。`;
  run();
}

/** 取り出した資料から、質問と語が重なる文を抜き出して回答らしく並べる */
function extractAnswer(question, picked) {
  const qVec = index.queryVector(question).vec; // 語 → 重み
  const scored = [];
  picked.forEach((r, i) => {
    for (const sent of r.chunk.text.split(/(?<=。)/).map((t) => t.trim()).filter(Boolean)) {
      const seen = new Set();
      let score = 0;
      for (let j = 0; j + 1 < sent.length; j++) {
        const t = sent.slice(j, j + 2);
        if (seen.has(t)) continue;
        seen.add(t);
        score += qVec.get(t) || 0; // 質問側で重い語（＝珍しい語）ほど効く
      }
      scored.push({ sent, score, cite: i + 1, rank: i });
    }
  });
  scored.sort((a, b) => b.score - a.score || a.rank - b.rank);
  const take = scored.slice(0, 2).sort((a, b) => a.rank - b.rank || b.score - a.score);
  return take.map((s) => `${s.sent} [${s.cite}]`).join(' ');
}

function run() {
  const q = state.question.trim();
  if (!q) return;
  const res = index.search(q);
  const scores = new Array(index.chunks.length).fill(0);
  for (const r of res) scores[r.index] = r.score;
  const picked = res.filter((r) => r.score >= state.threshold).slice(0, state.k);
  const pickedSet = new Set(picked.map((r) => r.index));
  state.results = { q, res, scores, picked, pickedSet };

  const [qx, qy] = index.projectQuery(q);
  map.draw({ coords: index.coords, chunks: index.chunks, query: [qx, qy], picked: pickedSet, scores });

  // 検索結果の一覧
  $('results').innerHTML = res.slice(0, 12).map((r, i) => {
    const on = pickedSet.has(r.index);
    const pct = Math.round(r.score * 100);
    return `<div class="result ${on ? 'on' : ''}" data-index="${r.index}">
      <div class="r-head"><b>${on ? `[${picked.findIndex((p) => p.index === r.index) + 1}] ` : ''}${r.chunk.title}</b><span>${r.score.toFixed(3)}</span></div>
      <div class="r-bar"><i style="width:${Math.min(100, pct * 2)}%"></i></div>
      <div class="r-text">${r.chunk.text}</div>
      ${r.terms.length ? `<div class="r-terms">一致した語：${r.terms.slice(0, 6).map((t) => `<code>${t.term}</code>`).join(' ')}</div>` : '<div class="r-terms">一致した語：なし</div>'}
    </div>`;
  }).join('');

  // プロンプトと回答
  const prompt = RagIndex.buildPrompt(q, picked);
  $('prompt').textContent = prompt;
  const top = res[0] ? res[0].score : 0;
  if (!picked.length) {
    $('answer').innerHTML = `<b>資料には記載がありません。</b><br>いちばん近い資料でも類似度 ${top.toFixed(3)} で、設定した下限 ${state.threshold.toFixed(2)} に届きませんでした。`;
    $('answerHint').textContent = '該当なし';
  } else {
    $('answer').innerHTML = extractAnswer(q, picked) +
      `<div class="cites">${picked.map((r, i) => `[${i + 1}] ${r.chunk.title}`).join('　')}</div>`;
    $('answerHint').textContent = `資料 ${picked.length} 件を根拠にしています`;
  }
}

function init() {
  $('question').value = state.question;
  $('presets').innerHTML = PRESET_QUESTIONS
    .map((p, i) => `<button class="preset" data-i="${i}">${p.q}<span>${p.note}</span></button>`).join('');
  $('presets').addEventListener('click', (e) => {
    const b = e.target.closest('.preset');
    if (!b) return;
    state.question = PRESET_QUESTIONS[+b.dataset.i].q;
    $('question').value = state.question;
    run();
  });
  $('ask').addEventListener('click', () => { state.question = $('question').value; run(); });
  $('question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { state.question = $('question').value; run(); }
  });

  const bind = (id, outId, get, set, fmt) => {
    const el = $(id);
    el.value = get();
    el.addEventListener('input', () => { set(+el.value); $(outId).textContent = fmt(+el.value); run(); });
    $(outId).textContent = fmt(+el.value);
  };
  bind('k', 'kOut', () => state.k, (v) => (state.k = v), String);
  bind('th', 'thOut', () => state.threshold * 100, (v) => (state.threshold = v / 100), (v) => (v / 100).toFixed(2));
  $('chunkMode').addEventListener('change', (e) => { state.chunkMode = e.target.value; rebuild(); });
  $('weighting').addEventListener('change', (e) => { state.weighting = e.target.value; rebuild(); });
  $('expand').addEventListener('change', (e) => { state.expand = e.target.checked; rebuild(); });

  $('results').addEventListener('pointerover', (e) => {
    const r = e.target.closest('.result');
    map.hover = r ? +r.dataset.index : null;
    if (state.results) map.draw({ coords: index.coords, chunks: index.chunks, query: index.projectQuery(state.results.q), picked: state.results.pickedSet, scores: state.results.scores });
  });

  const canvas = $('map');
  const tip = $('mapTip');
  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const i = map.pick(px, py, index.coords);
    map.hover = i >= 0 ? i : null;
    if (i >= 0) {
      const c = index.chunks[i];
      tip.innerHTML = `<div class="k">${c.title}</div><div style="max-width:260px;white-space:normal">${c.text.slice(0, 60)}…</div>` +
        (state.results ? `<div>類似度 <b>${state.results.scores[i].toFixed(3)}</b></div>` : '');
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = `${px + 14 + tw > rect.width ? px - tw - 14 : px + 14}px`;
      tip.style.top = `${Math.max(4, py - 50)}px`;
    } else {
      tip.hidden = true;
    }
    if (state.results) map.draw({ coords: index.coords, chunks: index.chunks, query: index.projectQuery(state.results.q), picked: state.results.pickedSet, scores: state.results.scores });
  });
  canvas.addEventListener('pointerleave', () => { tip.hidden = true; map.hover = null; if (state.results) run(); });
  window.addEventListener('resize', () => { if (state.results) run(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.results) run(); });

  rebuild();
}

init();
