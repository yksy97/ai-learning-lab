// 自己回帰 Transformer ページ：状態管理・操作パネル・描画ループ
import { ARModel, AR_DEFAULTS, ORDERS, T, V, BINS, CELL, binLo, toBin, slotName } from './ar.js';
import { DATASETS } from './datasets.js';
import { mulberry32 } from './nn.js';
import { LineChart } from './viz.js';
import { ARSpaceView } from './ar-viz.js';
import { History, initialDataset, syncDataset, fillDatasetSelect, mountHeader, mountGoal, enableGlossary } from './common.js';
import { LiveFormula } from './formula.js';
import { codeBox } from './codebox.js';
import { AR_CODE } from './code-snippets.js';

const $ = (id) => document.getElementById(id);

mountHeader({ id: 'ar', mode: 'detail', sub: '点をトークンの列に変えて、Transformer が「次の1つ」を予測することだけを繰り返します。競う相手はいません' });
mountGoal('ar');
const LR_STEPS = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01];
const N_VIEW = 500;
const TABLE_INTERVAL = 300; // 学習中に確率表を作り直す間隔（ms）
const GEN_DELAY = 650;      // 1点を生成するアニメーションの1トークンあたりの時間（ms）

const state = {
  cfg: { ...AR_DEFAULTS, dataset: initialDataset(AR_DEFAULTS.dataset) },
  running: false,
  speed: 4,
  show: {},
  rateSteps: 0,
  rateT: performance.now(),
  tablesT: 0,
  gen: [],        // 生成中（または調べている）トークン列
  trail: [],      // 1点ずつ生成した点
  genTimer: null,
};

let model, hist, crn, realView, samples;

function reset() {
  stopGen();
  model = new ARModel(state.cfg);
  hist = new History(['nll']);
  // 表示用サンプルは同じ乱数（共通乱数）で引き直すので、学習につれて点が滑らかに移る
  const r = mulberry32(state.cfg.seed * 97 + 3);
  crn = Array.from({ length: N_VIEW }, () => [r(), r(), r(), r(), r(), r()]);
  realView = model.sampleRealPoints(N_VIEW);
  state.gen = [];
  state.trail = [];
  $('paramCount').textContent = `パラメータ数：${model.net.paramCount().toLocaleString()}`;
  $('stH').textContent = model.entropy.toFixed(2);
  refreshTables();
}

function refreshTables() {
  const t = model.computeTables();
  state.tablesT = performance.now();
  space.updateDensity(t.density);
  samples = new Float64Array(N_VIEW * 2);
  for (let i = 0; i < N_VIEW; i++) {
    const u = crn[i];
    const tk = [];
    for (let k = 0; k < T; k++) tk.push(ARModel.draw(model.conditional(tk, t), u[k]));
    const [i1, i2] = model.binsOf(tk);
    samples[2 * i] = binLo(i1) + u[4] * CELL;
    samples[2 * i + 1] = binLo(i2) + u[5] * CELL;
  }
  updatePanels();
  render();
}

// ---- 描画 ----
const space = new ARSpaceView($('main'));
const chart = new LineChart($('chartNll'), {
  series: [{ key: 'nll', label: 'NLL', color: '--ar' }],
  refs: [{ y: 0 }],
  fmt: (v) => v.toFixed(2),
});

const formula = new LiveFormula($('arFormula'), {
  rows: [
    {
      label: '連鎖',
      tpl: 'p(x) = p(t₁) · p(t₂|t₁) · p(t₃|t₁,t₂) · p(t₄|…) = {{p1}} × {{p2}} × {{p3}} × {{p4}} = {{px}}',
      slots: {
        p1: { hl: '#genRows', tip: '1つ目のトークンの確率' },
        p2: { hl: '#genRows', tip: '1つ目を見たうえでの2つ目の確率' },
        p3: { hl: '#genRows', tip: '3つ目' },
        p4: { hl: '#genRows', tip: '4つ目' },
        px: { hl: '#main', tip: 'このマスの確率。背景の濃さ' },
      },
      note: '下の「1点ずつ生成する」で選んだトークン列の確率',
    },
    {
      label: '損失',
      tpl: 'NLL = −log p(x) = {{nll}}　／　下限（データのエントロピー） = {{h}}',
      slots: {
        nll: { hl: '#chartNll', tip: '小さいほど、正解に高い確率を与えられている' },
        h: { hl: '#chartNll', tip: 'どんなモデルでもこれより下がらない' },
      },
    },
    {
      label: '注意',
      tpl: 'Attention(Q,K,V) = softmax(QKᵀ / √d) V　　d = {{d}}、ヘッド数 = {{heads}}',
      slots: {
        d: { hl: '#attn', tip: '埋め込みの次元' },
        heads: { hl: '#attn', tip: '同時に走る「蛍光ペン」の本数' },
      },
    },
  ],
});

function genOverlay() {
  const g = state.gen;
  if (!g.length && !state.genTimer && !state.genArmed) return null;
  const t = model.tables;
  const o = { region: g.length ? model.region(g) : null, complete: g.length === T };
  if (g.length < T) {
    o.candidates = model.candidates(g);
    o.probs = Array.from(model.conditional(g, t));
  }
  return o;
}

function render() {
  space.draw({ show: state.show, real: realView, samples, gen: genOverlay(), trail: state.trail });
  chart.opt.refs = [{ y: model.entropy }];
  chart.draw(hist);
  const g = state.gen;
  const probs = [0, 1, 2, 3].map((k) => (k < g.length ? model.conditional(g.slice(0, k), model.tables)[g[k]] : NaN));
  const px = probs.every((v) => !Number.isNaN(v)) ? probs.reduce((a, b) => a * b, 1) : NaN;
  const nd = (v) => (Number.isNaN(v) ? '–' : v);
  const tail = hist.points[hist.points.length - 1];
  formula.update({
    p1: nd(probs[0]), p2: nd(probs[1]), p3: nd(probs[2]), p4: nd(probs[3]),
    px: Number.isNaN(px) ? '–' : px.toExponential(2),
    nll: tail ? tail.nll : '–', h: model.entropy,
    d: state.cfg.d, heads: state.cfg.heads,
  });
  $('stStep').textContent = model.step.toLocaleString();
  const last = hist.points[hist.points.length - 1];
  $('stNll').textContent = last ? last.nll.toFixed(2) : '–';
}

/** 右側の「1点ずつ生成」とアテンションの表示を更新 */
function updatePanels() {
  const g = state.gen;
  const t = model.tables;
  const rows = $('genRows');
  rows.innerHTML = '';
  let p = 1;
  const factors = [];
  for (let k = 0; k < T; k++) {
    const known = k <= g.length;
    const probs = known ? model.conditional(g.slice(0, k), t) : null;
    const chosen = k < g.length ? g[k] : -1;
    if (chosen >= 0) { p *= probs[chosen]; factors.push(probs[chosen]); }
    const row = document.createElement('div');
    row.className = 'gen-row' + (known ? '' : ' pending');
    const bars = probs
      ? Array.from(probs, (q, j) => `<i class="${j === chosen ? 'on' : ''}" style="height:${Math.max(2, q * 100)}%" title="${j}: ${(q * 100).toFixed(1)}%"></i>`).join('')
      : Array.from({ length: V }, () => '<i style="height:2px"></i>').join('');
    row.innerHTML = `<div class="lbl">t${k + 1}<b>${slotName(model.slots[k])}</b></div>
      <div><div class="bars">${bars}</div><div class="bars-axis">${Array.from({ length: V }, (_, j) => `<span>${j}</span>`).join('')}</div></div>
      <div class="val">${chosen >= 0 ? (probs[chosen] * 100).toFixed(1) + '%' : k === g.length ? '次' : '–'}</div>`;
    rows.appendChild(row);
  }
  const f = $('formula');
  if (g.length) {
    const [i1, i2] = model.binsOf([...g, 0, 0, 0].slice(0, T));
    f.innerHTML = `p = ${factors.map((q) => q.toFixed(3)).join(' × ')} = <b>${p.toExponential(2)}</b>` +
      (g.length === T
        ? `<br>このマス x₁∈[${binLo(i1).toFixed(2)}, ${(binLo(i1) + CELL).toFixed(2)}), x₂∈[${binLo(i2).toFixed(2)}, ${(binLo(i2) + CELL).toFixed(2)}) の確率。−log p = <b>${(-Math.log(p)).toFixed(2)}</b> nats`
        : `<br>残り ${T - g.length} トークン。データ空間の色つきの帯が次の候補。`);
  } else {
    f.innerHTML = '「1点を生成」または「次のトークン」で開始。データ空間をクリックすると、その点の確率の内訳を表示。';
  }
  drawAttention();
}

function drawAttention() {
  const g = state.gen;
  const { attention } = model.inspect([g[0] ?? 0, g[1] ?? 0, g[2] ?? 0]);
  const sub = ['₁', '₂', '₃', '₄'];
  const colLabels = ['BOS', ...[0, 1, 2].map((k) => (k < g.length ? `t${sub[k]}=${g[k]}` : `t${sub[k]}`))];
  const rowsKnown = Math.min(T, g.length + 1);
  const box = $('attn');
  box.innerHTML = '';
  attention.forEach((heads, l) => {
    heads.forEach((A, h) => {
      const fig = document.createElement('figure');
      let html = `<figcaption>層 ${l + 1} ・ ヘッド ${h + 1}</figcaption><table><tr><th></th>${colLabels.map((c) => `<th>${c}</th>`).join('')}</tr>`;
      for (let i = 0; i < T; i++) {
        html += `<tr><th class="r" title="${slotName(model.slots[i])} を予測">→t${sub[i]}</th>`;
        for (let j = 0; j < T; j++) {
          if (j > i || i >= rowsKnown) { html += '<td class="na"></td>'; continue; }
          const w = A[i * T + j];
          html += `<td style="background:color-mix(in oklab, var(--ar) ${(w * 100).toFixed(0)}%, var(--surface-2));color:${w > 0.55 ? '#fff' : 'var(--text-2)'}" title="${(w * 100).toFixed(1)}%">${(w * 100).toFixed(0)}</td>`;
        }
        html += '</tr>';
      }
      fig.innerHTML = html + '</table>';
      box.appendChild(fig);
    });
  });
}

// ---- 1点ずつ生成 ----
function stopGen() {
  if (state.genTimer) clearInterval(state.genTimer);
  state.genTimer = null;
}

function nextToken() {
  if (state.gen.length === T) state.gen = [];
  const t = model.tables;
  state.gen = [...state.gen, ARModel.draw(model.conditional(state.gen, t), Math.random())];
  if (state.gen.length === T) {
    state.trail.push(model.pointOf(state.gen, Math.random));
    if (state.trail.length > 30) state.trail.shift();
  }
  updatePanels();
  render();
}

function playGen() {
  stopGen();
  state.gen = [];
  state.genArmed = true;
  updatePanels();
  render();
  state.genTimer = setInterval(() => {
    nextToken();
    if (state.gen.length === T) stopGen();
  }, GEN_DELAY);
}

// ---- 学習ループ ----
function train(steps) {
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    hist.push(model.trainStep());
    state.rateSteps++;
    if (performance.now() - t0 > 22) break;
  }
}

function loop() {
  const now = performance.now();
  if (state.running) {
    train(state.speed);
    if (now - state.tablesT > TABLE_INTERVAL) refreshTables();
    else render();
  }
  if (now - state.rateT > 500) {
    $('stRate').textContent = state.running ? `${Math.round((state.rateSteps * 1000) / (now - state.rateT))}/秒` : '停止中';
    state.rateSteps = 0;
    state.rateT = now;
  }
  requestAnimationFrame(loop);
}

function setRunning(v) {
  state.running = v;
  $('btnPlay').textContent = v ? '❚❚ 一時停止' : model.step ? '▶ 再開' : '▶ 学習開始';
  if (!v && model.stale) refreshTables();
}

function bindRange(id, outId, get, set, fmt = String) {
  const el = $(id);
  el.value = get();
  const upd = () => { set(+el.value); $(outId).textContent = fmt(+el.value); };
  el.addEventListener('input', upd);
  $(outId).textContent = fmt(+el.value);
}

function init() {
  const dsSel = $('dataset');
  fillDatasetSelect(dsSel, state.cfg.dataset, $('datasetNote'));
  syncDataset(state.cfg.dataset);
  const rebuild = (k, v) => { state.cfg[k] = v; reset(); setRunning(state.running); };
  dsSel.addEventListener('change', () => {
    $('datasetNote').textContent = DATASETS[dsSel.value].note;
    syncDataset(dsSel.value);
    rebuild('dataset', dsSel.value);
  });

  const ordSel = $('order');
  for (const [k, v] of Object.entries(ORDERS)) ordSel.add(new Option(v.label, k));
  ordSel.value = state.cfg.order;
  ordSel.addEventListener('change', () => rebuild('order', ordSel.value));

  const live = (k, v) => { state.cfg[k] = v; if (model) model.cfg[k] = v; };
  bindRange('lr', 'lrOut', () => LR_STEPS.indexOf(state.cfg.lr), (i) => live('lr', LR_STEPS[i]), (i) => String(LR_STEPS[i]));
  $('batch').value = String(state.cfg.batch);
  $('batch').addEventListener('change', (e) => live('batch', +e.target.value));
  bindRange('speed', 'speedOut', () => state.speed, (v) => (state.speed = v));

  $('d').value = String(state.cfg.d);
  $('d').addEventListener('change', (e) => {
    const d = +e.target.value;
    // ヘッド数で割り切れない組み合わせは避ける
    if (d % state.cfg.heads !== 0) { state.cfg.heads = 1; $('heads').value = '1'; }
    rebuild('d', d);
  });
  $('heads').value = String(state.cfg.heads);
  $('heads').addEventListener('change', (e) => {
    const h = +e.target.value;
    if (state.cfg.d % h !== 0) { e.target.value = String(state.cfg.heads); return; }
    rebuild('heads', h);
  });
  $('layers').value = state.cfg.layers;
  $('layersOut').textContent = state.cfg.layers;
  $('layers').addEventListener('input', (e) => { $('layersOut').textContent = e.target.value; });
  $('layers').addEventListener('change', (e) => rebuild('layers', +e.target.value));
  $('seed').value = state.cfg.seed;
  $('seed').addEventListener('change', (e) => rebuild('seed', Math.max(1, Math.round(+e.target.value) || 1)));

  const showMap = { showDensity: 'density', showReal: 'real', showSamples: 'samples', showGrid: 'grid' };
  for (const [id, key] of Object.entries(showMap)) {
    state.show[key] = $(id).checked;
    $(id).addEventListener('change', (e) => { state.show[key] = e.target.checked; render(); });
  }

  $('btnPlay').addEventListener('click', () => setRunning(!state.running));
  $('btnStep').addEventListener('click', () => { setRunning(false); hist.push(model.trainStep()); refreshTables(); });
  $('btnReset').addEventListener('click', () => { reset(); setRunning(false); });
  $('genPlay').addEventListener('click', playGen);
  $('genNext').addEventListener('click', () => { stopGen(); state.genArmed = true; nextToken(); });
  $('genClear').addEventListener('click', () => {
    stopGen(); state.gen = []; state.trail = []; state.genArmed = false; updatePanels(); render();
  });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input[type="number"], select')) return;
    if (e.code === 'Space') { e.preventDefault(); setRunning(!state.running); }
    else if (e.key === 's' || e.key === 'S') $('btnStep').click();
    else if (e.key === 'r' || e.key === 'R') $('btnReset').click();
    else if (e.key === 'g' || e.key === 'G') playGen();
  });

  // ホバー：その位置のマスとトークン、確率
  const tip = $('mainTip');
  const canvas = $('main');
  const cellAt = (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const [x, y] = space.toWorld(px, py);
    return { px, py, x, y, i1: toBin(x), i2: toBin(y), rect };
  };
  canvas.addEventListener('pointermove', (e) => {
    const c = cellAt(e);
    const tk = model.tokensOf(c.i1, c.i2);
    const p = model.tables.density[c.i1 * BINS + c.i2];
    space.hover = [c.px, c.py];
    tip.innerHTML = `<div class="k">x = (${c.x.toFixed(2)}, ${c.y.toFixed(2)})</div>` +
      `<div>トークン [${tk.join(', ')}]</div><div>このマスの確率 <b>${p < 1e-4 ? p.toExponential(1) : (p * 100).toFixed(2) + '%'}</b></div>`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${c.px + 14 + tw > c.rect.width ? c.px - tw - 14 : c.px + 14}px`;
    tip.style.top = `${Math.max(4, c.py - 60)}px`;
    if (!state.running) render();
  });
  canvas.addEventListener('pointerleave', () => { tip.hidden = true; space.hover = null; if (!state.running) render(); });
  canvas.addEventListener('click', (e) => {
    const c = cellAt(e);
    stopGen();
    state.genArmed = true;
    state.gen = model.tokensOf(c.i1, c.i2);
    updatePanels();
    render();
  });

  window.addEventListener('resize', render);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { space.updateDensity(model.tables.density); render(); });

  formula.linkSources({
    '#genRows': ['p1', 'p2', 'p3', 'p4'],
    '#attn': ['d', 'heads'],
    '#chartNll': ['nll', 'h'],
    '.stats .stat:nth-child(3)': ['nll'],
    '.stats .stat:nth-child(4)': ['h'],
  });
  $('main').addEventListener('click', () => formula.focus(['px']));

  codeBox($('arCode'), { items: AR_CODE });
  enableGlossary();
  reset();
  setRunning(false);
  requestAnimationFrame(loop);
}

init();
