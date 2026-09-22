// アプリ本体：状態管理・操作パネル・描画ループ
import { GAN, DEFAULTS } from './gan.js';
import { DATASETS } from './datasets.js';
import { mulberry32, gaussian } from './nn.js';
import { SpaceView, LineChart, drawLatent, zColor } from './viz.js';

const $ = (id) => document.getElementById(id);
const LR_STEPS = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01];
const N_VIEW = 500;       // 表示用サンプル数
const GRID_MAX = 2.5;     // 潜在空間の格子の範囲
const GRID_VALS = Array.from({ length: 11 }, (_, i) => -GRID_MAX + i * 0.5);

// ---- 履歴（長時間学習しても点数が増えすぎないよう間引き平均） ----
class History {
  constructor() { this.points = []; this.every = 1; this.acc = null; this.cnt = 0; }
  push(r) {
    if (!this.acc) this.acc = { lossD: 0, lossG: 0, dReal: 0, dFake: 0 };
    for (const k in this.acc) this.acc[k] += r[k];
    this.cnt++;
    if (this.cnt >= this.every) {
      const p = { step: r.step };
      for (const k in this.acc) p[k] = this.acc[k] / this.cnt;
      this.points.push(p);
      this.acc = null; this.cnt = 0;
      if (this.points.length > 600) {
        const merged = [];
        for (let i = 0; i + 1 < this.points.length; i += 2) {
          const a = this.points[i], b = this.points[i + 1], m = { step: b.step };
          for (const k of ['lossD', 'lossG', 'dReal', 'dFake']) m[k] = (a[k] + b[k]) / 2;
          merged.push(m);
        }
        this.points = merged;
        this.every *= 2;
      }
    }
  }
}

// ---- 状態 ----
const state = {
  cfg: { ...DEFAULTS },
  running: false,
  speed: 8,
  show: {},
  frame: 0,
  rateSteps: 0,
  rateT: performance.now(),
};

let gan, hist, viewZ, jitter, fakeColors, realView;

function reset() {
  gan = new GAN(state.cfg);
  hist = new History();
  const r = mulberry32(state.cfg.seed * 31 + 5);
  const d = state.cfg.noiseDim;
  viewZ = new Float64Array(N_VIEW * d);
  for (let i = 0; i < viewZ.length; i++) viewZ[i] = gaussian(r);
  jitter = Array.from({ length: N_VIEW }, () => r());
  fakeColors = Array.from({ length: N_VIEW }, (_, i) => zColor(viewZ[i * d], d >= 2 ? viewZ[i * d + 1] : 0));
  realView = gan.sampleReal(N_VIEW);
  $('paramCount').textContent =
    `パラメータ数：G ${gan.G.paramCount().toLocaleString()} ／ D ${gan.D.paramCount().toLocaleString()}`;
  state.heatDirty = true;
  render(true);
}

// ---- 格子の写像 ----
function mapGrid() {
  const d = state.cfg.noiseDim;
  const seg = 48;
  const lines = [];
  if (d >= 2) {
    for (const v of GRID_VALS) {
      lines.push((t) => [v, t]);
      lines.push((t) => [t, v]);
    }
  } else {
    lines.push((t) => [t]);
  }
  const n = lines.length * (seg + 1);
  const Z = new Float64Array(n * d);
  lines.forEach((f, li) => {
    for (let i = 0; i <= seg; i++) {
      const t = -GRID_MAX + (i / seg) * 2 * GRID_MAX * (d >= 2 ? 1 : 1.2);
      const zz = f(t);
      const row = (li * (seg + 1) + i) * d;
      Z[row] = zz[0];
      if (d >= 2) Z[row + 1] = zz[1];
    }
  });
  const X = gan.generate(Z, n);
  return lines.map((_, li) => X.subarray(li * (seg + 1) * 2, (li + 1) * (seg + 1) * 2));
}

// ---- 描画 ----
const space = new SpaceView($('main'));
const chartLoss = new LineChart($('chartLoss'), {
  series: [
    { key: 'lossD', label: 'D の損失', color: '--lossD' },
    { key: 'lossG', label: 'G の損失', color: '--lossG' },
  ],
  refs: [{ y: 2 * Math.LN2 }, { y: Math.LN2 }],
});
const chartD = new LineChart($('chartD'), {
  series: [
    { key: 'dReal', label: 'D(本物)', color: '--real' },
    { key: 'dFake', label: 'D(生成)', color: '--fake' },
  ],
  yMin: 0,
  yMax: 1,
  refs: [{ y: 0.5 }],
});

function render(force = false) {
  const n = N_VIEW;
  const fake = gan.generate(viewZ, n);
  let grad = null;
  if (state.show.grad) grad = gan.dGradient(fake, n);

  if (state.show.heat && (force || state.heatDirty || state.frame % 3 === 0)) {
    const res = space.heatRes;
    space.updateHeat(gan.discriminate(space.heatPoints, res * res));
    state.heatDirty = false;
  }

  space.draw({
    show: state.show,
    real: realView,
    fake,
    grad,
    fakeColors,
    gridLines: state.show.grid ? mapGrid() : null,
  });
  drawLatent($('latent'), {
    show: state.show,
    z: viewZ,
    noiseDim: state.cfg.noiseDim,
    gridVals: GRID_VALS,
    gridMax: GRID_MAX,
    fakeColors,
    jitter,
  });
  chartLoss.draw(hist);
  chartD.draw(hist);

  $('stStep').textContent = gan.step.toLocaleString();
  if (gan.last) {
    $('stDReal').textContent = gan.last.dReal.toFixed(2);
    $('stDFake').textContent = gan.last.dFake.toFixed(2);
  } else {
    $('stDReal').textContent = $('stDFake').textContent = '–';
  }
}

function train(steps) {
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    hist.push(gan.trainStep());
    state.rateSteps++;
    if (performance.now() - t0 > 22) break; // UI を固めない
  }
}

function loop() {
  state.frame++;
  if (state.running) {
    train(state.speed);
    render();
  }
  const now = performance.now();
  if (now - state.rateT > 500) {
    $('stRate').textContent = state.running ? `${Math.round((state.rateSteps * 1000) / (now - state.rateT))}/秒` : '停止中';
    state.rateSteps = 0;
    state.rateT = now;
  }
  requestAnimationFrame(loop);
}

// ---- 操作パネル ----
function setRunning(v) {
  state.running = v;
  $('btnPlay').textContent = v ? '❚❚ 一時停止' : gan.step ? '▶ 再開' : '▶ 学習開始';
}

function bindRange(id, outId, get, set, fmt = String) {
  const el = $(id);
  el.value = get();
  const upd = () => { set(+el.value); $(outId).textContent = fmt(+el.value); };
  el.addEventListener('input', upd);
  $(outId).textContent = fmt(+el.value);
  return upd;
}

function init() {
  const dsSel = $('dataset');
  for (const [k, v] of Object.entries(DATASETS)) dsSel.add(new Option(v.label, k));
  dsSel.value = state.cfg.dataset;
  $('datasetNote').textContent = DATASETS[state.cfg.dataset].note;
  dsSel.addEventListener('change', () => {
    state.cfg.dataset = dsSel.value;
    $('datasetNote').textContent = DATASETS[dsSel.value].note;
    reset();
    setRunning(state.running);
  });

  // 学習中でも即時反映できる設定
  const live = (k, v) => { state.cfg[k] = v; if (gan) gan.cfg[k] = v; };
  $('loss').value = state.cfg.loss;
  $('loss').addEventListener('change', (e) => live('loss', e.target.value));
  $('batch').value = String(state.cfg.batch);
  $('batch').addEventListener('change', (e) => live('batch', +e.target.value));
  const lrFmt = (i) => String(LR_STEPS[i]);
  $('lrD').value = LR_STEPS.indexOf(state.cfg.lrD);
  $('lrG').value = LR_STEPS.indexOf(state.cfg.lrG);
  bindRange('lrD', 'lrDOut', () => LR_STEPS.indexOf(state.cfg.lrD), (i) => live('lrD', LR_STEPS[i]), lrFmt);
  bindRange('lrG', 'lrGOut', () => LR_STEPS.indexOf(state.cfg.lrG), (i) => live('lrG', LR_STEPS[i]), lrFmt);
  bindRange('dSteps', 'dStepsOut', () => state.cfg.dSteps, (v) => live('dSteps', v));
  bindRange('speed', 'speedOut', () => state.speed, (v) => (state.speed = v));

  // 構造の設定（変更でリセット）
  const rebuild = (k, v) => { state.cfg[k] = v; reset(); setRunning(state.running); };
  $('noiseDim').value = String(state.cfg.noiseDim);
  $('noiseDim').addEventListener('change', (e) => rebuild('noiseDim', +e.target.value));
  $('hidden').value = String(state.cfg.hidden);
  $('hidden').addEventListener('change', (e) => rebuild('hidden', +e.target.value));
  $('depth').value = state.cfg.depth;
  $('depthOut').textContent = state.cfg.depth;
  $('depth').addEventListener('change', (e) => { $('depthOut').textContent = e.target.value; rebuild('depth', +e.target.value); });
  $('depth').addEventListener('input', (e) => { $('depthOut').textContent = e.target.value; });
  $('seed').value = state.cfg.seed;
  $('seed').addEventListener('change', (e) => rebuild('seed', Math.max(1, Math.round(+e.target.value) || 1)));

  // 表示切り替え
  const showMap = { showHeat: 'heat', showReal: 'real', showFake: 'fake', showGrad: 'grad', showGrid: 'grid', colorByZ: 'colorByZ' };
  for (const [id, key] of Object.entries(showMap)) {
    state.show[key] = $(id).checked;
    $(id).addEventListener('change', (e) => { state.show[key] = e.target.checked; render(true); });
  }

  // ボタンとキー操作
  $('btnPlay').addEventListener('click', () => setRunning(!state.running));
  $('btnStep').addEventListener('click', () => { setRunning(false); hist.push(gan.trainStep()); render(true); });
  $('btnReset').addEventListener('click', () => { reset(); setRunning(false); });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input[type="number"], select')) return;
    if (e.code === 'Space') { e.preventDefault(); setRunning(!state.running); }
    else if (e.key === 's' || e.key === 'S') $('btnStep').click();
    else if (e.key === 'r' || e.key === 'R') $('btnReset').click();
  });

  // データ空間のホバー：その位置での D(x)
  const tip = $('mainTip');
  $('main').addEventListener('pointermove', (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const [x, y] = space.toWorld(px, py);
    const p = gan.discriminate(Float64Array.from([x, y]), 1)[0];
    space.hover = [px, py];
    tip.innerHTML = `<div class="k">x = (${x.toFixed(2)}, ${y.toFixed(2)})</div><div>D(x) = <b>${p.toFixed(3)}</b>　${p >= 0.5 ? '本物寄り' : '偽物寄り'}</div>`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${px + 14 + tw > rect.width ? px - tw - 14 : px + 14}px`;
    tip.style.top = `${Math.max(4, py - 44)}px`;
    if (!state.running) render();
  });
  $('main').addEventListener('pointerleave', () => { tip.hidden = true; space.hover = null; if (!state.running) render(); });

  window.addEventListener('resize', () => render(true));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => render(true));

  reset();
  setRunning(false);
  requestAnimationFrame(loop);
}

init();
