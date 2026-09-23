// 比較ページ：GAN と自己回帰 Transformer を同じステップ数で学習させて並べる
import { GAN, DEFAULTS } from './gan.js';
import { ARModel, AR_DEFAULTS, T, CELL, binLo } from './ar.js';
import { VAE, VAE_DEFAULTS } from './vae.js';
import { DATASETS } from './datasets.js';
import { mulberry32, gaussian } from './nn.js';
import { SpaceView, LineChart } from './viz.js';
import { ARSpaceView } from './ar-viz.js';
import { VAESpaceView } from './vae-viz.js';
import { precisionRecall, modesCovered } from './metrics.js';
import { History, initialDataset, syncDataset, fillDatasetSelect, mountHeader, enableGlossary } from './common.js';

const $ = (id) => document.getElementById(id);

mountHeader({ id: 'compare', mode: 'detail', sub: '同じ分布を、GAN・自己回帰 Transformer・VAE に同じステップ数だけ学習させて並べる' });
const GAN_LR = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01];
const AR_LR = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01];
const BETAS = [0, 0.3, 1, 2, 4, 10, 50];
const N = 500;
const METRIC_INTERVAL = 500;
const TABLE_INTERVAL = 300;

const state = {
  dataset: initialDataset('grid25'),
  seed: 1,
  ganLoss: 'nonsat',
  ganLr: DEFAULTS.lrG,
  arLr: AR_DEFAULTS.lr,
  beta: VAE_DEFAULTS.beta,
  speed: 3,
  running: false,
  show: { bg: true, real: true },
  metricT: 0,
  tablesT: 0,
  frame: 0,
};

let gan, ar, vae, hist, viewZ, crn, vaeZ, realRef, ganSamples, arSamples, vaeSamples;

const ganView = new SpaceView($('ganCanvas'));
const arView = new ARSpaceView($('arCanvas'));
const vaeView = new VAESpaceView($('vaeCanvas'));
const series = [
  { key: 'gan', label: 'GAN', color: '--fake' },
  { key: 'ar', label: 'Transformer', color: '--ar' },
  { key: 'vae', label: 'VAE', color: '--vae' },
];
const pct = (v) => `${(v * 100).toFixed(0)}%`;
const chartP = new LineChart($('chartP'), { series: series.map((s) => ({ ...s, key: 'p' + s.key })), yMin: 0, yMax: 1, fmt: pct });
const chartR = new LineChart($('chartR'), { series: series.map((s) => ({ ...s, key: 'r' + s.key })), yMin: 0, yMax: 1, fmt: pct });

function reset() {
  gan = new GAN({ dataset: state.dataset, seed: state.seed, loss: state.ganLoss, lrG: state.ganLr });
  ar = new ARModel({ dataset: state.dataset, seed: state.seed, lr: state.arLr });
  vae = new VAE({ dataset: state.dataset, seed: state.seed, beta: state.beta });
  hist = new History(['pgan', 'rgan', 'par', 'rar', 'pvae', 'rvae'], 400);
  const r = mulberry32(state.seed * 131 + 11);
  viewZ = new Float64Array(N * gan.cfg.noiseDim);
  for (let i = 0; i < viewZ.length; i++) viewZ[i] = gaussian(r);
  crn = Array.from({ length: N }, () => [r(), r(), r(), r(), r(), r()]);
  vaeZ = new Float64Array(N * vae.cfg.latentDim);
  for (let i = 0; i < vaeZ.length; i++) vaeZ[i] = gaussian(r);
  // 指標用の本物の参照サンプル（学習とは別の乱数）
  const rr = mulberry32(state.seed * 17 + 999);
  realRef = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) {
    const [x, y] = DATASETS[state.dataset].sample(rr);
    realRef[2 * i] = x;
    realRef[2 * i + 1] = y;
  }
  refreshAR();
  refreshGAN(true);
  refreshVAE(true);
  measure();
  render();
}

function refreshGAN(heat) {
  ganSamples = gan.generate(viewZ, N);
  if (heat && state.show.bg) ganView.updateHeat(gan.discriminate(ganView.heatPoints, ganView.heatRes ** 2));
}

function refreshAR() {
  const t = ar.computeTables();
  state.tablesT = performance.now();
  arView.updateDensity(t.density);
  arSamples = new Float64Array(N * 2);
  for (let i = 0; i < N; i++) {
    const u = crn[i];
    const tk = [];
    for (let k = 0; k < T; k++) tk.push(ARModel.draw(ar.conditional(tk, t), u[k]));
    const [i1, i2] = ar.binsOf(tk);
    arSamples[2 * i] = binLo(i1) + u[4] * CELL;
    arSamples[2 * i + 1] = binLo(i2) + u[5] * CELL;
  }
}

function refreshVAE(density) {
  vaeSamples = vae.decode(vaeZ, N);
  if (density && state.show.bg) vaeView.updateDensity(vae.density(vaeView.points, vaeView.res ** 2, 120));
}

function measure() {
  state.metricT = performance.now();
  const g = precisionRecall(realRef, ganSamples, N);
  const a = precisionRecall(realRef, arSamples, N);
  const v = precisionRecall(realRef, vaeSamples, N);
  const modes = DATASETS[state.dataset].modes;
  const mg = modesCovered(ganSamples, N, modes);
  const ma = modesCovered(arSamples, N, modes);
  const mv = modesCovered(vaeSamples, N, modes);
  hist.push({ step: gan.step, pgan: g.precision, rgan: g.recall, par: a.precision, rar: a.recall, pvae: v.precision, rvae: v.recall });
  $('ganP').textContent = pct(g.precision);
  $('ganR').textContent = pct(g.recall);
  $('arP').textContent = pct(a.precision);
  $('arR').textContent = pct(a.recall);
  $('ganM').textContent = mg ? `${mg.covered} / ${mg.total}` : '–';
  $('arM').textContent = ma ? `${ma.covered} / ${ma.total}` : '–';
  $('vaeP').textContent = pct(v.precision);
  $('vaeR').textContent = pct(v.recall);
  $('vaeM').textContent = mv ? `${mv.covered} / ${mv.total}` : '–';
}

function render() {
  ganView.draw({
    show: { heat: state.show.bg, real: state.show.real, fake: true, grid: false, grad: false, colorByZ: false },
    real: realRef,
    fake: ganSamples,
  });
  arView.draw({ show: { density: state.show.bg, real: state.show.real, samples: true, grid: false }, real: realRef, samples: arSamples });
  vaeView.draw({ show: { density: state.show.bg, real: state.show.real, samples: true, recon: false, grid: false }, real: realRef, samples: vaeSamples });
  chartP.draw(hist);
  chartR.draw(hist);
}

function trainBoth(steps) {
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    gan.trainStep();
    ar.trainStep();
    vae.trainStep();
    if (performance.now() - t0 > 28) break;
  }
}

function loop() {
  state.frame++;
  const now = performance.now();
  if (state.running) {
    trainBoth(state.speed);
    refreshGAN(state.frame % 3 === 0);
    refreshVAE(state.frame % 6 === 0);
    if (now - state.tablesT > TABLE_INTERVAL) refreshAR();
    if (now - state.metricT > METRIC_INTERVAL) measure();
    render();
  }
  requestAnimationFrame(loop);
}

function setRunning(v) {
  state.running = v;
  $('btnPlay').textContent = v ? '❚❚ 一時停止' : gan.step ? `▶ 再開（${gan.step.toLocaleString()} ステップ）` : '▶ 学習開始';
  if (!v && gan) { refreshAR(); refreshGAN(true); refreshVAE(true); measure(); render(); }
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
  fillDatasetSelect(dsSel, state.dataset, $('datasetNote'));
  syncDataset(state.dataset);
  dsSel.addEventListener('change', () => {
    state.dataset = dsSel.value;
    $('datasetNote').textContent = DATASETS[dsSel.value].note;
    syncDataset(dsSel.value);
    reset();
    setRunning(state.running);
  });
  bindRange('speed', 'speedOut', () => state.speed, (v) => (state.speed = v));
  $('seed').value = state.seed;
  $('seed').addEventListener('change', (e) => { state.seed = Math.max(1, Math.round(+e.target.value) || 1); reset(); setRunning(state.running); });
  $('ganLoss').value = state.ganLoss;
  $('ganLoss').addEventListener('change', (e) => { state.ganLoss = e.target.value; gan.cfg.loss = state.ganLoss; });
  bindRange('ganLr', 'ganLrOut', () => GAN_LR.indexOf(state.ganLr), (i) => { state.ganLr = GAN_LR[i]; if (gan) gan.cfg.lrG = GAN_LR[i]; }, (i) => String(GAN_LR[i]));
  bindRange('vaeBeta', 'vaeBetaOut', () => BETAS.indexOf(state.beta), (i) => { state.beta = BETAS[i]; if (vae) vae.cfg.beta = BETAS[i]; }, (i) => String(BETAS[i]));
  bindRange('arLr', 'arLrOut', () => AR_LR.indexOf(state.arLr), (i) => { state.arLr = AR_LR[i]; if (ar) ar.cfg.lr = AR_LR[i]; }, (i) => String(AR_LR[i]));
  $('showBg').addEventListener('change', (e) => { state.show.bg = e.target.checked; refreshGAN(true); refreshVAE(true); render(); });
  $('showReal').addEventListener('change', (e) => { state.show.real = e.target.checked; render(); });

  $('btnPlay').addEventListener('click', () => setRunning(!state.running));
  $('btnStep').addEventListener('click', () => { gan.trainStep(); ar.trainStep(); vae.trainStep(); setRunning(false); });
  $('btnReset').addEventListener('click', () => { reset(); setRunning(false); });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input[type="number"], select')) return;
    if (e.code === 'Space') { e.preventDefault(); setRunning(!state.running); }
    else if (e.key === 's' || e.key === 'S') $('btnStep').click();
    else if (e.key === 'r' || e.key === 'R') $('btnReset').click();
  });
  window.addEventListener('resize', render);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { refreshAR(); refreshGAN(true); refreshVAE(true); render(); });

  enableGlossary();
  reset();
  setRunning(false);
  requestAnimationFrame(loop);
}

init();
