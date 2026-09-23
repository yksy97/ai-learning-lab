// VAE ページ：状態管理・操作パネル・描画ループ
import { VAE, VAE_DEFAULTS } from './vae.js';
import { DATASETS } from './datasets.js';
import { mulberry32, gaussian } from './nn.js';
import { LineChart } from './viz.js';
import { VAESpaceView, drawLatent } from './vae-viz.js';
import { History, initialDataset, syncDataset, fillDatasetSelect, mountHeader, mountGoal, enableGlossary } from './common.js';
import { LiveFormula } from './formula.js';
import { codeBox } from './codebox.js';
import { VAE_CODE } from './code-snippets.js';

const $ = (id) => document.getElementById(id);

mountHeader({
  id: 'vae', mode: 'detail',
  sub: '点をいったんメモ（潜在変数）に変えてから描き直します。メモの置き場所が整っていく様子を見ます',
});
mountGoal('vae');

const BETAS = [0, 0.1, 0.3, 1, 2, 4, 10, 50, 150];
const SIGMAS = [0.05, 0.1, 0.2, 0.4];
const LRS = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.01];
const N_VIEW = 400;      // 表示用の本物サンプル
const N_GEN = 400;       // 表示用の生成サンプル
const N_ELLIPSE = 120;   // 潜在空間に描く楕円の数
const DENSITY_INTERVAL = 250;
const GRID_MAX = 2.5;
const GRID_VALS = Array.from({ length: 11 }, (_, i) => -GRID_MAX + i * 0.5);

const state = {
  cfg: { ...VAE_DEFAULTS, dataset: initialDataset(VAE_DEFAULTS.dataset) },
  running: false,
  speed: 10,
  show: {},
  rateSteps: 0,
  rateT: performance.now(),
  densityT: 0,
  pick: null,
};

let vae, hist, realView, genZ, samples;
const space = new VAESpaceView($('main'));
const chart = new LineChart($('chartLoss'), {
  series: [
    { key: 'recon', label: '再構成', color: '--vae' },
    { key: 'klw', label: 'β × KL', color: '--real' },
  ],
  fmt: (v) => v.toFixed(2),
  robustY: true,
  yMin: 0,
});

const formula = new LiveFormula($('vaeFormula'), {
  rows: [
    {
      label: '全体',
      tpl: 'L = 再構成 + β × KL = {{rec}} + {{beta}} × {{kl}} = {{tot}}',
      slots: {
        rec: { hl: '#main', tip: '元の点に戻せていないほど大きい' },
        beta: { tip: 'KL 項の重み（左のスライダー）' },
        kl: { hl: '#latent', tip: 'メモを標準の円へそろえる力' },
        tot: { hl: '#chartLoss', tip: 'この合計を小さくするのが学習' },
      },
    },
    {
      label: '再構成',
      tpl: '再構成 = ‖x − x̂‖² / (2σₓ²) = {{mse}} / (2 × {{sx}}²)',
      slots: {
        mse: { hl: '#main', tip: '元の点と復元の差の2乗（平均）' },
        sx: { tip: '復元の厳しさ σₓ' },
      },
    },
    {
      label: 'KL',
      tpl: 'KL = ½ Σ (σ² + μ² − 1 − 2 log σ) = {{kl2}}',
      slots: { kl2: { hl: '#latent', tip: '楕円が標準の円と一致すると 0' } },
      note: 'μ=0, σ=1 で 0。0 に張り付いたら事後崩壊',
    },
  ],
});

function reset() {
  vae = new VAE(state.cfg);
  hist = new History(['recon', 'klw']);
  const r = mulberry32(state.cfg.seed * 53 + 7);
  realView = vae.sampleReal(N_VIEW);
  genZ = new Float64Array(N_GEN * state.cfg.latentDim);
  for (let i = 0; i < genZ.length; i++) genZ[i] = gaussian(r);
  state.pick = null;
  $('paramCount').textContent =
    `パラメータ数：エンコーダ ${vae.enc.paramCount().toLocaleString()} ／ デコーダ ${vae.dec.paramCount().toLocaleString()}`;
  refreshDensity(true);
  updatePick();
}

function refreshDensity(force) {
  if (state.show.density || force) {
    space.updateDensity(vae.density(space.points, space.res * space.res));
  }
  state.densityT = performance.now();
}

/** デコーダが潜在空間の格子をどう写すか */
function mapGrid() {
  const L = state.cfg.latentDim;
  const seg = 40;
  const lines = [];
  if (L >= 2) {
    for (const v of GRID_VALS) {
      lines.push((t) => [v, t]);
      lines.push((t) => [t, v]);
    }
  } else {
    lines.push((t) => [t]);
  }
  const n = lines.length * (seg + 1);
  const Z = new Float64Array(n * L);
  lines.forEach((f, li) => {
    for (let i = 0; i <= seg; i++) {
      const t = -GRID_MAX + (i / seg) * 2 * GRID_MAX;
      const z = f(t);
      Z[(li * (seg + 1) + i) * L] = z[0];
      if (L >= 2) Z[(li * (seg + 1) + i) * L + 1] = z[1];
    }
  });
  const X = vae.decode(Z, n);
  return lines.map((_, li) => X.subarray(li * (seg + 1) * 2, (li + 1) * (seg + 1) * 2));
}

function render() {
  samples = vae.decode(genZ, N_GEN);
  const { mu, sd } = vae.encode(realView, N_VIEW);
  const recon = state.show.recon ? vae.decode(mu.slice(0, N_ELLIPSE * state.cfg.latentDim), N_ELLIPSE) : null;
  if (state.pick) {
    const p = vae.encode(Float64Array.from(state.pick.x), 1);
    state.pick.mu = Array.from(p.mu);
    state.pick.sd = Array.from(p.sd);
    state.pick.xh = Array.from(vae.decode(p.mu, 1));
  }
  space.draw({
    show: state.show,
    real: realView,
    samples,
    recon,
    pick: state.pick,
    gridLines: state.show.grid ? mapGrid() : null,
  });
  drawLatent($('latent'), {
    latentDim: state.cfg.latentDim,
    mu: mu.slice(0, N_ELLIPSE * state.cfg.latentDim),
    sd: sd.slice(0, N_ELLIPSE * state.cfg.latentDim),
    pick: state.pick,
  });
  chart.draw(hist);
  if (vae.last) {
    formula.update({
      rec: vae.last.recon, beta: state.cfg.beta, kl: vae.last.kl,
      tot: vae.last.recon + state.cfg.beta * vae.last.kl,
      mse: vae.last.mse, sx: state.cfg.sigmaX, kl2: vae.last.kl,
    });
  }
  $('stStep').textContent = vae.step.toLocaleString();
  if (vae.last) {
    $('stMse').textContent = Math.sqrt(vae.last.mse).toFixed(3);
    $('stKl').textContent = vae.last.kl.toFixed(2);
  } else {
    $('stMse').textContent = $('stKl').textContent = '–';
  }
}

function updatePick() {
  const p = state.pick;
  const L = state.cfg.latentDim;
  if (!p || !p.mu) {
    $('pickBody').innerHTML = 'データ空間の点をクリックすると、その点がどんなメモ（μ と σ）になり、どう復元されるかを表示。';
    return;
  }
  const f = (a) => a.map((v) => v.toFixed(2)).join(', ');
  const err = Math.hypot(p.xh[0] - p.x[0], p.xh[1] - p.x[1]);
  $('pickBody').innerHTML =
    `元の点 x = (${f(p.x)})<br>` +
    `メモの中心 μ = (${f(p.mu.slice(0, L))})<br>` +
    `メモの広がり σ = (${f(p.sd.slice(0, L))})<br>` +
    `復元 x̂ = (${f(p.xh)})　ズレ <b>${err.toFixed(3)}</b>`;
}

function train(steps) {
  const t0 = performance.now();
  for (let i = 0; i < steps; i++) {
    const r = vae.trainStep();
    hist.push({ step: r.step, recon: r.recon, klw: state.cfg.beta * r.kl });
    state.rateSteps++;
    if (performance.now() - t0 > 20) break;
  }
}

function loop() {
  const now = performance.now();
  if (state.running) {
    train(state.speed);
    if (now - state.densityT > DENSITY_INTERVAL) refreshDensity();
    render();
    updatePick();
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
  $('btnPlay').textContent = v ? '❚❚ 止める' : vae.step ? '▶ 続きから学習' : '▶ この条件で学習する';
  if (!v) { refreshDensity(); render(); }
}

function bindRange(id, outId, get, set, fmt = String) {
  const el = $(id);
  el.value = get();
  el.addEventListener('input', () => { set(+el.value); $(outId).textContent = fmt(+el.value); });
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

  const live = (k, v) => { state.cfg[k] = v; if (vae) vae.cfg[k] = v; };
  bindRange('beta', 'betaOut', () => BETAS.indexOf(state.cfg.beta), (i) => live('beta', BETAS[i]), (i) => String(BETAS[i]));
  bindRange('sigma', 'sigmaOut', () => SIGMAS.indexOf(state.cfg.sigmaX), (i) => live('sigmaX', SIGMAS[i]), (i) => String(SIGMAS[i]));
  bindRange('lr', 'lrOut', () => LRS.indexOf(state.cfg.lr), (i) => live('lr', LRS[i]), (i) => String(LRS[i]));
  bindRange('speed', 'speedOut', () => state.speed, (v) => (state.speed = v));
  $('batch').value = String(state.cfg.batch);
  $('batch').addEventListener('change', (e) => live('batch', +e.target.value));

  $('latentDim').value = String(state.cfg.latentDim);
  $('latentDim').addEventListener('change', (e) => rebuild('latentDim', +e.target.value));
  $('hidden').value = String(state.cfg.hidden);
  $('hidden').addEventListener('change', (e) => rebuild('hidden', +e.target.value));
  $('depth').value = state.cfg.depth;
  $('depthOut').textContent = state.cfg.depth;
  $('depth').addEventListener('input', (e) => { $('depthOut').textContent = e.target.value; });
  $('depth').addEventListener('change', (e) => rebuild('depth', +e.target.value));
  $('seed').value = state.cfg.seed;
  $('seed').addEventListener('change', (e) => rebuild('seed', Math.max(1, Math.round(+e.target.value) || 1)));

  const showMap = { showDensity: 'density', showReal: 'real', showSamples: 'samples', showRecon: 'recon', showGrid: 'grid' };
  for (const [id, key] of Object.entries(showMap)) {
    state.show[key] = $(id).checked;
    $(id).addEventListener('change', (e) => { state.show[key] = e.target.checked; if (key === 'density') refreshDensity(); render(); });
  }

  $('btnPlay').addEventListener('click', () => setRunning(!state.running));
  $('btnStep').addEventListener('click', () => { setRunning(false); train(1); refreshDensity(); render(); updatePick(); });
  $('btnReset').addEventListener('click', () => { reset(); setRunning(false); render(); });
  window.addEventListener('keydown', (e) => {
    if (e.target.matches('input[type="number"], select')) return;
    if (e.code === 'Space') { e.preventDefault(); setRunning(!state.running); }
    else if (e.key === 's' || e.key === 'S') $('btnStep').click();
    else if (e.key === 'r' || e.key === 'R') $('btnReset').click();
  });

  const canvas = $('main');
  const tip = $('mainTip');
  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const [x, y] = space.toWorld(px, py);
    const p = vae.encode(Float64Array.from([x, y]), 1);
    const xh = vae.decode(p.mu, 1);
    tip.innerHTML = `<div class="k">x = (${x.toFixed(2)}, ${y.toFixed(2)})</div>` +
      `<div>メモ μ = (${Array.from(p.mu).map((v) => v.toFixed(2)).join(', ')})</div>` +
      `<div>復元 → (${xh[0].toFixed(2)}, ${xh[1].toFixed(2)})</div>`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${px + 14 + tw > rect.width ? px - tw - 14 : px + 14}px`;
    tip.style.top = `${Math.max(4, py - 58)}px`;
  });
  canvas.addEventListener('pointerleave', () => { tip.hidden = true; });
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const [x, y] = space.toWorld(e.clientX - rect.left, e.clientY - rect.top);
    state.pick = { x: [x, y] };
    render();
    updatePick();
  });

  window.addEventListener('resize', render);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { refreshDensity(true); render(); });

  formula.linkSources({
    '.stage-head .legend span:nth-child(1)': ['rec', 'mse'],
    '.stage-head .legend span:nth-child(2)': ['tot'],
    '#latent': ['kl', 'kl2'],
    '.side .chart-head .legend span:nth-child(1)': ['rec', 'mse'],
    '.side .chart-head .legend span:nth-child(2)': ['beta', 'kl'],
    '#chartLoss': ['tot'],
    '.stats .stat:nth-child(3)': ['mse', 'rec'],
    '.stats .stat:nth-child(4)': ['kl', 'kl2'],
    '.controls section:nth-child(2) .field:nth-child(2)': ['beta'],
    '.controls section:nth-child(2) .field:nth-child(4)': ['sx'],
  });

  codeBox($('vaeCode'), { items: VAE_CODE });
  enableGlossary();
  reset();
  setRunning(false);
  render();
  requestAnimationFrame(loop);
}

init();
