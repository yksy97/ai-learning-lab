// 実験カード：予想する → 実験する → 観察する → 説明する → 比べる。
// つまみを触って結果を見るだけだと「操作」で終わる。先に予想を口に出させ、
// 結果を保存して並べられるようにすると、はじめて実験になる。
import { renderQuiz } from './quiz.js';
import { saveRun, listRuns, removeRun, diffConfig, saveNote, getNote } from './runs.js';
import { DATASETS } from './datasets.js';

const CONFIG_LABEL = {
  dataset: 'データ', lrG: 'G の学習率', lrD: 'D の学習率', dSteps: 'D の回数',
  loss: '損失の種類', hidden: '隠れ層の幅', depth: '層の数', noiseDim: 'z の次元',
  batch: 'バッチ', seed: '乱数の種', beta: 'β', sigmaX: 'σₓ', latentDim: 'z の次元',
  lr: '学習率', heads: 'ヘッド数', layers: '層の数', d: '埋め込みの幅', dModel: '埋め込みの幅',
  k: '取り出す件数', threshold: 'しきい値', chunk: 'チャンク', expand: '言い換え',
};
const label = (k) => CONFIG_LABEL[k] || k;

const VALUE_LABEL = { nonsat: '非飽和', minimax: 'ミニマックス', doc: '条文ごと', sentence: '文ごと', coarse: '粗→細', axis: '軸ごと' };
/** 設定の値を、画面で使っている言葉に直す */
function value(k, v) {
  if (k === 'dataset') return (DATASETS[v] || {}).label || v;
  if (typeof v === 'boolean') return v ? 'あり' : 'なし';
  return VALUE_LABEL[v] || v;
}

/**
 * @param {HTMLElement} el
 * @param {{pageId:string, experiments:Object[],
 *          apply:(cfg:Object)=>void, run:(steps:number, cb:{onProgress:Function,onDone:Function})=>void,
 *          stop:()=>void, capture:()=>{img:string, stats:{k:string,v:string}[], config:Object, step:number}}} opt
 */
export function mountLab(el, opt) {
  const { pageId, experiments } = opt;
  let cur = 0;
  let phase = 'predict';   // predict → ready → running → done
  let runsEl, cardEl;

  el.className = 'panel lab';
  el.innerHTML = `
    <div class="lab-head">
      <h2>実験する</h2>
      <div class="lab-tabs" id="labTabs"></div>
    </div>
    <div id="labCard"></div>
    <div id="labRuns"></div>`;
  cardEl = el.querySelector('#labCard');
  runsEl = el.querySelector('#labRuns');

  el.querySelector('#labTabs').innerHTML = experiments
    .map((e, i) => `<button data-i="${i}" aria-pressed="${i === 0}">実験 ${String(i + 1).padStart(2, '0')}</button>`).join('');
  el.querySelector('#labTabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || phase === 'running') return;
    cur = +b.dataset.i;
    for (const x of el.querySelectorAll('#labTabs button')) x.setAttribute('aria-pressed', String(+x.dataset.i === cur));
    phase = 'predict';
    renderCard();
  });

  function renderCard() {
    const ex = experiments[cur];
    cardEl.innerHTML = `
      <p class="lab-theme"><span>テーマ</span>${ex.title}</p>
      <div class="lab-body">${ex.body || ''}</div>
      <div id="labQuiz"></div>
      <div class="lab-actions">
        <button id="labRun" class="act" disabled>この条件で実験する（${ex.steps.toLocaleString()} ステップ）</button>
        <span class="lab-cond">${Object.entries(ex.config).map(([k, v]) => `${label(k)} = <b>${value(k, v)}</b>`).join(' ・ ')}</span>
      </div>
      <div class="lab-progress" id="labProg" hidden><i></i><span></span></div>
      <div class="lab-result" id="labResult" hidden></div>`;

    renderQuiz(cardEl.querySelector('#labQuiz'), {
      id: `lab:${pageId}:${ex.id}`, kind: 'predict', q: ex.question,
      options: ex.options, answer: ex.answer, explain: ex.explain,
    }, () => {
      phase = 'ready';
      cardEl.querySelector('#labRun').disabled = false;
    });

    cardEl.querySelector('#labRun').addEventListener('click', start);
  }

  function start() {
    const ex = experiments[cur];
    phase = 'running';
    const prog = cardEl.querySelector('#labProg');
    const bar = prog.querySelector('i');
    const txt = prog.querySelector('span');
    prog.hidden = false;
    cardEl.querySelector('#labRun').disabled = true;
    opt.apply({ ...ex.config });
    opt.run(ex.steps, {
      onProgress: (done) => {
        bar.style.width = `${Math.round((done / ex.steps) * 100)}%`;
        txt.textContent = `${done.toLocaleString()} / ${ex.steps.toLocaleString()} ステップ`;
      },
      onDone: () => {
        phase = 'done';
        prog.hidden = true;
        showResult(ex);
      },
    });
  }

  function showResult(ex) {
    const box = cardEl.querySelector('#labResult');
    const snap = opt.capture();
    box.hidden = false;
    box.innerHTML = `
      <h3>何が起きたか</h3>
      <div class="lab-observe">
        <img alt="実験の結果" src="${snap.img}" />
        <div>
          <ul>${ex.observe.map((t) => `<li>${t}</li>`).join('')}</ul>
          <dl class="lab-stats">${snap.stats.map((s) => `<div><dt>${s.k}</dt><dd>${s.v}</dd></div>`).join('')}</dl>
        </div>
      </div>
      <div class="lab-actions">
        <button id="labSave" class="act">この結果を保存して比べる</button>
        <button id="labAgain">別の条件でもう一度</button>
        <span class="note">保存すると、下で2つ並べて見られます</span>
      </div>`;
    box.querySelector('#labSave').addEventListener('click', () => {
      const s = opt.capture();
      saveRun(pageId, { label: ex.short || ex.title, img: s.img, stats: s.stats, config: s.config, step: s.step });
      renderRuns();
      box.querySelector('#labSave').disabled = true;
      box.querySelector('#labSave').textContent = '保存しました';
      runsEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
    box.querySelector('#labAgain').addEventListener('click', () => {
      phase = 'predict';
      cur = (cur + 1) % experiments.length;
      for (const x of el.querySelectorAll('#labTabs button')) x.setAttribute('aria-pressed', String(+x.dataset.i === cur));
      renderCard();
    });
  }

  // ---- 保存した実験を並べる ----
  let pick = [];
  function renderRuns() {
    const runs = listRuns(pageId);
    if (!runs.length) { runsEl.innerHTML = ''; return; }
    if (pick.length < 2) pick = runs.slice(-2).map((r) => r.id);
    pick = pick.filter((id) => runs.some((r) => r.id === id));

    runsEl.innerHTML = `
      <h3 class="lab-sub">保存した実験（${runs.length}）<span class="hint">2つ選ぶと並べて比べられます</span></h3>
      <div class="run-list">${runs.map((r) => `
        <button class="run${pick.includes(r.id) ? ' on' : ''}" data-id="${r.id}">
          <img alt="" src="${r.img}" />
          <b>${r.label}</b>
          <span>${r.step.toLocaleString()} ステップ</span>
          <i class="run-del" data-del="${r.id}" title="消す">×</i>
        </button>`).join('')}</div>
      <div id="runCompare"></div>`;

    runsEl.querySelector('.run-list').addEventListener('click', (e) => {
      const del = e.target.closest('[data-del]');
      if (del) { removeRun(pageId, del.dataset.del); pick = []; renderRuns(); return; }
      const b = e.target.closest('.run');
      if (!b) return;
      const id = b.dataset.id;
      if (pick.includes(id)) pick = pick.filter((x) => x !== id);
      else pick = [...pick, id].slice(-2);
      renderRuns();
    });
    renderCompare(runs);
  }

  function renderCompare(runs) {
    const box = runsEl.querySelector('#runCompare');
    if (pick.length < 2) { box.innerHTML = '<p class="note">もう1つ選ぶと、違いが表に出ます。</p>'; return; }
    const [a, b] = pick.map((id) => runs.find((r) => r.id === id));
    const d = diffConfig(a.config, b.config);
    const pair = [a.id, b.id].sort().join('_');
    const note = getNote(pageId, pair) || {};

    box.innerHTML = `
      <div class="cmp">
        <figure><img alt="A の結果" src="${a.img}" /><figcaption>A：${a.label}</figcaption></figure>
        <figure><img alt="B の結果" src="${b.img}" /><figcaption>B：${b.label}</figcaption></figure>
      </div>
      <table class="cmp-table">
        <thead><tr><th>項目</th><th>A</th><th>B</th></tr></thead>
        <tbody>
          ${d.length ? d.map((x) => `<tr class="diff"><td>${label(x.k)}</td><td>${value(x.k, x.a)}</td><td>${value(x.k, x.b)}</td></tr>`).join('')
            : '<tr><td colspan="3" class="note">設定は同じです（乱数の違いだけ）</td></tr>'}
          <tr><td>学習したステップ</td><td>${a.step.toLocaleString()}</td><td>${b.step.toLocaleString()}</td></tr>
          ${a.stats.map((s, i) => `<tr><td>${s.k}</td><td>${s.v}</td><td>${(b.stats[i] || {}).v ?? '–'}</td></tr>`).join('')}
        </tbody>
      </table>
      <div class="cmp-note">
        <label>何が違いましたか？<textarea id="noteWhat" rows="2" placeholder="見た目・数字のどこが変わったかを書きます">${note.what || ''}</textarea></label>
        <label>なぜそうなったと思いますか？<textarea id="noteWhy" rows="3" placeholder="設定の違いと結果を、損失や勾配の言葉でつなげてみてください">${note.why || ''}</textarea></label>
        <span class="ex-saved" hidden>保存しました</span>
      </div>`;

    let t = null;
    const tag = box.querySelector('.ex-saved');
    for (const id of ['noteWhat', 'noteWhy']) {
      box.querySelector(`#${id}`).addEventListener('input', () => {
        clearTimeout(t);
        t = setTimeout(() => {
          saveNote(pageId, pair, { what: box.querySelector('#noteWhat').value, why: box.querySelector('#noteWhy').value });
          tag.hidden = false;
          setTimeout(() => { tag.hidden = true; }, 1400);
        }, 600);
      });
    }
  }

  renderCard();
  renderRuns();
  return { renderRuns };
}

/** キャンバスを小さな画像にして保存できる形にする */
export function thumb(canvas, size = 190) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim() || '#fff';
  ctx.fillRect(0, 0, size, size);
  const s = Math.min(canvas.width, canvas.height);
  ctx.drawImage(canvas, (canvas.width - s) / 2, (canvas.height - s) / 2, s, s, 0, 0, size, size);
  return c.toDataURL('image/png');
}
