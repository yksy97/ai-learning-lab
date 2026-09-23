// 自分の言葉で説明する（自己説明）。
// 読む・予想する・手で計算するの次に、いちばん効くのが「出力する」こと。
// 書いてから模範例と見くらべる順番を守るため、書く前は模範例を開けないようにしてある。
import { getExplain, saveExplain } from './progress.js';

const MIN = 60; // これくらい書かないと、思い出す負荷がかからない

/**
 * @param {HTMLElement} el
 * @param {{id:string, q:string, lead?:string,
 *          words:{t:string, any?:string[]}[],
 *          model:string[], checks:string[]}} spec
 */
export function renderExplain(el, spec) {
  const saved = getExplain(spec.id) || { text: '', checks: [], opened: false };
  el.className = 'explain';
  el.innerHTML = `
    <h3>${spec.q}</h3>
    ${spec.lead ? `<p class="ex-lead">${spec.lead}</p>` : ''}
    <div class="ex-words">${spec.words.map((w, i) => `<i data-i="${i}">${w.t}</i>`).join('')}</div>
    <textarea placeholder="たとえ話でも、箇条書きでもかまいません。上の言葉をできるだけ使って書いてみてください。"></textarea>
    <div class="ex-meta">
      <span class="ex-count"></span>
      <span class="ex-actions">
        <button class="ex-open">模範例と見くらべる</button>
        <span class="ex-saved" hidden>保存しました</span>
      </span>
    </div>
    <div class="ex-model" hidden>
      <h4>説明の一例（これが唯一の正解ではありません）</h4>
      ${spec.model.map((p) => `<p>${p}</p>`).join('')}
      <h4>自分の説明と見くらべる</h4>
      <div class="ex-check">${spec.checks.map((c, i) => `<label><input type="checkbox" data-i="${i}" /><span>${c}</span></label>`).join('')}</div>
    </div>`;

  const ta = el.querySelector('textarea');
  const count = el.querySelector('.ex-count');
  const openBtn = el.querySelector('.ex-open');
  const model = el.querySelector('.ex-model');
  const savedTag = el.querySelector('.ex-saved');
  ta.value = saved.text || '';

  const used = () => {
    const t = ta.value;
    let n = 0;
    for (const [i, w] of spec.words.entries()) {
      const hit = (w.any || [w.t]).some((s) => t.includes(s));
      el.querySelector(`.ex-words i[data-i="${i}"]`).classList.toggle('used', hit);
      if (hit) n++;
    }
    return n;
  };

  const sync = () => {
    const n = used();
    const len = ta.value.trim().length;
    count.textContent = `${len} 文字・使えた言葉 ${n} / ${spec.words.length}`;
    openBtn.disabled = len < MIN && !saved.opened;
    openBtn.title = openBtn.disabled ? `まず ${MIN} 文字くらい自分で書いてみてください` : '';
  };

  let timer = null;
  ta.addEventListener('input', () => {
    sync();
    clearTimeout(timer);
    timer = setTimeout(() => {
      saveExplain(spec.id, { text: ta.value, checks: current(), opened: !model.hidden });
      savedTag.hidden = false;
      setTimeout(() => { savedTag.hidden = true; }, 1400);
    }, 600);
  });

  const current = () => [...el.querySelectorAll('.ex-check input')].map((c) => c.checked);

  el.querySelector('.ex-check').addEventListener('change', () => {
    saveExplain(spec.id, { text: ta.value, checks: current(), opened: true });
  });

  openBtn.addEventListener('click', () => {
    model.hidden = false;
    openBtn.hidden = true;
    saveExplain(spec.id, { text: ta.value, checks: current(), opened: true });
  });

  if (saved.opened) {
    model.hidden = false;
    openBtn.hidden = true;
    for (const [i, c] of (saved.checks || []).entries()) {
      const box = el.querySelector(`.ex-check input[data-i="${i}"]`);
      if (box) box.checked = !!c;
    }
  }
  sync();
}
