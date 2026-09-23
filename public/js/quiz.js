// 予想クイズ／確認問題のウィジェット
// 「動かす前に予想する」と記憶に残りやすい。正解でも外しても、そのあと実物を見るのが大事。
import { recordQuiz } from './progress.js';

/**
 * @param {HTMLElement} el
 * @param {{id:string, q:string, options:(string|{t:string, why?:string})[], answer:number,
 *          explain:string, kind?:'predict'|'check'}} spec
 *   options は文字列でも { t: 表示文, why: 選んだときに返す理由 } でもよい。
 * @param {(correct:boolean)=>void} [onDone]
 */
export function renderQuiz(el, spec, onDone) {
  el.className = 'quiz';
  const opts = spec.options.map((o) => (typeof o === 'string' ? { t: o } : o));
  el.innerHTML = `
    <div class="q-head">${spec.kind === 'check' ? '確認問題' : '動かす前に予想してみる'}</div>
    <div class="q-body">${spec.q}</div>
    <div class="q-options">${opts.map((o, i) => `<button data-i="${i}">${o.t}</button>`).join('')}</div>
    <div class="q-feedback" hidden></div>`;
  const fb = el.querySelector('.q-feedback');
  el.querySelector('.q-options').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || el.dataset.answered === '1') return;
    el.dataset.answered = '1';
    const i = +b.dataset.i;
    const ok = i === spec.answer;
    recordQuiz(spec.id, ok);
    for (const btn of el.querySelectorAll('.q-options button')) {
      btn.disabled = true;
      if (+btn.dataset.i === spec.answer) btn.classList.add('correct');
      else if (+btn.dataset.i === i) btn.classList.add('wrong');
    }
    fb.hidden = false;
    const why = !ok && opts[i].why ? `<div class="q-why">選んだ「${opts[i].t}」が違う理由：${opts[i].why}</div>` : '';
    fb.innerHTML = `<b>${ok ? '正解' : 'おしい'}：${opts[spec.answer].t}</b><br>${spec.explain}${why}`;
    if (onDone) onDone(ok);
  });
}
