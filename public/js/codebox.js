// 「このページのコード」：実際に動いている JavaScript と、PyTorch で書いた場合の対応。
// 画面で見ていたものが、実務のコードのどの行にあたるかをつなぐ。

export function codeBox(el, { title = 'この画面を動かしているコード', items }) {
  el.innerHTML = items.map((it, i) => `
    <details class="codebox"${i === 0 ? '' : ''}>
      <summary>${it.label}</summary>
      <div class="cb-grid">
        <div>
          <h4>このアプリ（手書きの JavaScript）</h4>
          <pre class="prompt">${esc(it.js)}</pre>
          ${it.file ? `<p class="note">${it.file}</p>` : ''}
        </div>
        <div>
          <h4>PyTorch で書くなら</h4>
          <pre class="prompt">${esc(it.py)}</pre>
          ${it.note ? `<p class="note">${it.note}</p>` : ''}
        </div>
      </div>
    </details>`).join('');
  el.insertAdjacentHTML('afterbegin', `<h2>${title}</h2>`);
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
