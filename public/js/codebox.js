// 「コードで見る」：実務で書く Python（PyTorch）と、この画面の中で動いている JavaScript の対応。
// 主役は Python。画面で見ていたものが、実務のコードのどの行にあたるかをつなぐ。

export function codeBox(el, { title = 'コードで見る（Python）', items }) {
  el.innerHTML = items.map((it) => `
    <details class="codebox">
      <summary>${it.label}</summary>
      <div class="cb-grid">
        <div>
          <h4>実務ではこう書く（PyTorch）</h4>
          <pre class="prompt">${esc(it.py)}</pre>
          ${it.note ? `<p class="note">${it.note}</p>` : ''}
        </div>
        <div>
          <h4>この画面の中では（JavaScript・参考）</h4>
          <pre class="prompt">${esc(it.js)}</pre>
          ${it.file ? `<p class="note">${it.file}</p>` : ''}
        </div>
      </div>
    </details>`).join('');
  el.insertAdjacentHTML('afterbegin',
    `<h2>${title}</h2><p class="note" style="margin-top:0">左が、実務で書くことになる Python のコードです。` +
    `右は同じ処理をこの画面の中で書いたもので、ライブラリが内部でやっていることを確かめたいときに見てください。</p>`);
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
