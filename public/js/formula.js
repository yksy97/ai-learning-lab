// 「生きた数式」：式の中に、いま画面で動いている実際の数値を入れて表示する。
// 項にマウスを乗せると、画面の対応する場所が光る。
//
// 使い方:
//   const f = new LiveFormula(el, {
//     rows: [{ label: 'D の損失', tpl: 'L_D = −log {{dr}} − log(1 − {{df}}) = {{ld}}',
//              slots: { dr: { hl: '#main', tip: '本物への判定 D(x)' }, ... } }],
//     onHighlight: (target, on) => { ... },   // 省略可
//   });
//   f.update({ dr: 0.62, df: 0.34, ld: 0.89 });

const fmt = (v) => (typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)) : String(v));

export class LiveFormula {
  /**
   * @param {HTMLElement} el
   * @param {{rows: {label?:string, tpl:string, note?:string, slots?:Object}[], onHighlight?:Function}} opt
   */
  constructor(el, opt) {
    this.el = el;
    this.opt = opt;
    this.slots = new Map();
    el.classList.add('formula-box');
    el.innerHTML = opt.rows.map((row, ri) => {
      const body = row.tpl.replace(/\{\{(\w+)\}\}/g, (_, id) => {
        const s = (row.slots && row.slots[id]) || {};
        return `<span class="f-val" data-row="${ri}" data-id="${id}"${s.hl ? ` data-hl="${s.hl}"` : ''}${s.tip ? ` title="${s.tip}"` : ''}>–</span>`;
      });
      return `<div class="f-row">${row.label ? `<span class="f-label">${row.label}</span>` : ''}<span class="f-body">${body}</span>${row.note ? `<span class="f-note">${row.note}</span>` : ''}</div>`;
    }).join('');

    for (const span of el.querySelectorAll('.f-val')) {
      this.slots.set(`${span.dataset.row}:${span.dataset.id}`, span);
      const hl = span.dataset.hl;
      if (!hl) continue;
      span.addEventListener('pointerenter', () => this.highlight(hl, true, span));
      span.addEventListener('pointerleave', () => this.highlight(hl, false, span));
    }
  }

  highlight(target, on, span) {
    span.classList.toggle('f-active', on);
    if (this.opt.onHighlight) this.opt.onHighlight(target, on);
    if (target.startsWith('#') || target.startsWith('.')) {
      for (const t of document.querySelectorAll(target)) t.classList.toggle('hl-target', on);
    }
  }

  /**
   * 画面 → 式の逆引き。指定した項を一時的に光らせ、式を画面内までスクロールする。
   * @param {string[]} ids 光らせる項（row をまたいで同じ id をすべて対象にする）
   */
  focus(ids) {
    clearTimeout(this._flashTimer);
    const targets = [];
    for (const [key, span] of this.slots) {
      if (ids.includes(key.split(':')[1])) targets.push(span);
    }
    if (!targets.length) return;
    for (const t of targets) t.classList.add('f-flash');
    this.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    this._flashTimer = setTimeout(() => {
      for (const t of targets) t.classList.remove('f-flash');
    }, 2200);
  }

  /**
   * 「画面のこの部分は、式のどこ？」を押せるようにする。
   * @param {Object<string, string[]>} map セレクタ → 項の id
   */
  linkSources(map) {
    for (const [sel, ids] of Object.entries(map)) {
      for (const el of document.querySelectorAll(sel)) {
        el.classList.add('f-source');
        el.title = '式のどこにあたるかを見る';
        el.addEventListener('click', (e) => {
          e.stopPropagation();
          this.focus(ids);
        });
      }
    }
  }

  /** @param {Object} values 行ごとに分けない場合は { id: value }、分ける場合は [{id: value}, …] */
  update(values) {
    const rows = Array.isArray(values) ? values : [values];
    for (const [key, span] of this.slots) {
      const [ri, id] = key.split(':');
      const src = rows.length === 1 ? rows[0] : rows[+ri] || {};
      if (src && id in src) span.textContent = fmt(src[id]);
    }
  }
}
