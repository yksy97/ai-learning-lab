// 進捗と復習キュー（このブラウザの中だけに保存される）
const KEY = 'mlviz.progress.v1';
const DAY = 24 * 60 * 60 * 1000;
// 間隔反復：思い出せたら次はこの日数だけ先送り
const INTERVALS = [1, 3, 7, 16, 35];

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || { done: {}, quiz: {}, review: {} };
  } catch {
    return { done: {}, quiz: {}, review: {} };
  }
}

function save(p) {
  try {
    localStorage.setItem(KEY, JSON.stringify(p));
  } catch {
    /* プライベートモードなどで保存できない場合は、その場かぎりで動く */
  }
}

export function getProgress() {
  return load();
}

export function isDone(id) {
  return !!load().done[id];
}

export function setDone(id, on, terms = []) {
  const p = load();
  if (on) {
    p.done[id] = Date.now();
    // 覚えるべき用語を復習キューに入れる
    for (const t of terms) if (!p.review[t]) p.review[t] = { level: 0, due: Date.now() };
  } else {
    delete p.done[id];
  }
  save(p);
  return p;
}

export function recordQuiz(id, correct) {
  const p = load();
  const q = p.quiz[id] || { tries: 0, correct: 0 };
  q.tries++;
  if (correct) q.correct++;
  q.last = Date.now();
  p.quiz[id] = q;
  save(p);
}

export function quizStats() {
  const p = load();
  const ids = Object.keys(p.quiz);
  const tries = ids.reduce((s, i) => s + p.quiz[i].tries, 0);
  const correct = ids.reduce((s, i) => s + p.quiz[i].correct, 0);
  return { tries, correct };
}

/** 今日ぶんの復習（期限が来ている用語） */
export function dueTerms(limit = 5) {
  const p = load();
  const now = Date.now();
  return Object.entries(p.review)
    .filter(([, v]) => v.due <= now)
    .sort((a, b) => a[1].due - b[1].due)
    .slice(0, limit)
    .map(([id, v]) => ({ id, level: v.level }));
}

export function reviewCount() {
  return Object.keys(load().review).length;
}

/** 思い出せたら次の間隔へ、あやしければ最初に戻す */
export function reviewed(id, ok) {
  const p = load();
  const r = p.review[id] || { level: 0 };
  r.level = ok ? Math.min(INTERVALS.length - 1, r.level + 1) : 0;
  r.due = Date.now() + INTERVALS[r.level] * DAY;
  p.review[id] = r;
  save(p);
}

export function addReviewTerms(terms) {
  const p = load();
  for (const t of terms) if (!p.review[t]) p.review[t] = { level: 0, due: Date.now() };
  save(p);
}

export function resetAll() {
  save({ done: {}, quiz: {}, review: {} });
}
