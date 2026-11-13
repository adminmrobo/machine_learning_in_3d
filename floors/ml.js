// 머신러닝 연구소 · 실험에서 함께 쓰는 계산 도구와 가상의 데이터
// 모든 숫자는 이 파일과 demos*.js 안에서 계산한다. 바깥 라이브러리는 쓰지 않는다.
export const C = { fg: 0xe9edf9, mu: 0x9aa7c7, a: 0x6fb1ff, b: 0xff9f6b, y: 0xffd76a, p: 0xff8fc7, g: 0x7fe0a8, r: 0xff6b5e, v: 0xb79cff };
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
export const gauss = r => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(Math.PI * 2 * r());
export const f2 = (v, n = 2) => Number(v).toFixed(n).replace('-', '−');
export const pct = v => Math.round(v * 100) + '%';
export const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
export const sgn = v => (v < 0 ? ' − ' : ' + ') + Math.abs(v).toFixed(2);
export const memo = fn => { const c = {}; return x => (x in c ? c[x] : (c[x] = fn(x))); };
export const sig = z => 1 / (1 + Math.exp(-z));
export const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

/* ---------- 선형대수 ---------- */
export function solve(A, b) { const n = b.length; A = A.map(r => r.slice()); b = b.slice();
  for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k; [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    const d = A[i][i] || 1e-12; for (let k = i + 1; k < n; k++) { const f = A[k][i] / d; if (!f) continue; for (let j = i; j < n; j++) A[k][j] -= f * A[i][j]; b[k] -= f * b[i]; } }
  const x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / (A[i][i] || 1e-12); } return x; }
export function lsq(Phi, y, lam, skip0) { const m = Phi[0].length, A = Array.from({ length: m }, () => new Array(m).fill(0)), b = new Array(m).fill(0);
  Phi.forEach((row, k) => { for (let i = 0; i < m; i++) { b[i] += row[i] * y[k]; for (let j = 0; j < m; j++) A[i][j] += row[i] * row[j]; } }); for (let i = skip0 ? 1 : 0; i < m; i++) A[i][i] += lam; return solve(A, b); }

/* ---------- 3D 판 위에 그리기 ---------- */
// 판(가로·세로 −R~R) 안에 들어오는 직선 a·x + b·y + c = 0 의 두 끝
export function clipLine(a, b, c, R = 1.2) { const P = [];
  if (Math.abs(b) > 1e-9) for (const x of [-R, R]) { const y = (-c - a * x) / b; if (Math.abs(y) <= R + 1e-9) P.push([x, y]); }
  if (Math.abs(a) > 1e-9) for (const y of [-R, R]) { const x = (-c - b * y) / a; if (Math.abs(x) <= R + 1e-9) P.push([x, y]); }
  let best = null, bd = 1e-6; for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) { const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]); if (d > bd) { bd = d; best = [...P[i], ...P[j]]; } } return best; }
// 말하는 식구가 서는 쪽을 비워 둔다: 모아(오른쪽 앞) → 판은 오른쪽·곁들이는 물체는 왼쪽, 시아(왼쪽 앞) → 반대
export const LAY = { moa: { sx: 1.9, wx: -4.1 }, sia: { sx: -1.9, wx: 4.1 } };
export const SURF = (host, o) => ({ id: 'f', type: 'surface', x: LAY[host].sx, z: -.7, w: 6.6, pr: .12, ...o });
export const SIDE = (host, o) => ({ x: LAY[host].wx, z: -1.6, ...o });
// 무거운 함수는 격자에 한 번 계산해 두고 사이를 이어 읽는다 (판은 1초에 30번 다시 그려진다)
export function gridFn(f, R = 1.2, n = 44) { const G = new Float32Array((n + 1) * (n + 1)), st = 2 * R / n;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) G[i * (n + 1) + j] = f(-R + j * st, -R + i * st);
  return (x, y) => { const u = clamp((x + R) / st, 0, n - 1e-6), v = clamp((y + R) / st, 0, n - 1e-6), j = Math.floor(u), i = Math.floor(v), a = u - j, b = v - i, k = i * (n + 1) + j;
    return G[k] * (1 - a) * (1 - b) + G[k + 1] * a * (1 - b) + G[k + n + 1] * (1 - a) * b + G[k + n + 2] * a * b; }; }
export const pts = (D, col) => D.map(d => ({ x: d.x, y: d.y, col: col ? col(d) : (d.c ? C.y : C.a) }));
export const curve = (f, c = C.b, w = .05, lo = -1.15, hi = 1.15, n = 60, R = 1.18) => { const L = []; let p = null;
  for (let i = 0; i <= n; i++) { const x = lo + (hi - lo) * i / n, y = f(x), q = Math.abs(y) <= R ? [x, y] : null; if (p && q) L.push([p[0], p[1], q[0], q[1], c, w]); p = q; } return L; };

/* ---------- 가상의 데이터 (x, y는 −1.1~1.1, c는 0 또는 1) ---------- */
const keep = (r, n, gen, ok, gap = .07) => { const o = []; let g = 0; while (o.length < n && g++ < 20000) { const p = gen(r); if (Math.abs(p.x) > 1.08 || Math.abs(p.y) > 1.08) continue; if (ok && !ok(p)) continue; if (o.some(q => Math.hypot(q.x - p.x, q.y - p.y) < gap)) continue; o.push(p); } return o; };
// 두 무리: 오른쪽 위가 노랑(1), 왼쪽 아래가 파랑(0). sp = 퍼짐
export function blobs2(seed, n = 24, sp = .3, dx = .45, dy = .3) { const r = rng(seed);
  return [...keep(r, n, r => ({ x: -dx + gauss(r) * sp, y: -dy + gauss(r) * sp, c: 0 })), ...keep(r, n, r => ({ x: dx + gauss(r) * sp, y: dy + gauss(r) * sp, c: 1 }))]; }
// 나무용: 노랑은 왼쪽 아래 칸, 파랑은 오른쪽 전체와 왼쪽 위 (직선 하나로는 못 가름)
export function treeData(seed, n = 70, noise = .06) { const r = rng(seed);
  return keep(r, n, r => { const x = r() * 2.1 - 1.05, y = r() * 2.1 - 1.05; let c = x < -.05 && y < .1 ? 1 : 0; if (r() < noise) c = 1 - c; return { x, y, c }; }, null, .1); }
export function moons(seed, n = 30, nz = .1) { const r = rng(seed), o = [];
  for (let i = 0; i < n; i++) { const a = Math.PI * i / (n - 1); o.push({ x: -.35 + .6 * Math.cos(a) + gauss(r) * nz, y: -.12 + .6 * Math.sin(a) + gauss(r) * nz, c: 0 }); o.push({ x: .35 - .6 * Math.cos(a) + gauss(r) * nz, y: .12 - .6 * Math.sin(a) + gauss(r) * nz, c: 1 }); }
  return o.map(p => ({ ...p, x: clamp(p.x, -1.1, 1.1), y: clamp(p.y, -1.1, 1.1) })); }
export function circles(seed, n = 30, nz = .08) { const r = rng(seed), o = [];
  for (let i = 0; i < n; i++) { const a = r() * 2 * Math.PI, ri = .32 * Math.sqrt(r()); o.push({ x: ri * Math.cos(a) + gauss(r) * nz * .5, y: ri * Math.sin(a) + gauss(r) * nz * .5, c: 1 });
    const b = r() * 2 * Math.PI, ro = .78 + gauss(r) * nz; o.push({ x: ro * Math.cos(b), y: ro * Math.sin(b), c: 0 }); }
  return o.map(p => ({ ...p, x: clamp(p.x, -1.1, 1.1), y: clamp(p.y, -1.1, 1.1) })); }
// 훈련용·시험용 나누기 (같은 씨앗이면 같은 나눔)
export function split(D, seed, nt) { const q = rng(seed * 7919 + 3), idx = D.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(q() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const te = new Set(idx.slice(0, nt)); return { tr: D.filter((_, i) => !te.has(i)), te: D.filter((_, i) => te.has(i)) }; }
export const acc = (D, p) => mean(D.map(d => ((p(d.x, d.y) > .5 ? 1 : 0) === d.c ? 1 : 0)));
// 1차원 회귀용 점: 물결 곡선 + 잡음
export function wave(seed, n = 30, nz = .14) { const r = rng(seed); return Array.from({ length: n }, (_, i) => { const x = -1.05 + 2.1 * (i + .2 + r() * .6) / n; return { x, y: clamp(.62 * Math.sin(2.8 * x + .3) + gauss(r) * nz, -1.1, 1.1) }; }); }

/* ---------- 결정 트리 (분류: 노랑일 확률 · 회귀: 평균) ----------
   o = { depth, minLeaf, feats(한 번에 볼 변수 수), rand(기준값 무작위), r(난수), reg(true면 회귀), lam(잎 값 벌점), gamma(나눌 최소 이득), w(점의 무게) } */
export function tree(D, o, depth = 0) { const w = o.w || (d => 1), val = ps => { let s = 0, t = 0; ps.forEach(d => { s += w(d) * (o.reg ? d.t : d.c); t += w(d); }); return o.reg ? s / (t + (o.lam || 0)) : (t ? s / t : .5); };
  const leaf = { leaf: 1, v: val(D), n: D.length };
  if (depth >= o.depth || D.length < 2 * (o.minLeaf || 1)) return leaf;
  const imp = ps => { if (!ps.length) return 0; if (o.reg) { let s = 0, t = 0; ps.forEach(d => { s += w(d) * d.t; t += w(d); }); return -(s * s) / (t + (o.lam || 0)); }
    let s = 0, t = 0; ps.forEach(d => { s += w(d) * d.c; t += w(d); }); const p = s / t; return t * 2 * p * (1 - p); };
  const base = imp(D); let best = null;
  let F = ['x', 'y']; if (o.feats === 1) F = [o.r() < .5 ? 'x' : 'y'];
  for (const f of F) { const vs = [...new Set(D.map(d => d[f]))].sort((a, b) => a - b); if (vs.length < 2) continue;
    let cands; if (o.rand) cands = [vs[0] + o.r() * (vs[vs.length - 1] - vs[0])]; else if (o.bins) { const lo = -1.1, st = 2.2 / o.bins; cands = []; for (let k = 1; k < o.bins; k++) cands.push(lo + k * st); } else cands = vs.slice(1).map((v, i) => (v + vs[i]) / 2);
    if (o.cnt) o.cnt.n += cands.length;
    for (const t of cands) { const L = D.filter(d => d[f] < t), R = D.filter(d => d[f] >= t); if (L.length < (o.minLeaf || 1) || R.length < (o.minLeaf || 1)) continue;
      const g = base - imp(L) - imp(R); if (!best || g > best.g) best = { f, t, g, L, R }; } }
  if (!best || best.g <= (o.gamma || 1e-12)) return leaf;
  return { f: best.f, t: best.t, g: best.g, n: D.length, v: leaf.v, l: tree(best.L, o, depth + 1), r: tree(best.R, o, depth + 1) }; }
export const tpred = (T, x, y) => { while (!T.leaf) T = (T.f === 'x' ? x : y) < T.t ? T.l : T.r; return T.v; };
export const leaves = T => T.leaf ? 1 : leaves(T.l) + leaves(T.r);
// 나무가 그은 선분들 (판 좌표, 범위 R 안)
export function tsegs(T, col = C.fg, box = [-1.2, 1.2, -1.2, 1.2], o = []) { if (T.leaf) return o; const [x0, x1, y0, y1] = box;
  if (T.f === 'x') { o.push([T.t, y0, T.t, y1, col]); tsegs(T.l, col, [x0, T.t, y0, y1], o); tsegs(T.r, col, [T.t, x1, y0, y1], o); }
  else { o.push([x0, T.t, x1, T.t, col]); tsegs(T.l, col, [x0, x1, y0, T.t], o); tsegs(T.r, col, [x0, x1, T.t, y1], o); } return o; }
export const bootstrap = (D, r) => D.map(() => D[Math.floor(r() * D.length)]);
