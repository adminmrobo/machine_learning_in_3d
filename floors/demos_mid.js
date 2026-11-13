// 머신러닝 연구소 · 중층부(10~18층) 실험과 구역 ①③④
// 형식은 demos.js 맨 위 설명과 같다. 점은 모두 설명을 위해 만든 가상의 데이터다.
import { C, clamp, rng, gauss, f2, pct, mean, memo, sig, d2, clipLine, SURF, SIDE, LAY, gridFn, pts, blobs2, treeData, moons, split, acc, tree, tpred, leaves, tsegs, bootstrap, bigTest } from './ml.js';

const K = { i: '#ffd76a', o: '#ff9f6b', bx: '#6fb1ff', gt: '#b79cff', g: '#7fe0a8', p: '#ff8fc7' };
const PAL = [C.y, C.a, C.v, C.g, C.p, C.b, C.r, 0x8fe3d0];
export const DEMOS = {}, EXTRA = {};
const ellSegs = (m, S, k, col, n = 36) => { // 공분산 S의 k배 타원 둘레
  const a = S[0][0], b = S[0][1], c = S[1][1], tr = (a + c) / 2, dt = Math.sqrt(Math.max(0, ((a - c) / 2) ** 2 + b * b)), l1 = tr + dt, l2 = Math.max(1e-9, tr - dt), ang = .5 * Math.atan2(2 * b, a - c);
  const p = t => { const u = k * Math.sqrt(l1) * Math.cos(t), v = k * Math.sqrt(l2) * Math.sin(t); return [m[0] + u * Math.cos(ang) - v * Math.sin(ang), m[1] + u * Math.sin(ang) + v * Math.cos(ang)]; };
  return Array.from({ length: n }, (_, i) => [...p(i / n * 2 * Math.PI), ...p((i + 1) / n * 2 * Math.PI), col]); };

/* ================= 10층 계층적 군집 (시아) ================= */
const HC = (() => { const r = rng(10), D = [];
  [[-.6, .45, 5], [-.35, -.5, 5], [.5, .15, 6]].forEach(([cx, cy, n]) => { for (let i = 0; i < n; i++) D.push({ x: clamp(cx + gauss(r) * .17, -1.05, 1.05), y: clamp(cy + gauss(r) * .17, -1.05, 1.05) }); });
  const link = { single: (A, B) => Math.min(...A.flatMap(i => B.map(j => Math.sqrt(d2([D[i].x, D[i].y], [D[j].x, D[j].y]))))), complete: (A, B) => Math.max(...A.flatMap(i => B.map(j => Math.sqrt(d2([D[i].x, D[i].y], [D[j].x, D[j].y]))))),
    average: (A, B) => mean(A.flatMap(i => B.map(j => Math.sqrt(d2([D[i].x, D[i].y], [D[j].x, D[j].y]))))) };
  const run = memo(kind => { let cl = D.map((_, i) => ({ m: [i], id: i, h: 0 })), next = D.length; const merges = [];
    while (cl.length > 1) { let best = null; for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const v = link[kind](cl[i].m, cl[j].m); if (!best || v < best.v) best = { i, j, v }; }
      const A = cl[best.i], B = cl[best.j], N = { m: [...A.m, ...B.m], id: next++, h: best.v, a: A, b: B }; merges.push(N); cl = cl.filter((_, k) => k !== best.i && k !== best.j); cl.push(N); }
    return merges; });
  const labels = (kind, k) => { const M = run(kind), roots = new Set(D.map((_, i) => i)); const node = {}; D.forEach((_, i) => node[i] = { m: [i], id: i, h: 0 });
    M.slice(0, D.length - k).forEach(N => { roots.delete(N.a.id); roots.delete(N.b.id); roots.add(N.id); node[N.id] = N; });
    const lab = new Array(D.length); [...roots].sort((a, b) => Math.min(...node[a].m) - Math.min(...node[b].m)).forEach((id, c) => node[id].m.forEach(i => lab[i] = c)); return lab; };
  return { D, run, labels }; })();
const cutH = (M, n, k) => { const lo = k >= n ? 0 : M[n - k - 1].h, hi = k <= 1 ? M[n - 2].h * 1.08 : M[n - k].h; return (lo + hi) / 2; };
const hcCut = memo(k => cutH(HC.run('average'), HC.D.length, k));
DEMOS.HAC = {
  vt: '가까운 묶음끼리 합치고, 원하는 높이에서 자르기',
  note: '점 16개는 설명을 위해 만든 가상의 데이터이고, 정답은 없습니다. 오른쪽 나무 그림(덴드로그램)에서 두 가지가 만나는 높이가 그 둘을 합칠 때의 거리입니다.',
  controls: [{ k: 'k', l: '묶음 수', min: 1, max: 8, step: 1, v: 3 }, { k: 'lk', l: '묶음 사이 거리', opts: [['average', '평균 거리'], ['single', '가장 가까운 두 점'], ['complete', '가장 먼 두 점']], v: 'average' }],
  init(s) { this.change(s); },
  change(s) { s.lab = HC.labels(s.lk, s.k); const M = HC.run(s.lk); s.cut = cutH(M, HC.D.length, s.k); s.top = M[M.length - 1].h * 1.08; },
  read(s) { return [['묶음 수', s.k], ['자르는 높이 (거리)', f2(s.cut, 2)]]; },
  say(s) { const nm = { average: '평균 거리', single: '가장 가까운 두 점', complete: '가장 먼 두 점' }[s.lk];
    if (s.k === 1) return '모든 점이 한 묶음입니다. 나무 그림의 꼭대기에서 자르면 이렇게 됩니다. 아래로 내려가며 잘라 보십시오.';
    return '나무 그림을 높이 ' + f2(s.cut) + '에서 가로로 자르면 가지가 <b>' + s.k + '개</b> 남습니다. 묶음 수를 정하지 않고 나무를 한 번 만들어 두면, <b>자르는 높이만 바꿔</b> 여러 답을 얻습니다. 지금은 "' + nm + '"로 거리를 잽니다.'; },
  view(s) { const host = 'sia', D = HC.D, M = HC.run(s.lk), n = D.length, order = [], walk = N => N.a ? (walk(N.a), walk(N.b)) : order.push(N.id); walk(M[M.length - 1]);
    const X = {}, Y = {}, W = 3.4, H = 3.2, nodes = [], links = [], idx = {}; order.forEach((id, i) => { X[id] = -W / 2 + W * i / (n - 1); Y[id] = .35; });
    const col = id => { const m = id < n ? [id] : M[id - n].m; const c = s.lab[m[0]]; return m.every(i => s.lab[i] === c) ? PAL[c] : C.mu; };
    order.forEach(id => { idx[id] = nodes.length; nodes.push({ p: [X[id], Y[id], 0], c: PAL[s.lab[id]], r: .1 }); });
    M.forEach(N => { X[N.id] = (X[N.a.id] + X[N.b.id]) / 2; Y[N.id] = .35 + N.h / s.top * H; idx[N.id] = nodes.length; nodes.push({ p: [X[N.id], Y[N.id], 0], c: col(N.id), r: .07, glow: .1 });
      [N.a.id, N.b.id].forEach(ch => { const k = nodes.length; nodes.push({ p: [X[ch], Y[N.id], 0], c: col(ch), r: .035, glow: 0 }); links.push([idx[ch], k, .25, col(ch)], [k, idx[N.id], .25, col(N.id)]); }); });
    const cy = .35 + s.cut / s.top * H, ca = nodes.length; nodes.push({ p: [-W / 2 - .3, cy, .05], c: C.b, r: .06, t: '자르는 높이' }, { p: [W / 2 + .3, cy, .05], c: C.b, r: .06 }); links.push([ca, ca + 1, .3, C.b]);
    return [SURF(host, { x: LAY[host].sx - .3, w: 6, pts: D.map((d, i) => ({ x: d.x, y: d.y, col: PAL[s.lab[i]] })) }),
      { id: 'n', type: 'nodes', x: LAY[host].wx + .2, z: -1.4, nodes, links, title: '나무 그림 (덴드로그램)', ty: H + 1.1 }]; },
};
EXTRA.HAC = { say: '정답 없는 점들을 넣으면, 묶음이 합쳐지는 순서를 담은 나무 그림이 나와요!', io: ['정답 없는 점들', '나무 그림 → 자르면 묶음'],
  part: { items: [['ball', K.i, '점마다 따로 한 묶음'], ['box', K.bx, '묶음 사이 거리 모두 재기'], ['gate', K.gt, '가장 가까운 두 묶음 합치기'], ['tiles', K.g, '나무 그림에 높이 기록'], ['ball', K.o, '원하는 높이에서 자르기']], loops: [[2, 1, '한 묶음이 될 때까지']],
    t: ['처음에는 점 하나하나가 각자 한 묶음입니다.', '가장 가까운 두 묶음을 합치고, 그때의 거리를 나무 그림의 높이로 적습니다.', '모두 한 묶음이 될 때까지 되풀이한 뒤, 원하는 높이에서 가로로 잘라 묶음을 얻습니다.'], say: '묶음 사이 거리를 어떻게 재느냐(가장 가까운 점, 가장 먼 점, 평균)에 따라 나무 모양이 달라집니다. 점이 많으면 거리를 모두 재야 해서 느립니다.' },
  curve: { f: hcCut, lo: 1, hi: 8, step: 1, v: 3, xl: '묶음 수', yl: '자르는 높이', title: '묶음 수와 자르는 높이 (평균 거리)', t: ['묶음 수를 하나 줄일 때마다 더 먼 두 묶음을 합쳐야 합니다.', '높이가 갑자기 뛰는 곳(여기서는 3 → 2)이 억지로 합치기 시작하는 곳, 즉 자연스러운 묶음 수입니다.'] } };

/* ================= 11층 선형 판별 분석 (모아) ================= */
const LD = (() => { const r = rng(111), D = [], t = 62 * Math.PI / 180, mk = (cx, cy, c) => { let g = 0; while (D.filter(d => d.c === c).length < 22 && g++ < 999) { const a = gauss(r) * .42, b = gauss(r) * .1, x = cx + a * Math.cos(t) - b * Math.sin(t), y = cy + a * Math.sin(t) + b * Math.cos(t); if (Math.abs(x) < 1.08 && Math.abs(y) < 1.08) D.push({ x, y, c }); } };
  mk(-.27, .05, 0); mk(.27, -.05, 1);
  const st = c => { const P = D.filter(d => d.c === c), m = [mean(P.map(d => d.x)), mean(P.map(d => d.y))]; return { P, m, S: [[mean(P.map(d => (d.x - m[0]) ** 2)), mean(P.map(d => (d.x - m[0]) * (d.y - m[1])))], [0, mean(P.map(d => (d.y - m[1]) ** 2))]] }; };
  const A = st(0), B = st(1); [A, B].forEach(q => q.S[1][0] = q.S[0][1]);
  const J = deg => { const th = deg * Math.PI / 180, u = [Math.cos(th), Math.sin(th)], pr = (m) => m[0] * u[0] + m[1] * u[1], v = S => u[0] * u[0] * S[0][0] + 2 * u[0] * u[1] * S[0][1] + u[1] * u[1] * S[1][1]; return (pr(B.m) - pr(A.m)) ** 2 / (v(A.S) + v(B.S)); };
  const Sw = [[A.S[0][0] + B.S[0][0], A.S[0][1] + B.S[0][1]], [A.S[0][1] + B.S[0][1], A.S[1][1] + B.S[1][1]]], dt = Sw[0][0] * Sw[1][1] - Sw[0][1] ** 2, dm = [B.m[0] - A.m[0], B.m[1] - A.m[1]];
  const w = [(Sw[1][1] * dm[0] - Sw[0][1] * dm[1]) / dt, (-Sw[0][1] * dm[0] + Sw[0][0] * dm[1]) / dt], best = ((Math.atan2(w[1], w[0]) * 180 / Math.PI) + 180) % 180;
  return { D, A, B, J, best }; })();
DEMOS.LDA = {
  vt: '두 무리가 가장 잘 갈려 보이는 방향',
  note: '노랑·파랑 점 44개는 설명을 위해 만든 가상의 데이터입니다. 두 무리 모두 비스듬히 길쭉합니다. 굵은 선은 "그림자를 비추는 방향", 오른쪽 판은 그 방향으로 비춘 그림자의 분포입니다.',
  controls: [{ k: 'th', l: '비추는 방향 (도)', min: 0, max: 179, step: 1, v: 0 }, { b: 'best', l: '가장 잘 갈리는 방향 찾기' }],
  init(s) { s.auto = 0; },
  change(s) { s.auto = 0; },
  click(s) { s.auto = 1; },
  step(s, dt) { if (!s.auto) return; const d = ((LD.best - s.th + 270) % 180) - 90; s.th = (s.th + d * Math.min(1, dt * 3) + 180) % 180; if (Math.abs(d) < .2) { s.th = LD.best; s.auto = 0; } },
  proj(s) { const t = s.th * Math.PI / 180, u = [Math.cos(t), Math.sin(t)], z = LD.D.map(d => d.x * u[0] + d.y * u[1]), za = mean(z.filter((_, i) => !LD.D[i].c)), zb = mean(z.filter((_, i) => LD.D[i].c)), mid = (za + zb) / 2, sgn = zb > za ? 1 : -1;
    return { u, z, mid, err: LD.D.filter((d, i) => ((z[i] - mid) * sgn > 0 ? 1 : 0) !== d.c).length }; },
  read(s) { return [['갈림 정도 J', f2(LD.J(s.th), 2)], ['잘못 갈린 점', this.proj(s).err + ' / 44']]; },
  say(s) { const j = LD.J(s.th), jb = LD.J(LD.best), mean0 = (((Math.atan2(LD.B.m[1] - LD.A.m[1], LD.B.m[0] - LD.A.m[0]) * 180 / Math.PI) + 180) % 180);
    if (j > jb * .97) return '가장 잘 갈리는 방향이에요! 그림자끼리 겹침이 가장 적어요. 신기한 건 이 방향이 <b>두 중심을 잇는 방향(' + Math.round(mean0) + '°)과 달라요.</b> 무리가 길쭉한 모양까지 따져서 고르거든요.';
    return '지금 방향에서는 갈림 정도가 가장 좋은 방향의 ' + pct(j / jb) + '예요. J는 <b>중심 사이 거리²</b>를 <b>무리 안 퍼짐</b>으로 나눈 값이에요. 중심은 멀리, 퍼짐은 작게 보이는 방향을 찾아보세요.'; },
  view(s) { const host = 'moa', P = this.proj(s), u = P.u, L = clipLine(-u[1], u[0], 0), bd = clipLine(u[0], u[1], -P.mid), nb = 16, lo = -1.2, hi = 1.2, h = [new Array(nb).fill(0), new Array(nb).fill(0)];
    P.z.forEach((z, i) => { const b = clamp(Math.floor((z - lo) / (hi - lo) * nb), 0, nb - 1); h[LD.D[i].c][b]++; });
    return [SURF(host, { pts: pts(LD.D), lines: [...(L ? [[...L, C.fg, .045]] : []), ...(bd ? [[...bd, C.b, .05]] : [])], segs: LD.D.map((d, i) => [d.x, d.y, P.z[i] * u[0], P.z[i] * u[1], d.c ? C.y : C.a]), marks: LD.D.map((d, i) => ({ x: P.z[i] * u[0], y: P.z[i] * u[1], c: d.c ? C.y : C.a, s: .05 })) }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: 0, ymax: 10, series: [{ v: h[0], c: C.a, n: '파랑 그림자', w: .07 }, { v: h[1], c: C.y, n: '노랑 그림자', w: .07 }], title: '그 방향으로 비춘 그림자' }]; },
};
EXTRA.LDA = { say: '점의 위치를 넣으면, 갈림이 가장 잘 보이는 축 위의 좌표와 어느 무리인지가 나와요!', io: ['점의 위치 (변수 여러 개)', '무리 + 갈림 축 위의 좌표'],
  part: { items: [['tiles', K.i, '무리별 데이터'], ['box', K.bx, '무리마다 중심 · 퍼짐'], ['box', K.gt, '갈림 방향 w = S⁻¹(중심 차)'], ['ball', K.o, '그 축에 비추기'], ['gate', K.g, '두 중심 가운데서 가르기']],
    t: ['무리마다 중심과 퍼짐(공분산)을 구합니다. 모든 무리의 퍼짐 모양이 같다고 가정합니다.', '중심 차이를 퍼짐으로 나눠 준 방향이 가장 잘 갈리는 방향입니다.', '점을 그 방향에 비춘 그림자로 바꾸고, 두 중심의 가운데에서 가릅니다.'], say: '주성분 분석처럼 축 하나를 고르지만, 정답 색을 씁니다. PCA는 "넓게 퍼진 방향", LDA는 "잘 갈리는 방향"입니다.' },
  curve: { f: LD.J, lo: 0, hi: 179, v: 0, xl: '비추는 방향 (도)', yl: '갈림 정도 J', title: '방향을 돌리며 잰 갈림 정도', t: ['봉우리가 LDA가 고르는 방향입니다.', '중심을 잇는 방향은 봉우리가 아닙니다. 길쭉한 퍼짐이 겹침을 만들기 때문입니다.'] } };

/* ================= 12층 서포트 벡터 머신 (시아) ================= */
const SV = (() => { const D = blobs2(5, 16, .2, .45, .3);
  const at = deg => { const t = deg * Math.PI / 180, u = [Math.cos(t), Math.sin(t)], z = D.map(d => d.x * u[0] + d.y * u[1]), a = Math.max(...z.filter((_, i) => !D[i].c)), b = Math.min(...z.filter((_, i) => D[i].c)); return { u, z, a, b, m: (b - a) / 2, mid: (a + b) / 2 }; };
  let best = 0, bm = -9; for (let d = 0; d < 360; d += .25) { const q = at(d); if (q.m > bm) { bm = q.m; best = d; } } return { D, at, best }; })();
DEMOS.SVM = {
  vt: '두 무리 사이의 길을 가장 넓게',
  note: '노랑·파랑 점 32개는 설명을 위해 만든 가상의 데이터이고, 직선 하나로 깨끗이 갈립니다. 가운데 굵은 선이 경계, 분홍 선 두 개가 길의 가장자리, 고리를 두른 점이 길에 닿은 점(서포트 벡터)입니다.',
  controls: [{ k: 'th', l: '경계에 수직인 방향 (도)', min: 0, max: 359, step: .5, v: 70 }, { b: 'best', l: '가장 넓은 길 찾기' }],
  init(s) { s.auto = 0; },
  change(s) { s.auto = 0; },
  click(s) { s.auto = 1; },
  step(s, dt) { if (!s.auto) return; const d = ((SV.best - s.th + 540) % 360) - 180; s.th = (s.th + d * Math.min(1, dt * 3) + 360) % 360; if (Math.abs(d) < .1) { s.th = SV.best; s.auto = 0; } },
  read(s) { const q = SV.at(s.th); return [['길 폭', q.m > 0 ? f2(2 * q.m, 3) : '겹침'], ['가장 넓은 길 폭', f2(2 * SV.at(SV.best).m, 3)]]; },
  say(s) { const q = SV.at(s.th), b = SV.at(SV.best).m;
    if (q.m <= 0) return '이 방향에서는 노랑과 파랑이 <b>겹칩니다.</b> 길을 낼 수 없습니다. 방향을 돌려 보십시오.';
    if (q.m > b * .99) return '가장 넓은 길입니다. 경계는 <b>길에 닿은 몇 개의 점</b>만으로 정해졌습니다. 나머지 점은 움직여도 경계가 바뀌지 않습니다. 그래서 이름이 "서포트(받치는) 벡터"입니다.';
    return '두 무리를 가르는 직선은 이것 말고도 많습니다. SVM은 그중 <b>길 폭이 가장 넓은</b> 선을 고릅니다. 지금 길 폭은 가장 넓은 길의 ' + pct(q.m / b) + '입니다.'; },
  view(s) { const host = 'sia', q = SV.at(s.th), u = q.u, L = (o, c, w) => { const l = clipLine(u[0], u[1], -o); return l ? [[...l, c, w]] : []; }, sv = SV.D.map((d, i) => q.m > 0 && Math.abs(Math.abs(q.z[i] - q.mid) - q.m) < 1e-6);
    return [SURF(host, { pts: pts(SV.D), ring: sv, lines: [...L(q.mid, C.b, .055), ...(q.m > 0 ? [...L(q.a, C.p, .03), ...L(q.b, C.p, .03)] : [])] }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [Math.max(0, 2 * q.m), 2 * SV.at(SV.best).m], colors: [C.p, C.g], max: .8, labels: ['지금 길', '가장 넓은 길'], title: '길 폭' }]; },
};
EXTRA.SVM = { say: '점의 위치를 넣으면, 경계의 어느 쪽인지(노랑/파랑)가 나와요!', io: ['점의 위치 (변수 여러 개)', '경계의 어느 쪽인가'],
  part: { items: [['ball', K.i, '입력 x'], ['box', K.bx, '점수 w·x + b'], ['gate', K.gt, '부호로 가르기'], ['ball', K.p, '길에 닿은 점(서포트 벡터)만 기억'], ['box', K.g, '길 폭 2/‖w‖ 최대로']],
    t: ['로지스틱 회귀처럼 점수 w·x + b의 부호로 가릅니다.', '고를 때는 "가장 가까운 점까지의 거리(길 폭)"를 가장 크게 하는 w, b를 찾습니다.', '답이 정해지면 경계는 길에 닿은 점 몇 개로만 결정됩니다.'], say: '틀린 점이 조금 있을 때는 "길을 침범한 만큼 벌점(C)"을 더해 타협합니다. 이 층은 깨끗이 갈리는 경우만 봅니다.' },
  curve: { f: d => 2 * SV.at(d).m, lo: 0, hi: 359, v: 70, xl: '경계에 수직인 방향 (도)', yl: '길 폭 (음수 = 겹침)', title: '방향마다 낼 수 있는 길 폭', t: ['0보다 위인 방향에서만 두 무리를 깨끗이 가를 수 있습니다.', '가장 높은 봉우리가 SVM이 고르는 경계입니다.'] } };

/* ================= 13·14층 랜덤 포레스트 · 엑스트라 트리 ================= */
const FD = (() => { const D = treeData(23, 110, .12); return { tr: split(D, 9, 36).tr, te: bigTest.tree(1023, .12) }; })();
const grow = memo(key => { const [mode, seed] = key.split(':'), r = rng(+seed * 131 + (mode === 'et' ? 7 : 0)), cnt = { n: 0 };
  const T = Array.from({ length: 40 }, () => tree(mode === 'et' ? FD.tr : bootstrap(FD.tr, r), { depth: 7, minLeaf: 2, feats: 1, rand: mode === 'et', r, cnt }));
  const G = T.map(t => gridFn((x, y) => tpred(t, x, y), 1.2, 40)); return { T, G, per: cnt.n / 40 }; });
const forestP = (F, n) => (x, y) => { let s = 0; for (let i = 0; i < n; i++) s += F.G[i](x, y); return s / n; };
const forestAcc = (F, n, D) => mean(D.map(d => { let s = 0; for (let i = 0; i < n; i++) s += tpred(F.T[i], d.x, d.y); return (s / n > .5 ? 1 : 0) === d.c ? 1 : 0; }));
const rfCurve = memo(n => forestAcc(grow('rf:1'), n, FD.te)), etCurve = memo(n => forestAcc(grow('et:1'), n, FD.te));
DEMOS.RF = {
  vt: '나무를 늘릴수록 경계가 차분해진다',
  note: '점 74개로 배우고, 같은 방식으로 따로 뽑은 점 400개로 채점합니다. 설명을 위해 만든 가상의 데이터이고, 잡음을 섞었습니다. 나무마다 훈련 점을 중복 허용으로 다시 뽑고, 질문마다 변수 하나만 무작위로 봅니다. 초록 선은 가장 최근에 심은 나무 한 그루의 경계입니다.',
  controls: [{ k: 'n', l: '나무 수', min: 1, max: 40, step: 1, v: 1 }, { b: 'new', l: '숲 다시 심기' }],
  init(s) { s.seed = 1; this.change(s); },
  change(s) { s.F = grow('rf:' + s.seed); s.fn = gridFn(forestP(s.F, s.n), 1.2, 40); s.one = forestAcc(s.F, 1, FD.te); s.te = forestAcc(s.F, s.n, FD.te); },
  click(s) { s.seed++; this.change(s); },
  read(s) { return [['나무 한 그루 · 숲 (시험)', pct(s.one) + ' · ' + pct(s.te)], ['나무 수', s.n]]; },
  say(s) { if (s.n === 1) return '나무 한 그루는 다시 뽑은 데이터에 맞춰져 있어서 <b>들쭉날쭉해요.</b> 나무 수를 올려 보세요.';
    return '나무 ' + s.n + '그루가 투표했어요. 나무마다 틀리는 곳이 달라서, 다수결을 하면 <b>실수가 서로 지워져요.</b> 바닥이 계단 대신 완만한 비탈이 됐어요. 시험 정확도 ' + pct(s.one) + ' → <b>' + pct(s.te) + '</b>.'; },
  view(s) { const host = 'moa';
    return [SURF(host, { fn: s.fn, h: .5, pts: pts(FD.tr), segs: tsegs(s.F.T[s.n - 1], C.g), segY: .56 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.one, s.te], colors: [C.g, C.b], max: 1, labels: ['한 그루', '숲 ' + s.n + '그루'], title: '시험 정확도' }]; },
};
EXTRA.RF = { say: '점의 위치를 넣으면, 나무 수백 그루가 투표한 결과가 나와요!', io: ['점의 위치 (변수 여러 개)', '나무들의 다수결 (또는 평균)'],
  part: { items: [['tiles', K.i, '훈련 데이터'], ['box', K.bx, '중복 허용으로 다시 뽑기'], ['gate', K.gt, '질문마다 변수 일부만'], ['tiles', K.g, '나무 한 그루'], ['ball', K.o, '모든 나무 투표']], loops: [[3, 1, '나무 수만큼']],
    t: ['훈련 데이터를 중복 허용으로 다시 뽑아(부트스트랩) 나무마다 조금 다른 데이터를 줍니다.', '나무가 질문을 고를 때도 변수 일부만 무작위로 보게 해서 나무끼리 더 달라지게 합니다.', '모든 나무에 물어 다수결로 답합니다. 서로 다른 실수가 상쇄됩니다.'], say: '4층 결정 트리를 여러 번 세운 것입니다. 나무끼리 서로 다를수록 숲이 강해집니다.' },
  curve: { f: rfCurve, lo: 1, hi: 40, step: 1, v: 1, xl: '나무 수', yl: '시험 정확도', title: '나무 수와 시험 정확도', t: ['처음 몇 그루에서 빠르게 오르고, 그 뒤로는 거의 평평합니다.', '나무를 더 심어도 과적합이 심해지지 않습니다. 대신 계산이 늘어납니다.'] } };
DEMOS.ET = {
  vt: '기준값까지 무작위로 고르면 어떻게 될까',
  note: '13층과 같은 점(훈련 74개, 따로 뽑은 시험 400개)입니다. 엑스트라 트리는 다시 뽑기 없이 원래 데이터를 쓰고, 기준값을 따져 보지 않고 무작위로 하나 고릅니다.',
  controls: [{ k: 'n', l: '나무 수', min: 1, max: 40, step: 1, v: 10 }, { k: 'mode', l: '기준값 고르는 법', opts: [['et', '무작위 (엑스트라 트리)'], ['rf', '가장 좋은 값 (랜덤 포레스트)']], v: 'et' }],
  init(s) { this.change(s); },
  change(s) { s.F = grow(s.mode + ':1'); s.fn = gridFn(forestP(s.F, s.n), 1.2, 40); s.te = forestAcc(s.F, s.n, FD.te); },
  read(s) { return [['시험 정확도', pct(s.te)], ['질문 하나에 따져 본 기준값', s.mode === 'et' ? '1개' : '평균 ' + Math.round(grow('rf:1').per / Math.max(1, mean(grow('rf:1').T.map(t => leaves(t) - 1)))) + '개']]; },
  say(s) { if (s.mode === 'et') return '기준값을 무작위로 골라서 나무 한 그루는 엉성하지만, ' + s.n + '그루를 모으면 경계가 <b>더 매끈해집니다.</b> 질문마다 기준값을 하나만 보니 <b>심는 속도가 빠릅니다.</b> 시험 정확도 ' + pct(s.te) + '.';
    return '랜덤 포레스트로 바꿨습니다. 질문마다 가능한 기준값을 모두 따져서 나무 한 그루는 더 정확하지만 계산이 많습니다. 시험 정확도 ' + pct(s.te) + '. 바닥의 계단 모양을 엑스트라 트리와 비교해 보십시오.'; },
  view(s) { const host = 'sia';
    return [SURF(host, { fn: s.fn, h: .5, pts: pts(FD.tr), segs: tsegs(s.F.T[s.n - 1], C.g), segY: .56 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [forestAcc(grow('et:1'), s.n, FD.te), forestAcc(grow('rf:1'), s.n, FD.te)], colors: [C.p, C.b], max: 1, labels: ['엑스트라 트리', '랜덤 포레스트'], title: '나무 ' + s.n + '그루의 시험 정확도' }]; },
};
EXTRA.ET = { say: '랜덤 포레스트와 똑같이, 점을 넣으면 나무들의 투표 결과가 나와요!', io: ['점의 위치 (변수 여러 개)', '나무들의 다수결 (또는 평균)'],
  part: { items: [['tiles', K.i, '훈련 데이터 (그대로)'], ['gate', K.gt, '변수 무작위'], ['gate', K.p, '기준값도 무작위'], ['tiles', K.g, '나무 한 그루'], ['ball', K.o, '모든 나무 투표']], loops: [[3, 1, '나무 수만큼']],
    t: ['다시 뽑기 없이 원래 데이터 전체로 나무를 세웁니다.', '변수뿐 아니라 기준값까지 무작위로 골라, 나무끼리 훨씬 달라지게 합니다.', '나무 한 그루는 약하지만 많이 모으면 랜덤 포레스트와 비슷하거나 더 부드러운 경계가 됩니다.'], say: '"가장 좋은 기준값 찾기" 칸을 "아무 값이나"로 바꿨습니다. 찾는 계산이 사라져서 빠릅니다.' },
  curve: { f: etCurve, lo: 1, hi: 40, step: 1, v: 10, xl: '나무 수', yl: '시험 정확도', title: '엑스트라 트리의 나무 수와 시험 정확도', t: ['나무가 적을 때는 무작위 기준값 탓에 흔들립니다.', '나무가 쌓이면 무작위성이 평균으로 지워지며 안정됩니다.'] } };

/* ================= 15층 DBSCAN (모아) ================= */
const DB = (() => { const r = rng(12), D = [], R = .65; for (let i = 0; i < 32; i++) { const t = Math.PI * (i + .5) / 32; D.push({ x: (Math.cos(t) - .5) * R + gauss(r) * .035, y: (Math.sin(t) - .25) * R + gauss(r) * .035 }, { x: (.5 - Math.cos(t)) * R + gauss(r) * .035, y: (.25 - Math.sin(t)) * R + gauss(r) * .035 }); }
  for (let i = 0; i < 7; i++) D.push({ x: r() * 2 - 1, y: r() * 2 - 1 });
  const run = (eps, mp) => { const n = D.length, nb = D.map((p, i) => D.map((q, j) => j).filter(j => Math.hypot(D[j].x - p.x, D[j].y - p.y) <= eps)), core = nb.map(l => l.length >= mp), lab = new Array(n).fill(-1); let c = 0;
    for (let i = 0; i < n; i++) { if (lab[i] !== -1 || !core[i]) continue; const st = [i]; lab[i] = c; while (st.length) { const p = st.pop(); if (!core[p]) continue; nb[p].forEach(q => { if (lab[q] === -1) { lab[q] = c; st.push(q); } }); } c++; }
    return { lab, core, k: c, noise: lab.filter(v => v < 0).length }; };
  return { D, run }; })();
const dbK = memo(e => DB.run(e, 4).k);
DEMOS.DBSCAN = {
  vt: '반경 ε 안에 이웃이 충분하면 이어 붙인다',
  note: '초승달 두 개와 아무 데나 흩뿌린 점 7개, 모두 71개의 가상 데이터입니다. 정답은 주지 않습니다. 고리를 두른 점이 "핵심 점"(반경 안에 이웃이 충분한 점), 회색 점이 잡음입니다.',
  controls: [{ k: 'eps', l: '반경 ε', min: .04, max: .45, step: .01, v: .1 }, { k: 'mp', l: '최소 이웃 수', min: 2, max: 8, step: 1, v: 4 }],
  init(s) { this.change(s); },
  change(s) { s.R = DB.run(s.eps, s.mp); },
  read(s) { return [['덩어리 수', s.R.k], ['잡음으로 남은 점', s.R.noise]]; },
  say(s) { const R = s.R;
    if (R.k === 0) return '반경이 너무 작아서 이웃을 ' + s.mp + '개 가진 점이 하나도 없어요. 전부 <b>잡음</b>이 됐어요.';
    if (R.k === 1 && R.noise < 5) return '반경이 너무 커서 초승달 두 개가 <b>한 덩어리</b>로 이어졌어요. 반경을 줄여 보세요.';
    if (R.k === 2) return '초승달 두 개를 따로 찾았어요! k-평균은 이런 휜 모양을 못 나누지만, DBSCAN은 <b>빽빽한 곳을 따라 이어 붙여서</b> 모양에 상관없이 찾아요. 외딴 점 ' + R.noise + '개는 잡음으로 남겼어요.';
    if (R.k > 2) return '반경이 작아서 초승달이 <b>' + R.k + '조각</b>으로 끊기고, 잡음이 ' + R.noise + '개나 생겼어요. 이웃이 반경 안에 충분히 안 들어와서 사슬이 끊긴 거예요. 반경을 조금 키워 보세요.';
    return '덩어리 ' + R.k + '개, 잡음 ' + R.noise + '개예요. 묶음 수를 정해 주지 않아도 <b>밀도만 보고</b> 알아서 정해져요.'; },
  view(s) { const host = 'moa', D = DB.D, segs = [];
    for (let i = 0; i < D.length; i++) for (let j = i + 1; j < D.length; j++) if ((s.R.core[i] || s.R.core[j]) && Math.hypot(D[i].x - D[j].x, D[i].y - D[j].y) <= s.eps && s.R.lab[i] === s.R.lab[j] && s.R.lab[i] >= 0) segs.push([D[i].x, D[i].y, D[j].x, D[j].y, PAL[s.R.lab[i] % PAL.length]]);
    const ex = [10, 41].filter(i => s.R.core[i]).map(i => ({ x: D[i].x, y: D[i].y, c: C.p, s: .04, r: s.eps }));
    return [SURF(host, { pts: D.map((d, i) => ({ x: d.x, y: d.y, col: s.R.lab[i] < 0 ? C.mu : PAL[s.R.lab[i] % PAL.length] })), ring: s.R.core, segs: segs.slice(0, 600), marks: ex }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.R.k, s.R.noise], colors: [C.y, C.mu], max: 12, labels: ['덩어리', '잡음'], title: '찾은 것' }]; },
};
EXTRA.DBSCAN = { say: '정답 없는 점들을 넣으면, 덩어리 번호와 "잡음" 딱지가 나와요!', io: ['정답 없는 점들', '덩어리 번호 또는 잡음'],
  part: { items: [['ball', K.i, '점 하나'], ['box', K.bx, '반경 ε 안의 이웃 세기'], ['gate', K.gt, '이웃 ≥ 최소 수 → 핵심 점'], ['box', K.g, '핵심 점끼리 이어 붙이기'], ['ball', K.o, '덩어리 · 잡음']], loops: [[3, 1, '모든 점']],
    t: ['점마다 반경 ε 안에 이웃이 몇 개인지 셉니다.', '이웃이 충분한 점은 핵심 점입니다. 서로 반경 안에 있는 핵심 점들을 사슬처럼 이어 한 덩어리로 만듭니다.', '어떤 덩어리에도 닿지 않은 점은 잡음으로 남깁니다.'], say: '묶음 수 대신 반경과 최소 이웃 수를 정합니다. 덩어리마다 빽빽함이 크게 다르면 한 반경으로는 다 잡기 어렵습니다.' },
  curve: { f: dbK, lo: .04, hi: .44, step: .02, v: .2, xl: '반경 ε (최소 이웃 4)', yl: '덩어리 수', title: '반경에 따라 바뀌는 덩어리 수', t: ['반경이 작으면 덩어리가 잘게 쪼개지거나 아예 없습니다.', '반경이 크면 덩어리가 하나로 이어집니다. 가운데 평평한 구간이 믿을 만한 답입니다.'] } };

/* ================= 16층 에이다부스트 (시아) ================= */
const AB = (() => { const D = treeData(21, 90, .05), sp = { tr: split(D, 6, 30).tr, te: bigTest.tree(1021, .05) }, tr = sp.tr, n = tr.length; let w = new Array(n).fill(1 / n); const R = [];
  for (let m = 0; m < 30; m++) { const W = w.slice(), T = tree(tr.map((d, i) => ({ ...d, i })), { depth: 1, w: d => W[d.i] }), h = (x, y) => tpred(T, x, y) > .5 ? 1 : -1;
    let e = 0; tr.forEach((d, i) => { if (h(d.x, d.y) !== (d.c ? 1 : -1)) e += W[i]; }); e = clamp(e, 1e-6, .499); const a = .5 * Math.log((1 - e) / e);
    w = W.map((v, i) => v * Math.exp(-a * (tr[i].c ? 1 : -1) * h(tr[i].x, tr[i].y))); const z = w.reduce((s, v) => s + v, 0); w = w.map(v => v / z); R.push({ T, h, a, e, W, w: w.slice() }); }
  const F = m => (x, y) => { let s = 0; for (let i = 0; i < m; i++) s += R[i].a * R[i].h(x, y); return s; };
  const accm = (m, D) => mean(D.map(d => ((F(m)(d.x, d.y) > 0 ? 1 : 0) === d.c ? 1 : 0)));
  return { tr, te: sp.te, R, F, accm }; })();
const abTe = memo(m => AB.accm(m, AB.te)), abTr = memo(m => AB.accm(m, AB.tr));
DEMOS.AdaBoost = {
  vt: '틀린 점에 무게를 실어 다음 나무에 넘기기',
  note: '점 60개로 배우고, 같은 방식으로 따로 뽑은 점 400개로 채점합니다. 설명을 위해 만든 가상의 데이터입니다. 나무는 질문 하나짜리(그루터기)입니다. 점의 크기가 다음 나무가 보는 무게, 초록 선이 방금 세운 그루터기입니다.',
  controls: [{ k: 'm', l: '나무 수', min: 1, max: 30, step: 1, v: 1 }],
  init(s) { this.change(s); },
  change(s) { s.fn = gridFn((x, y) => sig(2 * AB.F(s.m)(x, y)), 1.2, 40); },
  read(s) { const r = AB.R[s.m - 1]; return [['이번 나무의 틀린 무게', pct(r.e)], ['훈련 · 시험 정확도', pct(abTr(s.m)) + ' · ' + pct(abTe(s.m))]]; },
  say(s) { const r = AB.R[s.m - 1];
    if (s.m === 1) return '첫 그루터기는 질문 하나로 모눈을 반으로 나눴을 뿐입니다. 틀린 점들이 <b>커졌습니다.</b> 다음 나무는 이 큰 점들을 맞히는 데 집중합니다. 나무 수를 하나씩 올려 보십시오.';
    return s.m + '번째 그루터기는 무게로 따져 ' + pct(r.e) + '를 틀렸고, 그래서 발언권 α = ' + f2(r.a) + '를 받았습니다. 잘 맞힌 나무일수록 발언권이 큽니다. 지금까지 ' + s.m + '그루의 가중 투표로 훈련 정확도 <b>' + pct(abTr(s.m)) + '</b>.'; },
  view(s) { const host = 'sia', r = AB.R[s.m - 1], T = r.T, W = r.w, n = AB.tr.length;
    return [SURF(host, { fn: s.fn, h: .45, marks: AB.tr.map((d, i) => ({ x: d.x, y: d.y, c: d.c ? C.y : C.a, s: clamp(.045 + .11 * Math.sqrt(W[i] * n), .04, .3) })), lines: tsegs(T, C.g).map(q => [q[0], q[1], q[2], q[3], C.g, .05]), lineY: .5 }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: .5, ymax: 1, series: [{ v: Array.from({ length: s.m }, (_, i) => abTr(i + 1)), c: C.y, len: 30, n: '훈련' }, { v: Array.from({ length: s.m }, (_, i) => abTe(i + 1)), c: C.p, len: 30, n: '시험' }], title: '나무를 더할 때마다의 정확도' }]; },
};
EXTRA.AdaBoost = { say: '점의 위치를 넣으면, 그루터기들이 발언권만큼 가중 투표한 결과가 나와요!', io: ['점의 위치 (변수 여러 개)', '가중 투표 결과'],
  part: { items: [['tiles', K.i, '점마다 같은 무게'], ['gate', K.gt, '그루터기 하나 세우기'], ['box', K.p, '틀린 무게 합 e'], ['ball', K.o, '발언권 α = ½ln((1−e)/e)'], ['box', K.g, '틀린 점 무게 키우기']], loops: [[4, 1, '나무 수만큼']],
    t: ['처음에는 모든 점의 무게가 같습니다. 무게를 보며 질문 하나짜리 나무를 세웁니다.', '그 나무가 틀린 무게만큼 발언권을 정하고, 틀린 점의 무게는 키우고 맞힌 점은 줄입니다.', '다음 나무는 무거운 점에 집중합니다. 마지막에는 발언권을 곱해 다수결합니다.'], say: '랜덤 포레스트는 나무를 나란히 따로 세우지만, 에이다부스트는 <b>차례로</b> 세웁니다. 앞 나무의 실수가 뒤 나무의 숙제가 됩니다.' },
  curve: { f: abTe, lo: 1, hi: 30, step: 1, v: 1, xl: '나무 수', yl: '시험 정확도', title: '그루터기 수와 시험 정확도', t: ['그루터기 하나는 동전 던지기보다 조금 나은 정도입니다.', '수십 개를 쌓으면 질문 하나로는 못 그리던 꺾인 경계가 나옵니다.'] } };

/* ================= 17층 아이솔레이션 포레스트 (모아) ================= */
const IF = (() => { const r = rng(93), D = []; while (D.length < 56) { const x = -.15 + gauss(r) * .26, y = -.05 + gauss(r) * .22; if (Math.abs(x) < 1 && Math.abs(y) < 1) D.push({ x, y, o: 0 }); }
  [[.82, .78], [-.9, .7], [.85, -.75], [-.8, -.85], [.15, .95]].forEach(([x, y]) => D.push({ x, y, o: 1 }));
  const cn = n => n <= 1 ? 0 : 2 * (Math.log(n - 1) + .5772) - 2 * (n - 1) / n, LIM = 8;
  const build = (P, r, d) => { if (d >= LIM || P.length <= 1) return { n: P.length }; const f = r() < .5 ? 'x' : 'y', v = P.map(p => p[f]), lo = Math.min(...v), hi = Math.max(...v); if (hi - lo < 1e-9) return { n: P.length };
    const t = lo + r() * (hi - lo); return { f, t, l: build(P.filter(p => p[f] < t), r, d + 1), r: build(P.filter(p => p[f] >= t), r, d + 1) }; };
  const depth = (T, x, y, d = 0) => T.f ? depth((T.f === 'x' ? x : y) < T.t ? T.l : T.r, x, y, d + 1) : d + cn(T.n);
  const forest = memo(n => { const q = rng(77); return Array.from({ length: n }, () => { const S = []; for (let i = 0; i < 48; i++) S.push(D[Math.floor(q() * D.length)]); return build(S, q, 0); }); });
  const all = forest(60), score = (n, x, y) => { const E = mean(all.slice(0, n).map(T => depth(T, x, y))); return Math.pow(2, -E / cn(48)); }, avgD = (n, x, y) => mean(all.slice(0, n).map(T => depth(T, x, y)));
  return { D, score, avgD }; })();
DEMOS.iForest = {
  vt: '빨리 혼자 남는 점이 별난 점',
  note: '보통 점 56개와 일부러 멀리 둔 별난 점 5개, 모두 가상의 데이터입니다. 정답은 주지 않습니다. 바닥의 높이가 "이상 점수"입니다. 높이 솟은 곳일수록 무작위로 자르면 금방 혼자 남는 자리입니다.',
  controls: [{ k: 'n', l: '나무 수', min: 1, max: 60, step: 1, v: 20 }, { k: 'thr', l: '이상치로 볼 점수', min: .4, max: .75, step: .01, v: .5 }],
  init(s) { this.change(s, 'n'); },
  change(s, k) { if (k === 'n') { s.sc = IF.D.map(d => IF.score(s.n, d.x, d.y)); s.fn = gridFn((x, y) => clamp((IF.score(s.n, x, y) - .3) / .45, 0, 1), 1.2, 36); s.dn = mean(IF.D.filter(d => !d.o).map(d => IF.avgD(s.n, d.x, d.y))); s.do = mean(IF.D.filter(d => d.o).map(d => IF.avgD(s.n, d.x, d.y))); }
    s.flag = s.sc.map(v => v > s.thr); s.hit = IF.D.filter((d, i) => d.o && s.flag[i]).length; s.fp = IF.D.filter((d, i) => !d.o && s.flag[i]).length; },
  read(s) { return [['이상치로 표시', s.flag.filter(Boolean).length + '개'], ['별난 점 5개 중 찾음', s.hit + '개']]; },
  say(s) { if (s.n < 4) return '나무가 ' + s.n + '그루뿐이라 운이 많이 섞여요. 바닥이 울퉁불퉁하죠? 나무를 늘려 보세요.';
    return '보통 점은 평균 ' + f2(s.dn, 1) + '번 잘라야 혼자 남는데, 별난 점은 <b>' + f2(s.do, 1) + '번</b>이면 혼자 남아요. 점수 ' + f2(s.thr) + '를 넘는 점을 분홍으로 칠했어요.' + (s.fp ? ' 보통 점 ' + s.fp + '개도 걸렸어요. 기준을 올려 보세요.' : s.hit < 5 ? ' 별난 점을 다 못 찾았어요. 기준을 내려 보세요.' : ' 별난 점 5개를 딱 찾았어요!'); },
  view(s) { const host = 'moa';
    return [SURF(host, { fn: s.fn, h: 1.1, pts: IF.D.map((d, i) => ({ x: d.x, y: d.y, col: s.flag[i] ? C.p : C.mu })) }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.dn, s.do], colors: [C.mu, C.p], max: 10, labels: ['보통 점', '별난 점'], title: '혼자 남기까지 자른 횟수 (평균)' }]; },
};
EXTRA.iForest = { say: '점을 넣으면 0~1 사이 "이상 점수"가 나와요. 1에 가까울수록 별난 점이에요!', io: ['점의 위치 (변수 여러 개)', '이상 점수 (0~1)'],
  part: { items: [['tiles', K.i, '데이터 일부 뽑기'], ['gate', K.gt, '변수 · 자르는 값 무작위'], ['box', K.bx, '혼자 남을 때까지 자르기'], ['ball', K.p, '자른 횟수 h'], ['box', K.o, '점수 2^(−평균 h / c)']], loops: [[2, 1, '나무 수만큼']],
    t: ['데이터 일부를 뽑아 아무 변수, 아무 값에서나 자르는 나무를 세웁니다.', '보통 점은 이웃이 많아 여러 번 잘라야 혼자 남고, 별난 점은 몇 번 만에 혼자 남습니다.', '여러 나무의 평균 횟수를 0~1 점수로 바꿉니다. 짧을수록 1에 가깝습니다.'], say: '정답도, 거리 계산도 쓰지 않습니다. 무작위로 자르기만 해서 아주 빠르고, 변수가 많아도 잘 됩니다.' },
  curve: { f: d => IF.score(60, -.15 + d, -.05), lo: 0, hi: 1.1, v: .3, xl: '중심에서 오른쪽으로 떨어진 거리', yl: '이상 점수', title: '중심에서 멀어질수록 오르는 점수', t: ['점이 빽빽한 가운데에서는 점수가 낮습니다.', '무리의 가장자리를 지나면 점수가 빠르게 오릅니다.'] } };

/* ================= 18층 가우시안 혼합 (시아) ================= */
const GM = (() => { const r = rng(63), D = [], mk = (cx, cy, sx, sy, rh, n) => { let g = 0, c = 0; while (c < n && g++ < 999) { const a = gauss(r), b = gauss(r), x = cx + sx * a, y = cy + sy * (rh * a + Math.sqrt(1 - rh * rh) * b); if (Math.abs(x) < 1.08 && Math.abs(y) < 1.08) { D.push({ x, y }); c++; } } };
  mk(-.3, .1, .3, .17, .5, 30); mk(.35, -.2, .2, .3, -.3, 24);
  const nd = (p, m, S) => { const a = S[0][0], b = S[0][1], c = S[1][1], dt = a * c - b * b, dx = p[0] - m[0], dy = p[1] - m[1]; return Math.exp(-.5 * (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / dt) / (2 * Math.PI * Math.sqrt(dt)); };
  const init = (k, seed) => { const q = rng(seed * 13 + k), idx = []; while (idx.length < k) { const i = Math.floor(q() * D.length); if (!idx.includes(i)) idx.push(i); } return { m: idx.map(i => [D[i].x, D[i].y]), S: idx.map(() => [[.05, 0], [0, .05]]), w: idx.map(() => 1 / k) }; };
  const E = M => D.map(d => { const v = M.m.map((m, j) => M.w[j] * nd([d.x, d.y], m, M.S[j])), z = v.reduce((s, t) => s + t, 0) || 1e-300; return v.map(t => t / z); });
  const ll = M => D.reduce((s, d) => s + Math.log(M.m.reduce((t, m, j) => t + M.w[j] * nd([d.x, d.y], m, M.S[j]), 0) + 1e-300), 0);
  const Mstep = (M, R) => { const k = M.m.length; const o = { m: [], S: [], w: [] }; for (let j = 0; j < k; j++) { const nj = R.reduce((s, r) => s + r[j], 0) + 1e-9, mx = R.reduce((s, r, i) => s + r[j] * D[i].x, 0) / nj, my = R.reduce((s, r, i) => s + r[j] * D[i].y, 0) / nj;
    let a = 0, b = 0, c = 0; R.forEach((r, i) => { const dx = D[i].x - mx, dy = D[i].y - my; a += r[j] * dx * dx; b += r[j] * dx * dy; c += r[j] * dy * dy; }); o.m.push([mx, my]); o.S.push([[a / nj + 1e-3, b / nj], [b / nj, c / nj + 1e-3]]); o.w.push(nj / D.length); } return o; };
  const fit = (k, seed, it = 80) => { let M = init(k, seed); for (let i = 0; i < it; i++) M = Mstep(M, E(M)); return M; };
  const bic = memo(k => Math.min(...[1, 2, 3].map(sd => { const M = fit(k, sd); return -2 * ll(M) + (6 * k - 1) * Math.log(D.length); })));
  return { D, nd, init, E, Mstep, ll, bic }; })();
DEMOS.GMM = {
  vt: 'E 걸음과 M 걸음을 번갈아 산 모양 맞추기',
  note: '점 54개는 기울어진 종 모양 두 개를 섞어 만든 가상의 데이터이고, 정답은 주지 않습니다. 바닥의 산이 모델이 그린 밀도, 타원이 종 모양 하나하나의 2σ 둘레입니다. 점의 색은 가장 가능성이 큰 종입니다.',
  controls: [{ k: 'k', l: '종 모양 수 k', min: 1, max: 4, step: 1, v: 2 }, { b: 'one', l: '한 걸음 (E+M)' }, { b: 'run', l: '끝까지' }, { b: 'new', l: '처음부터' }],
  init(s) { s.seed = 1; this.change(s); },
  change(s) { s.M = GM.init(s.k, s.seed); s.it = 0; s.auto = 0; s.hist = [GM.ll(s.M)]; this.fresh(s); },
  fresh(s) { const M = s.M; let mx = 0; const f = (x, y) => M.m.reduce((t, m, j) => t + M.w[j] * GM.nd([x, y], m, M.S[j]), 0); for (let i = 0; i <= 20; i++) for (let j = 0; j <= 20; j++) mx = Math.max(mx, f(-1.2 + i * .12, -1.2 + j * .12)); s.fn = gridFn((x, y) => f(x, y) / mx, 1.2, 36); s.R = GM.E(M); },
  one(s) { s.M = GM.Mstep(s.M, GM.E(s.M)); s.it++; s.hist.push(GM.ll(s.M)); this.fresh(s); },
  click(s, b) { if (b === 'new') { s.seed++; this.change(s); } else if (b === 'one') this.one(s); else { s.auto = 1; s.t = 0; } },
  step(s, dt) { if (!s.auto) return; s.t += dt; if (s.t < .25) return; s.t = 0; this.one(s); const h = s.hist; if (s.it > 60 || (h.length > 2 && Math.abs(h[h.length - 1] - h[h.length - 2]) < 1e-4)) s.auto = 0; },
  read(s) { return [['걸음 수', s.it], ['로그 우도 (높을수록 잘 맞음)', f2(s.hist[s.hist.length - 1], 1)]]; },
  say(s) { if (s.it === 0) return '종 모양 ' + s.k + '개를 작은 원으로 아무 데나 놓았습니다. <b>E 걸음</b>은 점마다 "어느 종에서 나왔을 확률"을 계산하고, <b>M 걸음</b>은 그 확률을 무게로 종의 중심·모양·크기를 다시 구합니다.';
    const half = s.R.filter(r => Math.max(...r) < .8).length, h = s.hist, conv = h.length > 2 && Math.abs(h[h.length - 1] - h[h.length - 2]) < 1e-3;
    return s.it + '걸음째입니다. ' + (conv ? '로그 우도가 더 오르지 않아 <b>멈췄습니다.</b>' : '로그 우도가 걸음마다 올라갑니다.') + ' 두 종이 겹친 곳의 점 ' + half + '개는 한쪽에 80% 넘게 속하지 않습니다. k-평균과 달리 소속을 <b>확률로 나눠 가집니다.</b>'; },
  view(s) { const host = 'sia', M = s.M;
    return [SURF(host, { fn: s.fn, h: 1, pts: GM.D.map((d, i) => { const r = s.R[i], j = r.indexOf(Math.max(...r)); return { x: d.x, y: d.y, col: Math.max(...r) < .8 ? C.fg : PAL[j] }; }), segs: M.m.flatMap((m, j) => ellSegs(m, M.S[j], 2, PAL[j])), segY: 1.12 }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: Math.min(...s.hist) - 2, ymax: Math.max(...s.hist) + 2, series: [{ v: s.hist, c: C.b, len: Math.max(10, s.hist.length), n: '로그 우도' }], title: '걸음마다 오르는 로그 우도' }]; },
};
EXTRA.GMM = { say: '점을 넣으면 "종 모양마다 여기서 나왔을 확률"이 나와요. 딱 잘라 하나로 정하지 않아요!', io: ['정답 없는 점들', '점마다 종 모양별 확률 + 종 모양 k개'],
  part: { items: [['ball', K.o, '종 모양 k개 놓기'], ['box', K.bx, 'E: 점마다 소속 확률'], ['box', K.g, 'M: 중심 · 모양 · 크기 다시 구하기'], ['gate', K.gt, '우도가 안 오르면 멈춤']], loops: [[2, 1, '되풀이']],
    t: ['데이터가 종 모양(가우스 분포) 몇 개를 섞어서 나왔다고 봅니다.', 'E 걸음: 지금 종 모양들로 점마다 어느 종에서 왔을지 확률을 냅니다. M 걸음: 그 확률을 무게로 종의 중심·기울기·크기·비율을 다시 구합니다.', '두 걸음을 번갈아 하면 데이터가 나올 가능성(우도)이 계속 올라갑니다.'], say: 'k-평균의 "가까운 중심 고르기"를 "확률로 나눠 갖기"로, "평균 내기"를 "무게 준 평균과 퍼짐 구하기"로 바꾼 것입니다.' },
  curve: { f: GM.bic, lo: 1, hi: 4, step: 1, v: 2, xl: '종 모양 수 k', yl: 'BIC (낮을수록 좋음)', title: '종 모양 수 고르기 (BIC)', t: ['k를 늘리면 데이터에 더 잘 맞지만, BIC는 늘어난 변수 수만큼 벌점을 줍니다.', '가장 낮은 곳이 알맞은 k입니다. 이 데이터는 2입니다.'] } };
