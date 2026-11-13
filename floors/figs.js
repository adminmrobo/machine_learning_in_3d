// 머신러닝 연구소 · 층마다 위에 띄우는 "모델 그림" 명세
// 머신러닝 모델 도감(machine_learning_model_atlas)의 모눈 그림을 같은 좌표(가로 0~220, 세로 0~130, 아래가 +)와 같은 가상의 점으로 옮겼다.
// 여기서는 "무엇을 어디에"만 돌려주고, floors/index.html 의 buildFig 가 3D 물체로 만든다.
//   ['pt', x, y, 색, 크기]            공 (데이터 점)
//   ['ln', x1, y1, x2, y2, 색, 굵기]   선 (모눈 밖은 잘라 냄). 색이 M 이면 "모델이 그은 선"으로 보고 빛이 따라 흐른다.
//   ['poly', [[x, y], …], 색, 투명도]   칠한 영역
//   ['txt', x, y, 글]                  글자 (큰 그림에서만 보임)
// 색: A 노랑 · B 파랑 · C 보라 · U 회색 · N 분홍 · G 초록 · M 주황 · E 옅은 보조선
export const FW = 220, FH = 130;
export const FCOL = { A: '#ffd76a', B: '#6fb1ff', C: '#b79cff', U: '#9aa7c7', N: '#ff8fc7', G: '#7fe0a8', M: '#ff9f6b', E: '#93a4d8' };

function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const gauss = r => Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r());
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const cen = ps => [mean(ps.map(p => p[0])), mean(ps.map(p => p[1]))];
function blob(r, cx, cy, sx, sy, n, ok) { const o = []; let g = 0; while (o.length < n && g++ < 6000) { const x = cx + gauss(r) * sx, y = cy + gauss(r) * sy; if (x < 9 || x > 211 || y < 9 || y > 121) continue; if (ok && !ok(x, y)) continue; if (o.some(p => Math.hypot(p[0] - x, p[1] - y) < 7)) continue; o.push([x, y]); } return o; }

/* ---------- 그리기 도구: 모두 pt · ln · poly · txt 로 풀어 낸다 ---------- */
const dot = (x, y, c, r) => ['pt', x, y, c, r || 3.6];
const dots = (ps, c, r) => ps.map(p => dot(p[0], p[1], c, r));
const ln = (x1, y1, x2, y2, c = 'M', w = 1.8) => ['ln', x1, y1, x2, y2, c, w];
const dash = (x1, y1, x2, y2, c = 'E', w = 1.1, on = 5, off = 4) => { const L = Math.hypot(x2 - x1, y2 - y1), o = []; for (let t = 0; t < L; t += on + off) { const a = t / L, b = Math.min(L, t + on) / L; o.push(ln(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b, c, w)); } return o; };
const poly = (P, c, o) => ['poly', P, c, o == null ? .16 : o];
const rect = (x, y, w, h, c, o) => poly([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], c, o);
const txt = (x, y, s) => ['txt', x, y, s];
const line = (P, c = 'M', w = 1.8) => P.slice(1).map((q, i) => ln(P[i][0], P[i][1], q[0], q[1], c, w));
const ellP = (cx, cy, rx, ry, rot = 0, n = 40) => { const c = Math.cos(rot), s = Math.sin(rot); return Array.from({ length: n + 1 }, (_, i) => { const a = i / n * 2 * Math.PI, x = rx * Math.cos(a), y = ry * Math.sin(a); return [cx + x * c - y * s, cy + x * s + y * c]; }); };
const ellipse = (cx, cy, rx, ry, c, rot = 0, fill = .1, w = 1.2) => [poly(ellP(cx, cy, rx, ry, rot), c, fill), ...line(ellP(cx, cy, rx, ry, rot), c, w)];
const ringD = (cx, cy, r, c = 'N') => { const P = ellP(cx, cy, r, r, 0, 36), o = [poly(P, c, .07)]; for (let i = 0; i < 36; i += 2) o.push(ln(P[i][0], P[i][1], P[i + 1][0], P[i + 1][1], c, 1.1)); return o; };
const ring = (x, y, c = 'N') => line(ellP(x, y, 7, 7, 0, 18), c, 1.3);
const arrow = (x1, y1, x2, y2, c = 'M', w = 2) => { const a = Math.atan2(y2 - y1, x2 - x1), s = 6; return [ln(x1, y1, x2, y2, c, w), ln(x2, y2, x2 - s * Math.cos(a + .5), y2 - s * Math.sin(a + .5), c, w), ln(x2, y2, x2 - s * Math.cos(a - .5), y2 - s * Math.sin(a - .5), c, w)]; };

/* ---------- 공용 가상 데이터 (도감과 같은 씨앗) ---------- */
// 두 집단 (직선으로 갈리는 배치)
const MID = [107.5, 65], NV = [85 / 93.94, -40 / 93.94], UV = [40 / 93.94, 85 / 93.94];
const sd = (x, y) => (x - MID[0]) * NV[0] + (y - MID[1]) * NV[1];
const r1 = rng(11);
const CA = blob(r1, 62, 88, 17, 15, 9, (x, y) => sd(x, y) < -11), CB = blob(r1, 152, 44, 17, 15, 9, (x, y) => sd(x, y) > 11);
const bline = (m = 0, c = 'M', w = 1.8) => ln(MID[0] - 130 * UV[0] + m * NV[0], MID[1] - 130 * UV[1] + m * NV[1], MID[0] + 130 * UV[0] + m * NV[0], MID[1] + 130 * UV[1] + m * NV[1], c, w);
const bdash = (m, c = 'M', w = 1.2) => dash(MID[0] - 130 * UV[0] + m * NV[0], MID[1] - 130 * UV[1] + m * NV[1], MID[0] + 130 * UV[0] + m * NV[0], MID[1] + 130 * UV[1] + m * NV[1], c, w);
const halves = () => [poly([[0, 0], [76.9, 0], [138.1, 130], [0, 130]], 'A', .13), poly([[76.9, 0], [220, 0], [220, 130], [138.1, 130]], 'B', .13)];
const two = () => [...dots(CA, 'A'), ...dots(CB, 'B')];
// 회귀용 점
const r2 = rng(5);
const RP = Array.from({ length: 12 }, (_, i) => { const x = 18 + i * 16.6 + (r2() - .5) * 6; return [x, 104 - .36 * (x - 18) + gauss(r2) * 8]; });
const fitLine = ps => { const mx = mean(ps.map(p => p[0])), my = mean(ps.map(p => p[1])), b = ps.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / ps.reduce((s, p) => s + (p[0] - mx) ** 2, 0); return [my - b * mx, b]; };
// 나무용 점 (직선 하나로는 못 가르는 배치)
const r3 = rng(23);
const TA = blob(r3, 58, 94, 14, 11, 8, (x, y) => x < 95 && y > 65), TB = blob(r3, 160, 62, 18, 22, 9, (x, y) => x > 116), TB2 = blob(r3, 52, 28, 13, 8, 4, (x, y) => x < 95 && y < 45);
const treePts = big => [...dots(TA, 'A'), ...dots(TB, 'B'), ...dots(TB2, 'B', big ? 6.5 : 3.6)];
const treeShade = () => [rect(0, 0, 105, 55, 'B', .13), rect(0, 55, 105, 75, 'A', .13), rect(105, 0, 115, 130, 'B', .13)];
// 곡선 회귀용 점
const r4 = rng(8);
const GP_ = Array.from({ length: 15 }, (_, i) => { const x = 16 + i * 13.4; return [x, 65 - 36 * Math.sin((x - 16) / 188 * Math.PI * 2) + gauss(r4) * 4]; });

export const FIG = {
  /* 로비 ②의 왼쪽: 사람이 기준을 정한 프로그램 */
  rule() { return [ln(6, 66, 214, 66, 'U', 1.8), ...dots(RP, 'U'), txt(110, 16, '사람이 정한 기준선')]; },
  /* 로비: 선을 긋기 전, 데이터를 훈련용과 시험용으로 나눈 모습 */
  split() { const test = [2, 5, 9, 11]; return [...RP.map((p, i) => dot(p[0], p[1], test.includes(i) ? 'N' : 'A', test.includes(i) ? 4.4 : 3.6)), txt(62, 16, '노랑: 훈련용 (배울 때 봄)'), txt(160, 120, '분홍: 시험용 (채점할 때만 봄)')]; },
  /* 1 선형 회귀: 점과 직선 사이 세로 거리(오차) */
  lr() { const [a, b] = fitLine(RP); return [...RP.map(p => ln(p[0], p[1], p[0], a + b * p[0], 'E', 1.3)), ln(6, a + b * 6, 214, a + b * 214, 'M', 2.2), ...dots(RP, 'A'), txt(62, 14, '주황: 모델이 그은 직선'), txt(168, 118, '옅은 세로선: 오차')]; },
  /* 7·8 릿지 · 라쏘: 계수 막대 (점선 = 벌점 없는 원래 크기). kind = ols 이면 원래 크기만 */
  coef(kind) { const v = [.9, -.62, .5, .16, -.12, .3], bw = 20, x0 = 28, y0 = 66, s = 52, o = [ln(14, y0, 206, y0, 'E', 1)];
    v.forEach((c, i) => { const x = x0 + i * 29, full = c * s, sh = kind === 'ridge' ? c * .55 : kind === 'lasso' ? Math.sign(c) * Math.max(Math.abs(c) - .25, 0) : c, hh = sh * s;
      const top = full > 0 ? y0 - full : y0, bot = full > 0 ? y0 : y0 - full;
      if (kind !== 'ols') o.push(...dash(x, top, x + bw, top, 'E', 1, 3, 2), ...dash(x, top, x, bot, 'E', 1, 3, 2), ...dash(x + bw, top, x + bw, bot, 'E', 1, 3, 2));
      if (Math.abs(hh) > .01) o.push(rect(x, hh > 0 ? y0 - hh : y0, bw, Math.abs(hh), 'M', .85)); else o.push(...ring(x + bw / 2, y0, 'N'));
      o.push(txt(x + bw / 2, 126, 'w' + (i + 1))); });
    return [...o, txt(150, 12, kind === 'ridge' ? '모두 조금씩 줄어든다' : kind === 'lasso' ? '작은 것은 0이 된다' : '벌점 없이 구한 계수')]; },
  /* 3 로지스틱 회귀: 경계 하나 + 확률 등고선 */
  logr() { return [...halves(), bline(-14, 'E', 1), bline(14, 'E', 1), bline(-28, 'E', .7), bline(28, 'E', .7), bline(), ...two(), txt(40, 14, '가운데 선 = 확률 0.5')]; },
  /* 11 선형 판별 분석: 두 집단의 타원과 중심 */
  lda() { const a = cen(CA), b = cen(CB), plus = c => [ln(c[0] - 5, c[1], c[0] + 5, c[1], 'U', 1.6), ln(c[0], c[1] - 5, c[0], c[1] + 5, 'U', 1.6)];
    return [...ellipse(a[0], a[1], 36, 22, 'A', -18 * Math.PI / 180), ...ellipse(b[0], b[1], 36, 22, 'B', -18 * Math.PI / 180), ...dash(a[0], a[1], b[0], b[1], 'E', 1, 3, 3), bline(), ...two(), ...plus(a), ...plus(b)]; },
  /* 2 k-최근접 이웃: 새 점과 가장 가까운 3개 */
  knn() { const q = [114, 72], all = [...CA.map(p => [...p, 'A']), ...CB.map(p => [...p, 'B'])].map(p => [...p, Math.hypot(p[0] - q[0], p[1] - q[1])]).sort((a, b) => a[3] - b[3]), k = all.slice(0, 3);
    return [...ringD(q[0], q[1], k[2][3] + 5), ...k.map(p => ln(q[0], q[1], p[0], p[1], 'N', 1.2)), ...two(), dot(q[0], q[1], 'N', 4.6), txt(q[0] - 14, q[1] + 18, '새 점')]; },
  /* 12 서포트 벡터 머신: 가장 넓은 길과 그 가장자리 점 */
  svm() { const all = [...CA, ...CB], m = Math.min(...all.map(p => Math.abs(sd(p[0], p[1]))));
    return [...halves(), ...bdash(-m), ...bdash(m), bline(), ...two(), ...all.filter(p => Math.abs(sd(p[0], p[1])) < m + 2.5).flatMap(p => ring(p[0], p[1]))]; },
  /* 19 커널 SVM: 안쪽 무리를 감싸는 곡선 경계 */
  ksvm() { const r = rng(31), inn = blob(r, 110, 65, 13, 9, 8, (x, y) => ((x - 110) / 34) ** 2 + ((y - 65) / 22) ** 2 < 1);
    const out = Array.from({ length: 13 }, (_, i) => { const a = i / 13 * 2 * Math.PI + .2; return [110 + (78 + (r() - .5) * 14) * Math.cos(a), 65 + (46 + (r() - .5) * 8) * Math.sin(a)]; });
    return [rect(0, 0, 220, 130, 'B', .1), poly(ellP(110, 65, 52, 31), 'A', .2), ...line(ellP(110, 65, 52, 31), 'M', 1.8), ...dots(inn, 'A'), ...dots(out, 'B')]; },
  /* 6 나이브 베이즈: 변수마다 따로 그린 종 모양 */
  nb() { const bell = (mu, s, k, horiz) => { const P = []; for (let t = -3; t <= 3.01; t += .25) { const u = mu + t * s, v = 22 * Math.exp(-.5 * t * t); P.push(horiz ? [u, 128 - v] : [2 + v, u]); } return [poly(P, k, .28), ...line(P, k, 1.2)]; };
    const st = (ps, i) => { const m = mean(ps.map(p => p[i])); return [m, Math.sqrt(mean(ps.map(p => (p[i] - m) ** 2)))]; };
    return [...bell(...st(CA, 0), 'A', 1), ...bell(...st(CB, 0), 'B', 1), ...bell(...st(CA, 1), 'A', 0), ...bell(...st(CB, 1), 'B', 0), ...two(), txt(130, 112, '가로 변수 따로'), txt(40, 14, '세로 변수 따로')]; },
  /* 24 가우시안 프로세스: 점이 없는 곳은 띠가 넓다 */
  gp() { const xs = [28, 52, 80, 152, 174], m = x => 66 + 26 * Math.sin(x / 30), w = x => 3 + 25 * (1 - Math.exp(-((Math.min(...xs.map(v => Math.abs(v - x))) / 30) ** 2)));
    const up = [], dn = [], mid = []; for (let x = 4; x <= 216; x += 4) { up.push([x, m(x) - w(x)]); dn.unshift([x, m(x) + w(x)]); mid.push([x, m(x)]); }
    return [poly([...up, ...dn], 'M', .18), ...line(up, 'E', .8), ...line(dn.slice().reverse(), 'E', .8), ...line(mid, 'M', 1.8), ...xs.map(x => dot(x, m(x), 'A', 4)), txt(112, 14, '점이 없는 곳은 띠가 넓다')]; },
  /* 4 결정 트리: 질문 두 개로 나눈 네모 칸 */
  dt() { return [...treeShade(), ln(105, 0, 105, 130), ln(0, 55, 105, 55), ...treePts(), txt(124, 124, '질문 1'), txt(22, 50, '질문 2')]; },
  /* 13·14 랜덤 포레스트 · 엑스트라 트리: 여러 그루의 경계(초록)와 다수결(주황) */
  forest(seed, k, jx, jy) { const r = rng(seed), o = [...treeShade()];
    for (let i = 0; i < k; i++) { const vx = 105 + (r() - .5) * 2 * jx, hy = 55 + (r() - .5) * 2 * jy; o.push(ln(0, hy, vx, hy, 'G', 1), ln(vx, hy, vx, 130, 'G', 1)); }
    const Q = [[0, 55], [90, 55]]; for (let t = 1; t <= 6; t++) { const a = t / 6 * Math.PI / 2; Q.push([90 + 15 * Math.sin(a), 70 - 15 * Math.cos(a)]); } Q.push([105, 130]);
    return [...o, ...line(Q, 'M', 2.2), ...treePts(), txt(150, 124, seed === 7 ? '나무마다 조금씩 다른 선' : '기준값까지 무작위')]; },
  /* 16 에이다부스트: 앞 나무가 틀린 점을 크게 */
  ada() { return [rect(0, 0, 105, 130, 'A', .13), rect(105, 0, 115, 130, 'B', .13), ln(105, 0, 105, 130), ...dash(0, 52, 105, 52, 'G', 1.4), ...treePts(true), txt(124, 124, '나무 1'), txt(22, 62, '나무 2'), txt(52, 12, '틀린 점은 크게')]; },
  /* 20 그래디언트 부스팅: 계단이 점점 잘게 */
  gbm() { const step = k => { const bw = 200 / k, P = []; for (let i = 0; i < k; i++) { const a = 10 + i * bw, b = a + bw, ps = GP_.filter(p => p[0] >= a && p[0] < b), y = mean(ps.map(p => p[1])); P.push([a, y], [b, y]); } return P; };
    return [...line(step(2), 'E', 1.2), ...line(step(4), 'G', 1.3), ...line(step(8), 'M', 2), ...dots(GP_, 'A'), txt(60, 124, '옅음 2칸 → 초록 4칸 → 주황 8칸')]; },
  /* 21·22·23 XGBoost(층별) · LightGBM(잎별) · CatBoost(대칭) 나무 모양 */
  tree(kind) { const e = [], n = [], ys = [16, 48, 80, 112], lv = ['G', 'B', 'C'];
    if (kind === 'leaf') { const P = [[70, 14], [40, 42, 0], [100, 42, 0], [75, 70, 2], [130, 70, 2], [105, 98, 4], [160, 98, 4], [140, 122, 6], [180, 122, 6]], leaf = [1, 3, 5, 7, 8];
      P.forEach((p, i) => { if (i) e.push(ln(P[p[2]][0], P[p[2]][1], p[0], p[1], 'E', 1.1)); });
      P.forEach((p, i) => n.push(dot(p[0], p[1], leaf.includes(i) ? 'M' : 'G', leaf.includes(i) ? 4 : 5.2))); return [...e, ...n, txt(36, 122, '급한 잎부터 깊게')]; }
    const X = [Array.from({ length: 8 }, (_, i) => 24 + i * 24.6)]; for (let d = 0; d < 3; d++) { const p = X[0], q = []; for (let i = 0; i < p.length; i += 2) q.push((p[i] + p[i + 1]) / 2); X.unshift(q); }
    if (kind === 'sym') for (let d = 0; d < 3; d++) { e.push(...dash(18, ys[d], 216, ys[d], lv[d], .8, 2, 4)); n.push(txt(8, ys[d] + 3, 'Q' + (d + 1))); }
    for (let d = 0; d < 3; d++) X[d].forEach((x, i) => e.push(ln(x, ys[d], X[d + 1][2 * i], ys[d + 1], 'E', 1.1), ln(x, ys[d], X[d + 1][2 * i + 1], ys[d + 1], 'E', 1.1)));
    X.forEach((row, d) => row.forEach(x => n.push(dot(x, ys[d], d === 3 ? 'M' : (kind === 'sym' ? lv[d] : 'G'), d === 3 ? 4 : 5.2))));
    return [...e, ...n, txt(170, 8, kind === 'sym' ? '같은 층 = 같은 질문' : '한 층씩 고르게')]; },
  /* 5 k-평균: 중심(X)과 거기 모인 점 */
  kmeans() { const r = rng(41), g = [blob(r, 52, 92, 14, 12, 8), blob(r, 112, 34, 17, 10, 8), blob(r, 172, 88, 14, 13, 8)], k = ['A', 'B', 'C'], o = [], t = [];
    g.forEach((ps, i) => { const c = cen(ps); ps.forEach(p => o.push(ln(p[0], p[1], c[0], c[1], 'E', .9))); t.push(...dots(ps, k[i]), ln(c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5, 'M', 2.6), ln(c[0] - 5, c[1] + 5, c[0] + 5, c[1] - 5, 'M', 2.6)); });
    return [...o, ...t, txt(110, 124, 'X = 묶음의 중심')]; },
  /* 10 계층적 군집: 나무 그림(덴드로그램)과 자르는 높이 */
  hac() { const x = i => 22 + i * 25, L = 112, o = [], U = (a, b, h) => { o.push(ln(a[0], a[1], a[0], h, 'E', 1.3), ln(a[0], h, b[0], h, 'E', 1.3), ln(b[0], h, b[0], b[1], 'E', 1.3)); return [(a[0] + b[0]) / 2, h]; };
    const lf = i => [x(i), L], l = U(U(lf(0), lf(1), 98), U(lf(2), lf(3), 92), 68), rr = U(U(lf(4), lf(5), 100), U(lf(6), lf(7), 88), 60); U(l, rr, 20);
    return [...o, ...dash(6, 42, 214, 42, 'M', 1.6, 6, 4), txt(160, 36, '여기서 자르면 2묶음'), ...[0, 1, 2, 3].map(i => dot(x(i), L + 6, 'A')), ...[4, 5, 6, 7].map(i => dot(x(i), L + 6, 'B'))]; },
  /* 15 DBSCAN: 초승달 두 개와 외딴 점(잡음) */
  dbscan() { const r = rng(52), moon = (cx, cy, a0, a1) => Array.from({ length: 13 }, (_, i) => { const a = a0 + (a1 - a0) * i / 12; return [cx + 45 * Math.cos(a) + (r() - .5) * 5, cy + 40 * Math.sin(a) + (r() - .5) * 5]; });
    const a = moon(85, 84, Math.PI * 1.05, Math.PI * 1.95), b = moon(138, 58, Math.PI * .05, Math.PI * .95), nz = [[16, 16], [40, 118], [204, 14]];
    return [...[a[3], a[4], a[5], b[7], b[8]].flatMap(p => ringD(p[0], p[1], 12)), ...dots(a, 'A'), ...dots(b, 'B'), ...dots(nz, 'U'), txt(30, 26, '잡음'), txt(150, 124, '분홍 원 = 반경 ε')]; },
  /* 18 가우시안 혼합: 겹친 타원, 경계의 점은 반반 */
  gmm() { const r = rng(63), a = blob(r, 82, 72, 24, 13, 10), b = blob(r, 142, 58, 20, 18, 10), el = (cx, cy, sx, sy, k) => [1.1, 2].flatMap(s => ellipse(cx, cy, sx * s, sy * s, k, 0, .09, 1.1));
    const za = p => Math.hypot((p[0] - 82) / 24, (p[1] - 72) / 13), zb = p => Math.hypot((p[0] - 142) / 20, (p[1] - 58) / 18);
    const pt = (p, k) => Math.abs(za(p) - zb(p)) < 1 ? [dot(p[0] - 1.8, p[1], 'A', 3), dot(p[0] + 1.8, p[1], 'B', 3)] : [dot(p[0], p[1], k)];
    return [...el(82, 72, 24, 13, 'A'), ...el(142, 58, 20, 18, 'B'), ...a.flatMap(p => pt(p, 'A')), ...b.flatMap(p => pt(p, 'B')), txt(110, 124, '겹친 곳의 점은 두 색 반반')]; },
  /* 9 주성분 분석: 가장 넓게 퍼진 방향 PC1 */
  pca() { const r = rng(71), u = [.906, -.423], v = [.423, .906], ps = []; while (ps.length < 17) { const a = gauss(r) * 36, b = gauss(r) * 8, x = 110 + a * u[0] + b * v[0], y = 68 + a * u[1] + b * v[1]; if (x > 12 && x < 208 && y > 12 && y < 118) ps.push([x, y]); }
    return [...dots(ps, 'U'), ...arrow(110, 68, 110 + 84 * u[0], 68 + 84 * u[1]), ...arrow(110, 68, 110 + 30 * v[0], 68 + 30 * v[1], 'G', 2), txt(170, 30, 'PC1'), txt(128, 106, 'PC2')]; },
  /* 25·26 t-SNE · UMAP: 고차원(왼쪽 원)을 2차원 지도(오른쪽)로 */
  embed(kind) { const r = rng(kind === 'umap' ? 88 : 81), k = ['A', 'B', 'C'], L = []; while (L.length < 15) { const a = r() * 2 * Math.PI, d = Math.sqrt(r()) * 33, p = [46 + d * Math.cos(a), 62 + d * Math.sin(a), k[L.length % 3]]; if (!L.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < 8)) L.push(p); }
    const cs = kind === 'umap' ? [[146, 40], [176, 70], [150, 100]] : [[150, 30], [196, 66], [146, 96]], R = []; cs.forEach((c, i) => blob(r, c[0], c[1], 6.5, 6.5, 5).forEach(p => R.push([...p, k[i]])));
    const e = []; if (kind === 'umap') { const near = (P, p) => P.filter(q => q !== p && q[2] === p[2]).sort((a, b) => Math.hypot(a[0] - p[0], a[1] - p[1]) - Math.hypot(b[0] - p[0], b[1] - p[1]))[0];
      [L, R].forEach(P => P.forEach(p => { const q = near(P, p); e.push(ln(p[0], p[1], q[0], q[1], 'E', 1)); })); e.push(...dash(cs[0][0], cs[0][1], cs[1][0], cs[1][1], 'E', .9, 2, 3), ...dash(cs[1][0], cs[1][1], cs[2][0], cs[2][1], 'E', .9, 2, 3)); }
    const circ = ellP(46, 62, 42, 42, 0, 36), cd = []; for (let i = 0; i < 36; i += 2) cd.push(ln(circ[i][0], circ[i][1], circ[i + 1][0], circ[i + 1][1], 'E', 1));
    return [...cd, ...e, ...arrow(96, 62, 118, 62, 'M', 1.6), ...[...L, ...R].map(p => dot(p[0], p[1], p[2])), txt(46, 124, '고차원'), txt(170, 124, '2차원 지도')]; },
  /* 17 아이솔레이션 포레스트: 별난 점은 금방 혼자 남는다 */
  iso() { const r = rng(93), ps = blob(r, 92, 74, 17, 14, 15, (x, y) => x < 150);
    return [ln(60, 0, 60, 130, 'G', 1), ln(0, 74, 166, 74, 'G', 1), ln(92, 74, 92, 130, 'G', 1), ln(92, 100, 166, 100, 'G', 1), ln(118, 0, 118, 74, 'G', 1),
      ln(166, 0, 166, 130), ln(166, 48, 220, 48), ...dots(ps, 'U'), dot(192, 24, 'N', 4.8), txt(176, 120, '①'), txt(206, 60, '②'), txt(126, 14, '2번 만에 고립')]; },
  /* 27 다층 퍼셉트론: 입력 → 은닉 → 은닉 → 출력 */
  mlp() { const L = [3, 4, 4, 2], xs = [38, 86, 134, 182], k = ['A', 'G', 'G', 'M'], pos = L.map((n, i) => Array.from({ length: n }, (_, j) => [xs[i], 65 + (j - (n - 1) / 2) * 27])), e = [];
    for (let i = 0; i < 3; i++) pos[i].forEach(a => pos[i + 1].forEach(b => e.push(ln(a[0], a[1], b[0], b[1], 'E', 1))));
    return [...e, ...pos.flatMap((p, i) => dots(p, k[i], 8)), txt(38, 124, '입력'), txt(110, 124, '은닉층 2개'), txt(182, 124, '출력')]; },
};
export const fig = spec => FIG[spec[0]](...spec.slice(1));
