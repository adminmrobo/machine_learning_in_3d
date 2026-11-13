// 머신러닝 연구소 · 층별 실험(⑤)과 구역 ①③④의 내용
// 층마다 DEMOS[약어] = { controls, init, change, click, step, read, say, view } 와 EXTRA[약어] = { say, io, part, curve } 를 둔다.
// 숫자는 모두 이 파일 안에서 계산한다. view(s) 는 widgets3d.js 가 그릴 3D 물체 목록을 돌려준다(좌표는 ⑤ 구역 바닥판 기준, 가로 −6~6).
import * as LO from './demos_lo.js';
import * as MID from './demos_mid.js';
import * as HI from './demos_hi.js';
const C = { fg: 0xe9edf9, mu: 0x9aa7c7, a: 0x6fb1ff, b: 0xff9f6b, y: 0xffd76a, p: 0xff8fc7, g: 0x7fe0a8, r: 0xff6b5e };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const gauss = r => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(Math.PI * 2 * r());
const f2 = (v, n = 2) => Number(v).toFixed(n).replace('-', '−');
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const sgn = v => (v < 0 ? ' − ' : ' + ') + Math.abs(v).toFixed(2);      // 식에 넣는 '+ 0.30' / '− 0.11'
const memo = fn => { const c = {}; return x => (x in c ? c[x] : (c[x] = fn(x))); };

/* ---------- 작은 계산 도구 ---------- */
function solve(A, b) { const n = b.length; A = A.map(r => r.slice()); b = b.slice();
  for (let i = 0; i < n; i++) { let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k; [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    const d = A[i][i] || 1e-12; for (let k = i + 1; k < n; k++) { const f = A[k][i] / d; if (!f) continue; for (let j = i; j < n; j++) A[k][j] -= f * A[i][j]; b[k] -= f * b[i]; } }
  const x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / (A[i][i] || 1e-12); } return x; }
function lsq(Phi, y, lam) { const m = Phi[0].length, A = Array.from({ length: m }, () => new Array(m).fill(0)), b = new Array(m).fill(0);
  Phi.forEach((row, k) => { for (let i = 0; i < m; i++) { b[i] += row[i] * y[k]; for (let j = 0; j < m; j++) A[i][j] += row[i] * row[j]; } }); for (let i = 0; i < m; i++) A[i][i] += lam; return solve(A, b); }
// 판(가로·세로 −R~R) 안에 들어오는 직선 a·x + b·y + c = 0 의 두 끝
function clipLine(a, b, c, R = 1.2) { const P = [];
  if (Math.abs(b) > 1e-9) for (const x of [-R, R]) { const y = (-c - a * x) / b; if (Math.abs(y) <= R + 1e-9) P.push([x, y]); }
  if (Math.abs(a) > 1e-9) for (const y of [-R, R]) { const x = (-c - b * y) / a; if (Math.abs(x) <= R + 1e-9) P.push([x, y]); }
  let best = null, bd = 1e-6; for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) { const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]); if (d > bd) { bd = d; best = [...P[i], ...P[j]]; } } return best; }
const SURF = o => ({ id: 'f', type: 'surface', x: 0, z: -.7, w: 7.2, ...o });

/* ---------- 로비: 복잡도와 과적합 ---------- */
const LB = (() => { const r = rng(47), N = 24, all = Array.from({ length: N }, (_, i) => { const x = -1 + 2 * (i + r() * .8) / N; return { x, y: clamp(.5 * Math.sin(2.6 * x + .5) + .28 * x + gauss(r) * .16, -1.1, 1.1) }; });
  const split = seed => { const q = rng(seed * 7919 + 3), idx = all.map((_, i) => i); for (let i = N - 1; i > 0; i--) { const j = Math.floor(q() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } const te = new Set(idx.slice(0, 10)); return { tr: all.filter((_, i) => !te.has(i)), te: all.filter((_, i) => te.has(i)) }; };
  const row = (x, d) => Array.from({ length: d + 1 }, (_, k) => x ** k);
  const fit = (tr, d) => { const w = lsq(tr.map(p => row(p.x, d)), tr.map(p => p.y), 1e-7); return x => w.reduce((s, c, k) => s + c * x ** k, 0); };
  const mse = (ps, f) => mean(ps.map(p => (p.y - f(p.x)) ** 2));
  return { split, fit, mse }; })();
const lobbyTest = memo(d => { const sp = LB.split(3); return LB.mse(sp.te, LB.fit(sp.tr, d)); });

/* ---------- 1층: 선형 회귀 ---------- */
const LRD = (() => { const r = rng(4), base = Array.from({ length: 14 }, (_, i) => { const x = -1 + 2 * (i + .5) / 14 + (r() - .5) * .08; return { x, y: clamp(.55 * x + .05 + gauss(r) * .17, -1.1, 1.1) }; });
  const ols = d => { const mx = mean(d.map(p => p.x)), my = mean(d.map(p => p.y)), w = d.reduce((s, p) => s + (p.x - mx) * (p.y - my), 0) / d.reduce((s, p) => s + (p.x - mx) ** 2, 0); return { w, b: my - w * mx }; };
  const mse = (d, w, b) => mean(d.map(p => (p.y - w * p.x - b) ** 2));
  return { base, out: [...base, { x: .86, y: -.98, o: 1 }], ols, mse }; })();
const lrCurve = w => { const d = LRD.base, b = mean(d.map(p => p.y)) - w * mean(d.map(p => p.x)); return LRD.mse(d, w, b); };

export const DEMOS = {
  /* 로비 */
  로비: {
    vt: '복잡도를 올리면 훈련 오차와 시험 오차가 어떻게 달라지나',
    note: '점 24개는 물결 모양 곡선에 잡음을 섞어 만든 가상의 데이터입니다. 14개로 배우고 10개로 채점합니다. 모델은 차수를 고를 수 있는 다항식입니다.',
    controls: [{ k: 'deg', l: '복잡도 (차수)', min: 1, max: 9, step: 1, v: 1 }, { b: 're', l: '훈련용·시험용 다시 나누기' }],
    init(s) { s.seed = 3; this.change(s); },
    change(s) { s.sp = LB.split(s.seed); s.f = LB.fit(s.sp.tr, s.deg); s.tr = LB.mse(s.sp.tr, s.f); s.te = LB.mse(s.sp.te, s.f);
      s.all = Array.from({ length: 9 }, (_, i) => LB.mse(s.sp.te, LB.fit(s.sp.tr, i + 1))); s.best = s.all.indexOf(Math.min(...s.all)) + 1; },
    click(s) { s.seed++; this.change(s); },
    read(s) { return [['훈련 오차', f2(s.tr, 3)], ['시험 오차', f2(s.te, 3)]]; },
    say(s) { const r = s.te / Math.max(s.tr, 1e-6);
      if (s.deg < s.best && s.te > s.all[s.best - 1] * 1.3) return '차수 ' + s.deg + '에서는 모델이 너무 단순합니다. 훈련 점도 시험 점도 못 맞히는 <b>과소적합</b>입니다. 차수를 올려 보십시오.';
      if (s.deg > s.best && s.te > s.all[s.best - 1] * 1.3) return '훈련 오차는 ' + f2(s.tr, 3) + '인데 시험 오차는 ' + f2(s.te, 3) + ', <b>' + f2(r, 1) + '배</b>입니다. 훈련 점만 외운 <b>과적합</b>입니다.';
      return '훈련 오차와 시험 오차가 함께 낮습니다. 이 나눔에서는 차수 <b>' + s.best + '</b> 근처가 알맞은 복잡도입니다.'; },
    view(s) { const lines = [], n = 60; let p = null;
      for (let i = 0; i <= n; i++) { const x = -1.1 + 2.2 * i / n, y = s.f(x), q = Math.abs(y) <= 1.18 ? [x, y] : null; if (p && q) lines.push([p[0], p[1], q[0], q[1], C.b, .045]); p = q; }
      // 말하는 식구(시아)가 왼쪽 앞에 서므로 막대는 오른쪽에 둔다
      return [SURF({ x: -1.9, w: 6.6, fn: null, pr: .12, lines, pts: [...s.sp.tr.map(d => ({ x: d.x, y: d.y, col: C.y })), ...s.sp.te.map(d => ({ x: d.x, y: d.y, col: C.p }))],
          marks: [{ x: -1.05, y: 1.08, c: C.y, s: .1, t: '노랑: 훈련용 14개' }, { x: .45, y: 1.08, c: C.p, s: .1, t: '분홍: 시험용 10개' }] }),
        { id: 'b', type: 'bars', x: 4.1, z: -1.6, w: 3, h: 3.4, v: [s.tr, s.te], colors: [C.y, C.p], max: .12, labels: ['훈련 오차', '시험 오차'], title: '오차 (낮을수록 좋음)' }]; },
  },

  /* 1층 선형 회귀 */
  LR: {
    vt: '오차 네모의 넓이 합이 가장 작은 직선 찾기',
    note: '점 14개는 설명을 위해 만든 가상의 데이터입니다. 분홍 네모 하나의 넓이가 그 점의 오차 제곱입니다.',
    controls: [{ k: 'w', l: '기울기 w', min: -1.5, max: 1.5, step: .01, v: -.2 }, { k: 'b', l: '절편 b', min: -1, max: 1, step: .01, v: .3 },
      { k: 'out', l: '멀리 튄 점', opts: [['0', '없음'], ['1', '하나 넣기 (오른쪽 아래)']], v: '0' }, { b: 'fit', l: '가장 좋은 선 찾기' }],
    init(s) { s.auto = 0; this.change(s, 'out'); },
    change(s, k) { if (k === 'out') { s.d = s.out === '1' ? LRD.out : LRD.base; s.best = LRD.ols(s.d); s.bm = LRD.mse(s.d, s.best.w, s.best.b); } s.auto = 0; },
    click(s) { s.auto = 1; },
    step(s, dt) { if (!s.auto) return; const t = Math.min(1, dt * 3.5); s.w += (s.best.w - s.w) * t; s.b += (s.best.b - s.b) * t; if (Math.abs(s.w - s.best.w) + Math.abs(s.b - s.best.b) < .004) { s.w = s.best.w; s.b = s.best.b; s.auto = 0; } },
    read(s) { return [['지금 선의 오차 (MSE)', f2(LRD.mse(s.d, s.w, s.b), 3)], ['가장 좋은 선의 오차', f2(s.bm, 3)]]; },
    say(s) { const m = LRD.mse(s.d, s.w, s.b), r = m / s.bm;
      if (r < 1.03) return s.out === '1' ? '가장 좋은 선이에요. 그런데 튄 점 하나 때문에 선이 <b>오른쪽 아래로 끌려갔어요.</b> 기울기가 ' + f2(LRD.ols(LRD.base).w) + '에서 ' + f2(s.best.w) + '로 바뀌었어요.' : '분홍 네모가 가장 작아졌어요! 이 선이 <b>최소제곱 직선</b>이에요. ŷ = ' + f2(s.w) + '·x' + sgn(s.b);
      return '분홍 네모의 넓이를 다 더한 게 오차예요. 지금은 가장 좋은 선보다 <b>' + f2(r, 1) + '배</b> 커요. 기울기와 절편을 움직이거나 <b>가장 좋은 선 찾기</b>를 눌러 보세요.'; },
    view(s) { const L = clipLine(s.w, -1, s.b), segs = [], m = LRD.mse(s.d, s.w, s.b);
      s.d.forEach(p => { const yh = s.w * p.x + s.b, e = Math.abs(p.y - yh); if (e < .004) return; segs.push([p.x, p.y, p.x, yh, C.fg], [p.x, p.y, p.x + e, p.y, C.p], [p.x + e, p.y, p.x + e, yh, C.p], [p.x, yh, p.x + e, yh, C.p]); });
      // 말하는 식구(모아)가 오른쪽 앞에 서므로 막대는 왼쪽에 둔다
      return [SURF({ x: 1.9, w: 6.6, fn: null, pr: .12, pts: s.d.map(p => ({ x: p.x, y: p.y, col: p.o ? C.r : C.y })), lines: L ? [[...L, C.b, .06]] : [], segs }),
        { id: 'b', type: 'bars', x: -4.1, z: -1.6, w: 3, h: 3.4, v: [m, s.bm], colors: [C.p, C.g], max: .4, labels: ['지금 선', '가장 좋은 선'], title: '오차 제곱의 평균' }]; },
  },
};

/* ================= 구역 ①③④의 내용 =================
   say: ①에서 조수(하루)가 하는 말 · io: [넣는 것, 나오는 것]
   part: ③ 부품 줄. items = [모양(ball·box·gate·tiles), 색, 이름] · loops = [[어디서, 어디로, 이름]] · t: 세 줄 · say: 주판봇의 말
   curve: ④ 그래프 판. f(x), 범위 lo~hi, 처음 값 v, (step: 눈금 간격), 축 이름 xl·yl, 제목 title, 설명 t */
const K = { i: '#ffd76a', o: '#ff9f6b', bx: '#6fb1ff', gt: '#b79cff', g: '#7fe0a8', p: '#ff8fc7' };
export const EXTRA = {
  로비: { say: '문제와 정답을 잔뜩 넣으면, 처음 보는 문제에 답을 내는 모델이 나와요!', io: ['데이터 (문제와 정답 묶음)', '새 문제에 대한 예측'],
    part: { items: [['tiles', K.i, '데이터 표'], ['gate', K.gt, '훈련용 · 시험용 나누기'], ['box', K.bx, '크기 맞추기 (스케일링)'], ['box', K.g, '모델 학습 (훈련용만)'], ['ball', K.o, '시험용으로 채점']],
      t: ['데이터를 먼저 훈련용과 시험용으로 나눕니다.', '변수마다 단위가 다르면 크기를 맞춥니다. 기준(평균·표준편차)은 훈련용에서만 구합니다.', '훈련용으로만 배우고, 시험용으로 한 번 채점합니다.'], say: '순서가 중요합니다. 나누기가 먼저, 크기 맞추기와 학습은 그다음입니다. 시험용은 채점 때만 꺼냅니다.' },
    curve: { f: lobbyTest, lo: 1, hi: 9, step: 1, v: 1, xl: '모델의 복잡도 (차수)', yl: '시험 오차', title: '복잡도와 시험 오차', t: ['복잡도를 올리면 시험 오차는 내려가다가 다시 올라갑니다.', '가장 낮은 곳이 알맞은 복잡도입니다. 훈련 오차는 계속 내려가기만 해서, 훈련 오차만 보면 이 지점을 알 수 없습니다.'] } },
  LR: { say: '가로 값을 넣으면 직선 위의 세로 값이 나와요. 숫자를 넣으면 숫자가 나오는 거예요!', io: ['숫자 하나 (가로 값 x)', '숫자 하나 (예측한 세로 값 ŷ)'],
    part: { items: [['ball', K.i, '입력 x'], ['box', K.bx, '기울기 w 곱하고 절편 b 더하기'], ['ball', K.o, '예측 ŷ'], ['box', K.p, '정답과의 차이를 제곱'], ['gate', K.gt, '제곱 합이 가장 작은 w, b 고르기']],
      loops: [[4, 1, 'w와 b를 정함']], t: ['입력에 기울기를 곱하고 절편을 더하면 예측이 나옵니다.', '예측과 정답의 차이를 제곱해 모든 점에 대해 더합니다.', '그 합이 가장 작은 기울기와 절편을 고릅니다. 반복 없이 계산 한 번으로 구합니다.'], say: '부품은 곱하기 하나, 더하기 하나입니다. 나머지는 가장 좋은 w와 b를 고르는 계산입니다.' },
    curve: { f: lrCurve, lo: -1, hi: 2, v: -.2, xl: '기울기 w', yl: '평균 제곱 오차', title: '기울기를 바꾸면 오차는 그릇 모양', t: ['기울기가 알맞은 값에서 멀어질수록 오차가 제곱으로 커집니다.', '그릇의 바닥은 하나뿐입니다. 그래서 선형 회귀는 헤매지 않고 답을 바로 구합니다.'] } },
};

// 2~27층은 층 묶음별 파일에 있다: demos_lo.js(2~9) · demos_mid.js(10~18) · demos_hi.js(19~27). 계산 도구는 ml.js.
[LO, MID, HI].forEach(M => { Object.assign(DEMOS, M.DEMOS); Object.assign(EXTRA, M.EXTRA); });
