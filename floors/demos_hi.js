// 머신러닝 연구소 · 고층부(19~27층) 실험과 구역 ①③④
// 형식은 demos.js 맨 위 설명과 같다. 점은 모두 설명을 위해 만든 가상의 데이터다.
import { C, clamp, rng, gauss, f2, pct, mean, memo, sig, solve, SURF, SIDE, LAY, gridFn, pts, curve, treeData, moons, circles, split, acc, wave, tree, tpred, leaves, tsegs, bigTest } from './ml.js';

const K = { i: '#ffd76a', o: '#ff9f6b', bx: '#6fb1ff', gt: '#b79cff', g: '#7fe0a8', p: '#ff8fc7' };
const PAL = [C.y, C.a, C.v, C.g, C.p];
export const DEMOS = {}, EXTRA = {};

/* ================= 19층 커널 SVM (모아) ================= */
const KS = (() => { const D = circles(14, 32, .18), sp = { tr: split(D, 3, 24).tr, te: circles(1014, 200, .18) };
  const ker = (kind, g) => kind === 'lin' ? (a, b) => a.x * b.x + a.y * b.y : kind === 'poly' ? (a, b) => (a.x * b.x + a.y * b.y + 1) ** 2 : (a, b) => Math.exp(-g * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2));
  const fit = memo(key => { const [kind, g] = key.split(':'), kf = ker(kind, +g), X = sp.tr, n = X.length, y = X.map(d => d.c ? 1 : -1), Km = X.map(a => X.map(b => kf(a, b))), Cc = 5, r = rng(9);
    const al = new Array(n).fill(0); let b = 0, passes = 0, loops = 0; const f = i => { let s = b; for (let k = 0; k < n; k++) if (al[k]) s += al[k] * y[k] * Km[k][i]; return s; };
    while (passes < 8 && loops++ < 300) { let ch = 0;
      for (let i = 0; i < n; i++) { const Ei = f(i) - y[i]; if (!((y[i] * Ei < -1e-3 && al[i] < Cc) || (y[i] * Ei > 1e-3 && al[i] > 0))) continue;
        let j = Math.floor(r() * (n - 1)); if (j >= i) j++; const Ej = f(j) - y[j], ai = al[i], aj = al[j];
        const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - Cc), H = y[i] !== y[j] ? Math.min(Cc, Cc + aj - ai) : Math.min(Cc, ai + aj); if (L >= H) continue;
        const eta = 2 * Km[i][j] - Km[i][i] - Km[j][j]; if (eta >= 0) continue;
        al[j] = clamp(aj - y[j] * (Ei - Ej) / eta, L, H); if (Math.abs(al[j] - aj) < 1e-5) continue; al[i] = ai + y[i] * y[j] * (aj - al[j]);
        const b1 = b - Ei - y[i] * (al[i] - ai) * Km[i][i] - y[j] * (al[j] - aj) * Km[i][j], b2 = b - Ej - y[i] * (al[i] - ai) * Km[i][j] - y[j] * (al[j] - aj) * Km[j][j];
        b = al[i] > 0 && al[i] < Cc ? b1 : al[j] > 0 && al[j] < Cc ? b2 : (b1 + b2) / 2; ch++; }
      passes = ch ? 0 : passes + 1; }
    const dec = (x, yy) => { const p = { x, y: yy }; let s = b; for (let k = 0; k < n; k++) if (al[k] > 1e-7) s += al[k] * y[k] * kf(X[k], p); return s; };
    const P = (x, yy) => dec(x, yy) > 0 ? 1 : 0;
    return { dec, sv: al.map(a => a > 1e-7), tr: acc(X, P), te: acc(sp.te, P) }; });
  return { ...sp, fit }; })();
const ksKey = s => s.kind + ':' + (s.kind === 'rbf' ? s.g : 0);
DEMOS['Kernel SVM'] = {
  vt: '평평한 칼로는 못 자르는 무리를 커널로 가르기',
  note: '안쪽 노랑, 바깥 파랑 고리 모양의 가상 데이터입니다. 40개로 배우고, 같은 방식으로 따로 뽑은 400개로 채점합니다. 바닥의 높이는 모델의 점수(경계로부터의 거리)를 0~1로 눌러 담은 것이고, 경계는 그 언덕의 허리(0.5)입니다.',
  controls: [{ k: 'kind', l: '커널', opts: [['lin', '선형 (커널 없음)'], ['poly', '다항 (2차)'], ['rbf', 'RBF (가우스)']], v: 'lin' }, { k: 'g', l: 'RBF 폭 γ', min: 1, max: 40, step: 1, v: 6 }],
  init(s) { this.change(s); },
  change(s) { s.M = KS.fit(ksKey(s)); s.fn = gridFn((x, y) => sig(1.6 * s.M.dec(x, y)), 1.2, 40); },
  read(s) { return [['서포트 벡터', s.M.sv.filter(Boolean).length + '개'], ['훈련 · 시험 정확도', pct(s.M.tr) + ' · ' + pct(s.M.te)]]; },
  say(s) { if (s.kind === 'lin') return '평평한 경계로는 안쪽 노랑을 둘러쌀 수 없어요. 시험 정확도 <b>' + pct(s.M.te) + '</b>. 커널을 바꿔 보세요!';
    if (s.kind === 'poly') return '2차 다항 커널은 점을 x², xy, y² 같은 <b>새 변수가 있는 공간으로 옮긴 셈</b>이에요. 거기서는 평평한 칼 하나로 갈리고, 원래 바닥에서는 <b>타원 경계</b>로 보여요. 시험 정확도 ' + pct(s.M.te) + '.';
    if (s.g > 25) return 'γ가 너무 커서 점마다 작은 언덕이 따로 솟았어요. 훈련 점은 다 맞히지만(' + pct(s.M.tr) + ') 시험 정확도는 ' + pct(s.M.te) + '예요. <b>과적합</b>이에요.';
    return 'RBF 커널은 "가까울수록 닮았다"를 점수로 써요. 서포트 벡터 ' + s.M.sv.filter(Boolean).length + '개 주변에 언덕을 쌓아 <b>둥근 경계</b>를 만들었어요. 시험 정확도 ' + pct(s.M.te) + '.'; },
  view(s) { const host = 'moa';
    return [SURF(host, { fn: s.fn, h: 1, pts: pts(KS.tr), ring: s.M.sv }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.M.tr, s.M.te], colors: [C.y, C.p], max: 1, labels: ['훈련 ' + pct(s.M.tr), '시험 ' + pct(s.M.te)], title: '정확도' }]; },
};
EXTRA['Kernel SVM'] = { say: 'SVM처럼 점을 넣으면 어느 쪽인지가 나와요. 그런데 경계가 휠 수 있어요!', io: ['점의 위치 (변수 여러 개)', '경계의 어느 쪽인가'],
  part: { items: [['ball', K.i, '입력 x'], ['box', K.gt, '커널 K(x, 서포트 벡터)'], ['box', K.bx, '닮음 점수 × α 더하기'], ['gate', K.g, '부호로 가르기'], ['ball', K.o, '답']],
    t: ['새 점과 서포트 벡터 사이의 "닮음"을 커널 함수로 잽니다.', '닮음에 서포트 벡터마다의 무게 α와 색(±1)을 곱해 더합니다.', '점을 정말 높은 차원으로 옮기지 않고도, 옮긴 것과 같은 계산이 됩니다(커널 트릭).'], say: '12층 SVM에서 "점끼리 곱하기" 칸 하나를 커널로 바꿨습니다. 나머지 부품은 그대로입니다.' },
  curve: { f: g => KS.fit('rbf:' + g).te, lo: 1, hi: 40, step: 3, v: 6, xl: 'RBF 폭 γ', yl: '시험 정확도', title: 'γ와 시험 정확도', t: ['γ가 작으면 언덕이 넓고 완만해서 경계가 거의 직선입니다.', 'γ가 크면 점마다 뾰족한 언덕이 솟아 훈련 점만 외웁니다. 이 데이터에서는 γ가 작을 때 가장 좋고, 커질수록 시험 정확도가 조금씩 내려갑니다.'] } };

/* ================= 20·21층 그래디언트 부스팅 · XGBoost (1차원 회귀) ================= */
const GB = (() => { const tr = wave(9, 30, .15), te = wave(109, 60, .15), XS = Array.from({ length: 121 }, (_, i) => -1.15 + 2.3 * i / 120);
  const boost = memo(key => { const [lr, n, dp, lam, gam] = key.split(':').map(Number), F = tr.map(() => 0), Fx = XS.map(() => 0), Ft = te.map(() => 0), H = [], T = [];
    for (let m = 0; m < n; m++) { const t = tree(tr.map((d, i) => ({ x: d.x, y: 0, t: d.y - F[i] })), { depth: dp, minLeaf: 2, reg: 1, lam, gamma: 2 * gam + 1e-12 });
      tr.forEach((d, i) => F[i] += lr * tpred(t, d.x, 0)); te.forEach((d, i) => Ft[i] += lr * tpred(t, d.x, 0)); XS.forEach((x, i) => Fx[i] += lr * tpred(t, x, 0));
      T.push(t); H.push({ fx: Fx.slice(), F: F.slice(), tr: mean(tr.map((d, i) => (d.y - F[i]) ** 2)), te: mean(te.map((d, i) => (d.y - Ft[i]) ** 2)) }); }
    return { T, H }; });
  const at = (fx, x) => fx[clamp(Math.round((x + 1.15) / 2.3 * 120), 0, 120)];
  return { tr, te, XS, boost, at }; })();
DEMOS.GBM = {
  vt: '남은 오차를 다음 나무가 맞히게 하기',
  note: '노랑 점 30개로 배우고 보이지 않는 점 60개로 채점합니다. 물결 곡선에 잡음을 섞은 가상의 데이터입니다. 나무는 깊이 2짜리 작은 회귀 나무이고, 분홍 선은 지금 남은 오차(잔차)입니다.',
  controls: [{ k: 'n', l: '나무 수', min: 1, max: 80, step: 1, v: 1 }, { k: 'lr', l: '학습률', min: .05, max: 1, step: .05, v: .3 }],
  init(s) { this.change(s); },
  change(s) { s.B = GB.boost(s.lr.toFixed(2) + ':80:2:0:0'); s.h = s.B.H[s.n - 1]; },
  read(s) { return [['훈련 오차', f2(s.h.tr, 3)], ['시험 오차', f2(s.h.te, 3)]]; },
  say(s) { const H = s.B.H, best = H.reduce((b, h, i) => h.te < H[b].te ? i : b, 0) + 1;
    if (s.n === 1) return '첫 나무 하나는 계단 몇 칸뿐입니다. 분홍 선(남은 오차)이 깁니다. 다음 나무는 <b>원래 점이 아니라 이 분홍 선</b>을 맞히도록 배웁니다. 나무 수를 올려 보십시오.';
    if (s.n > best + 15 && s.h.te > H[best - 1].te * 1.08) return '훈련 오차는 ' + f2(s.h.tr, 3) + '까지 내려갔지만 시험 오차는 나무 ' + best + '그루 때가 가장 낮았습니다. 나무가 잡음까지 따라가기 시작했습니다. 실제로는 따로 떼어 둔 <b>검증용 데이터</b>의 오차가 오르기 시작하면 일찍 멈춥니다(조기 종료).';
    return '나무 ' + s.n + '그루가 차례로 오차를 조금씩 메웠습니다. 학습률 ' + s.lr + '은 나무 하나의 말을 <b>' + Math.round(s.lr * 100) + '%만</b> 듣는다는 뜻입니다. 작게 하면 느리지만 차분하게 다가갑니다.'; },
  view(s) { const host = 'sia', h = s.h, prev = s.n > 1 ? s.B.H[s.n - 2].F : GB.tr.map(() => 0), t = s.B.T[s.n - 1];
    const L = []; for (let i = 0; i < 120; i++) L.push([GB.XS[i], h.fx[i], GB.XS[i + 1], h.fx[i + 1], C.b, .05]);
    const tl = []; for (let i = 0; i < 120; i++) tl.push([GB.XS[i], .92 + .4 * tpred(t, GB.XS[i], 0), GB.XS[i + 1], .92 + .4 * tpred(t, GB.XS[i + 1], 0), C.g]);
    return [SURF(host, { pts: GB.tr.map(d => ({ x: d.x, y: d.y, col: C.y })), lines: L, segs: [...GB.tr.map((d, i) => [d.x, d.y, d.x, h.F[i], C.p]), ...tl], marks: [{ x: -.2, y: 1.15, c: C.g, s: .05, t: '초록: 방금 나무가 배운 잔차 (줄여서 뒤쪽에)' }] }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: 0, ymax: .3, series: [{ v: s.B.H.slice(0, s.n).map(q => q.tr), c: C.y, len: 80, n: '훈련' }, { v: s.B.H.slice(0, s.n).map(q => q.te), c: C.p, len: 80, n: '시험' }], title: '나무 수에 따른 오차' }]; },
};
EXTRA.GBM = { say: '숫자를 넣으면, 나무들이 내놓은 작은 고침을 다 더한 숫자가 나와요!', io: ['숫자 (변수 여러 개)', '나무들의 답을 모두 더한 값'],
  part: { items: [['ball', K.i, '처음 예측 (0 또는 평균)'], ['box', K.p, '잔차 = 정답 − 지금 예측'], ['tiles', K.g, '잔차를 맞히는 작은 나무'], ['box', K.bx, '예측 += 학습률 × 나무'], ['ball', K.o, '다 더한 예측']], loops: [[3, 1, '나무 수만큼']],
    t: ['지금까지의 예측이 틀린 만큼(잔차)을 구합니다.', '그 잔차를 맞히는 작은 나무를 하나 세웁니다. 잔차는 오차 제곱의 기울기(그래디언트)와 같아서 이름이 그래디언트 부스팅입니다.', '나무의 답에 학습률을 곱해 예측에 더하고, 다시 잔차를 구합니다.'], say: '에이다부스트는 틀린 점의 무게를 키웠고, 그래디언트 부스팅은 틀린 양(잔차) 자체를 다음 나무에 넘깁니다. 손실 함수만 바꾸면 분류에도 씁니다.' },
  curve: { f: n => GB.boost('0.30:80:2:0:0').H[n - 1].te, lo: 1, hi: 80, step: 1, v: 1, xl: '나무 수 (학습률 0.3)', yl: '시험 오차', title: '나무 수와 시험 오차', t: ['처음 몇 그루에서 오차가 크게 줄어듭니다.', '그 뒤로는 거의 평평하거나 조금씩 다시 오릅니다. 가장 낮은 곳에서 멈추는 것을 조기 종료라고 합니다.'] } };

DEMOS.XGB = {
  vt: '잎 값 벌점 λ와 가지치기 기준 γ',
  note: '20층과 같은 점(훈련 30, 시험 60)입니다. 나무는 깊이 4까지, 30그루, 학습률 0.3으로 고정했습니다. λ는 잎에 적는 값을 작게 누르고, γ는 "이 정도 이득이 안 되면 가지를 치지 마라"는 기준입니다.',
  controls: [{ k: 'lam', l: '잎 값 벌점 λ', min: 0, max: 20, step: .5, v: 0 }, { k: 'gam', l: '가지치기 기준 γ', min: 0, max: .5, step: .01, v: 0 }],
  init(s) { this.change(s); },
  change(s) { s.B = GB.boost('0.30:30:4:' + s.lam + ':' + s.gam); s.h = s.B.H[29]; s.lv = s.B.T.reduce((t, q) => t + leaves(q), 0); },
  read(s) { return [['잎 수 (30그루 합)', s.lv + '개'], ['훈련 · 시험 오차', f2(s.h.tr, 3) + ' · ' + f2(s.h.te, 3)]]; },
  say(s) { const b = GB.boost('0.30:30:4:0:0'), bl = b.T.reduce((t, q) => t + leaves(q), 0);
    if (!s.lam && !s.gam) return '벌점이 없으면 20층의 그래디언트 부스팅과 같아요. 깊은 나무 30그루가 잎 ' + s.lv + '개로 잡음까지 따라가서 곡선이 <b>들쭉날쭉</b>해요. λ나 γ를 올려 보세요.';
    return '잎이 ' + bl + '개에서 <b>' + s.lv + '개</b>로 줄었어요. 시험 오차 ' + f2(b.H[29].te, 3) + ' → <b>' + f2(s.h.te, 3) + '</b>. XGBoost는 이렇게 <b>나무의 복잡함에 벌점</b>을 매겨서 과적합을 막아요.'; },
  view(s) { const host = 'moa', h = s.h, L = []; for (let i = 0; i < 120; i++) L.push([GB.XS[i], h.fx[i], GB.XS[i + 1], h.fx[i + 1], C.b, .05]);
    const base = GB.boost('0.30:30:4:0:0').H[29], L0 = []; for (let i = 0; i < 120; i++) L0.push([GB.XS[i], base.fx[i], GB.XS[i + 1], base.fx[i + 1], C.mu]);
    return [SURF(host, { pts: GB.tr.map(d => ({ x: d.x, y: d.y, col: C.y })), lines: L, segs: L0 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.h.tr, s.h.te], colors: [C.y, C.p], max: .1, labels: ['훈련 오차', '시험 오차'], title: '오차 (회색 선 = 벌점 없을 때)' }]; },
};
EXTRA.XGB = { say: '그래디언트 부스팅과 똑같이 숫자를 넣으면 숫자가 나와요. 다른 건 나무를 세우는 규칙이에요!', io: ['숫자 (변수 여러 개)', '나무들의 답을 모두 더한 값'],
  part: { items: [['box', K.p, '잔차의 1차·2차 정보 g, h'], ['gate', K.gt, '가지 이득 계산 (λ · γ 벌점 포함)'], ['tiles', K.g, '한 층씩 고르게 키운 나무'], ['box', K.bx, '잎 값 −G/(H+λ)'], ['ball', K.o, '예측 += 학습률 × 나무']], loops: [[4, 0, '나무 수만큼']],
    t: ['오차의 기울기(g)뿐 아니라 굽은 정도(h)까지 써서 잎 값과 가지 이득을 계산합니다.', 'λ는 잎 값을 0 쪽으로 누르고, 이득이 γ보다 작은 가지는 만들지 않습니다.', '계산을 병렬로 나누고 빈 값(결측)을 알아서 처리하는 등 빠르게 돌리는 장치가 많습니다.'], say: '20층의 부품에 "벌점 계산기"를 붙였습니다. 이 페이지는 제곱 오차만 쓰므로 h는 점마다 1입니다.' },
  curve: { f: l => GB.boost('0.30:30:4:' + l + ':0').H[29].te, lo: 0, hi: 20, step: 1, v: 0, xl: '잎 값 벌점 λ', yl: '시험 오차', title: 'λ와 시험 오차 (γ = 0)', t: ['λ를 조금 주면 깊은 나무의 지나친 잎 값이 눌려 시험 오차가 줄어듭니다.', '너무 크면 나무가 아무 말도 못 해서 다시 나빠집니다.'] } };

/* ================= 22층 LightGBM (시아) ================= */
const LG = (() => { const D = treeData(22, 110, .08), sp = { tr: split(D, 5, 36).tr, te: bigTest.tree(1022, .08) };
  const gini = P => { if (!P.length) return 0; const p = mean(P.map(d => d.c)); return P.length * 2 * p * (1 - p); };
  const best = (P, cand) => { let b = null, nc = 0; for (const f of ['x', 'y']) { const vs = cand(P, f); nc += vs.length; for (const t of vs) { const L = P.filter(d => d[f] < t), R = P.filter(d => d[f] >= t); if (L.length < 2 || R.length < 2) continue; const g = gini(P) - gini(L) - gini(R); if (!b || g > b.g) b = { f, t, g, L, R }; } } return { b, nc }; };
  const bins = nb => (P, f) => Array.from({ length: nb - 1 }, (_, k) => -1.1 + (k + 1) * 2.2 / nb);
  const exact = (P, f) => { const v = [...new Set(P.map(d => d[f]))].sort((a, b) => a - b); return v.slice(1).map((x, i) => (x + v[i]) / 2); };
  const grow = memo(key => { const [nl, nb] = key.split(':').map(Number), root = { leaf: 1, v: mean(sp.tr.map(d => d.c)), P: sp.tr }, L = [root]; let nc = 0, ne = 0;
    const plan = n => { if (n.plan) return; const q = best(n.P, bins(nb)); nc += q.nc; ne += best(n.P, exact).nc; n.plan = q.b; };
    while (L.length < nl) { L.forEach(plan); const c = L.filter(n => n.plan && n.plan.g > 1e-9).sort((a, b) => b.plan.g - a.plan.g)[0]; if (!c) break;
      const p = c.plan; delete c.leaf; c.f = p.f; c.t = p.t; c.l = { leaf: 1, v: mean(p.L.map(d => d.c)), P: p.L }; c.r = { leaf: 1, v: mean(p.R.map(d => d.c)), P: p.R }; L.splice(L.indexOf(c), 1, c.l, c.r); }
    const P = (x, y) => tpred(root, x, y); return { root, n: L.length, nc, ne, te: acc(sp.te, P), tr: acc(sp.tr, P) }; });
  return { ...sp, grow }; })();
DEMOS.LGBM = {
  vt: '값을 구간으로 묶고, 이득이 가장 큰 잎부터 키우기',
  note: '점 74개로 배우고, 같은 방식으로 따로 뽑은 점 400개로 채점합니다. 설명을 위해 만든 가상의 데이터입니다. 옅은 격자가 구간(빈)의 경계이고, 나무는 이 격자선 위에서만 자를 수 있습니다. 실제 LightGBM은 이런 나무 수백 그루로 부스팅합니다. 여기서는 나무 한 그루가 자라는 방식만 봅니다.',
  controls: [{ k: 'nl', l: '잎 수', min: 2, max: 16, step: 1, v: 4 }, { k: 'nb', l: '구간 수', min: 2, max: 32, step: 1, v: 8 }],
  init(s) { this.change(s); },
  change(s) { s.G = LG.grow(s.nl + ':' + s.nb); s.fn = gridFn((x, y) => tpred(s.G.root, x, y), 1.2, 40); },
  read(s) { return [['따져 본 기준값 (구간 · 모든 값)', s.G.nc + ' · ' + s.G.ne], ['시험 정확도', pct(s.G.te)]]; },
  say(s) { const G = s.G;
    if (s.nb <= 3) return '구간이 ' + s.nb + '개뿐이라 자를 수 있는 자리가 너무 적습니다. 경계를 정확한 곳에 긋지 못합니다. 시험 정확도 ' + pct(G.te) + '.';
    if (G.nc >= G.ne) return '점이 74개뿐이라 구간(' + G.nc + '개)이 모든 값(' + G.ne + '개)보다 오히려 많습니다. 데이터가 수백만 줄이면 모든 값은 수백만 개로 늘지만 구간은 그대로 몇백 개라, 그때 차이가 커집니다. 잎 ' + G.n + '개로 시험 정확도 ' + pct(G.te) + '.';
    return '모든 값을 따지면 기준값 ' + G.ne + '개를 봐야 하는데, 구간으로 묶어서 <b>' + G.nc + '개</b>만 봤습니다. 그리고 깊이를 고르게 키우는 대신 <b>이득이 가장 큰 잎 하나씩</b> 쪼갰습니다. 잎 ' + G.n + '개로 시험 정확도 ' + pct(G.te) + '.'; },
  view(s) { const host = 'sia', g = []; for (let k = 1; k < s.nb; k++) { const v = -1.1 + k * 2.2 / s.nb; g.push([v, -1.2, v, 1.2, 0x34406a], [-1.2, v, 1.2, v, 0x34406a]); }
    return [SURF(host, { fn: s.fn, h: .45, pts: pts(LG.tr), segs: [...g, ...tsegs(s.G.root, C.fg)], segY: .5 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.G.nc, s.G.ne], colors: [C.g, C.mu], max: Math.max(400, s.G.ne), labels: ['구간', '모든 값'], title: '따져 본 기준값 수' }]; },
};
EXTRA.LGBM = { say: '그래디언트 부스팅처럼 점을 넣으면 나무들의 답을 더한 값이 나와요. 다른 건 속도예요!', io: ['점의 위치 (변수 여러 개)', '나무들의 답을 모두 더한 값'],
  part: { items: [['tiles', K.i, '값을 구간(빈)으로 묶기'], ['box', K.bx, '구간별 합계표(히스토그램)'], ['gate', K.gt, '이득 가장 큰 잎 고르기'], ['tiles', K.g, '그 잎만 쪼개기'], ['ball', K.o, '예측 += 학습률 × 나무']], loops: [[3, 2, '잎 수 한도까지']],
    t: ['변수마다 값을 몇십~몇백 개 구간으로 미리 묶어 둡니다. 기준값 후보가 구간 경계로 줄어듭니다.', '나무를 층마다 고르게 키우지 않고, 쪼갰을 때 이득이 가장 큰 잎 하나만 골라 쪼갭니다.', '같은 잎 수에서 오차를 더 빨리 줄이지만, 잎 수를 너무 늘리면 한쪽으로 깊어져 과적합하기 쉽습니다.'], say: 'XGBoost와 같은 부스팅인데, "후보 줄이기"와 "급한 잎부터" 두 가지로 빠르게 만들었습니다. 데이터가 수백만 줄일 때 차이가 큽니다.' },
  curve: { f: nl => LG.grow(nl + ':16').te, lo: 2, hi: 16, step: 1, v: 4, xl: '잎 수 (구간 16)', yl: '시험 정확도', title: '잎 수와 시험 정확도', t: ['잎이 너무 적으면 경계를 다 못 그립니다.', '잎이 많아지면 잡음까지 칸으로 떼어 내서 정확도가 다시 떨어질 수 있습니다.'] } };

/* ================= 23층 CatBoost (모아) ================= */
const CB = (() => { const mk = (K, seed) => { const r = rng(seed), rate = Array.from({ length: K }, (_, i) => r() < .5 ? .3 : .7), row = () => { const c = Math.floor(r() * K); return { c, y: r() < rate[c] ? 1 : 0, j: r() * 2 - 1 }; };
    return { tr: Array.from({ length: 160 }, row), te: Array.from({ length: 400 }, row) }; };
  const run = memo(key => { const [K, mode] = key.split(':'), Kn = +K, { tr, te } = mk(Kn, 23), P = mean(tr.map(d => d.y)), sum = new Array(Kn).fill(0), cnt = new Array(Kn).fill(0);
    tr.forEach(d => { sum[d.c] += d.y; cnt[d.c]++; }); const r = rng(5), perm = tr.map((_, i) => i); for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const enc = new Array(tr.length); if (mode === 'mean') tr.forEach((d, i) => enc[i] = sum[d.c] / cnt[d.c]);
    else { const s2 = new Array(Kn).fill(0), c2 = new Array(Kn).fill(0); perm.forEach(i => { const d = tr[i]; enc[i] = (s2[d.c] + P) / (c2[d.c] + 1); s2[d.c] += d.y; c2[d.c]++; }); }
    const tenc = te.map(d => (sum[d.c] + P) / (cnt[d.c] + 1)), ok = (e, y) => ((e > .5 ? 1 : 0) === y ? 1 : 0);
    return { tr, enc, tra: mean(tr.map((d, i) => ok(enc[i], d.y))), tea: mean(te.map((d, i) => ok(tenc[i], d.y))) }; });
  return { run }; })();
DEMOS.CatBoost = {
  vt: '범주를 숫자로 바꿀 때 생기는 커닝 막기',
  note: '가상의 주문 160건(훈련)과 400건(시험)입니다. 변수는 "가게 이름" 하나뿐이고, 가게마다 재주문률이 30% 또는 70%로 정해져 있습니다. 가게 이름을 그 가게의 평균 재주문률로 바꾼 뒤, 0.5를 넘으면 "재주문"으로 답합니다. 판의 가로 위치가 바꾼 숫자입니다.',
  controls: [{ k: 'K', l: '가게 수', min: 4, max: 80, step: 4, v: 60 }, { k: 'mode', l: '숫자로 바꾸는 법', opts: [['mean', '그 가게 전체 평균 (자기 답 포함)'], ['ord', '순서대로, 앞 줄들만 보고 (CatBoost)']], v: 'mean' }],
  init(s) { this.change(s); },
  change(s) { s.R = CB.run(s.K + ':' + s.mode); },
  read(s) { return [['훈련 정확도', pct(s.R.tra)], ['시험 정확도', pct(s.R.tea)]]; },
  say(s) { const R = s.R;
    if (s.mode === 'mean') return '훈련 정확도 ' + pct(R.tra) + ', 시험 정확도 ' + pct(R.tea) + '. 가게당 주문이 ' + f2(160 / s.K, 1) + '건뿐이라, 평균 안에 <b>그 줄 자신의 정답이 섞여</b> 들어갔어요. 정답을 보고 숫자를 만든 셈이라 훈련에서만 잘 맞아요. 이걸 <b>타깃 누수</b>라고 해요.';
    return '주문을 무작위 순서로 줄 세우고, 각 줄은 <b>자기보다 앞 줄들만</b> 보고 평균을 내요. 자기 정답은 안 봐요. 훈련 ' + pct(R.tra) + ' · 시험 ' + pct(R.tea) + '. 훈련 성적이 더 이상 <b>부풀지 않아요.</b> 앞 줄이 적은 초반 줄은 숫자가 거칠어서 훈련 쪽이 오히려 조금 낮게 나오기도 해요.'; },
  view(s) { const host = 'moa', R = s.R;
    return [SURF(host, { pts: R.tr.map((d, i) => ({ x: R.enc[i] * 2.2 - 1.1, y: d.j, col: d.y ? C.y : C.a })), lines: [[0, -1.2, 0, 1.2, C.b, .05]], marks: [{ x: -.95, y: 1.1, c: C.a, s: .05, t: '← 재주문률 낮게 바꿈' }, { x: .55, y: 1.1, c: C.y, s: .05, t: '높게 바꿈 →' }] }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [R.tra, R.tea], colors: [C.y, C.p], max: 1, labels: ['훈련 ' + pct(R.tra), '시험 ' + pct(R.tea)], title: '정확도' }]; },
};
EXTRA.CatBoost = { say: '"서울점", "부산점" 같은 글자 변수를 그대로 넣으면, 예측한 숫자가 나와요!', io: ['범주(글자) 변수 + 숫자 변수', '예측 (부스팅)'],
  part: { items: [['tiles', K.i, '범주 변수 (가게 이름)'], ['gate', K.gt, '무작위 순서로 줄 세우기'], ['box', K.bx, '앞 줄들만 보고 평균 (순서 있는 타깃 통계)'], ['tiles', K.g, '대칭 나무: 같은 층 = 같은 질문'], ['ball', K.o, '예측 += 학습률 × 나무']], loops: [[4, 1, '나무 수만큼']],
    t: ['범주를 숫자로 바꿀 때, 줄마다 자기보다 앞 줄들의 평균만 써서 정답이 새지 않게 합니다.', '부스팅의 잔차를 계산할 때도 같은 "순서" 원리를 써서 자기 정답을 미리 본 효과를 막습니다.', '나무는 같은 층에서 모두 같은 질문을 쓰는 대칭 모양이라 예측이 아주 빠릅니다.'], say: '부스팅 부품은 20층과 같습니다. 앞에 "커닝 막는 숫자 변환기"를 달고, 나무 모양을 대칭으로 고정했습니다.' },
  curve: { f: k => { const a = CB.run(k + ':mean'); return a.tra - a.tea; }, lo: 4, hi: 80, step: 4, v: 60, xl: '가게 수', yl: '훈련 − 시험 정확도', title: '그 가게 전체 평균으로 바꿨을 때의 틈', t: ['가게가 많을수록 가게당 줄이 적어서, 평균에 자기 정답이 차지하는 몫이 커집니다.', '그만큼 훈련 성적이 부풀고 시험 성적과의 틈이 벌어집니다.'] } };

/* ================= 24층 가우스 과정 (시아) ================= */
const GPD = (() => { const r = rng(24), X0 = [-.85, -.55, -.3, .3, .7], EX = [0, .95, -1.02, .5, -.1], f = x => .6 * Math.sin(2.5 * x + .4), noise = [...X0, ...EX].map(() => gauss(r) * .05), XS = Array.from({ length: 81 }, (_, i) => -1.2 + 2.4 * i / 80);
  const data = n => [...X0, ...EX].slice(0, n).map((x, i) => ({ x, y: f(x) + noise[i] }));
  const chol = A => { const n = A.length, L = A.map(() => new Array(n).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k]; L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]; } return L; };
  const post = (D, l, sn) => { const k = (a, b) => .36 * Math.exp(-((a - b) ** 2) / (2 * l * l)), n = D.length, Km = D.map(a => D.map(b => k(a.x, b.x) + (a === b ? sn * sn : 0))), al = solve(Km, D.map(d => d.y)), L = chol(Km);
    const mu = [], sd = []; XS.forEach(x => { const ks = D.map(d => k(d.x, x)), v = solve(Km, ks); mu.push(ks.reduce((t, q, i) => t + q * al[i], 0)); sd.push(Math.sqrt(Math.max(1e-9, .36 - ks.reduce((t, q, i) => t + q * v[i], 0)) + sn * sn)); });
    const lml = -.5 * D.reduce((t, d, i) => t + d.y * al[i], 0) - L.reduce((t, row, i) => t + Math.log(row[i]), 0) - n / 2 * Math.log(2 * Math.PI); return { mu, sd, lml }; };
  const at = (A, x) => { const u = clamp((x + 1.2) / 2.4 * 80, 0, 79.999), i = Math.floor(u), t = u - i; return A[i] * (1 - t) + A[i + 1] * t; };
  return { data, post, XS, at }; })();
DEMOS.GP = {
  vt: '점이 없는 곳에서는 모른다고 말하기',
  note: '물결 곡선 위에 잡음을 조금 섞은 가상의 점들입니다. 바닥의 능선이 예측의 분포입니다. 능선이 좁고 높으면 자신 있고, 넓고 낮으면 자신 없습니다. 분홍 선은 ±2σ 띠입니다.',
  controls: [{ k: 'l', l: '길이 척도 ℓ', min: .05, max: 1, step: .01, v: .3 }, { k: 'sn', l: '잡음 크기 σₙ', min: .01, max: .4, step: .01, v: .05 }, { b: 'add', l: '점 하나 더' }, { b: 'reset', l: '처음 5개로' }],
  init(s) { s.n = 5; this.change(s); },
  change(s) { s.D = GPD.data(s.n); s.P = GPD.post(s.D, s.l, s.sn); const P = s.P; s.fn = gridFn((x, y) => { const m = GPD.at(P.mu, x), d = GPD.at(P.sd, x); return Math.exp(-((y - m) ** 2) / (2 * d * d)) * clamp(.12 / d, .15, 1); }, 1.2, 48); },
  click(s, b) { s.n = b === 'add' ? Math.min(10, s.n + 1) : 5; this.change(s); },
  read(s) { const w = Math.max(...s.P.sd) * 4; return [['가장 넓은 띠 폭 (±2σ)', f2(w, 2)], ['데이터 점', s.n + '개']]; },
  say(s) { if (s.l < .1) return 'ℓ이 너무 짧아서 점 바로 옆만 믿고, 조금만 떨어져도 <b>모른다</b>고 합니다. 곡선이 점 사이에서 0으로 주저앉습니다.';
    if (s.l > .7) return 'ℓ이 너무 길어서 멀리 떨어진 점끼리도 닮았다고 봅니다. 곡선이 지나치게 <b>뻣뻣</b>해서 물결을 따라가지 못합니다.';
    return '점 근처는 띠가 좁고, <b>점이 없는 곳은 띠가 넓습니다.</b> 이 모델은 예측과 함께 "얼마나 모르는지"를 내놓습니다. <b>점 하나 더</b>를 눌러 빈 곳을 채우면 그 자리 띠가 좁아집니다.'; },
  view(s) { const host = 'sia', P = s.P, X = GPD.XS, L = [], U = [];
    for (let i = 0; i < 80; i++) { L.push([X[i], P.mu[i], X[i + 1], P.mu[i + 1], C.b, .045]); for (const sg of [1, -1]) U.push([X[i], P.mu[i] + sg * 2 * P.sd[i], X[i + 1], P.mu[i + 1] + sg * 2 * P.sd[i + 1], C.p]); }
    return [SURF(host, { fn: s.fn, h: .8, pts: s.D.map(d => ({ x: d.x, y: d.y, col: C.y })), lines: L, lineY: .86, segs: U.filter(q => Math.abs(q[1]) < 1.2 && Math.abs(q[3]) < 1.2), segY: .05 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [Math.max(...P.sd) * 4, Math.min(...P.sd) * 4], colors: [C.p, C.g], max: 3, labels: ['가장 넓은 곳', '가장 좁은 곳'], title: '±2σ 띠 폭' }]; },
};
EXTRA.GP = { say: '숫자를 넣으면 예측한 숫자와 함께 "이 정도는 틀릴 수 있다"는 폭이 나와요!', io: ['숫자 (변수 여러 개)', '예측 평균 + 불확실성(±σ)'],
  part: { items: [['ball', K.i, '새 x'], ['box', K.gt, '커널: 데이터 점들과의 닮음'], ['tiles', K.bx, '데이터끼리의 닮음 표 K'], ['box', K.g, '평균 = kᵀK⁻¹y, 분산 = k(x,x) − kᵀK⁻¹k'], ['ball', K.o, '예측 + 띠']],
    t: ['"가까운 x끼리는 y도 닮았다"는 믿음을 커널로 적습니다. ℓ은 얼마나 멀리까지 닮았다고 볼지입니다.', '데이터 점들과 새 x 사이의 닮음으로 평균과 분산을 계산합니다.', '데이터 점에서 멀어질수록 닮음이 줄어 분산(띠)이 커집니다.'], say: '배운다기보다 데이터를 들고 있다가 계산합니다. 점이 n개면 n×n 표를 풀어야 해서, 점이 수만 개면 느립니다.' },
  curve: { f: l => GPD.post(GPD.data(5), l, .05).lml, lo: .05, hi: 1, step: .05, v: .3, xl: '길이 척도 ℓ', yl: '로그 주변 우도', title: 'ℓ 고르기 (점 5개, σₙ = 0.05)', t: ['로그 주변 우도는 "이 ℓ이면 지금 데이터가 나올 법한가"입니다.', '가장 높은 곳의 ℓ을 고르면 너무 들쭉날쭉하지도, 뻣뻣하지도 않습니다.'] } };

/* ================= 25·26층 t-SNE · UMAP (같은 8차원 데이터) ================= */
const HD = (() => { const r = rng(2525), P = 8, cen = [0, 1, 2].map(() => Array.from({ length: P }, () => gauss(r) * 2.2)), X = [], lab = [];
  cen.forEach((c, k) => { for (let i = 0; i < 20; i++) { X.push(c.map(v => v + gauss(r) * (k === 2 ? 1.3 : .8))); lab.push(k); } });
  const n = X.length, D2 = X.map(a => X.map(b => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0)));
  const knn = (i, k) => D2[i].map((d, j) => [d, j]).filter(q => q[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, k).map(q => q[1]);
  const keep = (Y, k) => { let t = 0; for (let i = 0; i < n; i++) { const h = new Set(knn(i, k)), l = Y.map((y, j) => [(y[0] - Y[i][0]) ** 2 + (y[1] - Y[i][1]) ** 2, j]).filter(q => q[1] !== i).sort((a, b) => a[0] - b[0]).slice(0, k); t += l.filter(q => h.has(q[1])).length / k; } return t / n; };
  return { X, lab, n, D2, knn, keep }; })();
const norm = Y => { const mx = Math.max(...Y.map(y => (y[0] - mean(Y.map(q => q[0]))) ** 2 + (y[1] - mean(Y.map(q => q[1]))) ** 2).map(Math.sqrt)) || 1, cx = mean(Y.map(q => q[0])), cy = mean(Y.map(q => q[1])); return Y.map(y => [(y[0] - cx) / mx * .95, (y[1] - cy) / mx * .95]); };
const TS = (() => { const n = HD.n;
  const Pm = memo(perp => { const P = HD.D2.map(() => new Array(n).fill(0)), H = Math.log(perp);
    for (let i = 0; i < n; i++) { let lo = 0, hi = 1e4, beta = 1; for (let it = 0; it < 60; it++) { let s = 0, sh = 0; for (let j = 0; j < n; j++) if (j !== i) { const p = Math.exp(-HD.D2[i][j] * beta); P[i][j] = p; s += p; sh += p * HD.D2[i][j] * beta; }
        const h = Math.log(s) + sh / s; if (h > H) { lo = beta; beta = hi >= 1e4 ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; } }
      const s = P[i].reduce((a, b) => a + b, 0); for (let j = 0; j < n; j++) P[i][j] /= s; }
    return P.map((row, i) => row.map((v, j) => Math.max((v + P[j][i]) / (2 * n), 1e-12))); });
  const start = seed => { const r = rng(seed); return { Y: HD.X.map(() => [gauss(r) * 1e-2, gauss(r) * 1e-2]), V: HD.X.map(() => [0, 0]), it: 0, kl: [] }; };
  const iter = (S, P) => { const Y = S.Y, ex = S.it < 60 ? 4 : 1, W = Y.map(() => new Array(n).fill(0)); let Z = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const w = 1 / (1 + (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2); W[i][j] = W[j][i] = w; Z += 2 * w; }
    let kl = 0; const mo = S.it < 60 ? .5 : .8;
    for (let i = 0; i < n; i++) { let gx = 0, gy = 0; for (let j = 0; j < n; j++) if (j !== i) { const q = W[i][j] / Z, c = (ex * P[i][j] - q) * W[i][j]; gx += 4 * c * (Y[i][0] - Y[j][0]); gy += 4 * c * (Y[i][1] - Y[j][1]); if (ex === 1) kl += P[i][j] * Math.log(P[i][j] / Math.max(q, 1e-12)); }
      S.V[i][0] = mo * S.V[i][0] - 100 * gx; S.V[i][1] = mo * S.V[i][1] - 100 * gy; }
    Y.forEach((y, i) => { y[0] += S.V[i][0]; y[1] += S.V[i][1]; }); S.it++; if (ex === 1) S.kl.push(kl); };
  const ref = memo(() => { const S = start(1), P = Pm(10); for (let i = 0; i < 400; i++) iter(S, P); return S.kl; });
  return { Pm, start, iter, ref }; })();
DEMOS['t-SNE'] = {
  vt: '가까웠던 점끼리 가깝게, 2차원에 다시 앉히기',
  note: '변수 8개짜리 가상의 점 60개(무리 셋, 20개씩)입니다. 색은 무리를 알아보라고 칠했을 뿐이고, t-SNE는 색을 보지 않습니다. 처음에는 모든 점이 한가운데 뭉쳐 있습니다.',
  controls: [{ k: 'perp', l: '퍼플렉서티 (이웃 수 감각)', min: 2, max: 30, step: 1, v: 10 }, { b: 'run', l: '돌리기 / 멈추기' }, { b: 'reset', l: '처음부터' }],
  init(s) { s.seed = 1; this.change(s); },
  change(s) { s.S = TS.start(s.seed); s.P = TS.Pm(s.perp); s.auto = 0; s.Y = norm(s.S.Y); },
  click(s, b) { if (b === 'reset') { s.seed++; this.change(s); } else s.auto = s.auto ? 0 : 1; },
  step(s) { if (!s.auto) return; for (let k = 0; k < 4; k++) TS.iter(s.S, s.P); s.Y = norm(s.S.Y); if (s.S.it >= 400) s.auto = 0; },
  read(s) { const k = s.S.kl; return [['반복', s.S.it + ' / 400'], ['KL (낮을수록 이웃을 잘 지킴)', k.length ? f2(k[k.length - 1], 3) : '—']]; },
  say(s) { if (s.S.it === 0) return '<b>돌리기</b>를 눌러 보세요! 점마다 8차원에서 가까웠던 이웃을 기억하고, 2차원에서도 그 이웃 곁으로 다가가요.';
    if (s.S.it < 60) return '처음 60번은 이웃 끌림을 4배로 키워서 무리들이 <b>먼저 크게 갈라지게</b> 해요(과장 단계).';
    if (s.perp < 5) return '퍼플렉서티가 너무 작아서 바로 옆 몇 개만 이웃으로 봐요. 한 무리가 <b>작은 조각들로</b> 쪼개져 보일 수 있어요.';
    return s.S.it + '번 돌았어요. 같은 색끼리 모였죠? 그런데 무리 사이의 <b>거리와 크기는 믿으면 안 돼요.</b> t-SNE는 가까운 이웃만 지키고 먼 거리는 신경 쓰지 않거든요.'; },
  view(s) { const host = 'moa';
    return [SURF(host, { pts: s.Y.map((y, i) => ({ x: y[0], y: y[1], col: PAL[HD.lab[i]] })) }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: 0, ymax: Math.max(1, ...(s.S.kl.length ? [s.S.kl[0]] : [1])) * 1.05, series: [{ v: s.S.kl.filter((_, i) => i % 4 === 0), c: C.b, len: 86, n: 'KL' }], title: '돌수록 줄어드는 KL' }]; },
};
EXTRA['t-SNE'] = { say: '변수가 많은 점들을 넣으면, 눈으로 볼 수 있는 2차원 지도가 나와요!', io: ['변수 여러 개 (예: 8개)', '2차원 좌표 (그림용)'],
  part: { items: [['tiles', K.i, '고차원 점들'], ['box', K.bx, '고차원 이웃 확률 P (종 모양)'], ['ball', K.o, '2차원 점 아무렇게나'], ['box', K.g, '2차원 이웃 확률 Q (꼬리 두꺼운 t)'], ['gate', K.gt, 'P와 Q 차이(KL) 줄이기']], loops: [[4, 2, '반복']],
    t: ['고차원에서 점마다 "누가 내 이웃일 확률"을 종 모양으로 구합니다. 퍼플렉서티는 이웃을 몇 개쯤으로 볼지입니다.', '2차원에서도 같은 확률을 구하되, 꼬리가 두꺼운 t 분포를 써서 먼 점들이 서로 넉넉히 떨어지게 합니다.', '두 확률의 차이(KL)를 줄이는 쪽으로 2차원 점을 조금씩 옮깁니다.'], say: '그림용입니다. 새 점이 오면 처음부터 다시 돌려야 하고, 무리 사이 거리에는 뜻이 없습니다.' },
  curve: { f: it => { const k = TS.ref(); return k[clamp(Math.round(it) - 61, 0, k.length - 1)]; }, lo: 60, hi: 400, step: 10, v: 100, xl: '반복 횟수', yl: 'KL', title: '반복에 따라 줄어드는 KL (퍼플렉서티 10)', t: ['과장 단계가 끝난 직후 KL이 빠르게 줄어듭니다.', '그 뒤로는 천천히 줄며 지도가 자리를 잡습니다.'] } };

const UM = (() => { const n = HD.n, D = HD.D2.map(r => r.map(Math.sqrt));
  const ab = memo(md => { let best = null; for (let a = .2; a < 8; a *= 1.08) for (let b = .4; b <= 1.6; b += .05) { let e = 0; for (let d = .02; d <= 3; d += .06) { const t = d < md ? 1 : Math.exp(-(d - md)), p = 1 / (1 + a * d ** (2 * b)); e += (p - t) ** 2; } if (!best || e < best.e) best = { a, b, e }; } return best; });
  const graph = memo(k => { const W = new Map(), key = (i, j) => i < j ? i * n + j : j * n + i, A = [];
    for (let i = 0; i < n; i++) { const nb = HD.knn(i, k), rho = D[i][nb[0]]; let lo = 0, hi = 1e3, sg = 1; for (let it = 0; it < 64; it++) { const s = nb.reduce((t, j) => t + Math.exp(-Math.max(0, D[i][j] - rho) / sg), 0); if (s > Math.log2(k)) { hi = sg; sg = (lo + sg) / 2; } else { lo = sg; sg = hi >= 1e3 ? sg * 2 : (sg + hi) / 2; } }
      nb.forEach(j => A.push([i, j, Math.exp(-Math.max(0, D[i][j] - rho) / sg)])); }
    const M = new Map(); A.forEach(([i, j, w]) => M.set(i * n + j, w)); A.forEach(([i, j, w]) => { const v = M.get(j * n + i) || 0, s = w + v - w * v; W.set(key(i, j), s); });
    return [...W.entries()].map(([kk, w]) => [Math.floor(kk / n), kk % n, w]); });
  const start = seed => { const r = rng(seed); return { Y: HD.X.map(() => [r() * 20 - 10, r() * 20 - 10]), ep: 0, hist: [] }; };
  const epoch = (S, E, a, b, r, N = 200) => { const lr = 1 - S.ep / N, Y = S.Y, cl = v => clamp(v, -4, 4);
    E.forEach(([i, j, w]) => { if (r() > w) return; let dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1], d2 = dx * dx + dy * dy;
      if (d2 > 0) { const c = -2 * a * b * d2 ** (b - 1) / (1 + a * d2 ** b); Y[i][0] += lr * cl(c * dx); Y[i][1] += lr * cl(c * dy); Y[j][0] -= lr * cl(c * dx); Y[j][1] -= lr * cl(c * dy); }
      for (let t = 0; t < 3; t++) { const q = Math.floor(r() * n); if (q === i) continue; dx = Y[i][0] - Y[q][0]; dy = Y[i][1] - Y[q][1]; d2 = dx * dx + dy * dy; const c = 2 * b / ((.001 + d2) * (1 + a * d2 ** b)); Y[i][0] += lr * cl(c * dx); Y[i][1] += lr * cl(c * dy); } });
    S.ep++; };
  const full = memo(key => { const [k, md] = key.split(':').map(Number), S = start(1), E = graph(k), { a, b } = ab(md), r = rng(3); for (let e = 0; e < 200; e++) epoch(S, E, a, b, r); return HD.keep(S.Y, k); });
  return { ab, graph, start, epoch, full }; })();
DEMOS.UMAP = {
  vt: '이웃 그물을 엮고, 그물 모양대로 2차원에 펼치기',
  note: '25층 t-SNE와 같은 8차원 가상 데이터(무리 셋, 60개)입니다. 색은 알아보기용이고 UMAP은 보지 않습니다. 옅은 선이 고차원에서 엮은 이웃 그물입니다. 여기 구현은 원 논문의 핵심만 줄인 교육용입니다.',
  controls: [{ k: 'k', l: '이웃 수', min: 2, max: 20, step: 1, v: 8 }, { k: 'md', l: '최소 거리', min: 0, max: .8, step: .05, v: .1 }, { b: 'run', l: '돌리기 / 멈추기' }, { b: 'reset', l: '처음부터' }],
  init(s) { s.seed = 1; this.change(s); },
  change(s) { s.S = UM.start(s.seed); s.E = UM.graph(s.k); s.AB = UM.ab(s.md); s.r = rng(s.seed * 7 + 3); s.auto = 0; s.Y = norm(s.S.Y); s.kp = HD.keep(s.S.Y, s.k); },
  click(s, b) { if (b === 'reset') { s.seed++; this.change(s); } else s.auto = s.auto ? 0 : 1; },
  step(s) { if (!s.auto) return; for (let k = 0; k < 2; k++) UM.epoch(s.S, s.E, s.AB.a, s.AB.b, s.r); s.Y = norm(s.S.Y); if (s.S.ep % 10 === 0) { s.kp = HD.keep(s.S.Y, s.k); s.S.hist.push(s.kp); } if (s.S.ep >= 200) { s.auto = 0; s.kp = HD.keep(s.S.Y, s.k); } },
  read(s) { return [['반복', s.S.ep + ' / 200'], ['이웃 유지 비율', pct(s.kp)]]; },
  say(s) { if (s.S.ep === 0) return '점을 아무 데나 흩어 놓았습니다. <b>돌리기</b>를 누르면 그물로 이어진 이웃은 끌어당기고, 아무 점이나 몇 개 골라 밀어냅니다.';
    if (s.S.ep < 200) return s.S.ep + '번 돌았습니다. 이웃 유지 비율(8차원에서 이웃이던 점이 2차원에서도 이웃인 비율)이 ' + pct(s.kp) + '입니다.';
    return '다 돌았습니다. 이웃 유지 비율 <b>' + pct(s.kp) + '</b>. 이웃 수를 키우면 큰 그림(무리 사이 배치)을, 줄이면 동네 모양을 더 챙깁니다. 최소 거리는 점들이 얼마나 빽빽하게 뭉칠지입니다.'; },
  view(s) { const host = 'sia', Y = s.Y;
    return [SURF(host, { pts: Y.map((y, i) => ({ x: y[0], y: y[1], col: PAL[HD.lab[i]] })), segs: s.E.filter(e => e[2] > .5).map(([i, j]) => [Y[i][0], Y[i][1], Y[j][0], Y[j][1], 0x4a5a8a]) }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: 0, ymax: 1, series: [{ v: s.S.hist, c: C.g, len: 20, n: '이웃 유지 비율' }], title: '돌수록 오르는 이웃 유지 비율' }]; },
};
EXTRA.UMAP = { say: 't-SNE처럼 2차원 지도가 나와요. 더 빠르고, 새 점도 지도에 얹을 수 있어요!', io: ['변수 여러 개 (예: 8개)', '2차원(또는 몇 차원) 좌표'],
  part: { items: [['tiles', K.i, '고차원 점들'], ['box', K.bx, '이웃 k개 찾기'], ['tiles', K.g, '거리 → 연결 세기 (흐릿한 그물)'], ['ball', K.o, '2차원 배치'], ['box', K.gt, '이웃은 당기고, 아무 점은 밀기']], loops: [[4, 3, '반복']],
    t: ['점마다 가까운 이웃 k개를 찾고, 거리를 0~1 사이 연결 세기로 바꿔 그물을 엮습니다.', '2차원에서도 비슷한 그물이 되도록, 연결된 점은 끌어당기고 무작위로 고른 점은 밀어냅니다.', '최소 거리는 2차원에서 점끼리 얼마나 붙을 수 있는지를 정합니다.'], say: 't-SNE와 목표는 같지만, 모든 점 쌍 대신 이웃 그물과 무작위 몇 쌍만 계산해서 빠릅니다.' },
  curve: { f: k => UM.full(k + ':0.1'), lo: 2, hi: 20, step: 2, v: 8, xl: '이웃 수 k', yl: '이웃 유지 비율', title: '이웃 수에 따른 이웃 유지 비율', t: ['k가 아주 작으면 그물이 끊어져 무리가 조각나고, 이웃 유지 비율이 낮습니다.', 'k를 키우면 무리 전체가 하나로 엮여 비율이 오릅니다. 이 데이터는 무리마다 점이 20개라 k가 20에 가까우면 무리 전체가 서로 이웃입니다.'] } };

/* ================= 27층 다층 퍼셉트론 (모아) ================= */
const MP = (() => { const D = moons(27, 34, .12), sp = { tr: split(D, 4, 24).tr, te: moons(1027, 200, .12) };
  const make = (H, seed) => { const r = rng(seed * 31 + H); return { H, W1: Array.from({ length: H }, () => [gauss(r) * 1.2, gauss(r) * 1.2]), b1: Array.from({ length: H }, () => gauss(r) * .5), W2: Array.from({ length: H }, () => gauss(r) * .5), b2: 0, ep: 0, V: null, loss: [] }; };
  const fwd = (N, x, y) => { const h = N.W1.map((w, k) => Math.tanh(w[0] * x + w[1] * y + N.b1[k])); return [sig(h.reduce((s, v, k) => s + v * N.W2[k], N.b2)), h]; };
  const train = (N, E, lr) => { const T = sp.tr, n = T.length; if (!N.V) N.V = { W1: N.W1.map(() => [0, 0]), b1: N.b1.map(() => 0), W2: N.W2.map(() => 0), b2: 0 };
    for (let e = 0; e < E; e++) { const g = { W1: N.W1.map(() => [0, 0]), b1: N.b1.map(() => 0), W2: N.W2.map(() => 0), b2: 0 }; let L = 0;
      T.forEach(d => { const [p, h] = fwd(N, d.x, d.y), dz = p - d.c; L -= d.c ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12); g.b2 += dz; h.forEach((v, k) => { g.W2[k] += dz * v; const dh = dz * N.W2[k] * (1 - v * v); g.W1[k][0] += dh * d.x; g.W1[k][1] += dh * d.y; g.b1[k] += dh; }); });
      N.V.b2 = .9 * N.V.b2 - lr * g.b2 / n; N.b2 += N.V.b2;
      for (let k = 0; k < N.H; k++) { N.V.W2[k] = .9 * N.V.W2[k] - lr * g.W2[k] / n; N.W2[k] += N.V.W2[k]; N.V.b1[k] = .9 * N.V.b1[k] - lr * g.b1[k] / n; N.b1[k] += N.V.b1[k];
        for (let j = 0; j < 2; j++) { N.V.W1[k][j] = .9 * N.V.W1[k][j] - lr * g.W1[k][j] / n; N.W1[k][j] += N.V.W1[k][j]; } }
      N.ep++; if (N.ep % 20 === 0) N.loss.push(L / n); } };
  const P = N => (x, y) => fwd(N, x, y)[0];
  const final = memo(H => Math.max(...[1, 2].map(sd => { const N = make(H, sd); train(N, 1500, .5); return acc(sp.te, P(N)); })));
  return { ...sp, make, train, P, final }; })();
DEMOS.MLP = {
  vt: '은닉 뉴런이 직선 여러 개를 엮어 곡선 경계를 만든다',
  note: '초승달 두 개가 맞물린 가상의 데이터 44개로 배우고, 같은 방식으로 따로 뽑은 400개로 채점합니다. 바닥의 높이가 "노랑일 확률", 오른쪽은 이 신경망의 모양입니다. 선 굵기가 연결의 세기, 파랑은 +, 하늘은 − 입니다.',
  controls: [{ k: 'H', l: '은닉 뉴런 수', min: 1, max: 12, step: 1, v: 4 }, { k: 'lr', l: '학습률', min: .05, max: 1.5, step: .05, v: .5 }, { b: 'run', l: '학습 시작 / 멈춤' }, { b: 'reset', l: '처음부터' }],
  init(s) { s.seed = 1; this.change(s, 'H'); },
  change(s, k) { if (k === 'lr') return; s.N = MP.make(s.H, s.seed); s.auto = 0; this.fresh(s); },
  fresh(s) { s.fn = gridFn(MP.P(s.N), 1.2, 36); s.te = acc(MP.te, MP.P(s.N)); s.tr = acc(MP.tr, MP.P(s.N)); s.t = 0; },
  click(s, b) { if (b === 'reset') { s.seed++; this.change(s, 'H'); } else s.auto = s.auto ? 0 : 1; },
  step(s, dt) { if (!s.auto) return; MP.train(s.N, 20, s.lr); s.t += dt; if (s.t > .25) this.fresh(s); if (s.N.ep >= 3000) { s.auto = 0; this.fresh(s); } },
  read(s) { const L = s.N.loss; return [['반복 · 손실', s.N.ep + ' · ' + (L.length ? f2(L[L.length - 1], 3) : '—')], ['훈련 · 시험 정확도', pct(s.tr) + ' · ' + pct(s.te)]]; },
  say(s) { if (s.N.ep === 0) return '가중치를 아무렇게나 놓은 신경망이에요. <b>학습 시작</b>을 누르면 역전파로 틀린 만큼 가중치를 고쳐요.';
    if (s.H === 1) return '은닉 뉴런이 하나면 결국 S자 언덕 하나, <b>3층 로지스틱 회귀와 같아요.</b> 경계가 직선 하나라서 초승달 끝부분은 못 맞혀요. 시험 정확도 ' + pct(s.te) + '. 뉴런을 늘려 보세요.';
    return s.N.ep + '번 고쳤어요. 은닉 뉴런 하나하나가 직선 하나를 긋고, 출력이 그 직선들을 <b>섞어서 휜 경계</b>를 만들어요. 시험 정확도 ' + pct(s.te) + '. 이 건물은 여기까지예요. 신경망을 더 깊이 쌓는 이야기는 옆 건물 <b>신경망 연구소</b>에서 이어져요!'; },
  view(s) { const host = 'moa', N = s.N, X0 = LAY[host].wx - 2, nodes = [], links = [], Y = (i, m) => 2.1 + (i - (m - 1) / 2) * Math.min(.62, 3.6 / Math.max(1, m));
    nodes.push({ p: [X0, Y(0, 2), 0], c: C.y, r: .24, t: 'x₁' }, { p: [X0, Y(1, 2), 0], c: C.y, r: .24, t: 'x₂' }); N.W1.forEach((_, k) => nodes.push({ p: [X0 + 1.9, Y(k, N.H), 0], c: C.g, r: .2 })); nodes.push({ p: [X0 + 3.8, Y(0, 1), 0], c: C.b, r: .28, t: '노랑일 확률' });
    const mx = Math.max(1e-6, ...N.W1.flat().map(Math.abs), ...N.W2.map(Math.abs));
    N.W1.forEach((w, k) => w.forEach((v, j) => links.push([j, 2 + k, Math.abs(v) / mx, v > 0 ? C.b : C.a]))); N.W2.forEach((v, k) => links.push([2 + k, 2 + N.H, Math.abs(v) / mx, v > 0 ? C.b : C.a]));
    return [SURF(host, { fn: s.fn, h: .7, pts: pts(MP.tr) }), { id: 'n', type: 'nodes', x: 0, z: -1.4, nodes, links, title: '2 → ' + N.H + ' → 1 신경망', tx: X0 + 1.9, ty: 4.9 }]; },
};
EXTRA.MLP = { say: '숫자 여러 개를 넣으면, 층을 지나며 바뀌어서 확률(또는 숫자)이 나와요!', io: ['숫자 여러 개 (점의 위치)', '노랑일 확률'],
  part: { items: [['ball', K.i, '입력층 x'], ['tiles', K.g, '은닉층: tanh(W₁x + b₁)'], ['tiles', K.g, '(더 쌓으면 은닉층 더)'], ['gate', K.gt, '출력층: σ(W₂h + b₂)'], ['box', K.p, '로그 손실']], loops: [[4, 1, '역전파로 W 고치기']],
    t: ['은닉 뉴런마다 가중합을 구해 휘는 함수(tanh)에 넣습니다. 뉴런 하나가 직선 하나를 그은 셈입니다.', '출력층이 은닉층의 결과를 다시 가중합해 확률로 바꿉니다. 직선들이 섞여 휜 경계가 됩니다.', '손실의 기울기를 출력층부터 거꾸로 전해(역전파) 모든 가중치를 고칩니다.'], say: '3층 로지스틱 회귀를 여러 개 나란히 놓고, 그 위에 하나를 더 얹은 것입니다. 여기서부터가 신경망입니다.' },
  curve: { f: MP.final, lo: 1, hi: 12, step: 1, v: 4, xl: '은닉 뉴런 수', yl: '시험 정확도', title: '은닉 뉴런 수와 시험 정확도 (1500번 학습, 시작점 2개 중 좋은 쪽)', t: ['뉴런이 하나면 로지스틱 회귀와 같아서 경계가 직선 하나뿐입니다.', '몇 개만 있어도 휜 경계를 그릴 수 있고, 그 뒤로는 크게 달라지지 않습니다.'] } };
