// 머신러닝 연구소 · 저층부(2~9층) 실험과 구역 ①③④
// 형식은 demos.js 맨 위 설명과 같다. 점은 모두 설명을 위해 만든 가상의 데이터다.
import { C, clamp, rng, gauss, f2, pct, mean, sgn, memo, sig, d2, lsq, clipLine, SURF, SIDE, gridFn, pts, curve, blobs2, treeData, split, acc, wave, tree, tpred, leaves, tsegs, bigTest } from './ml.js';

const K = { i: '#ffd76a', o: '#ff9f6b', bx: '#6fb1ff', gt: '#b79cff', g: '#7fe0a8', p: '#ff8fc7' };
const PAL = [C.y, C.a, C.v, C.g, C.p, C.b];
export const DEMOS = {}, EXTRA = {};

/* ================= 2층 k-최근접 이웃 (시아) ================= */
const KN = (() => { const D = blobs2(36, 30, .42, .3, .22), sp = { tr: split(D, 2, 24).tr, te: bigTest.blobs(1036, .42, .3, .22) };
  const prob = (tr, k) => (x, y) => { const ds = tr.map(d => [(d.x - x) ** 2 + (d.y - y) ** 2, d.c]).sort((a, b) => a[0] - b[0]); let s = 0; for (let i = 0; i < k; i++) s += ds[i][1]; return s / k; };
  return { ...sp, prob }; })();
const knnAcc = memo(k => acc(KN.te, KN.prob(KN.tr, k)));
DEMOS.kNN = {
  vt: '새 점 주변 k개의 다수결',
  note: '노랑·파랑 점 36개로 배우고, 같은 방식으로 따로 뽑은 보이지 않는 점 400개로 채점합니다. 모두 설명을 위해 만든 가상의 데이터입니다. 바닥의 높낮이는 그 자리에 새 점이 오면 노랑으로 답할 비율입니다.',
  controls: [{ k: 'k', l: '이웃 수 k', min: 1, max: 15, step: 2, v: 3 }, { k: 'qx', l: '새 점 가로', min: -1, max: 1, step: .01, v: .08 }, { k: 'qy', l: '새 점 세로', min: -1, max: 1, step: .01, v: .02 }],
  init(s) { this.change(s, 'k'); },
  change(s, k) { if (k === 'k') { s.fn = gridFn(KN.prob(KN.tr, s.k)); s.acc = knnAcc(s.k); }
    s.nb = KN.tr.map(d => ({ d, r: Math.hypot(d.x - s.qx, d.y - s.qy) })).sort((a, b) => a.r - b.r).slice(0, s.k); s.ny = s.nb.filter(q => q.d.c).length; },
  read(s) { return [['이웃 중 노랑', s.ny + ' / ' + s.k], ['시험 정확도', pct(s.acc)]]; },
  say(s) { const yes = s.ny * 2 > s.k, w = yes ? '<b>노랑</b>' : '<b>파랑</b>';
    if (s.k === 1) return '가장 가까운 점 하나만 따라 ' + w + '으로 답합니다. k = 1이면 바닥이 들쭉날쭉해지고 시험 정확도는 <b>' + pct(s.acc) + '</b>입니다. k를 올려 보십시오.';
    if (s.k >= 13) return '이웃 ' + s.k + '개의 다수결로 ' + w + '입니다. k가 너무 크면 먼 점까지 투표해 경계가 뭉개집니다. 시험 정확도는 <b>' + pct(s.acc) + '</b>입니다.';
    return '분홍 원 안의 이웃 ' + s.k + '개 중 노랑이 ' + s.ny + '개입니다. 다수결로 ' + w + '입니다. 이 모델은 미리 배우는 것이 없고, 물어볼 때마다 거리를 잽니다.'; },
  view(s) { const host = 'sia', R = s.nb[s.nb.length - 1].r;
    return [SURF(host, { fn: s.fn, h: .45, pts: pts(KN.tr), marks: [{ x: s.qx, y: s.qy, c: C.p, s: .16, r: R + .03, t: '새 점' }], segs: s.nb.map(q => [s.qx, s.qy, q.d.x, q.d.y, C.p]), segY: .5 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.ny, s.k - s.ny], colors: [C.y, C.a], max: 15, labels: ['노랑 표', '파랑 표'], title: '이웃 ' + s.k + '개의 투표' }]; },
};
EXTRA.kNN = { say: '새 점의 위치를 넣으면, 가까운 이웃들이 많이 가진 색이 나와요!', io: ['새 점의 위치 (가로, 세로)', '노랑 또는 파랑'],
  part: { items: [['ball', K.p, '새 점'], ['box', K.bx, '모든 훈련 점까지 거리 재기'], ['gate', K.gt, '가장 가까운 k개 고르기'], ['tiles', K.i, '색깔별로 표 세기'], ['ball', K.o, '많은 쪽 색']],
    t: ['새 점에서 모든 훈련 점까지 거리를 잽니다.', '가장 가까운 k개만 남깁니다.', '그 k개가 가진 색을 세어 많은 쪽으로 답합니다. 미리 만들어 두는 식이 없습니다.'], say: '부품에 "배우는" 칸이 없습니다. 훈련 점을 통째로 들고 있다가 물어볼 때마다 다시 잽니다. 그래서 점이 많으면 느립니다.' },
  curve: { f: knnAcc, lo: 1, hi: 15, step: 2, v: 3, xl: '이웃 수 k', yl: '시험 정확도', title: 'k에 따라 달라지는 시험 정확도', t: ['k가 너무 작으면 잡음 하나에 휘둘리고, 너무 크면 경계가 뭉개집니다.', '가운데 어딘가에 알맞은 k가 있습니다. 로비에서 본 복잡도와 같은 이야기입니다.'] } };

/* ================= 3층 로지스틱 회귀 (모아) ================= */
const LG = (() => { const D = blobs2(31, 24, .4, .36, .24);
  const loss = (w1, w2, b) => mean(D.map(d => { const p = clamp(sig(w1 * d.x + w2 * d.y + b), 1e-9, 1 - 1e-9); return -(d.c ? Math.log(p) : Math.log(1 - p)); }));
  let w1 = 0, w2 = 0, b = 0; for (let it = 0; it < 4000; it++) { let g1 = 0, g2 = 0, g0 = 0; D.forEach(d => { const e = sig(w1 * d.x + w2 * d.y + b) - d.c; g1 += e * d.x; g2 += e * d.y; g0 += e; }); const n = D.length;
    w1 -= .8 * (g1 / n + .002 * w1); w2 -= .8 * (g2 / n + .002 * w2); b -= .8 * g0 / n; }
  const s = Math.hypot(w1, w2); return { D, loss, best: { th: ((Math.atan2(w2, w1) * 180 / Math.PI) + 360) % 360, s, b: -b / s } }; })();
const lgP = s => { const t = s.th * Math.PI / 180; return [s.s * Math.cos(t), s.s * Math.sin(t), -s.s * s.b]; };
DEMOS.LogR = {
  vt: 'S자 언덕의 방향과 가파름 맞추기',
  note: '노랑·파랑 점 48개는 설명을 위해 만든 가상의 데이터입니다. 바닥의 언덕 높이가 "노랑일 확률"입니다. 가운데 굵은 선에서 확률이 정확히 0.5입니다.',
  controls: [{ k: 'th', l: '언덕의 방향 (도)', min: 0, max: 359, step: 1, v: 150 }, { k: 'b', l: '경계의 위치', min: -.8, max: .8, step: .01, v: .35 }, { k: 's', l: '가파름', min: .5, max: 12, step: .1, v: 2 }, { b: 'fit', l: '가장 좋은 언덕 찾기' }],
  init(s) { s.auto = 0; },
  change(s) { s.auto = 0; },
  click(s) { s.auto = 1; },
  step(s, dt) { if (!s.auto) return; const B = LG.best, t = Math.min(1, dt * 3); let dth = ((B.th - s.th + 540) % 360) - 180;
    s.th = (s.th + dth * t + 360) % 360; s.b += (B.b - s.b) * t; s.s += (B.s - s.s) * t;
    if (Math.abs(dth) < .3 && Math.abs(B.b - s.b) < .004 && Math.abs(B.s - s.s) < .03) { s.th = B.th; s.b = B.b; s.s = B.s; s.auto = 0; } },
  read(s) { const [a, b, c] = lgP(s); return [['로그 손실', f2(LG.loss(a, b, c), 3)], ['정확도', pct(acc(LG.D, (x, y) => sig(a * x + b * y + c)))]]; },
  say(s) { const [a, b, c] = lgP(s), L = LG.loss(a, b, c), Lb = LG.loss(...lgP(LG.best));
    if (L < Lb * 1.02) return '가장 좋은 언덕이에요! 경계 근처는 확률이 0.5 언저리라 <b>자신 없음</b>, 멀어질수록 0이나 1에 가까워져요.';
    if (s.s > 8) return '너무 가팔라요. 경계 근처 점도 0 아니면 1로 딱 잘라서, <b>틀린 점에 큰 벌</b>을 받아요. 로그 손실 ' + f2(L, 3) + '.';
    return '노랑 점은 언덕 위에, 파랑 점은 언덕 아래에 있어야 해요. 지금 로그 손실은 가장 좋은 언덕의 <b>' + f2(L / Lb, 1) + '배</b>예요. <b>가장 좋은 언덕 찾기</b>를 눌러 보세요.'; },
  view(s) { const host = 'moa', [a, b, c] = lgP(s), L = clipLine(a, b, c);
    return [SURF(host, { fn: (x, y) => sig(a * x + b * y + c), h: 1, pts: pts(LG.D), lines: L ? [[...L, C.fg, .05]] : [], lineY: .55 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [LG.loss(a, b, c), LG.loss(...lgP(LG.best))], colors: [C.p, C.g], max: 1.2, labels: ['지금 언덕', '가장 좋은 언덕'], title: '로그 손실 (낮을수록 좋음)' }]; },
};
EXTRA.LogR = { say: '점의 위치를 넣으면 0과 1 사이 숫자, "노랑일 확률"이 나와요!', io: ['점의 위치 (변수 여러 개)', '노랑일 확률 (0~1)'],
  part: { items: [['ball', K.i, '입력 x₁, x₂'], ['box', K.bx, '가중합 z = w·x + b'], ['gate', K.gt, 'S자 곡선 σ(z)'], ['ball', K.o, '확률 p'], ['box', K.p, '로그 손실']], loops: [[4, 1, '경사하강법으로 w, b 고침']],
    t: ['선형 회귀처럼 가중합 z를 먼저 계산합니다.', 'z를 S자 곡선에 넣어 0~1 사이 확률로 바꿉니다.', '틀린 쪽으로 자신만만할수록 크게 벌하는 로그 손실을 줄이도록 w와 b를 조금씩 고칩니다.'], say: '선형 회귀에 S자 문 하나를 단 것입니다. 이 문 하나 때문에 답을 한 번에 못 구하고, 경사하강법으로 조금씩 찾아갑니다.' },
  curve: { f: z => sig(z), lo: -6, hi: 6, v: 0, xl: '점수 z', yl: '노랑일 확률', title: 'S자 곡선 σ(z) = 1 / (1 + e⁻ᶻ)', t: ['z가 0이면 확률 0.5, 경계 위입니다.', 'z가 커질수록 1에, 작아질수록 0에 다가가지만 끝내 닿지는 않습니다.'] } };

/* ================= 4층 결정 트리 (시아) ================= */
const DTD = (() => { const D = treeData(12, 110, .1); return { tr: split(D, 4, 36).tr, te: bigTest.tree(1012, .1) }; })();
const dtFit = memo(dp => tree(DTD.tr, { depth: dp, minLeaf: 1 }));
const dtAcc = memo(dp => acc(DTD.te, (x, y) => tpred(dtFit(dp), x, y)));
DEMOS.DT = {
  vt: '질문을 몇 번까지 던질까 (나무의 깊이)',
  note: '점 74개로 배우고, 같은 방식으로 따로 뽑은 점 400개로 채점합니다. 설명을 위해 만든 가상의 데이터이고, 일부러 잡음(색이 뒤바뀐 점)을 섞었습니다. 바닥의 네모 칸 높이가 그 칸의 노랑 비율입니다.',
  controls: [{ k: 'dp', l: '나무의 깊이', min: 1, max: 9, step: 1, v: 2 }],
  init(s) { this.change(s); },
  change(s) { s.T = dtFit(s.dp); s.fn = gridFn((x, y) => tpred(s.T, x, y)); s.tra = acc(DTD.tr, (x, y) => tpred(s.T, x, y)); s.tea = dtAcc(s.dp); },
  read(s) { return [['잎(칸) 수', leaves(s.T)], ['훈련 · 시험 정확도', pct(s.tra) + ' · ' + pct(s.tea)]]; },
  say(s) { const best = [1, 2, 3, 4, 5, 6, 7, 8, 9].reduce((b, d) => dtAcc(d) > dtAcc(b) ? d : b, 1);
    if (s.dp === 1) return '질문 하나로는 모눈을 두 칸으로만 나눕니다. 왼쪽 아래 노랑 칸을 따로 떼어 내지 못합니다. 깊이를 올려 보십시오.';
    if (s.tra - s.tea > .12) return '훈련 정확도 ' + pct(s.tra) + ', 시험 정확도 ' + pct(s.tea) + '입니다. 잡음 점 하나까지 칸을 따로 만든 <b>과적합</b>입니다. 이 데이터에서는 깊이 <b>' + best + '</b> 근처가 알맞습니다.';
    return '질문 ' + (leaves(s.T) - 1) + '개로 모눈을 칸 ' + leaves(s.T) + '개로 나눴습니다. 경계가 언제나 <b>가로 아니면 세로</b>인 것이 나무의 특징입니다.'; },
  view(s) { const host = 'sia';
    return [SURF(host, { fn: s.fn, h: .5, pts: pts(DTD.tr), segs: tsegs(s.T, C.fg), segY: .56 }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [s.tra, s.tea], colors: [C.y, C.p], max: 1, labels: ['훈련 ' + pct(s.tra), '시험 ' + pct(s.tea)], title: '정확도' }]; },
};
EXTRA.DT = { say: '점의 위치를 넣으면, 예/아니오 질문을 따라 내려가서 도착한 칸의 답이 나와요!', io: ['점의 위치 (변수 여러 개)', '도착한 칸(잎)의 답'],
  part: { items: [['tiles', K.i, '데이터 전체'], ['gate', K.gt, '질문: 가로 < 기준?'], ['box', K.bx, '왼쪽 · 오른쪽으로 나누기'], ['gate', K.gt, '다음 질문'], ['ball', K.o, '잎: 다수결 답']], loops: [[3, 1, '칸마다 되풀이']],
    t: ['변수 하나와 기준값 하나로 예/아니오 질문을 만듭니다.', '질문은 나눈 뒤 칸이 가장 깨끗해지는(한 색만 남는) 것으로 고릅니다.', '나뉜 칸마다 같은 일을 되풀이하다가 깊이 한도에 닿으면 멈춥니다.'], say: '부품은 질문 하나뿐입니다. 그 질문을 칸마다 되풀이해 세우는 것이 나무입니다. 깨끗함은 지니 불순도로 잽니다.' },
  curve: { f: dtAcc, lo: 1, hi: 9, step: 1, v: 2, xl: '나무의 깊이', yl: '시험 정확도', title: '깊이에 따른 시험 정확도', t: ['얕으면 칸이 모자라고, 깊으면 잡음까지 칸으로 만듭니다.', '시험 정확도가 가장 높은 깊이가 알맞은 깊이입니다.'] } };

/* ================= 5층 k-평균 (모아) ================= */
const KM = (() => { const r = rng(41), D = [];
  [[-.55, -.4, .2], [.1, .55, .17], [.6, -.35, .19]].forEach(([cx, cy, s]) => { let g = 0; while (D.filter(d => d.g === cx).length < 16 && g++ < 999) { const x = cx + gauss(r) * s, y = cy + gauss(r) * s * .9; if (Math.abs(x) < 1.08 && Math.abs(y) < 1.08) D.push({ x, y, g: cx }); } });
  const assign = (cs) => D.map(d => { let b = 0, bd = 1e9; cs.forEach((c, i) => { const v = d2([d.x, d.y], c); if (v < bd) { bd = v; b = i; } }); return b; });
  const inertia = (cs, a) => D.reduce((t, d, i) => t + d2([d.x, d.y], cs[a[i]]), 0);
  const upd = (cs, a) => cs.map((c, i) => { const m = D.filter((_, j) => a[j] === i); return m.length ? [mean(m.map(d => d.x)), mean(m.map(d => d.y))] : c; });
  const start = (k, seed) => { const q = rng(seed * 97 + k), idx = []; while (idx.length < k) { const i = Math.floor(q() * D.length); if (!idx.includes(i)) idx.push(i); } return idx.map(i => [D[i].x, D[i].y]); };
  const solve = (k, seed) => { let cs = start(k, seed), a = assign(cs); for (let i = 0; i < 50; i++) { cs = upd(cs, a); const b = assign(cs); if (b.every((v, j) => v === a[j])) break; a = b; } return inertia(cs, a); };
  return { D, assign, inertia, upd, start, solve }; })();
const kmBest = memo(k => Math.min(...[1, 2, 3, 4, 5, 6].map(sd => KM.solve(k, sd))));
DEMOS['k-means'] = {
  vt: '중심 옮기기를 되풀이하면 묶음이 자리를 잡는다',
  note: '점 48개는 설명을 위해 만든 가상의 데이터이고, 정답(색)은 없습니다. 점의 색은 지금 가장 가까운 중심의 색입니다.',
  controls: [{ k: 'k', l: '묶음 수 k', min: 2, max: 6, step: 1, v: 3 }, { b: 'one', l: '한 바퀴' }, { b: 'run', l: '끝까지' }, { b: 'new', l: '중심 다시 놓기' }],
  init(s) { s.seed = 1; this.change(s); },
  change(s) { s.cs = KM.start(s.k, s.seed); s.a = KM.assign(s.cs); s.it = 0; s.done = 0; s.auto = 0; s.hist = [KM.inertia(s.cs, s.a)]; },
  one(s) { const cs = KM.upd(s.cs, s.a), a = KM.assign(cs); s.done = a.every((v, j) => v === s.a[j]) && cs.every((c, i) => d2(c, s.cs[i]) < 1e-12); s.cs = cs; s.a = a; if (!s.done) { s.it++; s.hist.push(KM.inertia(cs, a)); } },
  click(s, b) { if (b === 'new') { s.seed++; this.change(s); } else if (b === 'one') this.one(s); else { s.auto = 1; s.t = 0; } },
  step(s, dt) { if (!s.auto) return; s.t += dt; if (s.t < .45) return; s.t = 0; this.one(s); if (s.done || s.it > 30) s.auto = 0; },
  read(s) { return [['바퀴 수', s.it + (s.done ? ' (멈춤)' : '')], ['거리 제곱 합', f2(KM.inertia(s.cs, s.a), 2)]]; },
  say(s) { if (s.it === 0) return '점 ' + s.k + '개를 골라 그 자리에 중심을 놓았어요. <b>한 바퀴</b>를 누르면 ① 점마다 가까운 중심을 고르고 ② 중심을 자기 점들의 한가운데로 옮겨요.';
    if (s.done) { const b = kmBest(s.k), v = KM.inertia(s.cs, s.a); return '더 이상 바뀌지 않아요. ' + s.it + '바퀴 만에 멈췄어요.' + (v > b * 1.05 ? ' 그런데 거리 제곱 합이 가장 좋은 답(' + f2(b) + ')보다 커요. 시작 자리에 따라 <b>엉뚱한 곳에 멈출 수도</b> 있어요. 중심을 다시 놓아 보세요.' : ' 이 k에서 가장 좋은 묶음이에요.'); }
    return s.it + '바퀴째예요. 바퀴를 돌 때마다 거리 제곱 합이 <b>줄어들기만</b> 해요. 그래서 언젠가는 반드시 멈춰요.'; },
  view(s) { const host = 'moa';
    return [SURF(host, { pts: KM.D.map((d, i) => ({ x: d.x, y: d.y, col: PAL[s.a[i]] })), marks: s.cs.map((c, i) => ({ x: c[0], y: c[1], c: PAL[i], s: .2, t: 'X' })), segs: KM.D.map((d, i) => [d.x, d.y, s.cs[s.a[i]][0], s.cs[s.a[i]][1], C.mu]) }),
      { id: 'pl', type: 'plot', ...SIDE(host, { w: 3.4, h: 3 }), ymin: 0, ymax: Math.max(...s.hist) * 1.1, series: [{ v: s.hist, c: C.b, len: Math.max(8, s.hist.length), n: '거리 제곱 합' }], title: '바퀴마다 줄어드는 거리 제곱 합' }]; },
};
EXTRA['k-means'] = { say: '정답 없는 점들을 넣으면, 묶음 번호와 묶음마다 중심 하나가 나와요!', io: ['정답 없는 점들', '점마다 묶음 번호 + 중심 k개'],
  part: { items: [['ball', K.o, '중심 k개 놓기'], ['box', K.bx, '점마다 가까운 중심 고르기'], ['box', K.g, '중심을 한가운데로 옮기기'], ['gate', K.gt, '바뀐 게 없으면 멈춤']], loops: [[2, 1, '되풀이']],
    t: ['중심 k개를 먼저 놓습니다. 어디에 놓느냐에 따라 결과가 달라질 수 있습니다.', '점마다 가장 가까운 중심에 붙고, 중심은 붙은 점들의 평균 자리로 옮깁니다.', '아무것도 바뀌지 않을 때까지 두 걸음을 되풀이합니다.'], say: '부품은 "가까운 중심 고르기"와 "평균 내기" 두 개입니다. k는 사람이 정해 줘야 합니다.' },
  curve: { f: kmBest, lo: 1, hi: 6, step: 1, v: 3, xl: '묶음 수 k', yl: '거리 제곱 합', title: '팔꿈치 그래프', t: ['k를 늘리면 거리 제곱 합은 언제나 줄어듭니다.', '줄어드는 폭이 갑자기 작아지는 곳(팔꿈치)이 알맞은 k의 후보입니다. 이 데이터는 3입니다.'] } };

/* ================= 6층 나이브 베이즈 (시아) ================= */
const NBD = (() => { const D = blobs2(44, 24, .32, .36, .28), st = (c, f) => { const v = D.filter(d => d.c === c).map(d => d[f]), m = mean(v); return [m, Math.sqrt(mean(v.map(x => (x - m) ** 2))) + .02]; };
  const P = { 1: { x: st(1, 'x'), y: st(1, 'y') }, 0: { x: st(0, 'x'), y: st(0, 'y') } }, nd = (v, [m, s]) => Math.exp(-.5 * ((v - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const lik = (c, x, y) => nd(x, P[c].x) * nd(y, P[c].y);
  const post = (pr, x, y) => { const a = pr * lik(1, x, y), b = (1 - pr) * lik(0, x, y); return a / (a + b + 1e-300); };
  return { D, lik, post }; })();
DEMOS.NB = {
  vt: '사전 확률을 바꾸면 경계가 옮겨 간다',
  note: '점 48개는 설명을 위해 만든 가상의 데이터입니다. 무리마다, 변수마다 평균과 퍼짐만 구해 종 모양을 그렸습니다. 바닥의 높낮이가 "노랑일 확률"입니다.',
  controls: [{ k: 'pr', l: '노랑의 사전 확률', min: .05, max: .95, step: .01, v: .5 }, { k: 'qx', l: '새 점 가로', min: -1, max: 1, step: .01, v: 0 }, { k: 'qy', l: '새 점 세로', min: -1, max: 1, step: .01, v: 0 }],
  init(s) { this.change(s, 'pr'); },
  change(s, k) { if (k === 'pr') s.fn = gridFn((x, y) => NBD.post(s.pr, x, y)); s.p = NBD.post(s.pr, s.qx, s.qy); },
  read(s) { return [['P(노랑 | 새 점)', f2(s.p, 3)], ['판정', s.p > .5 ? '노랑' : '파랑']]; },
  say(s) { const a = s.pr * NBD.lik(1, s.qx, s.qy), b = (1 - s.pr) * NBD.lik(0, s.qx, s.qy);
    return '노랑 쪽 점수는 사전 확률 × 가로 종 높이 × 세로 종 높이입니다. 두 변수를 <b>서로 상관없다고 보고 그냥 곱합니다.</b> 노랑 ' + f2(a / (a + b) * 100, 0) + ' : 파랑 ' + f2(b / (a + b) * 100, 0) + '으로 <b>' + (s.p > .5 ? '노랑' : '파랑') + '</b>입니다.' + (Math.abs(s.pr - .5) > .3 ? ' 사전 확률을 치우치게 하면 경계가 소수 쪽으로 밀려납니다.' : ''); },
  view(s) { const host = 'sia', a = s.pr * NBD.lik(1, s.qx, s.qy), b = (1 - s.pr) * NBD.lik(0, s.qx, s.qy);
    return [SURF(host, { fn: s.fn, h: .8, pts: pts(NBD.D), marks: [{ x: s.qx, y: s.qy, c: C.p, s: .16, t: '새 점' }] }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [a / (a + b), b / (a + b)], colors: [C.y, C.a], max: 1, labels: ['노랑 ' + f2(a / (a + b), 2), '파랑 ' + f2(b / (a + b), 2)], title: '새 점은 어느 쪽일까' }]; },
};
EXTRA.NB = { say: '점의 위치를 넣으면, 무리마다 "이 무리일 확률"이 나와요!', io: ['점의 위치 (변수 여러 개)', '무리마다 확률'],
  part: { items: [['ball', K.i, '입력 x₁, x₂'], ['box', K.bx, '변수마다 종 높이 P(xᵢ | 무리)'], ['box', K.g, '모두 곱하고 사전 확률도 곱하기'], ['gate', K.gt, '점수 큰 무리 고르기'], ['ball', K.o, '답']],
    t: ['무리마다, 변수마다 따로 평균과 퍼짐을 구해 둡니다. 배우는 일은 이게 전부입니다.', '새 점이 오면 변수마다 종 높이를 읽어 모두 곱하고, 그 무리가 원래 얼마나 흔한지(사전 확률)도 곱합니다.', '점수가 큰 무리로 답합니다. 변수끼리 관계가 있어도 없다고 치는 것이 "나이브(순진한)"의 뜻입니다.'], say: '배우는 데 평균과 퍼짐만 셉니다. 그래서 아주 빠르고, 스팸 메일 거르기처럼 변수가 수천 개여도 잘 돕니다.' },
  curve: { f: pr => NBD.post(pr, 0, 0), lo: .05, hi: .95, v: .5, xl: '노랑의 사전 확률', yl: '가운데 점 (0, 0)이 노랑일 확률', title: '사전 확률과 답', t: ['같은 점이라도 노랑이 원래 흔하다고 믿으면 노랑일 확률이 올라갑니다.', '데이터가 애매할수록 사전 확률이 답을 크게 좌우합니다.'] } };

/* ================= 7층 릿지 회귀 (모아) ================= */
const RG = (() => { const tr = wave(6, 13, .17), te = wave(106, 40, .17), D = 9, row = x => Array.from({ length: D + 1 }, (_, k) => x ** k);
  const fit = lg => lsq(tr.map(d => row(d.x)), tr.map(d => d.y), 10 ** lg, true), f = w => x => w.reduce((s, c, k) => s + c * x ** k, 0), mse = (ps, g) => mean(ps.map(d => (d.y - g(d.x)) ** 2));
  return { tr, te, fit, f, mse }; })();
const rgTest = memo(lg => { const w = RG.fit(lg); return RG.mse(RG.te, RG.f(w)); });
DEMOS.Ridge = {
  vt: '벌점 λ로 9차 곡선 길들이기',
  note: '노랑 점 13개로 배우고 분홍 점 40개로 채점합니다. 모두 물결 곡선에 잡음을 섞어 만든 가상의 데이터입니다. 모델은 9차 다항식이고, 절편을 뺀 계수 9개에 벌점을 겁니다.',
  controls: [{ k: 'lg', l: '벌점 λ (10의 지수)', min: -7, max: 1, step: .1, v: -7, f: v => '10^' + v.toFixed(1) }],
  init(s) { this.change(s); },
  change(s) { s.w = RG.fit(s.lg); s.g = RG.f(s.w); s.tr = RG.mse(RG.tr, s.g); s.te = rgTest(s.lg); s.sz = s.w.slice(1).reduce((t, c) => t + c * c, 0); },
  read(s) { return [['계수 제곱 합', s.sz > 999 ? Math.round(s.sz).toLocaleString() : f2(s.sz, 2)], ['훈련 · 시험 오차', f2(s.tr, 3) + ' · ' + f2(s.te, 3)]]; },
  say(s) { if (s.lg < -5) return '벌점이 거의 없어요. 9차 곡선이 노랑 점을 다 지나가려고 <b>계수가 ' + Math.round(Math.sqrt(s.sz)) + ' 크기까지 커졌어요.</b> 점 사이에서 곡선이 크게 출렁여요.';
    if (s.lg > -.5) return '벌점이 너무 세요. 계수를 다 눌러서 곡선이 <b>거의 평평</b>해졌어요. 이번엔 너무 단순해요(과소적합).';
    return '벌점이 계수를 <b>조금씩 고르게</b> 줄였어요. 곡선이 얌전해지고 시험 오차가 ' + f2(s.te, 3) + '예요. 계수가 0이 되지는 않아요.'; },
  view(s) { const host = 'moa';
    return [SURF(host, { pts: [...RG.tr.map(d => ({ x: d.x, y: d.y, col: C.y })), ...RG.te.map(d => ({ x: d.x, y: d.y, col: C.p }))], lines: curve(s.g, C.b, .05) }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3.6, h: 3.2 }), v: s.w.slice(1).map(c => Math.min(3, Math.abs(c))), colors: s.w.slice(1).map(c => Math.abs(c) >= 3 ? C.r : C.b), max: 3, labels: s.w.slice(1).map((_, i) => String(i + 1)), title: '계수 w1~w9의 크기 (3 넘으면 빨강)' }]; },
};
EXTRA.Ridge = { say: '선형 회귀와 똑같이 숫자를 넣으면 숫자가 나와요. 다른 건 배우는 방법이에요!', io: ['숫자 (변수 여러 개)', '예측한 숫자'],
  part: { items: [['ball', K.i, '입력 x'], ['box', K.bx, '가중합 ŷ = Σ wᵢxᵢ + b'], ['box', K.p, '오차 제곱 합'], ['box', K.gt, '+ λ × 계수 제곱 합'], ['gate', K.g, '합이 가장 작은 w 고르기']],
    t: ['예측은 선형 회귀와 똑같습니다.', '고를 때 오차에 "계수 제곱 합 × λ"를 벌점으로 더합니다.', '계수를 키워 봤자 벌점이 커지므로, 모든 계수가 조금씩 0 쪽으로 줄어듭니다. 답은 여전히 계산 한 번으로 나옵니다.'], say: '부품은 1층과 같고, 고르는 기준에 벌점 칸 하나만 더했습니다. λ는 사람이 정합니다.' },
  curve: { f: rgTest, lo: -7, hi: 1, step: .5, v: -7, xl: '벌점 λ (10의 지수)', yl: '시험 오차', title: '벌점 크기와 시험 오차', t: ['벌점이 없으면 과적합, 너무 세면 과소적합입니다.', '골짜기 바닥의 λ가 알맞은 벌점입니다.'] } };

/* ================= 8층 라쏘 회귀 (시아) ================= */
const LS = (() => { const r = rng(88), P = 8, TRUE = [1.1, -.8, .55, 0, 0, 0, 0, 0];
  const mk = n => Array.from({ length: n }, () => { const z = gauss(r), x = Array.from({ length: P }, () => .55 * gauss(r) + .45 * z); return { x, y: x.reduce((s, v, j) => s + v * TRUE[j], 0) + gauss(r) * .55 }; });
  const tr = mk(30), te = mk(200), mu = Array.from({ length: P }, (_, j) => mean(tr.map(d => d.x[j]))), sd = mu.map((m, j) => Math.sqrt(mean(tr.map(d => (d.x[j] - m) ** 2)))), ym = mean(tr.map(d => d.y));
  const Z = d => d.x.map((v, j) => (v - mu[j]) / sd[j]);
  const fit = lam => { const w = new Array(P).fill(0), X = tr.map(Z), y = tr.map(d => d.y - ym), n = tr.length;
    for (let it = 0; it < 400; it++) for (let j = 0; j < P; j++) { let rho = 0; X.forEach((x, i) => { let pr = 0; for (let k = 0; k < P; k++) if (k !== j) pr += x[k] * w[k]; rho += x[j] * (y[i] - pr); }); rho /= n; w[j] = Math.sign(rho) * Math.max(0, Math.abs(rho) - lam); }
    return w; };
  const mse = (D, w) => mean(D.map(d => (d.y - ym - Z(d).reduce((s, v, j) => s + v * w[j], 0)) ** 2));
  return { tr, te, fit, mse, P, TRUE }; })();
const lsFit = memo(l => LS.fit(l)), lsNZ = memo(l => lsFit(l).filter(v => Math.abs(v) > 1e-9).length);
DEMOS.Lasso = {
  vt: '벌점을 올리면 계수가 하나씩 0이 된다',
  note: '변수 8개, 훈련용 30줄, 시험용 200줄의 가상 데이터입니다. 정답을 만들 때 실제로 쓴 변수는 w1~w3 세 개뿐이고, 나머지 다섯은 서로 비슷하게 움직이는 쓸모없는 변수입니다.',
  controls: [{ k: 'l', l: '벌점 λ', min: 0, max: 1, step: .01, v: 0 }],
  init(s) { this.change(s); },
  change(s) { s.w = lsFit(s.l); s.ols = lsFit(0); s.nz = lsNZ(s.l); s.tr = LS.mse(LS.tr, s.w); s.te = LS.mse(LS.te, s.w); },
  read(s) { return [['0이 아닌 계수', s.nz + ' / 8'], ['훈련 · 시험 오차', f2(s.tr, 3) + ' · ' + f2(s.te, 3)]]; },
  say(s) { const z = s.w.map((v, i) => Math.abs(v) < 1e-9 ? 'w' + (i + 1) : null).filter(Boolean);
    if (s.l === 0) return '벌점이 없으면 쓸모없는 w4~w8도 <b>조금씩 값을 가집니다.</b> 잡음에 맞춘 값입니다. λ를 올려 보십시오.';
    if (s.nz === 0) return '벌점이 너무 세서 <b>모든 계수가 0</b>입니다. 이제 평균만 말하는 모델입니다.';
    return z.length ? '<b>' + z.join(', ') + '</b>가 정확히 0이 됐습니다. 라쏘는 계수를 줄이다가 <b>아예 지워 버립니다.</b> 남은 변수가 이 모델이 고른 변수입니다.' : 'λ를 조금 올렸습니다. 아직 0이 된 계수는 없지만 모두 줄어들고 있습니다.'; },
  view(s) { const host = 'sia';
    return [{ id: 'w', type: 'bars', x: -.6, z: -.6, w: 5.4, h: 3.4, v: s.w.map(Math.abs), refs: s.ols.map(Math.abs), colors: s.w.map(v => Math.abs(v) < 1e-9 ? C.mu : v > 0 ? C.b : C.a), max: 1.4, labels: s.w.map((v, i) => 'w' + (i + 1) + (Math.abs(v) < 1e-9 ? ' = 0' : '')), title: '계수 8개의 크기 (흰 눈금 = 벌점 없을 때)' },
      { id: 'b', type: 'bars', ...SIDE(host, { w: 2.6, h: 3 }), v: [s.tr, s.te], colors: [C.y, C.p], max: 1.5, labels: ['훈련 오차', '시험 오차'], title: '오차' }]; },
};
EXTRA.Lasso = { say: '숫자를 넣으면 숫자가 나와요. 그런데 쓸모없는 변수는 아예 안 써요!', io: ['숫자 (변수 여러 개)', '예측한 숫자 + 살아남은 변수 목록'],
  part: { items: [['ball', K.i, '입력 x₁ … x₈'], ['box', K.bx, '가중합 ŷ = Σ wᵢxᵢ + b'], ['box', K.p, '오차 제곱 합'], ['box', K.gt, '+ λ × 계수 절댓값 합'], ['gate', K.g, '계수 하나씩 고치기']], loops: [[4, 4, '좌표 하강법']],
    t: ['예측은 선형 회귀와 같습니다.', '벌점을 제곱 대신 절댓값으로 겁니다. 이 차이 때문에 작은 계수는 0에서 딱 멈춥니다.', '계수를 하나씩 번갈아 고치는 좌표 하강법으로 답을 찾습니다.'], say: '릿지와 벌점 칸 하나만 다릅니다. 제곱이냐 절댓값이냐가 "줄이기"와 "지우기"를 가릅니다.' },
  curve: { f: lsNZ, lo: 0, hi: 1, step: .05, v: 0, xl: '벌점 λ', yl: '0이 아닌 계수 수', title: '벌점에 따라 살아남는 변수', t: ['λ를 올릴수록 살아남는 변수가 계단처럼 줄어듭니다.', '실제로 쓴 세 변수(w1~w3)가 가장 오래 버팁니다.'] } };

/* ================= 9층 주성분 분석 (모아) ================= */
const PC = (() => { const r = rng(71), D = [], th = 28 * Math.PI / 180; while (D.length < 34) { const a = gauss(r) * .48, b = gauss(r) * .13, x = a * Math.cos(th) - b * Math.sin(th), y = a * Math.sin(th) + b * Math.cos(th); if (Math.abs(x) < 1.05 && Math.abs(y) < 1.05) D.push({ x, y }); }
  const mx = mean(D.map(d => d.x)), my = mean(D.map(d => d.y)), sxx = mean(D.map(d => (d.x - mx) ** 2)), syy = mean(D.map(d => (d.y - my) ** 2)), sxy = mean(D.map(d => (d.x - mx) * (d.y - my)));
  const v = deg => { const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t); return c * c * sxx + 2 * c * s * sxy + s * s * syy; }, tot = sxx + syy;
  const best = ((Math.atan2(2 * sxy, sxx - syy) / 2) * 180 / Math.PI + 180) % 180;
  return { D, mx, my, v, tot, best }; })();
DEMOS.PCA = {
  vt: '가장 넓게 퍼진 방향 찾기',
  note: '점 34개는 설명을 위해 만든 가상의 데이터이고, 정답(색)은 없습니다. 분홍 선은 점에서 축까지 수직으로 내린 거리입니다. 점들이 축 위로 내려앉은 자리가 노랑 표시입니다.',
  controls: [{ k: 'th', l: '축의 방향 (도)', min: 0, max: 179, step: 1, v: 110 }, { b: 'best', l: '가장 넓은 방향 찾기' }],
  init(s) { s.auto = 0; },
  change(s) { s.auto = 0; },
  click(s) { s.auto = 1; },
  step(s, dt) { if (!s.auto) return; const d = ((PC.best - s.th + 270) % 180) - 90; s.th = (s.th + d * Math.min(1, dt * 3) + 180) % 180; if (Math.abs(d) < .2) { s.th = PC.best; s.auto = 0; } },
  read(s) { return [['이 축에 담긴 퍼짐', pct(PC.v(s.th) / PC.tot)], ['가장 많이 담는 방향', Math.round(PC.best) + '°']]; },
  say(s) { const r = PC.v(s.th) / PC.tot;
    if (r > .97 * PC.v(PC.best) / PC.tot) return '이 방향이 첫째 주성분(PC1)이에요! 점 두 개짜리 정보를 축 하나로 줄였는데 퍼짐의 <b>' + pct(r) + '</b>가 남았어요. 분홍 선(버린 정보)도 가장 짧아요.';
    return '지금 축에는 퍼짐의 ' + pct(r) + '만 담겨요. 분홍 선이 길수록 버리는 정보가 많아요. 축을 돌려서 <b>점들이 가장 넓게 펼쳐지는</b> 방향을 찾아보세요.'; },
  view(s) { const host = 'moa', t = s.th * Math.PI / 180, u = [Math.cos(t), Math.sin(t)], L = clipLine(-u[1], u[0], u[1] * PC.mx - u[0] * PC.my), pr = PC.D.map(d => { const a = (d.x - PC.mx) * u[0] + (d.y - PC.my) * u[1]; return [PC.mx + a * u[0], PC.my + a * u[1]]; });
    return [SURF(host, { pts: PC.D.map(d => ({ x: d.x, y: d.y, col: C.mu })), lines: L ? [[...L, C.b, .055]] : [], segs: PC.D.map((d, i) => [d.x, d.y, pr[i][0], pr[i][1], C.p]), marks: pr.map(p => ({ x: p[0], y: p[1], c: C.y, s: .06 })) }),
      { id: 'b', type: 'bars', ...SIDE(host, { w: 3, h: 3.2 }), v: [PC.v(s.th), PC.tot - PC.v(s.th)], colors: [C.b, C.p], max: PC.tot, labels: ['축에 담김', '버림'], title: '퍼짐(분산) 나누기' }]; },
};
EXTRA.PCA = { say: '변수가 많은 점을 넣으면, 변수가 적은 점이 나와요. 중요한 방향부터 남겨요!', io: ['변수 여러 개 (예: 2개)', '변수 더 적게 (예: 1개)'],
  part: { items: [['tiles', K.i, '데이터 표'], ['box', K.bx, '평균 빼기 (가운데로)'], ['tiles', K.gt, '공분산 행렬'], ['box', K.g, '가장 넓게 퍼진 방향 찾기'], ['ball', K.o, '그 축 위의 좌표']],
    t: ['변수마다 평균을 빼서 점 구름을 가운데로 옮깁니다.', '변수끼리 함께 움직이는 정도(공분산)를 표로 만들고, 그 표의 고유벡터가 퍼짐이 큰 방향입니다.', '퍼짐이 큰 방향 몇 개만 남기고 나머지는 버립니다.'], say: '정답을 쓰지 않습니다. 퍼짐만 보고 축을 고르므로, 고른 축이 분류에 좋은지는 따로 확인해야 합니다.' },
  curve: { f: th => PC.v(th) / PC.tot, lo: 0, hi: 179, v: 110, xl: '축의 방향 (도)', yl: '담긴 퍼짐 비율', title: '방향을 돌리며 잰 퍼짐', t: ['가장 높은 봉우리가 PC1, 거기서 90° 돌린 골짜기가 PC2입니다.', '봉우리와 골짜기의 높이 차가 클수록 축 하나로 줄여도 잃는 것이 적습니다.'] } };
