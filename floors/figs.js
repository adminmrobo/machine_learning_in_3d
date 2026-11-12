// 머신러닝 연구소 · 층마다 위에 띄우는 "모델 그림" 명세
// 머신러닝 모델 도감(machine_learning_model_atlas)의 모눈 그림을 같은 좌표(가로 0~220, 세로 0~130, 아래가 +)로 옮겼다.
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
const dots = (ps, c, r) => ps.map(p => ['pt', p[0], p[1], c, r || 3.6]);

// 회귀용 점 (도감의 선형 회귀 그림과 같은 점)
const r2 = rng(5);
const RP = Array.from({ length: 12 }, (_, i) => { const x = 18 + i * 16.6 + (r2() - .5) * 6; return [x, 104 - .36 * (x - 18) + gauss(r2) * 8]; });
const fitLine = ps => { const mx = mean(ps.map(p => p[0])), my = mean(ps.map(p => p[1])), b = ps.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / ps.reduce((s, p) => s + (p[0] - mx) ** 2, 0); return [my - b * mx, b]; };

export const FIG = {
  /* 로비 ②의 왼쪽: 사람이 기준을 정한 프로그램 */
  rule() { return [['ln', 6, 66, 214, 66, 'U', 1.8], ...dots(RP, 'U'), ['txt', 110, 16, '사람이 정한 기준선']]; },
  /* 로비: 선을 긋기 전, 데이터를 훈련용과 시험용으로 나눈 모습 */
  split() { const test = [2, 5, 9, 11]; return [...RP.map((p, i) => ['pt', p[0], p[1], test.includes(i) ? 'N' : 'A', test.includes(i) ? 4.4 : 3.6]), ['txt', 62, 16, '노랑: 훈련용 (배울 때 봄)'], ['txt', 160, 120, '분홍: 시험용 (채점할 때만 봄)']]; },
  /* 1층 선형 회귀: 점과 직선 사이 세로 거리(오차) */
  lr() { const [a, b] = fitLine(RP); return [...RP.map(p => ['ln', p[0], p[1], p[0], a + b * p[0], 'E', 1.3]), ['ln', 6, a + b * 6, 214, a + b * 214, 'M', 2.2], ...dots(RP, 'A'), ['txt', 62, 14, '주황: 모델이 그은 직선'], ['txt', 168, 118, '옅은 세로선: 오차']]; },
};
export const fig = spec => FIG[spec[0]](...spec.slice(1));
