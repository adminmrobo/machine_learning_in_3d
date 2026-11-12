// 머신러닝 연구소 · 층 데이터 (건물 첫 화면과 층 페이지가 함께 씀)
// 27종과 묶음은 머신러닝 모델 도감(machine_learning_model_atlas)을 따르고, 층 순서는 쉬운 것부터 놓았다.
export const GROUPS = [
  { n: '선형 모델', c: '#7fe0a8', d: '데이터에 직선(변수가 많으면 평면) 하나를 긋습니다. 모든 비교의 기준선입니다.' },
  { n: '이웃 · 마진 · 확률', c: '#6fb1ff', d: '가까운 점에게 묻거나, 경계와의 거리를 넓히거나, 확률을 계산합니다.' },
  { n: '나무와 숲', c: '#ffd76a', d: '예/아니오 질문을 이어 붙여 모눈을 네모 칸으로 나눕니다. 여러 그루를 모으면 숲이 됩니다.' },
  { n: '부스팅', c: '#ff9f6b', d: '약한 나무를 차례로 세웁니다. 앞 나무가 틀린 곳을 뒤 나무가 고쳐 나갑니다.' },
  { n: '군집', c: '#ff8fc7', d: '정답 없이 비슷한 점끼리 묶습니다.' },
  { n: '차원 축소', c: '#b79cff', d: '변수 수를 줄여 압축하거나 눈에 보이게 합니다.' },
  { n: '이상 탐지 · 신경망으로 가는 다리', c: '#8fe3d0', d: '별난 점을 찾는 숲 하나와, 신경망 연구소로 넘어가는 마지막 층입니다.' },
];
export const BANDS = [['저층부 · 쉬움', 1, 9], ['중층부 · 보통', 10, 18], ['고층부 · 어려움', 19, 27]];
export const band = n => n === 0 ? '모든 층의 공통 기초' : BANDS.find(b => n >= b[1] && n <= b[2])[0];

// 모델 그림에 쓰는 색과 뜻 (층마다 leg 로 골라 범례에 보임)
export const CELLS = {
  A: ['#ffd76a', '데이터 점'], B: ['#6fb1ff', '다른 무리의 점'], C: ['#b79cff', '셋째 무리의 점'], U: ['#9aa7c7', '정답 없는 점 · 보조선'],
  N: ['#ff8fc7', '시험용 점 · 강조'], G: ['#7fe0a8', '여러 그루가 그은 선'], M: ['#ff9f6b', '모델이 그은 선'], E: ['#93a4d8', '오차 (점과 선 사이 거리)'],
};

const ATLAS = 'https://adminmrobo.github.io/machine_learning_model_atlas/';
const FISH = 'https://adminmrobo.github.io/fish_market_learning_epoch_etc/';
const NNLAB = 'https://adminmrobo.github.io/ai_in_3d/';
export const LINKS = { ATLAS, FISH, NNLAB };

// 지금 열려 있는 가장 높은 층. 층을 새로 채우면 이 숫자만 올린다. (0 = 로비만)
export const OPEN = 1;

// 로비: 모든 층의 공통 기초. 층 페이지와 같은 틀(?f=0)로 보여 준다.
export const LOBBY = { ab: '로비', en: 'Machine Learning Basics', ko: '머신러닝의 기초', g: 0, host: 'doctor', fig: ['split'], leg: ['A', 'N'],
  line: '데이터에서 규칙을 찾고, 처음 보는 데이터로 채점합니다. 어느 층에서나 쓰는 틀입니다.',
  three: ['머신러닝은 사람이 규칙을 쓰는 대신 <b>데이터에서 규칙을 찾습니다.</b>', '데이터를 <b>훈련용</b>과 <b>시험용</b>으로 나누고, 채점은 시험용으로만 합니다.', '훈련 점만 지나치게 잘 맞히는 것이 <b>과적합</b>입니다. 모든 층이 이것과 씨름합니다.'],
  talk: [['doctor', 'E13', '위층으로 올라가기 전에 채점하는 법부터 정해 둡니다. 배울 때 본 문제로 채점하면 안 됩니다.'], ['moa', 'E01', '시험 문제는 숨겨 뒀다가 마지막에만 꺼내는 거네요!']],
  diff: '왼쪽은 사람이 기준선을 직접 정한 프로그램입니다. 오른쪽은 머신러닝이 시작하는 자리입니다. 선을 긋기 전에 점을 훈련용(노랑)과 시험용(분홍)으로 먼저 나눕니다.',
  prev: { name: '사람이 규칙을 쓰는 프로그램', fig: ['rule'] } };

// host: 그 층을 안내하는 식구. talk: [식구, 표정, 대사] 두 줄. fig: 위에 띄우는 모델 그림(figs.js). leg: 범례에 보일 색.
// cmp: ②에서 견줄 아래층 번호(0 = 로비). 아직 열지 않은 층은 ab·en·ko·g·task·line·fig 만 적어 둔다.
export const FLOORS = [
  { ab: 'LR', cmp: 0, en: 'Linear Regression', ko: '선형 회귀', g: 0, yr: 1805, task: '회귀', host: 'moa', fig: ['lr'], leg: ['A', 'M', 'E'],
    line: '점들과의 세로 거리(오차)를 제곱해 더한 값이 가장 작은 직선을 긋습니다.',
    three: ['데이터에 <b>직선 하나</b>를 긋습니다.', '점과 직선 사이 <b>세로 거리(오차)</b>를 제곱해 모두 더합니다.', '그 합이 <b>가장 작은</b> 직선이 답입니다.'],
    talk: [['moa', 'E01', '점들 사이로 막대 하나를 걸쳐 놓는 거예요. 가장 덜 휘청이는 자리에요!'], ['doctor', 'E13', '그 "덜 휘청이는 자리"를 오차 제곱의 합으로 정합니다. 이 건물의 모든 층이 이 직선과 비교됩니다.']],
    diff: '로비에서는 점을 나누기만 했습니다. 이 층에서 처음으로 선을 긋습니다. 긋는 것은 직선 하나, 기준은 오차 제곱의 합입니다.' },
  { ab: 'kNN', en: 'k-Nearest Neighbors', ko: 'k-최근접 이웃', g: 1, yr: 1951, task: '분류 · 회귀', fig: ['knn'], line: '미리 배우지 않고, 새 점에 가장 가까운 k개 이웃의 다수결로 답합니다.' },
  { ab: 'LogR', en: 'Logistic Regression', ko: '로지스틱 회귀', g: 0, yr: 1958, task: '분류', fig: ['logr'], line: '직선으로 계산한 점수를 S자 곡선에 넣어 0~1 사이 확률로 바꿉니다.' },
  { ab: 'DT', en: 'Decision Tree', ko: '결정 트리', g: 2, yr: 1984, task: '분류 · 회귀', fig: ['dt'], line: '예/아니오 질문을 차례로 던져 데이터를 네모 칸으로 쪼갭니다.' },
  { ab: 'k-means', en: 'k-Means Clustering', ko: 'k-평균', g: 4, yr: 1957, task: '군집', fig: ['kmeans'], line: '중심점 k개를 놓고, 가까운 점을 모았다가 중심을 옮기기를 되풀이합니다.' },
  { ab: 'NB', en: 'Naive Bayes', ko: '나이브 베이즈', g: 1, yr: '1960년대', task: '분류', fig: ['nb'], line: '변수들이 서로 무관하다고 가정하고, 따로 구한 확률을 곱해 고릅니다.' },
  { ab: 'Ridge', en: 'Ridge Regression', ko: '릿지 회귀', g: 0, yr: 1970, task: '회귀', fig: ['coef', 'ridge'], line: '선형 회귀에 "계수를 너무 키우지 말라"는 벌점을 더합니다.' },
  { ab: 'Lasso', en: 'Lasso Regression', ko: '라쏘 회귀', g: 0, yr: 1996, task: '회귀', fig: ['coef', 'lasso'], line: '벌점을 절댓값으로 바꿔, 덜 중요한 계수를 정확히 0으로 만듭니다.' },
  { ab: 'PCA', en: 'Principal Component Analysis', ko: '주성분 분석', g: 5, yr: 1901, task: '차원 축소', fig: ['pca'], line: '데이터가 가장 넓게 퍼진 방향을 찾아 그 축만 남깁니다.' },
  { ab: 'HAC', en: 'Hierarchical Agglomerative Clustering', ko: '계층적 군집', g: 4, yr: 1963, task: '군집', fig: ['hac'], line: '가장 가까운 두 묶음을 계속 합쳐 나무 그림을 만듭니다.' },
  { ab: 'LDA', en: 'Linear Discriminant Analysis', ko: '선형 판별 분석', g: 0, yr: 1936, task: '분류 · 차원 축소', fig: ['lda'], line: '두 집단의 중심이 가장 잘 벌어져 보이는 방향을 찾아 그 사이를 가릅니다.' },
  { ab: 'SVM', en: 'Support Vector Machine', ko: '서포트 벡터 머신', g: 1, yr: 1995, task: '분류', fig: ['svm'], line: '두 집단 사이의 길이 가장 넓어지도록 경계를 긋습니다.' },
  { ab: 'RF', en: 'Random Forest', ko: '랜덤 포레스트', g: 2, yr: 2001, task: '분류 · 회귀', fig: ['forest', 7, 4, 9, 8], line: '조금씩 다른 데이터로 나무 수백 그루를 키워 다수결로 답합니다.' },
  { ab: 'ET', en: 'Extremely Randomized Trees', ko: '엑스트라 트리', g: 2, yr: 2006, task: '분류 · 회귀', fig: ['forest', 19, 7, 22, 16], line: '랜덤 포레스트에서 질문의 기준값까지 무작위로 고릅니다.' },
  { ab: 'DBSCAN', en: 'Density-Based Spatial Clustering', ko: 'DBSCAN', g: 4, yr: 1996, task: '군집 · 이상 탐지', fig: ['dbscan'], line: '반경 안에 이웃이 충분하면 한 덩어리로 잇고, 외딴 점은 잡음으로 남깁니다.' },
  { ab: 'AdaBoost', en: 'Adaptive Boosting', ko: '에이다부스트', g: 3, yr: 1997, task: '분류', fig: ['ada'], line: '앞 나무가 틀린 점의 무게를 키워 다음 나무가 그 점에 집중하게 합니다.' },
  { ab: 'iForest', en: 'Isolation Forest', ko: '아이솔레이션 포레스트', g: 6, yr: 2008, task: '이상 탐지', fig: ['iso'], line: '무작위로 선을 그어 점을 떼어 내고, 빨리 혼자 남는 점을 이상치로 봅니다.' },
  { ab: 'GMM', en: 'Gaussian Mixture Model', ko: '가우시안 혼합 모델', g: 4, yr: 1977, task: '군집', fig: ['gmm'], line: '종 모양 분포 여러 개가 섞였다고 보고, 소속을 확률로 답합니다.' },
  { ab: 'Kernel SVM', en: 'Kernel Support Vector Machine', ko: '커널 SVM', g: 1, yr: 1992, task: '분류', fig: ['ksvm'], line: '점들을 더 높은 차원으로 옮긴 셈 치고 거기서 평평한 경계를 긋습니다.' },
  { ab: 'GBM', en: 'Gradient Boosting Machine', ko: '그래디언트 부스팅', g: 3, yr: 2001, task: '회귀 · 분류', fig: ['gbm'], line: '앞 나무들이 남긴 오차를 다음 나무가 맞히도록 차례로 세웁니다.' },
  { ab: 'XGB', en: 'XGBoost', ko: 'XGBoost', g: 3, yr: 2016, task: '회귀 · 분류', fig: ['tree', 'level'], line: '그래디언트 부스팅에 벌점과 더 정밀한 오차 정보, 빠른 계산을 더했습니다.' },
  { ab: 'LGBM', en: 'LightGBM', ko: 'LightGBM', g: 3, yr: 2017, task: '회귀 · 분류', fig: ['tree', 'leaf'], line: '값을 구간으로 묶어 빠르게 계산하고, 오차가 가장 큰 잎부터 키웁니다.' },
  { ab: 'CatBoost', en: 'CatBoost', ko: 'CatBoost', g: 3, yr: 2018, task: '회귀 · 분류', fig: ['tree', 'sym'], line: '범주형 변수를 그대로 받고, 같은 층에서 같은 질문을 쓰는 대칭 나무를 세웁니다.' },
  { ab: 'GP', en: 'Gaussian Process', ko: '가우시안 프로세스', g: 1, yr: '1990년대', task: '회귀', fig: ['gp'], line: '가능한 곡선들의 분포를 다뤄, 예측과 함께 자신 없는 정도를 내놓습니다.' },
  { ab: 't-SNE', en: 't-distributed Stochastic Neighbor Embedding', ko: 't-SNE', g: 5, yr: 2008, task: '차원 축소', fig: ['embed', 'tsne'], line: '가까웠던 점끼리 가깝게 남도록 2차원에 다시 배치합니다.' },
  { ab: 'UMAP', en: 'Uniform Manifold Approximation and Projection', ko: 'UMAP', g: 5, yr: 2018, task: '차원 축소', fig: ['embed', 'umap'], line: '이웃 관계를 그물로 엮고, 그 모양을 지키면서 2차원에 펼칩니다.' },
  { ab: 'MLP', en: 'Multi-Layer Perceptron', ko: '다층 퍼셉트론', g: 6, yr: 1986, task: '분류 · 회귀', fig: ['mlp'], line: '입력과 출력 사이에 은닉층을 쌓은 신경망입니다. 신경망 연구소로 건너가는 다리입니다.' },
];

/* ================= 층별 핵심 식 · 참고 문헌 · 확인 문제 =================
   eq: 핵심 식 몇 줄 · eqNote: 식 아래 한 줄 · refs: 대표 문헌 · quiz: [질문, [보기 3개], 정답 번호, 풀이] */
const INFO = {
  로비: { eq: ['데이터 = 훈련용 + 시험용', '평균 제곱 오차  MSE = (1/n) Σ (yᵢ − ŷᵢ)²', '정확도 = 맞힌 개수 / 전체 개수', '크기 맞추기(표준화)  z = (x − 평균) / 표준편차'], eqNote: 'y는 정답, ŷ는 모델의 예측입니다. 숫자를 맞히는 문제(회귀)는 오차로, 종류를 맞히는 문제(분류)는 정확도로 채점합니다.',
    refs: ['Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer. 7장.', 'James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). An Introduction to Statistical Learning (2nd ed.). Springer. 2장·5장.'],
    quiz: [['모델을 채점할 때 쓰는 데이터는?', ['배울 때 쓴 훈련용 데이터', '따로 떼어 둔 시험용 데이터', '아무 데이터나'], 1, '배울 때 본 문제로 채점하면 외운 것인지 배운 것인지 구별할 수 없습니다.'],
      ['훈련 오차는 아주 작은데 시험 오차가 크다면?', ['과소적합', '과적합', '알맞게 배운 것'], 1, '훈련 점만 외운 상태입니다. 이 로비의 ⑤ 실험에서 복잡도를 끝까지 올리면 볼 수 있습니다.']] },
  LR: { eq: ['ŷ = w·x + b', '손실 = (1/n) Σ (yᵢ − ŷᵢ)²', 'w = Σ (xᵢ − x̄)(yᵢ − ȳ) / Σ (xᵢ − x̄)²,   b = ȳ − w·x̄'], eqNote: 'w는 기울기, b는 절편, x̄·ȳ는 평균입니다. 셋째 줄이 손실을 가장 작게 하는 답(최소제곱법)이고, 계산 한 번으로 나옵니다.',
    refs: ['Legendre, A.-M. (1805). Nouvelles méthodes pour la détermination des orbites des comètes. Paris: Firmin Didot.', 'Gauss, C. F. (1809). Theoria motus corporum coelestium in sectionibus conicis solem ambientium. Hamburg: Perthes & Besser.'],
    quiz: [['선형 회귀가 가장 작게 만들려는 것은?', ['점과 직선 사이 세로 거리의 제곱 합', '직선의 기울기', '점의 개수'], 0, '오차를 제곱해 더한 값이 가장 작은 직선을 고릅니다. 그래서 최소제곱법이라고 합니다.'],
      ['멀리 튄 점 하나가 직선을 크게 끌어당기는 이유는?', ['오차를 제곱하기 때문에 큰 오차가 훨씬 크게 계산되어서', '튄 점은 두 번 세기 때문에', '직선은 언제나 가장 먼 점을 지나야 해서'], 0, '오차가 2배면 벌은 4배입니다. 이 층의 ⑤ 실험에서 튄 점을 켜고 확인할 수 있습니다.']] },
};
[LOBBY, ...FLOORS].forEach(f => Object.assign(f, INFO[f.ab] || {}));
export const floor = n => n === 0 ? LOBBY : FLOORS[n - 1];
