import type { DataPoint, CorrelationResult, TrendResult, MultipleOlsResult, AnomalyResult, DecomposeResult } from './types';

// ─────────────────────────────────────────────────────────────
// MATH UTILITIES
// ─────────────────────────────────────────────────────────────

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function variance(arr: number[]): number {
  const m = mean(arr);
  return arr.reduce((a, x) => a + (x - m) ** 2, 0) / (arr.length - 1);
}

function lgamma(z: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  let x = c[0];
  const zz = z - 1;
  for (let i = 1; i < g + 2; i++) x += c[i] / (zz + i);
  const t = zz + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (zz + 0.5) * Math.log(t) - t + Math.log(x);
}

function lbeta(a: number, b: number): number {
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

// Regularized incomplete beta function using Lentz's continued fraction
function betacf(x: number, a: number, b: number): number {
  const MAXIT = 200, EPS = 3e-7, FPMIN = 1e-30;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= MAXIT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

function regularizedIncompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta(a, b));
  if (x < (a + 1) / (a + b + 2)) return front * betacf(x, a, b) / a;
  return 1 - front * betacf(1 - x, b, a) / b;
}

function tDistCDF(t: number, df: number): number {
  const x = df / (df + t * t);
  const p = regularizedIncompleteBeta(x, df / 2, 0.5) / 2;
  return t >= 0 ? 1 - p : p;
}

function tDistPValue(t: number, df: number): number {
  return 2 * (1 - tDistCDF(Math.abs(t), df));
}

function fDistPValue(f: number, df1: number, df2: number): number {
  const x = df2 / (df2 + df1 * f);
  return regularizedIncompleteBeta(x, df2 / 2, df1 / 2);
}

function correlationPValue(r: number, n: number): number {
  if (n < 4) return 1;
  const rClamped = Math.max(-0.9999999, Math.min(0.9999999, r));
  const t = rClamped * Math.sqrt((n - 2) / (1 - rClamped * rClamped));
  return tDistPValue(t, n - 2);
}

function rankArray(arr: number[]): number[] {
  const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length - 1 && indexed[j + 1].v === indexed[j].v) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
    i = j + 1;
  }
  return ranks;
}

function linReg(x: number[], y: number[]): { slope: number; intercept: number; r2: number; fitted: number[] } {
  const n = x.length;
  const xm = mean(x), ym = mean(y);
  const ssxx = x.reduce((a, xi) => a + (xi - xm) ** 2, 0);
  const ssxy = x.reduce((a, xi, i) => a + (xi - xm) * (y[i] - ym), 0);
  const slope = ssxx === 0 ? 0 : ssxy / ssxx;
  const intercept = ym - slope * xm;
  const fitted = x.map(xi => slope * xi + intercept);
  const ssRes = y.reduce((a, yi, i) => a + (yi - fitted[i]) ** 2, 0);
  const ssTot = y.reduce((a, yi) => a + (yi - ym) ** 2, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  return { slope, intercept, r2, fitted };
}

// ─────────────────────────────────────────────────────────────
// LABEL HELPERS
// ─────────────────────────────────────────────────────────────

function r2Label(r2: number): string {
  if (r2 < 0.10) return 'No meaningful correlation';
  if (r2 < 0.30) return 'Weak';
  if (r2 < 0.50) return 'Moderate';
  if (r2 < 0.70) return 'Strong';
  return 'Very strong';
}

function pLabel(p: number): string {
  if (p > 0.05) return 'not statistically significant';
  if (p > 0.01) return 'statistically significant';
  return 'highly statistically significant';
}

export function summaryLabel(r2: number, p: number, coef: number): string {
  const dir = coef >= 0 ? 'positive' : 'negative';
  return `${r2Label(r2)} ${dir} — ${pLabel(p)}`;
}

// ─────────────────────────────────────────────────────────────
// PAIR + ALIGN HELPERS
// ─────────────────────────────────────────────────────────────

interface AlignedPair {
  dates: string[];
  a: number[];
  b: number[];
}

function alignPair(rowsA: DataPoint[], rowsB: DataPoint[]): AlignedPair {
  const mapB = new Map(rowsB.map(r => [r.date, r.value]));
  const result: AlignedPair = { dates: [], a: [], b: [] };
  for (const { date, value: va } of rowsA) {
    const vb = mapB.get(date);
    if (vb != null) {
      result.dates.push(date);
      result.a.push(va);
      result.b.push(vb);
    }
  }
  return result;
}

// ─────────────────────────────────────────────────────────────
// CORRELATION FUNCTIONS
// ─────────────────────────────────────────────────────────────

function makeCorrelationResult(
  dates: string[], a: number[], b: number[],
  r: number, slope: number, intercept: number
): CorrelationResult {
  const n = a.length;
  const r2 = r * r;
  const p = correlationPValue(r, n);
  return {
    r2: +r2.toFixed(4), r: +r.toFixed(4), p_value: +p.toFixed(6),
    coefficient: +slope.toFixed(6), intercept: +intercept.toFixed(6), n,
    label: summaryLabel(r2, p, slope),
    series_a: dates.map((date, i) => ({ date, value: a[i] })),
    series_b: dates.map((date, i) => ({ date, value: b[i] })),
  };
}

function pearsonR(a: number[], b: number[]): number {
  const n = a.length;
  const am = mean(a), bm = mean(b);
  const num = a.reduce((s, ai, i) => s + (ai - am) * (b[i] - bm), 0);
  const den = Math.sqrt(
    a.reduce((s, ai) => s + (ai - am) ** 2, 0) *
    b.reduce((s, bi) => s + (bi - bm) ** 2, 0)
  );
  return den === 0 ? 0 : num / den;
}

export function pearsonCorrelation(
  rawA: DataPoint[], rawB: DataPoint[]
): CorrelationResult | { error: string } {
  const { dates, a, b } = alignPair(rawA, rawB);
  if (a.length < 3) return { error: 'Insufficient data (need ≥ 3 paired observations)' };
  const r = pearsonR(a, b);
  const { slope, intercept } = linReg(a, b);
  return makeCorrelationResult(dates, a, b, r, slope, intercept);
}

export function spearmanCorrelation(
  rawA: DataPoint[], rawB: DataPoint[]
): CorrelationResult | { error: string } {
  const { dates, a, b } = alignPair(rawA, rawB);
  if (a.length < 3) return { error: 'Insufficient data (need ≥ 3 paired observations)' };
  const ra = rankArray(a), rb = rankArray(b);
  const r = pearsonR(ra, rb);
  const { slope, intercept } = linReg(a, b);
  return makeCorrelationResult(dates, a, b, r, slope, intercept);
}

export function laggedCorrelation(
  rawA: DataPoint[], rawB: DataPoint[], lag: number, method: 'pearson' | 'spearman' = 'pearson'
): CorrelationResult | { error: string } {
  // Shift B forward by lag: pair rawA[t] with rawB[t+lag]
  const mapB = new Map(rawB.map((r, i) => [i, r]));
  const sortedA = [...rawA].sort((x, y) => x.date.localeCompare(y.date));
  const sortedB = [...rawB].sort((x, y) => x.date.localeCompare(y.date));
  const dateToIdxB = new Map(sortedB.map((r, i) => [r.date, i]));

  const dates: string[] = [], a: number[] = [], b: number[] = [];
  for (const pa of sortedA) {
    const idxB = dateToIdxB.get(pa.date);
    if (idxB == null) continue;
    const shiftedIdx = idxB + lag;
    const pb = shiftedIdx < sortedB.length ? sortedB[shiftedIdx] : undefined;
    if (!pb) continue;
    dates.push(pa.date); a.push(pa.value); b.push(pb.value);
  }
  if (a.length < 3) return { error: 'Insufficient data after applying lag' };

  const xVals = method === 'spearman' ? rankArray(a) : a;
  const yVals = method === 'spearman' ? rankArray(b) : b;
  const r = pearsonR(xVals, yVals);
  const { slope, intercept } = linReg(a, b);
  return makeCorrelationResult(dates, a, b, r, slope, intercept);
}

export function rollingAvgCorrelation(
  rawA: DataPoint[], rawB: DataPoint[], window: number, method: 'pearson' | 'spearman' = 'pearson'
): CorrelationResult | { error: string } {
  const minPeriods = Math.max(3, Math.floor(window / 2));
  function rollingMean(pts: DataPoint[]): DataPoint[] {
    return pts.map((p, i) => {
      const slice = pts.slice(Math.max(0, i - window + 1), i + 1);
      if (slice.length < minPeriods) return { date: p.date, value: NaN };
      return { date: p.date, value: mean(slice.map(s => s.value)) };
    });
  }
  const ra = rollingMean([...rawA].sort((x, y) => x.date.localeCompare(y.date)));
  const rb = rollingMean([...rawB].sort((x, y) => x.date.localeCompare(y.date)));
  const validA = ra.filter(p => !isNaN(p.value));
  const validB = rb.filter(p => !isNaN(p.value));
  const { dates, a, b } = alignPair(validA, validB);
  if (a.length < 3) return { error: 'Insufficient data for rolling window' };

  const xVals = method === 'spearman' ? rankArray(a) : a;
  const yVals = method === 'spearman' ? rankArray(b) : b;
  const r = pearsonR(xVals, yVals);
  const { slope, intercept } = linReg(a, b);
  return makeCorrelationResult(dates, a, b, r, slope, intercept);
}

// ─────────────────────────────────────────────────────────────
// OLS TREND
// ─────────────────────────────────────────────────────────────

export function olsTrend(raw: DataPoint[]): TrendResult | { error: string } {
  const sorted = [...raw].sort((a, b) => a.date.localeCompare(b.date));
  const values = sorted.map(p => p.value);
  if (values.length < 3) return { error: 'Insufficient data (need ≥ 3 observations)' };

  const t = values.map((_, i) => i);
  const { slope, intercept, r2, fitted } = linReg(t, values);

  // t-test on slope
  const n = values.length;
  const resid = values.map((y, i) => y - fitted[i]);
  const sse = resid.reduce((a, r) => a + r * r, 0);
  const sst = values.reduce((a, y) => a + (y - mean(values)) ** 2, 0);
  const mse = sse / (n - 2);
  const sxx = t.reduce((a, ti) => a + (ti - mean(t)) ** 2, 0);
  const seSlope = Math.sqrt(mse / sxx);
  const tStat = slope / seSlope;
  const p = tDistPValue(tStat, n - 2);

  return {
    r2: +r2.toFixed(4), p_value: +p.toFixed(6), coefficient: +slope.toFixed(6), n,
    label: summaryLabel(r2, p, slope),
    series:  sorted.map((p) => ({ date: p.date, value: p.value })),
    fitted: sorted.map((_, i) => ({ date: sorted[i].date, value: fitted[i] })),
  };
}

// ─────────────────────────────────────────────────────────────
// MULTIPLE OLS
// ─────────────────────────────────────────────────────────────

function solveLinear(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M: number[][] = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) throw new Error('Singular matrix');
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = M[row][col] / pivot;
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }
    for (let k = col; k <= n; k++) M[col][k] /= pivot;
  }
  return M.map(row => row[n]);
}

function invertMatrix(A: number[][]): number[][] {
  const n = A.length;
  const M: number[][] = A.map((row, i) => {
    const aug = new Array(n).fill(0);
    aug[i] = 1;
    return [...row, ...aug];
  });
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) throw new Error('Singular matrix');
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = M[row][col] / pivot;
      for (let k = 0; k < 2 * n; k++) M[row][k] -= f * M[col][k];
    }
    for (let k = 0; k < 2 * n; k++) M[col][k] /= pivot;
  }
  return M.map(row => row.slice(n));
}

export function multipleOls(
  predictorSeries: DataPoint[][], predictorNames: string[],
  outcomeSeries: DataPoint[], outcomeName: string
): MultipleOlsResult | { error: string } {
  // Align all series on common dates
  const dateMap = new Map<string, { preds: (number | null)[]; outcome: number | null }>();
  for (const pt of outcomeSeries) dateMap.set(pt.date, { preds: predictorSeries.map(() => null), outcome: pt.value });
  for (let pi = 0; pi < predictorSeries.length; pi++) {
    for (const pt of predictorSeries[pi]) {
      if (dateMap.has(pt.date)) dateMap.get(pt.date)!.preds[pi] = pt.value;
    }
  }
  const dates = [...dateMap.keys()].sort();
  const rows = dates
    .map(d => dateMap.get(d)!)
    .filter(r => r.outcome != null && r.preds.every(p => p != null));
  const n = rows.length;
  const k = predictorSeries.length;
  if (n < k + 2) return { error: 'Insufficient data for number of predictors' };

  const Y = rows.map(r => r.outcome as number);
  // Design matrix with intercept
  const X = rows.map(r => [1, ...r.preds as number[]]);
  const p = X[0].length;

  // XtX and Xty
  const XtX: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => X.reduce((s, row) => s + row[i] * row[j], 0))
  );
  const Xty: number[] = Array.from({ length: p }, (_, i) => X.reduce((s, row, ri) => s + row[i] * Y[ri], 0));

  let beta: number[];
  try { beta = solveLinear(XtX, Xty); } catch { return { error: 'Singular matrix — predictors may be collinear' }; }

  const fitted = X.map(row => row.reduce((s, xi, i) => s + xi * beta[i], 0));
  const ym = mean(Y);
  const ssRes = Y.reduce((a, yi, i) => a + (yi - fitted[i]) ** 2, 0);
  const ssTot = Y.reduce((a, yi) => a + (yi - ym) ** 2, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const r2adj = 1 - (ssRes / (n - p)) / (ssTot / (n - 1));
  const mse = ssRes / (n - p);
  const F = ((ssTot - ssRes) / (p - 1)) / mse;
  const pOverall = fDistPValue(F, p - 1, n - p);

  let XtXinv: number[][];
  try { XtXinv = invertMatrix(XtX); } catch { XtXinv = XtX.map(row => row.map(() => NaN)); }
  const pValues = predictorNames.reduce((acc, name, i) => {
    const se = Math.sqrt(mse * (XtXinv[i + 1]?.[i + 1] ?? NaN));
    const tStat = beta[i + 1] / se;
    acc[name] = +tDistPValue(tStat, n - p).toFixed(6);
    return acc;
  }, {} as Record<string, number>);
  const coefficients = predictorNames.reduce((acc, name, i) => {
    acc[name] = +beta[i + 1].toFixed(6);
    return acc;
  }, {} as Record<string, number>);

  return {
    r2: +r2.toFixed(4), r2_adj: +r2adj.toFixed(4),
    p_value: +pOverall.toFixed(6), coefficients, p_values: pValues,
    n, actual: Y, fitted,
  };
}

// ─────────────────────────────────────────────────────────────
// ANOMALY DETECTION (rolling z-score)
// ─────────────────────────────────────────────────────────────

export function anomalyDetection(
  raw: DataPoint[], windowDays: number, threshold: number
): AnomalyResult | { error: string } {
  const sorted = [...raw].sort((a, b) => a.date.localeCompare(b.date));
  if (sorted.length < 7) return { error: 'Insufficient data (need ≥ 7 observations)' };

  const values = sorted.map(p => p.value);
  const rollingMeanArr: (number | null)[] = values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - windowDays + 1), i + 1);
    return slice.length >= 7 ? mean(slice) : null;
  });
  const rollingStdArr: (number | null)[] = values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - windowDays + 1), i + 1);
    if (slice.length < 7) return null;
    const m = mean(slice);
    const std = Math.sqrt(slice.reduce((a, x) => a + (x - m) ** 2, 0) / slice.length);
    return std;
  });

  const anomalies: DataPoint[] = [];
  sorted.forEach((p, i) => {
    const rm = rollingMeanArr[i], rs = rollingStdArr[i];
    if (rm != null && rs != null && rs > 0) {
      const z = Math.abs(p.value - rm) / rs;
      if (z > threshold) anomalies.push(p);
    }
  });

  return {
    n_anomalies: anomalies.length, n: sorted.length, threshold, window: windowDays,
    series: sorted,
    rolling_mean: sorted.map((p, i) => ({ date: p.date, value: rollingMeanArr[i] })),
    anomalies,
  };
}

// ─────────────────────────────────────────────────────────────
// SEASONAL DECOMPOSITION (additive, moving average)
// ─────────────────────────────────────────────────────────────

export function seasonalDecompose(
  raw: DataPoint[], period: number
): DecomposeResult | { error: string } {
  const sorted = [...raw].sort((a, b) => a.date.localeCompare(b.date));
  if (sorted.length < period * 2) return { error: `Need ≥ ${period * 2} days of data for decomposition` };

  const values = sorted.map(p => p.value);
  const n = values.length;

  // Trend: centred moving average
  const trend: (number | null)[] = values.map((_, i) => {
    const half = Math.floor(period / 2);
    const start = i - half, end = i + half;
    if (start < 0 || end >= n) return null;
    const slice = values.slice(start, end + 1);
    return mean(slice);
  });

  // Detrended
  const detrended: (number | null)[] = values.map((v, i) => trend[i] == null ? null : v - trend[i]!);

  // Seasonal: average of detrended values by period position
  const seasonalAvg: number[] = new Array(period).fill(0);
  const seasonalCount: number[] = new Array(period).fill(0);
  detrended.forEach((d, i) => {
    if (d != null) { seasonalAvg[i % period] += d; seasonalCount[i % period]++; }
  });
  const seasonal = seasonalAvg.map((s, i) => seasonalCount[i] > 0 ? s / seasonalCount[i] : 0);
  // Normalise so seasonal sums to ~0
  const seasonalMean = mean(seasonal);
  const seasonalNorm = seasonal.map(s => s - seasonalMean);

  // Residual
  const residual: (number | null)[] = values.map((v, i) =>
    trend[i] == null ? null : v - trend[i]! - seasonalNorm[i % period]
  );

  return {
    n,
    observed:  sorted,
    trend:     sorted.map((p, i) => ({ date: p.date, value: trend[i] })),
    seasonal:  sorted.map((p, i) => ({ date: p.date, value: seasonalNorm[i % period] })),
    residual:  sorted.map((p, i) => ({ date: p.date, value: residual[i] })),
  };
}

// ─────────────────────────────────────────────────────────────
// EXPERIMENT ANALYSIS
// ─────────────────────────────────────────────────────────────

export interface ExperimentAnalysisResult {
  r2: number; r: number; p_value: number; coefficient: number; intercept: number;
  n: number; label: string;
  pre: DataPoint[];
  during: DataPoint[];
  all_paired: DataPoint[];
  pre_b: DataPoint[];
  during_b: DataPoint[];
  all_paired_b: DataPoint[];
  full_slope: number; full_intercept: number;
  pre_avg_a: number | null; pre_avg_b: number | null;
  during_avg_a: number; during_avg_b: number;
}

export function runExperimentAnalysis(
  rawA: DataPoint[], rawB: DataPoint[],
  varA: string, varB: string,
  lag: number, method: 'pearson' | 'spearman',
  startDate: string, durationDays: number
): ExperimentAnalysisResult | { error: string } {
  // Shift B by lag
  const sortedB = [...rawB].sort((a, b) => a.date.localeCompare(b.date));
  const shiftedB = new Map<string, number>();
  [...rawA].sort((a, b) => a.date.localeCompare(b.date)).forEach((pa, i) => {
    const idxB = sortedB.findIndex(p => p.date === pa.date);
    if (idxB < 0) return;
    const shiftedPt = sortedB[idxB + lag];
    if (shiftedPt) shiftedB.set(pa.date, shiftedPt.value);
  });

  const allPaired: DataPoint[] = [], allPairedB: DataPoint[] = [];
  for (const pa of [...rawA].sort((a, b) => a.date.localeCompare(b.date))) {
    const vb = shiftedB.get(pa.date);
    if (vb != null) { allPaired.push(pa); allPairedB.push({ date: pa.date, value: vb }); }
  }

  const endDate = addDays(startDate, durationDays);
  const pre = allPaired.filter(p => p.date < startDate);
  const preB = allPairedB.filter((_, i) => allPaired[i].date < startDate);
  const during = allPaired.filter(p => p.date >= startDate && p.date < endDate);
  const duringB = allPairedB.filter((_, i) => { const d = allPaired[i].date; return d >= startDate && d < endDate; });

  if (during.length < 3) return { error: 'Not enough data in experiment window yet — check back soon.' };

  const a = during.map(p => p.value), b = duringB.map(p => p.value);
  const xVals = method === 'spearman' ? rankArray(a) : a;
  const yVals = method === 'spearman' ? rankArray(b) : b;
  const r = pearsonR(xVals, yVals);
  const r2 = r * r;
  const p = correlationPValue(r, a.length);
  const { slope, intercept } = linReg(a, b);
  const allA = allPaired.map(p => p.value), allB = allPairedB.map(p => p.value);
  const { slope: fs, intercept: fi } = linReg(allA, allB);

  return {
    r2: +r2.toFixed(4), r: +r.toFixed(4), p_value: +p.toFixed(6),
    coefficient: +slope.toFixed(6), intercept: +intercept.toFixed(6),
    n: a.length, label: summaryLabel(r2, p, slope),
    pre, during, all_paired: allPaired,
    pre_b: preB, during_b: duringB, all_paired_b: allPairedB,
    full_slope: fs, full_intercept: fi,
    pre_avg_a: pre.length ? mean(pre.map(p => p.value)) : null,
    pre_avg_b: preB.length ? mean(preB.map(p => p.value)) : null,
    during_avg_a: mean(a), during_avg_b: mean(b),
  };
}

function addDays(dateStr: string, days: number): string {
  const d = new Date(dateStr);
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}
