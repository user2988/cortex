export interface BiometricRow {
  date: string;
  sleep_duration_min: number | null;
  sleep_efficiency_pct: number | null;
  deep_sleep_min: number | null;
  rem_sleep_min: number | null;
  light_sleep_min: number | null;
  awake_min: number | null;
  time_in_bed_min: number | null;
  hrv_ms: number | null;
  hrv_deep_rmssd: number | null;
  rhr_bpm: number | null;
  spo2_avg_pct: number | null;
  spo2_min_pct: number | null;
  spo2_max_pct: number | null;
  respiratory_rate: number | null;
  steps: number | null;
  active_zone_min: number | null;
  very_active_min: number | null;
  fairly_active_min: number | null;
  lightly_active_min: number | null;
  sedentary_min: number | null;
  calories_burned: number | null;
  distance_km: number | null;
  vo2_max: number | null;
  time_in_fat_burn_min: number | null;
  time_in_cardio_min: number | null;
  time_in_peak_min: number | null;
}

export interface Finding {
  id: number;
  variable_a: string;
  variable_b: string | null;
  r_squared: number;
  p_value: number;
  coefficient: number;
  lag_days: number;
  analysis_type: string;
  sample_size: number;
  calculated_at: string;
  pinned: boolean;
}

export interface Experiment {
  id: number;
  name: string;
  variable_a: string;
  variable_b: string;
  lag_days: number;
  method: string;
  start_date: string;
  duration_days: number;
  status: string;
  interpretation: string | null;
  created_at: string;
  end_date: string;
  is_complete: boolean;
  elapsed_days: number;
}

export interface DailyScore {
  date: string;
  sleep_score: number | null;
  heart_score: number | null;
  duration_score: number | null;
  deep_score: number | null;
  rem_score: number | null;
  efficiency_score: number | null;
  hrv_score: number | null;
  rhr_score: number | null;
  spo2_score: number | null;
  sleep_duration_min: number | null;
  deep_pct: number | null;
  rem_pct: number | null;
  hrv_ms: number | null;
  rhr_bpm: number | null;
  spo2_avg_pct: number | null;
}

export interface Recommendation {
  target_score: string;
  activity_metric: string;
  activity_label: string;
  optimal_min: number | null;
  optimal_min_fmt: string;
  optimal_max_fmt: string;
  avg_score_in_range: number;
  avg_score_outside: number;
  score_delta: number;
  correlation: number;
  sample_size: number;
  recommendation_text: string;
}

export interface DataPoint {
  date: string;
  value: number;
}

export interface CorrelationResult {
  r2: number;
  r: number;
  p_value: number;
  coefficient: number;
  intercept: number;
  n: number;
  label: string;
  series_a: DataPoint[];
  series_b: DataPoint[];
}

export interface TrendResult {
  r2: number;
  p_value: number;
  coefficient: number;
  n: number;
  label: string;
  series: DataPoint[];
  fitted: DataPoint[];
}

export interface MultipleOlsResult {
  r2: number;
  r2_adj: number;
  p_value: number;
  coefficients: Record<string, number>;
  p_values: Record<string, number>;
  n: number;
  actual: number[];
  fitted: number[];
}

export interface AnomalyResult {
  n_anomalies: number;
  n: number;
  threshold: number;
  window: number;
  series: DataPoint[];
  rolling_mean: Array<{ date: string; value: number | null }>;
  anomalies: DataPoint[];
}

export interface DecomposeResult {
  n: number;
  observed: DataPoint[];
  trend: Array<{ date: string; value: number | null }>;
  seasonal: DataPoint[];
  residual: Array<{ date: string; value: number | null }>;
}

export type AnalysisPayload =
  | { type: 'Pearson Correlation' | 'Spearman Correlation'; var_a: string; var_b: string; days: number }
  | { type: 'Lagged Correlation'; var_a: string; var_b: string; lag: number; method: string; days: number }
  | { type: 'Rolling Average'; var_a: string; var_b: string; window: number; method: string; days: number }
  | { type: '30-Day Trend (OLS)'; var_a: string; days: number }
  | { type: 'Multiple OLS Regression'; predictors: string[]; outcome: string; days: number }
  | { type: 'Anomaly Detection'; var_a: string; window: number; threshold: number; days: number }
  | { type: 'Decomposition'; var_a: string; period: number; days: number };

export type AnalysisResult =
  | { type: 'correlation'; data: CorrelationResult }
  | { type: 'trend'; data: TrendResult }
  | { type: 'multiple_ols'; data: MultipleOlsResult }
  | { type: 'anomaly'; data: AnomalyResult }
  | { type: 'decompose'; data: DecomposeResult }
  | { type: 'error'; message: string };
