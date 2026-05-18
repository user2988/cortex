export const COL_LABELS: Record<string, string> = {
  sleep_duration_min:    'Sleep Duration (min)',
  sleep_efficiency_pct:  'Sleep Efficiency (%)',
  deep_sleep_min:        'Deep Sleep (min)',
  rem_sleep_min:         'REM Sleep (min)',
  light_sleep_min:       'Light Sleep (min)',
  awake_min:             'Awake Time (min)',
  time_in_bed_min:       'Time in Bed (min)',
  hrv_ms:                'HRV RMSSD (ms)',
  hrv_deep_rmssd:        'HRV Deep RMSSD (ms)',
  rhr_bpm:               'Resting Heart Rate (bpm)',
  spo2_avg_pct:          'SpO₂ Average (%)',
  spo2_min_pct:          'SpO₂ Min (%)',
  spo2_max_pct:          'SpO₂ Max (%)',
  respiratory_rate:      'Respiratory Rate (br/min)',
  steps:                 'Steps',
  active_zone_min:       'Active Zone Minutes',
  very_active_min:       'Very Active (min)',
  fairly_active_min:     'Fairly Active (min)',
  lightly_active_min:    'Lightly Active (min)',
  sedentary_min:         'Sedentary (min)',
  calories_burned:       'Calories Burned',
  distance_km:           'Distance (km)',
  vo2_max:               'VO₂ Max',
  time_in_fat_burn_min:  'Fat Burn Zone (min)',
  time_in_cardio_min:    'Cardio Zone (min)',
  time_in_peak_min:      'Peak Zone (min)',
};

export const VAR_A_TREE: Record<string, string[]> = {
  'Activity  ·  Volume':    ['steps', 'distance_km', 'calories_burned', 'sedentary_min', 'lightly_active_min'],
  'Activity  ·  Intensity': ['active_zone_min', 'very_active_min', 'fairly_active_min'],
  'Activity  ·  Zones':     ['time_in_fat_burn_min', 'time_in_cardio_min', 'time_in_peak_min'],
};

export const VAR_B_TREE: Record<string, string[]> = {
  'Sleep  ·  Primary':              ['sleep_efficiency_pct', 'sleep_duration_min'],
  'Sleep  ·  Architecture':         ['deep_sleep_min', 'rem_sleep_min', 'light_sleep_min', 'awake_min'],
  'Sleep  ·  Behavioural':          ['time_in_bed_min'],
  'Cardiovascular  ·  Heart':       ['hrv_ms', 'hrv_deep_rmssd', 'rhr_bpm'],
  'Cardiovascular  ·  Oxygen':      ['spo2_avg_pct', 'spo2_min_pct', 'spo2_max_pct'],
  'Cardiovascular  ·  Respiratory': ['respiratory_rate', 'vo2_max'],
};

export const VAR_TREE: Record<string, string[]> = { ...VAR_A_TREE, ...VAR_B_TREE };

export const A_CATS = ['Activity'];
export const A_SUBS: Record<string, string[]> = {
  Activity: ['Volume', 'Intensity', 'Zones'],
};
export const B_CATS = ['Sleep', 'Cardiovascular'];
export const B_SUBS: Record<string, string[]> = {
  Sleep:          ['Primary', 'Architecture', 'Behavioural'],
  Cardiovascular: ['Heart', 'Oxygen', 'Respiratory'],
};
export const ALL_CATS = ['Activity', 'Sleep', 'Cardiovascular'];
export const ALL_SUBS: Record<string, string[]> = { ...A_SUBS, ...B_SUBS };

export const BIOMETRIC_COLS: string[] = [
  'sleep_duration_min', 'sleep_efficiency_pct', 'deep_sleep_min',
  'rem_sleep_min', 'light_sleep_min', 'awake_min', 'time_in_bed_min',
  'hrv_ms', 'hrv_deep_rmssd', 'rhr_bpm',
  'spo2_avg_pct', 'spo2_min_pct', 'spo2_max_pct', 'respiratory_rate',
  'steps', 'active_zone_min', 'very_active_min', 'fairly_active_min',
  'lightly_active_min', 'sedentary_min', 'calories_burned',
  'distance_km', 'vo2_max',
  'time_in_fat_burn_min', 'time_in_cardio_min', 'time_in_peak_min',
];

export function colLabel(col: string): string {
  return COL_LABELS[col] ?? col;
}
