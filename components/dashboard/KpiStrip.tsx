'use client';

import type { BiometricRow } from '@/lib/types';

interface Props { rows: BiometricRow[] }

function latest(rows: BiometricRow[], col: keyof BiometricRow): number | null {
  for (let i = rows.length - 1; i >= 0; i--) {
    const v = rows[i][col];
    if (v != null) return Number(v);
  }
  return null;
}

function delta(rows: BiometricRow[], col: keyof BiometricRow): [number | null, string] {
  const vals = rows.map(r => r[col]).filter(v => v != null).map(Number);
  if (vals.length < 2) return [null, '#484F58'];
  const d = vals[vals.length - 1] - vals[vals.length - 2];
  const c = d > 0 ? '#10B981' : d < 0 ? '#EF4444' : '#484F58';
  return [d, c];
}

function fmt(v: number | null): string {
  if (v == null) return '—';
  if (v >= 10000) return v.toLocaleString('en', { maximumFractionDigits: 0 });
  return v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

function metricBg(col: keyof BiometricRow, v: number | null): string | undefined {
  if (v == null) return undefined;
  if (col === 'hrv_ms') return v >= 40 ? '#10B981' : v >= 25 ? '#F59E0B' : '#EF4444';
  if (col === 'rhr_bpm') return v <= 60 ? '#10B981' : v <= 75 ? '#F59E0B' : '#EF4444';
  if (col === 'spo2_avg_pct') return v >= 95 ? '#10B981' : v >= 90 ? '#F59E0B' : '#EF4444';
  if (col === 'steps') return v >= 10000 ? '#10B981' : v >= 5000 ? '#F59E0B' : '#EF4444';
  if (col === 'active_zone_min') return v >= 30 ? '#10B981' : v >= 15 ? '#F59E0B' : '#EF4444';
  return undefined;
}

interface KpiProps { label: string; value: number | null; unit?: string; delta?: number | null; deltaColor?: string; bg?: string }

function KpiCard({ label, value, unit = '', delta: d, deltaColor = '#484F58', bg }: KpiProps) {
  const arr = d == null ? '' : d > 0 ? '↑' : d < 0 ? '↓' : '→';
  return (
    <div className="kpi-card" style={bg ? { borderTop: `2px solid ${bg}` } : {}}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">
        {fmt(value)}<span className="kpi-unit">{unit}</span>
      </div>
      {d != null && (
        <div className="kpi-delta" style={{ color: deltaColor }}>
          {arr} {Math.abs(d).toFixed(1)}
        </div>
      )}
    </div>
  );
}

export default function KpiStrip({ rows }: Props) {
  const hrv    = latest(rows, 'hrv_ms');
  const rhr    = latest(rows, 'rhr_bpm');
  const spo2   = latest(rows, 'spo2_avg_pct');
  const steps  = latest(rows, 'steps');
  const azm    = latest(rows, 'active_zone_min');
  const kcal   = latest(rows, 'calories_burned');

  const [hrvD, hrvDc]   = delta(rows, 'hrv_ms');
  const [rhrD,]          = delta(rows, 'rhr_bpm');
  const [stepsD, stepsDc] = delta(rows, 'steps');

  const rhrDc = rhrD == null ? '#484F58' : rhrD > 0 ? '#EF4444' : '#10B981';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8, marginBottom: 12 }}>
      <KpiCard label="HRV RMSSD" value={hrv} unit="ms" delta={hrvD} deltaColor={hrvDc} bg={metricBg('hrv_ms', hrv)} />
      <KpiCard label="Resting HR" value={rhr} unit="bpm" delta={rhrD} deltaColor={rhrDc} bg={metricBg('rhr_bpm', rhr)} />
      <KpiCard label="SpO₂" value={spo2} unit="%" bg={metricBg('spo2_avg_pct', spo2)} />
      <KpiCard label="Steps" value={steps} delta={stepsD} deltaColor={stepsDc} bg={metricBg('steps', steps)} />
      <KpiCard label="Active Min" value={azm} unit="min" bg={metricBg('active_zone_min', azm)} />
      <KpiCard label="Calories Burnt" value={kcal} unit="kcal" />
    </div>
  );
}
