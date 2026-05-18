'use client';

import type { BiometricRow } from '@/lib/types';

type Col = keyof BiometricRow;
interface MetricCfg { col: Col; label: string; unit: string; invert: boolean }

const METRICS: MetricCfg[] = [
  { col: 'hrv_ms',               label: 'HRV',         unit: 'ms',   invert: false },
  { col: 'rhr_bpm',              label: 'RHR',         unit: 'bpm',  invert: true  },
  { col: 'spo2_avg_pct',         label: 'SpO₂',        unit: '%',    invert: false },
  { col: 'steps',                label: 'Steps',       unit: '',     invert: false },
  { col: 'sleep_efficiency_pct', label: 'Sleep Eff',   unit: '%',    invert: false },
  { col: 'calories_burned',      label: 'Calories',    unit: 'kcal', invert: false },
];

function getVals(rows: BiometricRow[], col: Col): number[] {
  return rows.map(r => r[col]).filter(v => v != null).map(Number);
}

export default function RollingStats({ rows }: { rows: BiometricRow[] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8, marginBottom: 8 }}>
      {METRICS.map(({ col, label, unit, invert }) => {
        const vals = getVals(rows, col);
        if (vals.length < 14) return (
          <div key={col} style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 4, padding: '5px 8px', textAlign: 'center' }}>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 8, fontWeight: 600, letterSpacing: '.1em', textTransform: 'uppercase', color: '#484F58' }}>{label}</div>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 12, color: '#484F58' }}>—</div>
          </div>
        );
        const r7  = vals.slice(-7).reduce((a, b) => a + b, 0) / 7;
        const r90 = vals.reduce((a, b) => a + b, 0) / vals.length;
        let pct = r90 !== 0 ? (r7 / r90 - 1) * 100 : 0;
        if (invert) pct = -pct;
        const color = pct >= 3 ? '#10B981' : pct <= -3 ? '#EF4444' : '#484F58';
        const arr   = pct >= 3 ? '↑' : pct <= -3 ? '↓' : '→';
        return (
          <div key={col} style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 4, padding: '5px 8px', textAlign: 'center' }}>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 8, fontWeight: 600, letterSpacing: '.1em', textTransform: 'uppercase', color: '#484F58' }}>{label}</div>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 15, fontWeight: 300, color }}>{arr} {Math.abs(pct).toFixed(1)}%</div>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 8, color: '#484F58' }}>7d avg {r7.toFixed(1)}{unit}</div>
          </div>
        );
      })}
    </div>
  );
}
