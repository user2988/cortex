'use client';

import { useState } from 'react';
import TrendCard from './TrendCard';
import PlotlyChart from '@/components/PlotlyChart';
import type { BiometricRow, DataPoint } from '@/lib/types';
import type { Data } from 'plotly.js';

function toSeries(rows: BiometricRow[], col: keyof BiometricRow): DataPoint[] {
  return rows.filter(r => r[col] != null).map(r => ({ date: r.date, value: Number(r[col]) }));
}

export default function CardioSection({ rows }: { rows: BiometricRow[] }) {
  const [days, setDays] = useState(90);
  const windowed = rows.slice(-days);

  const hrv   = toSeries(windowed, 'hrv_ms');
  const rhr   = toSeries(windowed, 'rhr_bpm');
  const spo2  = toSeries(windowed, 'spo2_avg_pct');
  const resp  = toSeries(windowed, 'respiratory_rate');
  const vo2   = toSeries(windowed, 'vo2_max');
  const sed   = toSeries(windowed, 'sedentary_min').map(p => ({ ...p, value: p.value / 60 }));

  const rhrVals = rhr.map(p => p.value);
  const rhrHistData: Data[] = [{
    x: rhrVals, type: 'histogram',
    xbins: { start: Math.min(...rhrVals) - 0.5, end: Math.max(...rhrVals) + 0.5, size: 2 },
    marker: { color: '#EF4444', opacity: 0.65 },
    hovertemplate: '%{x} bpm: %{y} days<extra></extra>',
  }];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="section-header" style={{ margin: 0 }}>Cardiovascular</div>
        <div style={{ display: 'flex', gap: 4 }}>
          {[7, 14, 30, 60, 90].map(d => (
            <button key={d} className={`window-btn ${days === d ? 'active' : ''}`} onClick={() => setDays(d)}>{d}d</button>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <TrendCard title="HRV RMSSD (ms)"           series={hrv}  color="#2DD4BF" fill="rgba(45,212,191,0.12)"  height={220} />
        <TrendCard title="Resting Heart Rate (bpm)"  series={rhr}  color="#EF4444" fill="rgba(239,68,68,0.12)"   height={220} />
        <TrendCard title="SpO₂ Average (%)"          series={spo2} color="#7EC8A4" fill="rgba(126,200,164,0.12)" height={220} refLine={95} refLabel="95%" />
        <TrendCard title="Respiratory Rate (br/min)" series={resp} color="#F59E0B" fill="rgba(245,158,11,0.12)"  height={220} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <div>
          {rhr.length >= 5 ? (
            <>
              <div className="chart-label">RHR Distribution</div>
              <PlotlyChart data={rhrHistData} height={200} layout={{ bargap: 0.06 }} />
            </>
          ) : <div className="empty-panel">RHR Distribution<br />Need ≥5 days</div>}
        </div>
        <TrendCard title="VO₂ Max (mL/kg/min)" series={vo2} color="#2DD4BF" fill="rgba(45,212,191,0.12)" height={220} />
        <TrendCard title="Sedentary Time (h)"   series={sed} color="#F59E0B" fill="rgba(245,158,11,0.12)" height={220} refLine={8} refLabel="8h" />
      </div>
    </div>
  );
}
