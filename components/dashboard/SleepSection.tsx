'use client';

import { useState } from 'react';
import PlotlyChart from '@/components/PlotlyChart';
import TrendCard from './TrendCard';
import type { BiometricRow, DataPoint } from '@/lib/types';
import type { Data } from 'plotly.js';

function filterWindow(rows: BiometricRow[], days: number): BiometricRow[] {
  return rows.slice(-days);
}

function toSeries(rows: BiometricRow[], col: keyof BiometricRow): DataPoint[] {
  return rows.filter(r => r[col] != null).map(r => ({ date: r.date, value: Number(r[col]) }));
}

export default function SleepSection({ rows }: { rows: BiometricRow[] }) {
  const [days, setDays] = useState(30);
  const windowed = filterWindow(rows, days);

  const stageCfg = [
    { col: 'deep_sleep_min' as keyof BiometricRow,  label: 'Deep (N3)', color: '#2DD4BF' },
    { col: 'rem_sleep_min' as keyof BiometricRow,   label: 'REM',       color: '#8B5CF6' },
    { col: 'light_sleep_min' as keyof BiometricRow, label: 'Light',     color: '#4A90D9' },
    { col: 'awake_min' as keyof BiometricRow,       label: 'Awake',     color: '#3D4451' },
  ];
  const stageCols = stageCfg.map(s => s.col);

  const stageData: Data[] = stageCfg.map(s => {
    const vals = windowed.map(r => r[s.col] != null ? Number(r[s.col]) : null);
    return {
      x: windowed.map(r => r.date), y: vals, type: 'bar' as const,
      name: s.label, marker: { color: s.color },
      hovertemplate: `${s.label}: %{y:.0f} min<extra></extra>`,
    };
  });

  // Donut: avg over window
  const avgVals = stageCfg.map(s => {
    const vals = windowed.filter(r => r[s.col] != null).map(r => Number(r[s.col]));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  });
  const donutData: Data[] = [{
    type: 'pie' as const,
    labels: ['Deep', 'REM', 'Light', 'Awake'],
    values: avgVals,
    marker: { colors: ['#2DD4BF', '#8B5CF6', '#4A90D9', '#3D4451'] },
    hole: 0.65, textinfo: 'percent',
    textfont: { size: 9, family: 'IBM Plex Mono' },
    hovertemplate: '%{label}: %{value:.0f} min<extra></extra>',
  }];

  const effSeries   = toSeries(windowed, 'sleep_efficiency_pct');
  const durRaw      = toSeries(windowed, 'sleep_duration_min');
  const durSeries   = durRaw.map(p => ({ date: p.date, value: p.value / 60 }));

  const durRolling = durSeries.map((_, i) => {
    const slice = durSeries.slice(Math.max(0, i - 6), i + 1);
    return slice.reduce((a, p) => a + p.value, 0) / slice.length;
  });

  const durBarData: Data[] = [
    { x: durSeries.map(p => p.date), y: durSeries.map(p => p.value), type: 'bar', marker: { color: '#8B5CF6', opacity: 0.5 }, showlegend: false, hovertemplate: '%{x|%b %-d}: %{y:.1f}h<extra></extra>' },
    { x: durSeries.map(p => p.date), y: durRolling, type: 'scatter', mode: 'lines', line: { color: '#8B5CF6', width: 2 }, showlegend: false },
  ];

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="section-header" style={{ margin: 0 }}>Sleep Architecture</div>
        <div style={{ display: 'flex', gap: 4 }}>
          {[7, 30, 90].map(d => (
            <button key={d} className={`window-btn ${days === d ? 'active' : ''}`} onClick={() => setDays(d)}>{d}d</button>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div className="chart-label">Sleep Stage Breakdown (min)</div>
          <PlotlyChart data={stageData} layout={{ barmode: 'stack', bargap: 0.15, legend: { orientation: 'h', y: 1.2, x: 0, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' } }} height={230} />
        </div>
        <div>
          <div className="chart-label">{days}d Avg</div>
          <PlotlyChart data={donutData} layout={{ showlegend: false }} height={230} />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 8 }}>
        <TrendCard title="Sleep Efficiency (%)" series={effSeries} color="#4A90D9" fill="rgba(74,144,217,0.12)" height={180} refLine={85} refLabel="85%" />
        <div>
          <div className="chart-label">Sleep Duration (h)</div>
          <PlotlyChart data={durBarData} layout={{ bargap: 0.2, shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 8, y1: 8, yref: 'y', line: { color: '#30363D', width: 1, dash: 'dot' } }], annotations: [{ x: 1, xref: 'paper', y: 8, yref: 'y', text: '8h', showarrow: false, xanchor: 'right', font: { color: '#484F58', size: 9 } }] }} height={180} />
        </div>
      </div>
    </div>
  );
}
