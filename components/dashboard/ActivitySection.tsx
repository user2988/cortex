'use client';

import { useState } from 'react';
import PlotlyChart from '@/components/PlotlyChart';
import type { BiometricRow, DataPoint } from '@/lib/types';
import type { Data } from 'plotly.js';

function toSeries(rows: BiometricRow[], col: keyof BiometricRow): DataPoint[] {
  return rows.filter(r => r[col] != null).map(r => ({ date: r.date, value: Number(r[col]) }));
}
function rolling7(vals: number[]): number[] {
  return vals.map((_, i) => { const s = vals.slice(Math.max(0, i - 6), i + 1); return s.reduce((a, b) => a + b, 0) / s.length; });
}

export default function ActivitySection({ rows }: { rows: BiometricRow[] }) {
  const [days, setDays] = useState(30);
  const windowed = rows.slice(-days);

  const steps = toSeries(windowed, 'steps');
  const kcal  = toSeries(windowed, 'calories_burned');
  const dist  = toSeries(windowed, 'distance_km');

  const stepsData: Data[] = [
    { x: steps.map(p => p.date), y: steps.map(p => p.value), type: 'bar', marker: { color: '#4A90D9', opacity: 0.5 }, showlegend: false, hovertemplate: '%{x|%b %-d}: %{y:,}<extra></extra>' },
    { x: steps.map(p => p.date), y: rolling7(steps.map(p => p.value)), type: 'scatter', mode: 'lines', line: { color: '#E6EDF3', width: 1.5 }, showlegend: false },
  ];

  const kcalData: Data[] = [
    { x: kcal.map(p => p.date), y: kcal.map(p => p.value), type: 'bar', marker: { color: '#F59E0B', opacity: 0.5 }, showlegend: false },
    { x: kcal.map(p => p.date), y: rolling7(kcal.map(p => p.value)), type: 'scatter', mode: 'lines', line: { color: '#F59E0B', width: 2 }, showlegend: false },
  ];
  const distData: Data[] = [
    { x: dist.map(p => p.date), y: dist.map(p => p.value), type: 'bar', marker: { color: '#7EC8A4', opacity: 0.5 }, showlegend: false },
    { x: dist.map(p => p.date), y: rolling7(dist.map(p => p.value)), type: 'scatter', mode: 'lines', line: { color: '#7EC8A4', width: 2 }, showlegend: false },
  ];

  // Zone donut
  const zoneCols: (keyof BiometricRow)[] = ['time_in_fat_burn_min', 'time_in_cardio_min', 'time_in_peak_min', 'lightly_active_min'];
  const zoneAvg = zoneCols.map(c => { const v = windowed.filter(r => r[c] != null).map(r => Number(r[c])); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0; });
  const zoneData: Data[] = [{
    type: 'pie', labels: ['Fat Burn', 'Cardio', 'Peak', 'Light'],
    values: zoneAvg, marker: { colors: ['#F59E0B', '#EF4444', '#8B5CF6', '#4A90D9'] },
    hole: 0.65, textinfo: 'percent', textfont: { size: 9, family: 'IBM Plex Mono' },
    hovertemplate: '%{label}: %{value:.0f} min/day avg<extra></extra>',
  }];

  // Intensity stacked bars
  const intCfg = [
    { col: 'lightly_active_min' as keyof BiometricRow, label: 'Light',    color: '#4A90D9' },
    { col: 'fairly_active_min'  as keyof BiometricRow, label: 'Moderate', color: '#F59E0B' },
    { col: 'very_active_min'    as keyof BiometricRow, label: 'Intense',  color: '#EF4444' },
  ];
  const intData: Data[] = intCfg.map(s => ({
    x: windowed.map(r => r.date),
    y: windowed.map(r => r[s.col] != null ? Number(r[s.col]) : 0),
    type: 'bar', name: s.label, marker: { color: s.color },
    hovertemplate: `${s.label}: %{y:.0f} min<extra></extra>`,
  }));

  // HR zones stacked bars
  const zoneCfg = [
    { col: 'time_in_fat_burn_min' as keyof BiometricRow, label: 'Fat Burn', color: '#F59E0B' },
    { col: 'time_in_cardio_min'   as keyof BiometricRow, label: 'Cardio',   color: '#EF4444' },
    { col: 'time_in_peak_min'     as keyof BiometricRow, label: 'Peak',     color: '#8B5CF6' },
  ];
  const hrZoneData: Data[] = zoneCfg.map(s => ({
    x: windowed.map(r => r.date),
    y: windowed.map(r => r[s.col] != null ? Number(r[s.col]) : 0),
    type: 'bar', name: s.label, marker: { color: s.color },
    hovertemplate: `${s.label}: %{y:.0f} min<extra></extra>`,
  }));

  // Steps heatmap
  const heatmapRows = rows.filter(r => r.steps != null);
  const weeklyData = buildWeeklyHeatmap(heatmapRows);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div className="section-header" style={{ margin: 0 }}>Activity</div>
        <div style={{ display: 'flex', gap: 4 }}>
          {[7, 14, 30, 60, 90].map(d => (
            <button key={d} className={`window-btn ${days === d ? 'active' : ''}`} onClick={() => setDays(d)}>{d}d</button>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div className="chart-label">Daily Steps</div>
          <PlotlyChart data={stepsData} layout={{ bargap: 0.2, shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 10000, y1: 10000, yref: 'y', line: { color: '#30363D', width: 1, dash: 'dot' } }], annotations: [{ x: 1, xref: 'paper', y: 10000, yref: 'y', text: '10k', showarrow: false, xanchor: 'right', font: { color: '#484F58', size: 9 } }] }} height={200} />
        </div>
        <div>
          <div className="chart-label">Avg Activity Zones</div>
          <PlotlyChart data={zoneData} layout={{ showlegend: false }} height={200} />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div className="chart-label">Calories Burned (kcal)</div>
          <PlotlyChart data={kcalData} layout={{ bargap: 0.2 }} height={180} />
        </div>
        <div>
          <div className="chart-label">Distance (km)</div>
          <PlotlyChart data={distData} layout={{ bargap: 0.2 }} height={180} />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <div className="chart-label">Activity Intensity (min/day)</div>
          <PlotlyChart data={intData} layout={{ barmode: 'stack', bargap: 0.15, legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' } }} height={220} />
        </div>
        <div>
          <div className="chart-label">HR Zones (min/day)</div>
          <PlotlyChart data={hrZoneData} layout={{ barmode: 'stack', bargap: 0.15, legend: { orientation: 'h', y: 1.18, x: 0, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' } }} height={220} />
        </div>
      </div>
      {weeklyData && (
        <div>
          <div className="chart-label">Step count by day-of-week × calendar week</div>
          <PlotlyChart data={weeklyData} layout={{ margin: { l: 40, r: 16, t: 30, b: 20 } }} height={220} />
        </div>
      )}
    </div>
  );
}

function buildWeeklyHeatmap(rows: BiometricRow[]): import('plotly.js').Data[] | null {
  if (rows.length < 14) return null;
  const DOW_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const weekMap = new Map<string, number[]>();
  rows.forEach(r => {
    const d = new Date(r.date);
    const dow = (d.getDay() + 6) % 7; // 0=Mon
    const year = d.getFullYear();
    const jan4 = new Date(year, 0, 4);
    const week = Math.ceil(((d.getTime() - jan4.getTime()) / 86400000 + jan4.getDay() + 1) / 7);
    const yw = `${year}-W${String(week).padStart(2, '0')}`;
    if (!weekMap.has(yw)) weekMap.set(yw, new Array(7).fill(null));
    weekMap.get(yw)![dow] = Number(r.steps);
  });
  const sortedWeeks = Array.from(weekMap.keys()).sort();
  const z = DOW_LABELS.map((_, dow) => sortedWeeks.map(yw => weekMap.get(yw)?.[dow] ?? null));
  return [{
    type: 'heatmap', z, x: sortedWeeks, y: DOW_LABELS,
    colorscale: [[0, '#161B22'], [0.3, '#1A4A6E'], [0.65, '#2DD4BF'], [1, '#A7F3D0']],
    showscale: false, hovertemplate: '%{x} %{y}: %{z:,} steps<extra></extra>',
  }] as import('plotly.js').Data[];
}
