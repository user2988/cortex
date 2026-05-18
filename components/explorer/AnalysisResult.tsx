'use client';

import { useState } from 'react';
import PlotlyChart from '@/components/PlotlyChart';
import type { AnalysisResult, CorrelationResult, TrendResult, MultipleOlsResult, AnomalyResult, DecomposeResult, AnalysisPayload } from '@/lib/types';
import type { Data } from 'plotly.js';
import { colLabel } from '@/lib/labels';

const BLUE   = '#1f77b4';
const ORANGE = '#ff7f0e';
const GREEN  = '#2ca02c';
const RED    = '#d62728';

interface Props {
  result: AnalysisResult;
  payload: AnalysisPayload;
  onSave?: () => void;
}

function StatBar({ r2, p, coef, n, label, extra }: { r2?: number; p?: number; coef?: number; n?: number; label?: string; extra?: [string, string][] }) {
  const items = [
    r2   != null && ['R²', r2.toFixed(4)],
    p    != null && ['p-value', p.toFixed(4)],
    coef != null && ['Coefficient', coef.toFixed(4)],
    n    != null && ['Data Points', String(n)],
    ...(extra ?? []),
  ].filter(Boolean) as [string, string][];
  return (
    <div>
      <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
        {items.map(([l, v]) => (
          <div key={l} className="metric-pill" style={{ minWidth: 90 }}>
            <div className="label">{l}</div>
            <div className="value" style={{ fontSize: 16 }}>{v}</div>
          </div>
        ))}
      </div>
      {label && <div style={{ fontSize: 13, fontWeight: 600, color: '#8B949E', marginBottom: 12 }}>{label}</div>}
    </div>
  );
}

function ScatterOls({ res, xLabel, yLabel }: { res: CorrelationResult; xLabel: string; yLabel: string }) {
  const xVals = res.series_a.map(p => p.value);
  const yVals = res.series_b.map(p => p.value);
  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  const data: Data[] = [
    {
      x: xVals, y: yVals, type: 'scatter', mode: 'markers',
      text: res.series_a.map(p => p.date) as string[],
      hovertemplate: '%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>',
      marker: { color: BLUE, size: 7, opacity: 0.8 },
    },
    {
      x: [xMin, xMax], y: [res.coefficient * xMin + res.intercept, res.coefficient * xMax + res.intercept],
      type: 'scatter', mode: 'lines',
      line: { color: ORANGE, dash: 'dash', width: 2 },
      showlegend: false,
    },
  ];
  return <PlotlyChart data={data} layout={{ xaxis: { title: xLabel } as Partial<import('plotly.js').LayoutAxis>, yaxis: { title: yLabel } as Partial<import('plotly.js').LayoutAxis> }} height={450} />;
}

export default function AnalysisResult({ result, payload, onSave }: Props) {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved]   = useState(false);

  if (result.type === 'error') {
    return <div style={{ color: '#EF4444', fontSize: 13 }}>{result.message}</div>;
  }

  // Determine title
  let title = '';
  if (result.type === 'multiple_ols') {
    title = `Multiple OLS — ${(payload as { outcome?: string }).outcome ? colLabel((payload as { outcome: string }).outcome) : ''}`;
  } else if (result.type === 'trend' || result.type === 'anomaly' || result.type === 'decompose') {
    title = `${payload.type} — ${(payload as { var_a?: string }).var_a ? colLabel((payload as { var_a: string }).var_a) : ''}`;
  } else if (result.type === 'correlation') {
    const p = payload as { var_a?: string; var_b?: string; lag?: number; window?: number };
    title = `${p.var_a ? colLabel(p.var_a) : ''} × ${p.var_b ? colLabel(p.var_b) : ''}`;
    if (payload.type === 'Lagged Correlation' && p.lag) title += ` (lag ${p.lag}d)`;
    if (payload.type === 'Rolling Average' && p.window) title += ` (${p.window}-day rolling)`;
  }

  const SAVEABLE = new Set(['Pearson Correlation', 'Spearman Correlation', 'Lagged Correlation', 'Rolling Average', '30-Day Trend (OLS)']);

  async function save() {
    if (result.type !== 'correlation' && result.type !== 'trend') return;
    setSaving(true);
    const p = payload as Record<string, unknown>;
    await fetch('/api/save-finding', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        variable_a: p.var_a, variable_b: p.var_b ?? null,
        r_squared: result.data.r2, p_value: result.data.p_value,
        coefficient: result.data.coefficient,
        lag_days: p.lag ?? 0, analysis_type: payload.type,
        sample_size: result.data.n,
      }),
    });
    setSaving(false); setSaved(true);
    onSave?.();
  }

  return (
    <div>
      <h2 style={{ fontFamily: 'Inter', fontSize: 18, fontWeight: 600, color: '#E6EDF3', marginBottom: 16 }}>{title}</h2>

      {result.type === 'correlation' && (() => {
        const d = result.data;
        const p = payload as { var_a?: string; var_b?: string; lag?: number; window?: number };
        const xLabel = p.var_a ? colLabel(p.var_a) : '';
        const yLabel = p.var_b ? (payload.type === 'Lagged Correlation' && p.lag ? `${colLabel(p.var_b)} (+${p.lag}d)` : colLabel(p.var_b)) : '';
        return (
          <>
            <StatBar r2={d.r2} p={d.p_value} coef={d.coefficient} n={d.n} label={d.label} />
            <ScatterOls res={d} xLabel={xLabel} yLabel={yLabel} />
          </>
        );
      })()}

      {result.type === 'trend' && (() => {
        const d = result.data;
        const varA = (payload as { var_a?: string }).var_a ?? '';
        const data: Data[] = [
          { x: d.series.map(p => p.date), y: d.series.map(p => p.value), type: 'scatter', mode: 'lines+markers', name: colLabel(varA), line: { color: BLUE }, marker: { size: 5 } },
          { x: d.fitted.map(p => p.date), y: d.fitted.map(p => p.value), type: 'scatter', mode: 'lines', name: 'Trend', line: { color: GREEN, dash: 'dash', width: 2 } },
        ];
        return (
          <>
            <StatBar r2={d.r2} p={d.p_value} coef={d.coefficient} n={d.n} label={d.label} extra={[['↑↓ per day', `${d.coefficient >= 0 ? '+' : ''}${d.coefficient.toFixed(4)}`]]} />
            <PlotlyChart data={data} layout={{ xaxis: { title: 'Date' } as Partial<import('plotly.js').LayoutAxis>, yaxis: { title: colLabel(varA) } as Partial<import('plotly.js').LayoutAxis>, legend: { orientation: 'h', y: 1.08 } }} height={450} />
          </>
        );
      })()}

      {result.type === 'multiple_ols' && (() => {
        const d = result.data as MultipleOlsResult;
        const outCol = (payload as { outcome?: string }).outcome ?? '';
        const mn = Math.min(...d.actual, ...d.fitted);
        const mx = Math.max(...d.actual, ...d.fitted);
        const preds = (payload as { predictors?: string[] }).predictors ?? [];
        const data: Data[] = [
          {
            x: d.actual, y: d.fitted, type: 'scatter', mode: 'markers',
            marker: { color: BLUE, size: 7, opacity: 0.8 },
            hovertemplate: 'Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>',
          },
          { x: [mn, mx], y: [mn, mx], type: 'scatter', mode: 'lines', line: { color: GREEN, dash: 'dash' }, showlegend: false },
        ];
        return (
          <>
            <StatBar r2={d.r2} p={d.p_value} n={d.n} extra={[['R² adj', d.r2_adj.toFixed(4)]]} />
            <PlotlyChart data={data} layout={{ xaxis: { title: `Actual ${colLabel(outCol)}` } as Partial<import('plotly.js').LayoutAxis>, yaxis: { title: `Predicted ${colLabel(outCol)}` } as Partial<import('plotly.js').LayoutAxis> }} height={450} />
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#E6EDF3', marginBottom: 10 }}>Coefficients</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, color: '#8B949E' }}>
                <thead><tr style={{ borderBottom: '1px solid #21262D' }}>{['Variable', 'Coefficient', 'p-value', 'Significant'].map(h => <th key={h} style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600, color: '#6E7681', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>)}</tr></thead>
                <tbody>{preds.map(p => <tr key={p} style={{ borderBottom: '1px solid #21262D' }}>
                  <td style={{ padding: '7px 8px', color: '#E6EDF3' }}>{colLabel(p)}</td>
                  <td style={{ padding: '7px 8px', fontFamily: 'IBM Plex Mono, monospace' }}>{d.coefficients[p]?.toFixed(4)}</td>
                  <td style={{ padding: '7px 8px', fontFamily: 'IBM Plex Mono, monospace' }}>{d.p_values[p]?.toFixed(4)}</td>
                  <td style={{ padding: '7px 8px', color: (d.p_values[p] ?? 1) < 0.05 ? '#10B981' : '#484F58' }}>{(d.p_values[p] ?? 1) < 0.05 ? '✓' : '✗'}</td>
                </tr>)}</tbody>
              </table>
            </div>
          </>
        );
      })()}

      {result.type === 'anomaly' && (() => {
        const d = result.data as AnomalyResult;
        const varA = (payload as { var_a?: string }).var_a ?? '';
        const data: Data[] = [
          { x: d.series.map(p => p.date), y: d.series.map(p => p.value), type: 'scatter', mode: 'lines', name: colLabel(varA), line: { color: BLUE } },
          { x: d.rolling_mean.filter(p => p.value != null).map(p => p.date), y: d.rolling_mean.filter(p => p.value != null).map(p => p.value as number), type: 'scatter', mode: 'lines', name: 'Baseline', line: { color: GREEN, dash: 'dot' } },
          ...(d.anomalies.length ? [{
            x: d.anomalies.map(p => p.date), y: d.anomalies.map(p => p.value), type: 'scatter' as const, mode: 'markers' as const, name: 'Anomaly',
            text: d.anomalies.map(p => p.date) as string[],
            hovertemplate: '%{text}: %{y:.2f}<extra></extra>',
            marker: { color: RED, size: 11, symbol: 'circle-open' as const, line: { width: 2, color: RED } },
          }] : []),
        ];
        return (
          <>
            <div className="metric-pill" style={{ display: 'inline-block', marginBottom: 12 }}>
              <div className="label">Anomalies</div>
              <div className="value">{d.n_anomalies} days</div>
            </div>
            <PlotlyChart data={data} layout={{ xaxis: { title: 'Date' } as Partial<import('plotly.js').LayoutAxis>, yaxis: { title: colLabel(varA) } as Partial<import('plotly.js').LayoutAxis>, legend: { orientation: 'h', y: 1.08 } }} height={450} />
          </>
        );
      })()}

      {result.type === 'decompose' && (() => {
        const d = result.data as DecomposeResult;
        const varA = (payload as { var_a?: string }).var_a ?? '';
        const makeData = (pts: Array<{ date: string; value: number | null }>, color: string): Data => ({
          x: pts.map(p => p.date), y: pts.map(p => p.value), type: 'scatter', mode: 'lines',
          line: { color }, showlegend: false,
        });
        const subplots = [
          { label: 'Observed', pts: d.observed, color: BLUE },
          { label: 'Trend',    pts: d.trend,    color: GREEN },
          { label: 'Seasonal', pts: d.seasonal, color: ORANGE },
          { label: 'Residual', pts: d.residual, color: RED },
        ];
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {subplots.map(({ label, pts, color }) => (
              <div key={label}>
                <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 4, fontFamily: 'IBM Plex Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label} — {colLabel(varA)}</div>
                <PlotlyChart data={[makeData(pts, color)]} height={160} />
              </div>
            ))}
          </div>
        );
      })()}

      {SAVEABLE.has(payload.type) && (
        <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid #21262D' }}>
          {saved ? (
            <div style={{ color: '#10B981', fontSize: 13 }}>✓ Saved to Findings</div>
          ) : (
            <button className="btn-secondary" onClick={save} disabled={saving}>
              {saving ? 'Saving…' : 'Save to Findings'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
