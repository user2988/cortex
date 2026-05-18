'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import PlotlyChart from '@/components/PlotlyChart';
import { COL_LABELS } from '@/lib/labels';
import type { Experiment } from '@/lib/types';
import type { ExperimentAnalysisResult } from '@/lib/stats';
import type { Data } from 'plotly.js';

interface PageData {
  experiment: Experiment;
  analysis: ExperimentAnalysisResult | { error: string };
}

export default function ExperimentDetailPage() {
  const { id }   = useParams() as { id: string };
  const router   = useRouter();
  const [data, setData]       = useState<PageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView]       = useState<'experiment' | 'gradient'>('experiment');
  const [showAll, setShowAll] = useState(true);
  const [interpreting, setInterpreting] = useState(false);
  const [interpretation, setInterp] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/experiments/${id}`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); });
  }, [id]);

  if (loading) return <div style={{ color: '#484F58', fontSize: 13, padding: 40 }}>Loading…</div>;
  if (!data) return <div style={{ color: '#EF4444', fontSize: 13, padding: 40 }}>Not found.</div>;

  const { experiment: exp, analysis: analysisRaw } = data;
  if ('error' in analysisRaw) {
    return (
      <div>
        <button className="btn-ghost" onClick={() => router.back()} style={{ marginBottom: 16 }}>← Back</button>
        <div style={{ color: '#F59E0B', fontSize: 13 }}>{analysisRaw.error}</div>
      </div>
    );
  }

  const analysis = analysisRaw as import('@/lib/stats').ExperimentAnalysisResult;
  const a = COL_LABELS[exp.variable_a] ?? exp.variable_a;
  const b = COL_LABELS[exp.variable_b] ?? exp.variable_b;
  const stColor = exp.is_complete ? '#F59E0B' : '#2DD4BF';
  const stLabel = exp.is_complete
    ? `COMPLETE — DAYS 1 TO ${exp.duration_days}`
    : `ACTIVE — DAYS 1 TO ${exp.elapsed_days} OF ${exp.duration_days}`;

  // Chart traces
  const traces: Data[] = [];
  const allA = analysis.all_paired;
  const allB = analysis.all_paired_b;

  if (view === 'experiment') {
    if (showAll && analysis.pre.length > 0) {
      traces.push({
        x: analysis.pre.map(p => p.value), y: analysis.pre_b.map(p => p.value), type: 'scatter', mode: 'markers',
        name: 'Before experiment',
        text: analysis.pre.map(p => p.date) as string[],
        hovertemplate: '%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>',
        marker: { color: 'rgba(150,150,150,0.45)', size: 7 },
      });
    }
    const dotColor = analysis.coefficient >= 0 ? 'rgba(0,230,120,1.0)' : '#ff4444';
    traces.push({
      x: analysis.during.map(p => p.value), y: analysis.during_b.map(p => p.value), type: 'scatter', mode: 'markers',
      name: 'During experiment',
      text: analysis.during.map(p => p.date) as string[],
      hovertemplate: '%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>',
      marker: { color: dotColor, size: 9, opacity: 0.9 },
    });
  } else {
    const n = allA.length;
    const positions = n > 1 ? allA.map((_, i) => i / (n - 1)) : [1];
    traces.push({
      x: allA.map(p => p.value), y: allB.map(p => p.value), type: 'scatter', mode: 'markers',
      name: 'All data',
      text: allA.map(p => p.date) as string[],
      hovertemplate: '%{text}<br>%{x:.2f} → %{y:.2f}<extra></extra>',
      marker: {
        color: positions,
        colorscale: [[0, 'rgba(120,120,120,0.35)'], [0.5, 'rgba(120,140,220,0.8)'], [1, 'rgba(0,230,120,1.0)']],
        size: 9, showscale: false,
      } as Record<string, unknown>,
    });
  }

  // OLS line
  if (allA.length > 0) {
    const xMin = Math.min(...allA.map(p => p.value));
    const xMax = Math.max(...allA.map(p => p.value));
    const xs = [xMin, xMax];
    traces.push({
      x: xs, y: xs.map(x => analysis.full_slope * x + analysis.full_intercept),
      type: 'scatter', mode: 'lines',
      line: { color: '#ff7f0e', dash: 'dash', width: 2 },
      showlegend: false,
    });
  }

  async function generateInterpretation() {
    setInterpreting(true);
    const res = await fetch('/api/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        var_a: exp.variable_a, var_b: exp.variable_b,
        r2: analysis.r2, p_value: analysis.p_value, coefficient: analysis.coefficient,
        lag: exp.lag_days, n: analysis.n,
        pre_avg_a: analysis.pre_avg_a, pre_avg_b: analysis.pre_avg_b,
        during_avg_a: analysis.during_avg_a, during_avg_b: analysis.during_avg_b,
      }),
    });
    const d = await res.json();
    const text = d.interpretation ?? d.error ?? '';
    setInterp(text);
    if (d.interpretation) {
      await fetch(`/api/experiments/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interpretation: text }),
      });
    }
    setInterpreting(false);
  }

  const storedInterp = exp.interpretation;

  return (
    <div>
      <button className="btn-ghost" onClick={() => router.push('/experiments')} style={{ marginBottom: 16 }}>← Back</button>

      <div style={{ color: stColor, fontSize: '0.85em', fontWeight: 600, marginBottom: 6 }}>{stLabel}</div>
      <h1 style={{ fontFamily: 'Inter', fontSize: 22, fontWeight: 600, color: '#E6EDF3', letterSpacing: '-0.02em', marginBottom: 4 }}>{exp.name}</h1>
      <div style={{ color: '#6E7681', fontSize: 13, marginBottom: 16 }}>
        {a} → {b}{exp.lag_days ? ` · ${exp.lag_days}-day lag` : ''}
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 12 }}>
        {[['R²', analysis.r2.toFixed(4)], ['p-value', analysis.p_value.toFixed(4)], ['Coefficient', analysis.coefficient.toFixed(4)]].map(([l, v]) => (
          <div key={l} className="metric-pill">
            <div className="label">{l}</div>
            <div className="value">{v}</div>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 13, fontWeight: 600, color: '#8B949E', marginBottom: 4 }}>{analysis.label}</div>
      <div style={{ fontSize: 12, color: '#484F58', marginBottom: 16 }}>Based on {analysis.n} days of experiment data</div>

      {/* Chart view toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {(['experiment', 'gradient'] as const).map(v => (
          <button key={v} className={view === v ? 'btn-primary' : 'btn-secondary'} onClick={() => setView(v)} style={{ padding: '5px 12px', fontSize: 12 }}>
            {v === 'experiment' ? 'Experiment view' : 'Gradient view'}
          </button>
        ))}
        {view === 'experiment' && analysis.pre.length > 0 && (
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#8B949E', cursor: 'pointer', marginLeft: 8 }}>
            <input type="checkbox" checked={showAll} onChange={e => setShowAll(e.target.checked)} />
            Show full history
          </label>
        )}
      </div>
      {view === 'gradient' && (
        <div style={{ fontSize: 12, color: '#484F58', marginBottom: 8 }}>● Oldest data &nbsp;&nbsp; ● Mid-period &nbsp;&nbsp; ● Most recent</div>
      )}

      <PlotlyChart
        data={traces}
        layout={{ xaxis_title: a, yaxis_title: b, legend: { orientation: 'h', y: 1.06 } } as Partial<import('plotly.js').Layout>}
        height={460}
      />

      {/* Interpretation */}
      {exp.is_complete && (
        <div style={{ marginTop: 20 }}>
          {storedInterp || interpretation ? (
            <div style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 6, padding: 14, fontSize: 13, color: '#8B949E' }}>
              <strong style={{ color: '#E6EDF3' }}>What this shows:</strong>{' '}{storedInterp ?? interpretation}
            </div>
          ) : (
            <button className="btn-secondary" onClick={generateInterpretation} disabled={interpreting}>
              {interpreting ? 'Generating interpretation…' : 'Generate interpretation'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
