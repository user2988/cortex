'use client';

import PlotlyChart from '@/components/PlotlyChart';
import type { Data } from 'plotly.js';
import type { DataPoint } from '@/lib/types';

interface Props {
  title: string;
  series: DataPoint[];
  color: string;
  fill: string;
  height?: number;
  refLine?: number;
  refLabel?: string;
}

export default function TrendCard({ title, series, color, fill, height = 190, refLine, refLabel }: Props) {
  if (series.length < 3) {
    return <div className="empty-panel" style={{ height }}>{title}<br /><span style={{ fontSize: 11 }}>No data yet.</span></div>;
  }

  const values = series.map(p => p.value);
  const dates  = series.map(p => p.date);
  const mean   = values.reduce((a, b) => a + b, 0) / values.length;
  const std    = Math.sqrt(values.reduce((a, x) => a + (x - mean) ** 2, 0) / values.length);

  // 7-day rolling average
  const rolling = values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - 6), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });

  // Anomalies (> 2 SD)
  const anomalyIdx = std > 0 ? values.reduce<number[]>((a, v, i) => {
    if (Math.abs(v - mean) > 2 * std) a.push(i);
    return a;
  }, []) : [];

  const data: Data[] = [
    {
      x: dates, y: values, type: 'scatter', mode: 'markers',
      marker: { color, size: 6, opacity: 0.55 },
      showlegend: false,
      hovertemplate: '%{x|%b %-d}: %{y:.1f}<extra></extra>',
    },
    {
      x: dates, y: rolling, type: 'scatter', mode: 'lines',
      line: { color, width: 2.5 },
      fill: 'tozeroy', fillcolor: fill,
      showlegend: false,
      hovertemplate: '%{x|%b %-d} (7d avg): %{y:.1f}<extra></extra>',
    },
  ];

  if (anomalyIdx.length > 0) {
    data.push({
      x: anomalyIdx.map(i => dates[i]),
      y: anomalyIdx.map(i => values[i]),
      type: 'scatter', mode: 'markers',
      marker: { color: '#F59E0B', size: 9, symbol: 'circle-open' as const, line: { width: 2, color: '#F59E0B' } },
      showlegend: false,
      hovertemplate: '%{x|%b %-d}: %{y:.1f} ⚠<extra></extra>',
    });
  }

  const chartLabel = title;
  const stat = `now ${values[values.length - 1]?.toFixed(1)} · min ${Math.min(...values).toFixed(1)} · avg ${mean.toFixed(1)} · max ${Math.max(...values).toFixed(1)}`;

  const layout: Partial<import('plotly.js').Layout> = { height };
  if (refLine != null) {
    (layout as Record<string, unknown>).shapes = [{
      type: 'line', x0: 0, x1: 1, xref: 'paper',
      y0: refLine, y1: refLine, yref: 'y',
      line: { color: '#484F58', width: 1, dash: 'dot' },
    }];
    (layout as Record<string, unknown>).annotations = [{
      x: 1, xref: 'paper', y: refLine, yref: 'y',
      text: refLabel, showarrow: false, xanchor: 'right',
      font: { color: '#6E7681', size: 9 },
    }];
  }

  return (
    <div>
      <div className="chart-label">{chartLabel}</div>
      <div className="chart-stats">{stat}</div>
      <PlotlyChart data={data} layout={layout} height={height} />
    </div>
  );
}
