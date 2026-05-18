'use client';

import dynamic from 'next/dynamic';
import type { Data, Layout, Config } from 'plotly.js';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false, loading: () => (
  <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#484F58', fontSize: 12 }}>
    Loading chart…
  </div>
)});

export const BASE_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: { family: 'Inter, sans-serif', size: 11, color: '#8B949E' },
  margin: { l: 6, r: 6, t: 36, b: 6 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.04)', linecolor: '#30363D',
    tickcolor: '#30363D', zeroline: false,
    tickfont: { size: 9, family: 'IBM Plex Mono, monospace' },
  } as Partial<Layout['xaxis']>,
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.04)', linecolor: '#30363D',
    tickcolor: '#30363D', zeroline: false,
    tickfont: { size: 9, family: 'IBM Plex Mono, monospace' },
  } as Partial<Layout['yaxis']>,
  hoverlabel: {
    bgcolor: '#1C2230', bordercolor: '#30363D',
    font: { family: 'Inter', size: 11, color: '#E6EDF3' },
  },
};

export const PLOT_CONFIG: Partial<Config> = { displayModeBar: false, responsive: true };

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  height?: number;
  style?: React.CSSProperties;
}

export default function PlotlyChart({ data, layout = {}, height = 220, style }: Props) {
  const merged: Partial<Layout> = {
    ...BASE_LAYOUT,
    ...layout,
    height,
    xaxis: { ...BASE_LAYOUT.xaxis, ...(layout.xaxis ?? {}) } as Partial<Layout['xaxis']>,
    yaxis: { ...BASE_LAYOUT.yaxis, ...(layout.yaxis ?? {}) } as Partial<Layout['yaxis']>,
  };
  return (
    <Plot
      data={data}
      layout={merged}
      config={PLOT_CONFIG}
      style={{ width: '100%', ...style }}
      useResizeHandler
    />
  );
}
