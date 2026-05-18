'use client';

import Link from 'next/link';
import { COL_LABELS } from '@/lib/labels';
import type { Finding, Experiment } from '@/lib/types';

interface Props {
  findings: Finding[];
  experiments: Experiment[];
}

export default function IntelligenceSection({ findings, experiments }: Props) {
  const autoFindings = findings.filter(f => !f.pinned).sort((a, b) => b.r_squared - a.r_squared).slice(0, 6);
  const activeExps   = experiments.filter(e => !e.is_complete).slice(0, 4);

  return (
    <div>
      <div className="section-header">Intelligence</div>
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 24 }}>
        <div>
          <div className="chart-label" style={{ marginBottom: 8 }}>Top Correlations</div>
          {autoFindings.length === 0 ? (
            <div className="empty-panel">Patterns computed weekly — check back after Sunday&apos;s analysis run.</div>
          ) : autoFindings.map(f => {
            const a   = COL_LABELS[f.variable_a] ?? f.variable_a;
            const b   = f.variable_b ? (COL_LABELS[f.variable_b] ?? f.variable_b) : '—';
            const r2c = f.r_squared >= 0.5 ? '#2DD4BF' : f.r_squared >= 0.3 ? '#F59E0B' : '#484F58';
            const dc  = f.coefficient > 0 ? '#10B981' : '#EF4444';
            return (
              <div key={f.id} className="finding-row">
                <div style={{ fontFamily: 'Inter', fontSize: 12, color: '#E6EDF3' }}>
                  {a}<span style={{ color: '#484F58' }}> → </span>{b}
                  <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, color: '#484F58', marginLeft: 8 }}>lag +{f.lag_days}d</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
                  <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, color: dc }}>{f.coefficient > 0 ? '↑' : '↓'}</span>
                  <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 14, fontWeight: 600, color: r2c }}>R²{f.r_squared.toFixed(2)}</span>
                </div>
              </div>
            );
          })}
        </div>
        <div>
          <div className="chart-label" style={{ marginBottom: 8 }}>Active Experiments</div>
          {activeExps.length === 0 ? (
            <div className="empty-panel">No active experiments. <Link href="/experiments" style={{ color: '#2DD4BF' }}>Create one →</Link></div>
          ) : activeExps.map(exp => {
            const a   = COL_LABELS[exp.variable_a] ?? exp.variable_a;
            const b   = COL_LABELS[exp.variable_b] ?? exp.variable_b;
            const pct = Math.min(exp.elapsed_days / exp.duration_days, 1);
            const bar = Math.round(pct * 18);
            return (
              <div key={exp.id} className="exp-card">
                <div style={{ fontFamily: 'Inter', fontSize: 12, color: '#E6EDF3', marginBottom: 3 }}>{exp.name}</div>
                <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, color: '#484F58', marginBottom: 6 }}>{a} → {b}</div>
                <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 10, color: '#2DD4BF' }}>{'█'.repeat(bar)}{'░'.repeat(18 - bar)}</div>
                <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, color: '#484F58', marginTop: 3 }}>Day {exp.elapsed_days} of {exp.duration_days}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
