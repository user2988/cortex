'use client';

import Link from 'next/link';
import { COL_LABELS } from '@/lib/labels';
import type { Experiment } from '@/lib/types';

interface Props {
  exp: Experiment;
  onDelete: (id: number) => void;
}

export default function ExperimentCard({ exp, onDelete }: Props) {
  const a        = COL_LABELS[exp.variable_a] ?? exp.variable_a;
  const b        = COL_LABELS[exp.variable_b] ?? exp.variable_b;
  const stColor  = exp.is_complete ? '#10B981' : '#F59E0B';
  const stLabel  = exp.is_complete ? 'Complete' : `Day ${exp.elapsed_days} of ${exp.duration_days}`;

  return (
    <div style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 6, padding: '12px 14px', marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
        <div>
          <div style={{ fontFamily: 'Inter', fontSize: 14, fontWeight: 600, color: '#E6EDF3' }}>{exp.name}</div>
          <div style={{ fontFamily: 'Inter', fontSize: 12, color: '#6E7681', marginTop: 2 }}>
            {a} → {b}{exp.lag_days ? ` · lag ${exp.lag_days}d` : ''}
          </div>
        </div>
        <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, color: stColor, textAlign: 'right' }}>{stLabel}</div>
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
        <Link href={`/experiments/${exp.id}`}>
          <button className="btn-secondary" style={{ padding: '5px 14px', fontSize: 12 }}>View details</button>
        </Link>
        <button className="btn-ghost" onClick={() => onDelete(exp.id)} style={{ fontSize: 12 }}>Delete</button>
      </div>
    </div>
  );
}
