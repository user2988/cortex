'use client';

import { useEffect, useState } from 'react';
import ExperimentCard from '@/components/experiments/ExperimentCard';
import NewExperimentForm from '@/components/experiments/NewExperimentForm';
import type { Experiment } from '@/lib/types';

export default function ExperimentsPage() {
  const [exps, setExps]   = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<'new' | 'active' | 'past'>('active');

  async function load() {
    const data = await fetch('/api/experiments').then(r => r.json());
    setExps(Array.isArray(data) ? data : []);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  async function handleDelete(id: number) {
    await fetch(`/api/experiments/${id}`, { method: 'DELETE' });
    setExps(prev => prev.filter(e => e.id !== id));
  }

  const active = exps.filter(e => !e.is_complete);
  const past   = exps.filter(e => e.is_complete);

  return (
    <div>
      <h1 style={{ fontFamily: 'Inter', fontSize: 22, fontWeight: 600, color: '#E6EDF3', letterSpacing: '-0.02em', marginBottom: 20 }}>Experiments</h1>

      <div className="tab-bar">
        <div className={`tab-item ${tab === 'new' ? 'active' : ''}`} onClick={() => setTab('new')}>New</div>
        <div className={`tab-item ${tab === 'active' ? 'active' : ''}`} onClick={() => setTab('active')}>Active ({active.length})</div>
        <div className={`tab-item ${tab === 'past' ? 'active' : ''}`} onClick={() => setTab('past')}>Past ({past.length})</div>
      </div>

      {tab === 'new' && (
        <div>
          {active.length >= 3 ? (
            <div style={{ color: '#F59E0B', fontSize: 13, marginBottom: 12 }}>
              You have 3 active experiments — complete or delete one before adding another.
            </div>
          ) : (
            <NewExperimentForm onCreated={() => { load(); setTab('active'); }} />
          )}
        </div>
      )}

      {tab === 'active' && (
        <div>
          {loading ? (
            <div style={{ color: '#484F58', fontSize: 12 }}>Loading…</div>
          ) : active.length === 0 ? (
            <div className="empty-panel">No active experiments. <span style={{ color: '#2DD4BF', cursor: 'pointer' }} onClick={() => setTab('new')}>Create one →</span></div>
          ) : (
            active.map(exp => <ExperimentCard key={exp.id} exp={exp} onDelete={handleDelete} />)
          )}
        </div>
      )}

      {tab === 'past' && (
        <div>
          {loading ? (
            <div style={{ color: '#484F58', fontSize: 12 }}>Loading…</div>
          ) : past.length === 0 ? (
            <div className="empty-panel">No completed experiments yet.</div>
          ) : (
            past.map(exp => <ExperimentCard key={exp.id} exp={exp} onDelete={handleDelete} />)
          )}
        </div>
      )}
    </div>
  );
}
