'use client';

import { useState } from 'react';
import { VAR_A_TREE, VAR_B_TREE, A_CATS, A_SUBS, B_CATS, B_SUBS } from '@/lib/labels';

interface Props { onCreated: () => void }

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function NewExperimentForm({ onCreated }: Props) {
  const [aCat, setACat] = useState(A_CATS[0]);
  const [aSub, setASub] = useState(A_SUBS[A_CATS[0]][0]);
  const [bCat, setBCat] = useState(B_CATS[0]);
  const [bSub, setBSub] = useState(B_SUBS[B_CATS[0]][0]);

  const aGrp = `${aCat}  ·  ${aSub}`;
  const bGrp = `${bCat}  ·  ${bSub}`;
  const aVars = VAR_A_TREE[aGrp] ?? [];
  const bVars = VAR_B_TREE[bGrp] ?? [];

  const [varA, setVarA]     = useState(aVars[0] ?? '');
  const [varB, setVarB]     = useState(bVars[0] ?? '');
  const [name, setName]     = useState('');
  const [lag, setLag]       = useState(0);
  const [method, setMethod] = useState('pearson');
  const [duration, setDur]  = useState(30);
  const [startDate, setSd]  = useState(today());
  const [submitting, setSub] = useState(false);
  const [error, setError]   = useState('');
  const [success, setSuccess] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) { setError('Give the experiment a name.'); return; }
    setSub(true); setError('');
    const res = await fetch('/api/experiments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, variable_a: varA, variable_b: varB, lag_days: lag, method, start_date: startDate, duration_days: duration }),
    });
    setSub(false);
    if (res.ok) { setSuccess(true); setName(''); onCreated(); }
    else { const d = await res.json(); setError(d.error ?? 'Failed'); }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Variable A */}
        <div>
          <label style={{ fontSize: 11, color: '#6E7681', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Variable A — Input / Driver</label>
          <select value={aCat} onChange={e => { setACat(e.target.value); setASub(A_SUBS[e.target.value][0]); }} style={{ marginBottom: 6 }}>
            {A_CATS.map(c => <option key={c}>{c}</option>)}
          </select>
          <select value={aSub} onChange={e => setASub(e.target.value)} style={{ marginBottom: 6 }}>
            {A_SUBS[aCat]?.map(s => <option key={s}>{s}</option>)}
          </select>
          <select value={varA} onChange={e => setVarA(e.target.value)}>
            {(VAR_A_TREE[`${aCat}  ·  ${aSub}`] ?? []).map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
        {/* Variable B */}
        <div>
          <label style={{ fontSize: 11, color: '#6E7681', display: 'block', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Variable B — Output / Target</label>
          <select value={bCat} onChange={e => { setBCat(e.target.value); setBSub(B_SUBS[e.target.value][0]); }} style={{ marginBottom: 6 }}>
            {B_CATS.map(c => <option key={c}>{c}</option>)}
          </select>
          <select value={bSub} onChange={e => setBSub(e.target.value)} style={{ marginBottom: 6 }}>
            {B_SUBS[bCat]?.map(s => <option key={s}>{s}</option>)}
          </select>
          <select value={varB} onChange={e => setVarB(e.target.value)}>
            {(VAR_B_TREE[`${bCat}  ·  ${bSub}`] ?? []).map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      </div>

      <div style={{ height: 1, background: '#21262D', margin: '16px 0' }} />

      <div style={{ marginBottom: 12 }}>
        <label style={{ fontSize: 12, color: '#8B949E', display: 'block', marginBottom: 6 }}>Hypothesis / name</label>
        <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="e.g. active minutes impact on deep sleep" />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
        <div>
          <label style={{ fontSize: 12, color: '#8B949E', display: 'block', marginBottom: 6 }}>Lag (days)</label>
          <select value={lag} onChange={e => setLag(Number(e.target.value))}>
            {[0,1,2,3].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: '#8B949E', display: 'block', marginBottom: 6 }}>Method</label>
          <select value={method} onChange={e => setMethod(e.target.value)}>
            <option value="pearson">Pearson</option>
            <option value="spearman">Spearman</option>
          </select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: '#8B949E', display: 'block', marginBottom: 6 }}>Duration (days)</label>
          <input type="number" value={duration} min={14} max={365} onChange={e => setDur(Number(e.target.value))} />
        </div>
        <div>
          <label style={{ fontSize: 12, color: '#8B949E', display: 'block', marginBottom: 6 }}>Start date</label>
          <input type="date" value={startDate} onChange={e => setSd(e.target.value)} />
        </div>
      </div>

      {error && <div style={{ color: '#EF4444', fontSize: 12, marginBottom: 10 }}>{error}</div>}
      {success && <div style={{ color: '#10B981', fontSize: 12, marginBottom: 10 }}>Experiment created.</div>}

      <button type="submit" className="btn-primary" disabled={submitting}>
        {submitting ? 'Creating…' : 'Create experiment'}
      </button>
    </form>
  );
}
