'use client';

import { useState } from 'react';
import { VAR_A_TREE, VAR_B_TREE, VAR_TREE, A_CATS, A_SUBS, B_CATS, B_SUBS, ALL_CATS, ALL_SUBS } from '@/lib/labels';
import type { AnalysisPayload } from '@/lib/types';

const ANALYSIS_TYPES = [
  'Pearson Correlation', 'Spearman Correlation', 'Lagged Correlation',
  'Rolling Average', 'Multiple OLS Regression', '30-Day Trend (OLS)',
  'Anomaly Detection', 'Decomposition',
];
const SINGLE_VAR = new Set(['30-Day Trend (OLS)', 'Anomaly Detection', 'Decomposition']);
const MULTI_PRED = new Set(['Multiple OLS Regression']);
const DAYS_MAP: Record<string, number> = { 'Last 30 days': 30, 'Last 60 days': 60, 'Last 90 days': 90, 'All data': 0 };

function VarPicker({ id, cats, subsMap, tree, label, value, onChange }: {
  id: string; cats: string[]; subsMap: Record<string, string[]>; tree: Record<string, string[]>;
  label: string; value: string; onChange: (v: string) => void;
}) {
  const [cat, setCat] = useState(cats[0]);
  const [sub, setSub] = useState(subsMap[cats[0]][0]);
  const grp = `${cat}  ·  ${sub}`;
  const vars = tree[grp] ?? [];
  return (
    <div>
      <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>{label}</div>
      <select value={cat} onChange={e => { setCat(e.target.value); setSub(subsMap[e.target.value]?.[0] ?? ''); onChange(tree[`${e.target.value}  ·  ${subsMap[e.target.value]?.[0]}`]?.[0] ?? ''); }} style={{ marginBottom: 6 }}>
        {cats.map(c => <option key={c}>{c}</option>)}
      </select>
      <select value={sub} onChange={e => { setSub(e.target.value); onChange(tree[`${cat}  ·  ${e.target.value}`]?.[0] ?? ''); }} style={{ marginBottom: 6 }}>
        {(subsMap[cat] ?? []).map(s => <option key={s}>{s}</option>)}
      </select>
      <select value={value} onChange={e => onChange(e.target.value)}>
        {vars.map(v => <option key={v} value={v}>{v}</option>)}
      </select>
    </div>
  );
}

interface Props {
  onRun: (payload: AnalysisPayload) => void;
  running: boolean;
}

export default function AnalysisForm({ onRun, running }: Props) {
  const [type, setType]         = useState(ANALYSIS_TYPES[0]);
  const [daysLabel, setDays]    = useState('All data');
  const [varA, setVarA]         = useState(Object.values(VAR_A_TREE)[0][0]);
  const [varB, setVarB]         = useState(Object.values(VAR_B_TREE)[0][0]);
  const [lag, setLag]           = useState(1);
  const [corrMethod, setCorrM]  = useState<'pearson' | 'spearman'>('pearson');
  const [window, setWindow]     = useState(7);
  const [period, setPeriod]     = useState(7);
  const [aWindow, setAWindow]   = useState(30);
  const [threshold, setThresh]  = useState(1.5);
  const [predictors, setPreds]  = useState<string[]>([Object.values(VAR_A_TREE)[0][0]]);
  const [outcome, setOutcome]   = useState(Object.values(VAR_B_TREE)[0][0]);

  const isSingle = SINGLE_VAR.has(type);
  const isMulti  = MULTI_PRED.has(type);
  const days     = DAYS_MAP[daysLabel];

  function buildPayload(): AnalysisPayload {
    switch (type) {
      case 'Pearson Correlation':     return { type, var_a: varA, var_b: varB, days };
      case 'Spearman Correlation':    return { type, var_a: varA, var_b: varB, days };
      case 'Lagged Correlation':      return { type, var_a: varA, var_b: varB, lag, method: corrMethod, days };
      case 'Rolling Average':         return { type, var_a: varA, var_b: varB, window, method: corrMethod, days };
      case '30-Day Trend (OLS)':      return { type, var_a: varA, days };
      case 'Multiple OLS Regression': return { type, predictors, outcome, days };
      case 'Anomaly Detection':       return { type, var_a: varA, window: aWindow, threshold, days };
      case 'Decomposition':           return { type, var_a: varA, period, days };
      default:                        return { type: 'Pearson Correlation', var_a: varA, var_b: varB, days };
    }
  }

  // Predictor multiselect for multiple OLS
  const [aOlsCat, setAOlsCat] = useState(A_CATS[0]);
  const [aOlsSub, setAOlsSub] = useState(A_SUBS[A_CATS[0]][0]);
  const aOlsVars = VAR_A_TREE[`${aOlsCat}  ·  ${aOlsSub}`] ?? [];

  return (
    <div style={{ padding: '16px 0' }}>
      {/* Analysis type */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Analysis Type</div>
        <select value={type} onChange={e => setType(e.target.value)}>
          {ANALYSIS_TYPES.map(t => <option key={t}>{t}</option>)}
        </select>
      </div>

      {/* Data range */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Data Range</div>
        <select value={daysLabel} onChange={e => setDays(e.target.value)}>
          {Object.keys(DAYS_MAP).map(k => <option key={k}>{k}</option>)}
        </select>
      </div>

      {/* Type-specific params */}
      {type === 'Lagged Correlation' && (
        <div style={{ marginBottom: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Lag (days)</div>
            <select value={lag} onChange={e => setLag(Number(e.target.value))}>
              {[0,1,2,3].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Method</div>
            <select value={corrMethod} onChange={e => setCorrM(e.target.value as 'pearson' | 'spearman')}>
              <option value="pearson">Pearson</option>
              <option value="spearman">Spearman</option>
            </select>
          </div>
        </div>
      )}
      {type === 'Rolling Average' && (
        <div style={{ marginBottom: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Window (days)</div>
            <select value={window} onChange={e => setWindow(Number(e.target.value))}>
              {[7, 14].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Method</div>
            <select value={corrMethod} onChange={e => setCorrM(e.target.value as 'pearson' | 'spearman')}>
              <option value="pearson">Pearson</option>
              <option value="spearman">Spearman</option>
            </select>
          </div>
        </div>
      )}
      {type === 'Decomposition' && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Period (days)</div>
          <select value={period} onChange={e => setPeriod(Number(e.target.value))}>
            {[7, 14, 30].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      )}
      {type === 'Anomaly Detection' && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Baseline window: {aWindow} days</div>
            <input type="range" min={14} max={60} value={aWindow} onChange={e => setAWindow(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
          <div>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Threshold (SD): {threshold}</div>
            <input type="range" min={1} max={3} step={0.1} value={threshold} onChange={e => setThresh(Number(e.target.value))} style={{ width: '100%' }} />
          </div>
        </div>
      )}

      <div style={{ height: 1, background: '#21262D', margin: '12px 0' }} />

      {/* Variable pickers */}
      {isSingle && (
        <div style={{ marginBottom: 12 }}>
          <VarPicker id="sv" cats={ALL_CATS} subsMap={ALL_SUBS} tree={VAR_TREE} label="Variable" value={varA} onChange={setVarA} />
        </div>
      )}
      {isMulti && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, color: '#6E7681', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'IBM Plex Mono, monospace' }}>Predictors (Variable A)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 6 }}>
              <select value={aOlsCat} onChange={e => { setAOlsCat(e.target.value); setAOlsSub(A_SUBS[e.target.value][0]); }}>
                {A_CATS.map(c => <option key={c}>{c}</option>)}
              </select>
              <select value={aOlsSub} onChange={e => setAOlsSub(e.target.value)}>
                {(A_SUBS[aOlsCat] ?? []).map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
            <div style={{ maxHeight: 120, overflowY: 'auto', border: '1px solid #21262D', borderRadius: 4, padding: 4 }}>
              {aOlsVars.map(v => (
                <label key={v} style={{ display: 'flex', gap: 8, padding: '4px 6px', cursor: 'pointer', fontSize: 12, color: '#8B949E' }}>
                  <input type="checkbox" checked={predictors.includes(v)}
                    onChange={e => setPreds(prev => e.target.checked ? [...prev, v] : prev.filter(p => p !== v))} />
                  {v}
                </label>
              ))}
            </div>
          </div>
          <VarPicker id="ols_b" cats={B_CATS} subsMap={B_SUBS} tree={VAR_B_TREE} label="Outcome (Variable B)" value={outcome} onChange={setOutcome} />
        </div>
      )}
      {!isSingle && !isMulti && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
          <VarPicker id="a" cats={A_CATS} subsMap={A_SUBS} tree={VAR_A_TREE} label="Variable A — Input / Driver" value={varA} onChange={setVarA} />
          <VarPicker id="b" cats={B_CATS} subsMap={B_SUBS} tree={VAR_B_TREE} label="Variable B — Output / Target" value={varB} onChange={setVarB} />
        </div>
      )}

      <button className="btn-primary" onClick={() => onRun(buildPayload())} disabled={running} style={{ width: '100%', marginTop: 4 }}>
        {running ? 'Running…' : 'Run Analysis'}
      </button>
    </div>
  );
}
