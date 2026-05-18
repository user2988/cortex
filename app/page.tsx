'use client';

import { useEffect, useState } from 'react';
import KpiStrip from '@/components/dashboard/KpiStrip';
import RollingStats from '@/components/dashboard/RollingStats';
import SleepSection from '@/components/dashboard/SleepSection';
import CardioSection from '@/components/dashboard/CardioSection';
import ActivitySection from '@/components/dashboard/ActivitySection';
import IntelligenceSection from '@/components/dashboard/IntelligenceSection';
import RecommendationsSection from '@/components/dashboard/RecommendationsSection';
import type { BiometricRow, Finding, Experiment, Recommendation } from '@/lib/types';

function streak(rows: BiometricRow[], col: keyof BiometricRow, threshold: number, dir: 'above' | 'below'): number {
  let count = 0;
  for (let i = rows.length - 1; i >= 0; i--) {
    const v = rows[i][col];
    if (v == null) break;
    const meets = dir === 'above' ? Number(v) >= threshold : Number(v) <= threshold;
    if (meets) count++; else break;
  }
  return count;
}

function latest<T extends keyof BiometricRow>(rows: BiometricRow[], col: T): number | null {
  for (let i = rows.length - 1; i >= 0; i--) {
    const v = rows[i][col];
    if (v != null) return Number(v);
  }
  return null;
}

export default function Dashboard() {
  const [rows, setRows]       = useState<BiometricRow[]>([]);
  const [findings, setFin]    = useState<Finding[]>([]);
  const [exps, setExps]       = useState<Experiment[]>([]);
  const [recs, setRecs]       = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/data?days=90').then(r => r.json()),
      fetch('/api/findings').then(r => r.json()),
      fetch('/api/experiments').then(r => r.json()),
      fetch('/api/scores').then(r => r.json()),
    ]).then(([bio, fin, exp, scores]) => {
      setRows(Array.isArray(bio) ? bio : []);
      setFin(Array.isArray(fin) ? fin : []);
      setExps(Array.isArray(exp) ? exp : []);
      setRecs(scores?.recommendations ?? []);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 12 }}>
        <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 13, color: '#484F58' }}>Loading Cortex…</div>
      </div>
    );
  }

  // Header
  const today = new Date();
  const todayStr = today.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric', year: 'numeric' });
  let syncColor = '#484F58', syncText = 'No data';
  if (rows.length > 0) {
    const last = new Date(rows[rows.length - 1].date);
    const hoursAgo = (Date.now() - last.getTime()) / 3600000;
    const daysAgo  = Math.floor(hoursAgo / 24);
    syncColor = hoursAgo < 6 ? '#10B981' : daysAgo === 0 ? '#F59E0B' : '#EF4444';
    syncText  = daysAgo === 0 ? `Synced ${hoursAgo.toFixed(0)}h ago` : daysAgo === 1 ? 'Synced yesterday' : `Synced ${daysAgo} days ago`;
  }

  const nBio  = rows.length;
  const nFind = findings.filter(f => !f.pinned).length;
  const nExp  = exps.length;

  // Streaks & last night
  const stkSteps = streak(rows, 'steps', 8000, 'above');
  const stkSleep = streak(rows, 'sleep_efficiency_pct', 80, 'above');
  const durNow   = latest(rows, 'sleep_duration_min');
  const effNow   = latest(rows, 'sleep_efficiency_pct');
  const hrvNow   = latest(rows, 'hrv_ms');
  const rhrNow   = latest(rows, 'rhr_bpm');

  const stkSc = stkSteps >= 3 ? '#2DD4BF' : stkSteps >= 1 ? '#F59E0B' : '#484F58';
  const stkEc = stkSleep >= 3 ? '#2DD4BF' : stkSleep >= 1 ? '#F59E0B' : '#484F58';

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
        <div>
          <span style={{ fontFamily: 'Inter', fontSize: 22, fontWeight: 600, color: '#E6EDF3', letterSpacing: '-0.02em' }}>Cortex</span>
          <span style={{ fontFamily: 'Inter', fontSize: 13, color: '#484F58', marginLeft: 12 }}>{todayStr}</span>
        </div>
        <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, color: syncColor, paddingTop: 4 }}>● {syncText}</div>
      </div>

      {/* Status bar */}
      <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 10, color: '#484F58', marginBottom: 16, paddingBottom: 6, borderBottom: '1px solid #21262D' }}>
        biometrics {nBio}d &nbsp;·&nbsp; findings {nFind} &nbsp;·&nbsp; experiments {nExp}
      </div>

      {/* Streaks + Last Night */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 16 }}>
        <div style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 4, padding: '12px 10px' }}>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 8, fontWeight: 600, letterSpacing: '.1em', textTransform: 'uppercase', color: '#484F58', marginBottom: 6 }}>Streaks</div>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 12, color: '#8B949E' }}>
            Steps ≥8k &nbsp;<span style={{ color: stkSc, fontSize: 14 }}>{stkSteps}d</span>
          </div>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 12, color: '#8B949E', marginTop: 4 }}>
            Sleep ≥80% &nbsp;<span style={{ color: stkEc, fontSize: 14 }}>{stkSleep}d</span>
          </div>
        </div>
        <div style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 4, padding: '12px 10px' }}>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 8, fontWeight: 600, letterSpacing: '.1em', textTransform: 'uppercase', color: '#484F58', marginBottom: 6 }}>Last Night</div>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, color: '#8B949E' }}>
            Sleep &nbsp;<span style={{ color: '#E6EDF3' }}>{durNow != null ? `${(durNow / 60).toFixed(1)}h` : '—'}</span>
            &nbsp; Eff &nbsp;<span style={{ color: '#E6EDF3' }}>{effNow != null ? `${effNow.toFixed(0)}%` : '—'}</span>
          </div>
          <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, color: '#8B949E', marginTop: 4 }}>
            HRV &nbsp;<span style={{ color: '#2DD4BF' }}>{hrvNow != null ? `${hrvNow.toFixed(0)}ms` : '—'}</span>
            &nbsp; RHR &nbsp;<span style={{ color: '#EF4444' }}>{rhrNow != null ? `${rhrNow.toFixed(0)}bpm` : '—'}</span>
          </div>
        </div>
      </div>

      {/* Yesterday's Stats */}
      <div className="section-header">Yesterday&apos;s Stats</div>
      <div style={{ height: 8 }} />
      <KpiStrip rows={rows} />
      <RollingStats rows={rows} />

      <SleepSection rows={rows} />
      <div style={{ height: 16 }} />
      <CardioSection rows={rows} />
      <div style={{ height: 16 }} />
      <ActivitySection rows={rows} />
      <div style={{ height: 16 }} />
      <IntelligenceSection findings={findings} experiments={exps} />
      <div style={{ height: 16 }} />
      <RecommendationsSection recommendations={recs} />
      <div style={{ height: 40 }} />
    </div>
  );
}
