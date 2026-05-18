'use client';

import type { Recommendation } from '@/lib/types';

const POSITIVE_METRICS = new Set([
  'steps', 'active_zone_min', 'very_active_min', 'fairly_active_min',
  'lightly_active_min', 'calories_burned', 'distance_km',
]);

const LABEL_MAP: Record<string, string> = {
  'lightly active minutes': 'light movement',
  'fairly active minutes':  'moderate exercise',
  'very active minutes':    'intense exercise',
  'active zone minutes':    'cardio time',
  'sedentary time':         'sitting time',
  'daily steps':            'daily steps',
  'distance walked/run':    'distance',
  'calories burned':        'calories burned',
};

const SCORE_EXPLAIN: Record<string, string> = {
  sleep: 'Your **sleep score** (0–100) is calculated each morning from how long you slept, how much time you spent in deep and REM sleep, and how well your heart rate and breathing recovered overnight. Higher is better.',
  heart: 'Your **heart score** (0–100) reflects cardiovascular health signals captured while you sleep — mainly resting heart rate and heart rate variability (HRV). A higher score means your heart is recovering efficiently.',
};

export default function RecommendationsSection({ recommendations }: { recommendations: Recommendation[] }) {
  let recs = [...recommendations];
  // Filter artifacts
  recs = recs.filter(r => !(POSITIVE_METRICS.has(r.activity_metric) && (r.optimal_min == null || r.optimal_min === 0)));

  if (recs.length === 0) {
    return (
      <div>
        <div className="section-header">Activity Recommendations</div>
        <div className="empty-panel">Recommendations appear after 14 days of data — keep syncing daily.</div>
      </div>
    );
  }

  const nDays = Math.max(...recs.map(r => r.sample_size));

  return (
    <div>
      <div className="section-header">Activity Recommendations</div>
      <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 10, color: '#484F58', marginBottom: 12 }}>
        Based on {nDays} days of your data. Updated daily.
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {recs.slice(0, 6).map((rec, i) => {
          const target     = rec.target_score;
          const scoreLabel = target === 'sleep' ? 'sleep' : 'heart';
          const tagColor   = target === 'sleep' ? '#4A90D9' : '#EF4444';
          const label      = LABEL_MAP[rec.activity_label.toLowerCase()] ?? rec.activity_label.toLowerCase();
          const explain    = SCORE_EXPLAIN[target] ?? '';
          return (
            <div key={i} style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 6, padding: '14px 16px' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: tagColor, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
                Improve your {scoreLabel} score
              </div>
              <div style={{ fontSize: 13, color: '#8B949E', marginBottom: 8 }}>
                {explain.split('**').map((part, j) => j % 2 === 1 ? <strong key={j} style={{ color: '#E6EDF3' }}>{part}</strong> : part)}
              </div>
              <div style={{ height: 1, background: '#21262D', margin: '10px 0' }} />
              <div style={{ fontSize: 13, color: '#8B949E' }}>
                <strong style={{ color: '#E6EDF3' }}>What your data shows:</strong> on days when you get{' '}
                <strong style={{ color: '#E6EDF3' }}>{rec.optimal_min_fmt}–{rec.optimal_max_fmt}</strong>{' '}
                of {label}, your {scoreLabel} score averages{' '}
                <strong style={{ color: '#E6EDF3' }}>{rec.avg_score_in_range.toFixed(0)}/100</strong>.{' '}
                On other days it averages <strong style={{ color: '#E6EDF3' }}>{rec.avg_score_outside.toFixed(0)}/100</strong>{' '}
                — a difference of <strong style={{ color: '#E6EDF3' }}>{rec.score_delta.toFixed(0)} points</strong>.
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
