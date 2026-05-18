import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { DailyScore, Recommendation } from '@/lib/types';

export async function GET(req: NextRequest) {
  const days = parseInt(req.nextUrl.searchParams.get('days') ?? '90', 10);
  try {
    const scores = await query<DailyScore>(
      `SELECT date::text, sleep_score, heart_score,
              duration_score, deep_score, rem_score, efficiency_score,
              hrv_score, rhr_score, spo2_score,
              sleep_duration_min, deep_pct, rem_pct, hrv_ms, rhr_bpm, spo2_avg_pct
       FROM daily_scores ORDER BY date DESC LIMIT $1`,
      [days]
    );
    const recs = await query<Recommendation>(
      `SELECT DISTINCT ON (target_score)
              target_score, activity_metric, activity_label,
              optimal_min, optimal_min_fmt, optimal_max_fmt,
              avg_score_in_range, avg_score_outside, score_delta,
              correlation, sample_size, recommendation_text
       FROM score_recommendations
       ORDER BY target_score, score_delta DESC`
    );
    return NextResponse.json({ scores, recommendations: recs });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
