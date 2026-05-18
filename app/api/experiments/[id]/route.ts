import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';
import { runExperimentAnalysis } from '@/lib/stats';
import type { BiometricRow, DataPoint } from '@/lib/types';

export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
  const id = parseInt(params.id, 10);
  try {
    const rows = await query(
      `SELECT id, name, variable_a, variable_b, lag_days, method,
              start_date::text, duration_days, status, interpretation, created_at::text
       FROM experiments WHERE id = $1`,
      [id]
    );
    if (!rows.length) return NextResponse.json({ error: 'Not found' }, { status: 404 });
    const exp = rows[0] as Record<string, unknown>;

    // Compute end date
    const startDate = exp.start_date as string;
    const duration = exp.duration_days as number;
    const end = new Date(startDate);
    end.setDate(end.getDate() + duration);
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const elapsed = Math.max(0, Math.floor((Math.min(today.getTime(), end.getTime()) - new Date(startDate).getTime()) / 86400000));
    const enriched = {
      ...exp,
      end_date: end.toISOString().slice(0, 10),
      is_complete: end <= today,
      elapsed_days: elapsed,
    };

    // Fetch all biometric data for analysis
    const bio = await query<BiometricRow>(`SELECT date::text, * FROM biometrics ORDER BY date`);
    const varA = exp.variable_a as string;
    const varB = exp.variable_b as string;
    const toSeries = (col: string): DataPoint[] =>
      bio.filter(r => r[col as keyof BiometricRow] != null)
         .map(r => ({ date: r.date, value: Number(r[col as keyof BiometricRow]) }));

    const result = runExperimentAnalysis(
      toSeries(varA), toSeries(varB),
      varA, varB,
      exp.lag_days as number,
      (exp.method as string) === 'spearman' ? 'spearman' : 'pearson',
      startDate, duration
    );

    return NextResponse.json({ experiment: enriched, analysis: result });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest, { params }: { params: { id: string } }) {
  const id = parseInt(params.id, 10);
  try {
    await query('DELETE FROM experiments WHERE id = $1', [id]);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  const id = parseInt(params.id, 10);
  const { interpretation } = await req.json();
  try {
    await query(
      `UPDATE experiments SET status = 'complete', interpretation = $1 WHERE id = $2`,
      [interpretation, id]
    );
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
