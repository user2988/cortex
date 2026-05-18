import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';

function computeExperimentFields(row: Record<string, unknown>) {
  const startDate = new Date(row.start_date as string);
  const endDate = new Date(startDate);
  endDate.setDate(endDate.getDate() + (row.duration_days as number));
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const elapsed = Math.max(0, Math.floor((Math.min(today.getTime(), endDate.getTime()) - startDate.getTime()) / 86400000));
  return {
    ...row,
    start_date: startDate.toISOString().slice(0, 10),
    end_date: endDate.toISOString().slice(0, 10),
    is_complete: endDate <= today,
    elapsed_days: elapsed,
  };
}

export async function GET() {
  try {
    const rows = await query(
      `SELECT id, name, variable_a, variable_b, lag_days, method,
              start_date::text, duration_days, status, interpretation, created_at::text
       FROM experiments ORDER BY created_at DESC`
    );
    return NextResponse.json(rows.map(computeExperimentFields));
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, variable_a, variable_b, lag_days, method, start_date, duration_days } = body;
    if (!name || !variable_a || !variable_b || !start_date || !duration_days) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }
    const rows = await query(
      `INSERT INTO experiments (name, variable_a, variable_b, lag_days, method, start_date, duration_days)
       VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id`,
      [name, variable_a, variable_b, lag_days ?? 0, method ?? 'pearson', start_date, duration_days]
    );
    return NextResponse.json({ id: (rows[0] as { id: number }).id });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
