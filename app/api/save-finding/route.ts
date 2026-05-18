import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { variable_a, variable_b, r_squared, p_value, coefficient, lag_days, analysis_type, sample_size } = body;
  try {
    await query(
      `INSERT INTO findings (variable_a, variable_b, r_squared, p_value, coefficient, lag_days, analysis_type, sample_size, pinned)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true)`,
      [variable_a, variable_b ?? null, r_squared, p_value, coefficient, lag_days ?? 0, analysis_type, sample_size]
    );
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
