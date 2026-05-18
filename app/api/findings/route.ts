import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { Finding } from '@/lib/types';

export async function GET() {
  try {
    const rows = await query<Finding>(
      `SELECT id, variable_a, variable_b, r_squared, p_value, coefficient,
              lag_days, analysis_type, sample_size, calculated_at::text, pinned
       FROM findings ORDER BY pinned DESC, r_squared DESC`
    );
    return NextResponse.json(rows);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest) {
  const id = req.nextUrl.searchParams.get('id');
  if (!id) return NextResponse.json({ error: 'Missing id' }, { status: 400 });
  try {
    await query('DELETE FROM findings WHERE id = $1', [parseInt(id, 10)]);
    return NextResponse.json({ ok: true });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
