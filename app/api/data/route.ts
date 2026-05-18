import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';
import { BIOMETRIC_COLS } from '@/lib/labels';
import type { BiometricRow } from '@/lib/types';

export async function GET(req: NextRequest) {
  const days = parseInt(req.nextUrl.searchParams.get('days') ?? '90', 10);
  const where = days > 0 ? `WHERE date >= CURRENT_DATE - INTERVAL '${days} days'` : '';
  const cols = BIOMETRIC_COLS.join(', ');
  try {
    const rows = await query<BiometricRow>(
      `SELECT date::text, ${cols} FROM biometrics ${where} ORDER BY date`
    );
    // Exclude device-failure rows (sleep_duration_min == 0)
    const clean = rows.filter(r => r.sleep_duration_min == null || r.sleep_duration_min > 0);
    return NextResponse.json(clean);
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
