import { NextRequest, NextResponse } from 'next/server';
import { COL_LABELS } from '@/lib/labels';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { var_a, var_b, r2, p_value, coefficient, lag, n,
    pre_avg_a, pre_avg_b, during_avg_a, during_avg_b } = body;

  const aLbl = COL_LABELS[var_a] ?? var_a;
  const bLbl = COL_LABELS[var_b] ?? var_b;
  const lagStr = lag ? `${lag}-day lag` : 'same day';
  const preCtx = pre_avg_a != null
    ? `Before: ${aLbl} avg ${pre_avg_a.toFixed(1)}, ${bLbl} avg ${pre_avg_b.toFixed(1)}. During: ${aLbl} avg ${during_avg_a.toFixed(1)}, ${bLbl} avg ${during_avg_b.toFixed(1)}. `
    : '';

  const prompt = `Variable A: ${aLbl} | Variable B: ${bLbl} | R²: ${r2} | p-value: ${p_value} | Coefficient: ${coefficient} | Lag: ${lagStr} | Sample: ${n} days. ${preCtx}`;

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return NextResponse.json({ error: 'ANTHROPIC_API_KEY not set' }, { status: 500 });

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-6',
        max_tokens: 200,
        system: 'You interpret statistical correlation results from a personal health tracking app. Never imply causation. Frame R² as explained variation. Always note correlation does not confirm causation. Do not give health advice or prescribe lifestyle changes. Describe what the data shows — nothing more. Maximum 3 sentences. If before/during averages differ noticeably, describe the visible shift.',
        messages: [{ role: 'user', content: prompt }],
      }),
    });
    const data = await response.json() as { content?: Array<{ text: string }> };
    const text = data.content?.[0]?.text ?? '';
    return NextResponse.json({ interpretation: text });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
