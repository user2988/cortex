import { NextRequest, NextResponse } from 'next/server';
import { query } from '@/lib/db';
import {
  pearsonCorrelation, spearmanCorrelation, laggedCorrelation,
  rollingAvgCorrelation, olsTrend, multipleOls, anomalyDetection, seasonalDecompose,
} from '@/lib/stats';
import type { BiometricRow, AnalysisPayload, DataPoint } from '@/lib/types';

export async function POST(req: NextRequest) {
  try {
    const body: AnalysisPayload = await req.json();
    const days = body.days || 0;
    const where = days > 0 ? `WHERE date >= CURRENT_DATE - INTERVAL '${days} days'` : '';
    const bio = await query<BiometricRow>(
      `SELECT date::text, * FROM biometrics ${where} ORDER BY date`
    );
    const clean = bio.filter(r => r.sleep_duration_min == null || r.sleep_duration_min > 0);

    const toSeries = (col: string): DataPoint[] =>
      clean
        .filter(r => r[col as keyof BiometricRow] != null)
        .map(r => ({ date: r.date, value: Number(r[col as keyof BiometricRow]) }));

    let result;
    switch (body.type) {
      case 'Pearson Correlation':
        result = pearsonCorrelation(toSeries(body.var_a), toSeries(body.var_b));
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'correlation', data: result });

      case 'Spearman Correlation':
        result = spearmanCorrelation(toSeries(body.var_a), toSeries(body.var_b));
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'correlation', data: result });

      case 'Lagged Correlation':
        result = laggedCorrelation(toSeries(body.var_a), toSeries(body.var_b), body.lag, body.method as 'pearson' | 'spearman');
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'correlation', data: result });

      case 'Rolling Average':
        result = rollingAvgCorrelation(toSeries(body.var_a), toSeries(body.var_b), body.window, body.method as 'pearson' | 'spearman');
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'correlation', data: result });

      case '30-Day Trend (OLS)':
        result = olsTrend(toSeries(body.var_a));
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'trend', data: result });

      case 'Multiple OLS Regression': {
        const predSeries = body.predictors.map(p => toSeries(p));
        result = multipleOls(predSeries, body.predictors, toSeries(body.outcome), body.outcome);
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'multiple_ols', data: result });
      }

      case 'Anomaly Detection':
        result = anomalyDetection(toSeries(body.var_a), body.window, body.threshold);
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'anomaly', data: result });

      case 'Decomposition':
        result = seasonalDecompose(toSeries(body.var_a), body.period);
        if ('error' in result) return NextResponse.json({ type: 'error', message: result.error });
        return NextResponse.json({ type: 'decompose', data: result });

      default:
        return NextResponse.json({ type: 'error', message: 'Unknown analysis type' });
    }
  } catch (err) {
    return NextResponse.json({ type: 'error', message: String(err) }, { status: 500 });
  }
}
