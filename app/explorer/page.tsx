'use client';

import { useState } from 'react';
import AnalysisForm from '@/components/explorer/AnalysisForm';
import AnalysisResult from '@/components/explorer/AnalysisResult';
import type { AnalysisResult as AnalysisResultType, AnalysisPayload } from '@/lib/types';

export default function ExplorerPage() {
  const [result, setResult]   = useState<AnalysisResultType | null>(null);
  const [payload, setPayload] = useState<AnalysisPayload | null>(null);
  const [running, setRunning] = useState(false);

  async function handleRun(p: AnalysisPayload) {
    setRunning(true); setResult(null); setPayload(p);
    const res = await fetch('/api/run-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(p),
    });
    const data: AnalysisResultType = await res.json();
    setResult(data);
    setRunning(false);
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 24, alignItems: 'start' }}>
      {/* Sidebar panel */}
      <div style={{ background: '#161B22', border: '1px solid #21262D', borderRadius: 6, padding: '4px 16px 16px' }}>
        <div style={{ fontFamily: 'Inter', fontSize: 13, fontWeight: 600, color: '#E6EDF3', paddingTop: 12, marginBottom: 4 }}>Explorer</div>
        <div style={{ height: 1, background: '#21262D', margin: '8px 0' }} />
        <AnalysisForm onRun={handleRun} running={running} />
      </div>

      {/* Result panel */}
      <div>
        {!result && !running && (
          <div className="empty-panel" style={{ marginTop: 40 }}>
            Configure your analysis and click <strong>Run Analysis</strong>.
          </div>
        )}
        {running && (
          <div className="empty-panel" style={{ marginTop: 40 }}>
            <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 12, color: '#484F58' }}>Running…</div>
          </div>
        )}
        {result && payload && (
          <AnalysisResult result={result} payload={payload} />
        )}
      </div>
    </div>
  );
}
