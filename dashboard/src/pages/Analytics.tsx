import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, LineChart, Line, Legend,
} from 'recharts';
import { useCollisions, useStats, useTsaForecast } from '../hooks/useQueries';
import ChartCard from '../components/ui/ChartCard';
import EmptyState from '../components/ui/EmptyState';
import { RISK_COLORS } from '../utils/constants';
import type { RiskLevel } from '../api/types';

const MODEL_COLORS: Record<string, string> = {
  'Naive':              '#64748b',
  'AR':                 '#38bdf8',
  'ARIMA':              '#818cf8',
  'VAR':                '#34d399',
  'LSTM':               '#fb923c',
  'PatchTST':           '#e879f9',
  'Hybrid VAR+LSTM':    '#f43f5e',
  'Hybrid VAR+PatchTST':'#a3e635',
};

const tooltipStyle = {
  background: '#0f172a', border: '1px solid #1e293b',
  borderRadius: '8px', color: '#f1f5f9', fontSize: 12,
};

export default function Analytics() {
  const { data: collisions }   = useCollisions({ page: 1, per_page: 200, sort_by: 'distance_km', sort_order: 'asc' });
  const { data: stats }        = useStats();
  const { data: forecastData } = useTsaForecast();

  // ── Collision analytics ───────────────────────────────────────────────────

  const distanceBuckets = (() => {
    if (!collisions?.collisions) return [];
    const buckets = [
      { range: '0-1 km',  min: 0,  max: 1,   count: 0 },
      { range: '1-5 km',  min: 1,  max: 5,   count: 0 },
      { range: '5-20 km', min: 5,  max: 20,  count: 0 },
      { range: '20-50',   min: 20, max: 50,  count: 0 },
      { range: '50-100',  min: 50, max: 100, count: 0 },
    ];
    collisions.collisions.forEach((c) => {
      if (c.distance_km == null) return;
      const b = buckets.find((b) => c.distance_km! >= b.min && c.distance_km! < b.max);
      if (b) b.count++;
    });
    return buckets;
  })();

  const scatter = collisions?.collisions
    ? collisions.collisions.slice(0, 150).map((c) => ({
        distance: c.distance_km,
        velocity: c.relative_velocity_kms,
        risk:     c.risk_level as RiskLevel,
      })).filter((d) => d.distance != null && d.velocity != null)
    : [];

  const riskBar = stats
    ? [
        { name: 'Critical', count: stats.critical_risk_collisions, fill: RISK_COLORS.CRITICAL },
        { name: 'High',     count: stats.high_risk_collisions,     fill: RISK_COLORS.HIGH },
        { name: 'Medium',   count: stats.medium_risk_collisions,   fill: RISK_COLORS.MEDIUM },
        { name: 'Low',      count: stats.low_risk_collisions,      fill: RISK_COLORS.LOW },
      ]
    : [];

  // ── TSA model benchmark data ──────────────────────────────────────────────

  const benchmarkBar = forecastData?.models
    ? [...forecastData.models].sort((a, b) => a.rmse_km - b.rmse_km)
    : [];

  // Line chart: actual vs. best model forecast (if available)
  const forecastLines = forecastData?.forecasts
    ? Object.entries(forecastData.forecasts).slice(0, 3) // show top 3 models
    : [];

  const lineChartData = (() => {
    if (!forecastLines.length) return [];
    const maxLen = Math.max(...forecastLines.map(([, pts]) => pts.length));
    return Array.from({ length: maxLen }, (_, i) => {
      const row: Record<string, number | null> = { step: i + 1 };
      forecastLines.forEach(([model, pts]) => {
        row[model] = pts[i]?.forecast ?? null;
        if (i === 0 || pts[i]?.actual != null) {
          row['actual'] = pts[i]?.actual ?? null;
        }
      });
      return row;
    });
  })();

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-text-primary">Analytics</h2>
        <p className="text-sm text-text-muted mt-0.5">Collision statistics and TSA model benchmark results</p>
      </div>

      {/* ── TSA Model Benchmark ────────────────────────────────────────── */}
      {benchmarkBar.length > 0 && (
        <>
          <div className="flex items-center gap-2 mt-2 mb-1">
            <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">Model Benchmarks</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* RMSE bar */}
            <ChartCard title="Forecast RMSE by Model" subtitle="Lower is better — test set (km)">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={benchmarkBar} layout="vertical" margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} unit=" km" />
                  <YAxis type="category" dataKey="model" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={{ stroke: '#1e293b' }} width={130} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number | undefined) => v != null ? [`${v.toFixed(0)} km`, 'RMSE'] : ['—', 'RMSE']} />
                  <Bar dataKey="rmse_km" radius={[0, 4, 4, 0]} barSize={16}>
                    {benchmarkBar.map((m) => (
                      <Cell key={m.model} fill={MODEL_COLORS[m.model] ?? '#38bdf8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Benchmark table */}
            <ChartCard title="Benchmark Table" subtitle="All evaluated models">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border-primary text-text-muted uppercase tracking-wider">
                      <th className="text-left pb-2 pr-4">Model</th>
                      <th className="text-right pb-2 pr-4">RMSE (km)</th>
                      <th className="text-right pb-2 pr-4">MAE (km)</th>
                      <th className="text-right pb-2">MAPE (%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-primary/40">
                    {benchmarkBar.map((m, i) => (
                      <tr key={m.model} className="hover:bg-bg-surface-hover/50 transition-colors">
                        <td className="py-2 pr-4 font-medium" style={{ color: MODEL_COLORS[m.model] ?? '#94a3b8' }}>
                          {i === 0 && <span className="mr-1 text-risk-low">★</span>}{m.model}
                        </td>
                        <td className="py-2 pr-4 font-mono text-text-primary text-right">{m.rmse_km.toFixed(0)}</td>
                        <td className="py-2 pr-4 font-mono text-text-secondary text-right">{m.mae_km.toFixed(0)}</td>
                        <td className="py-2 font-mono text-text-muted text-right">
                          {m.mape != null ? m.mape.toFixed(1) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {forecastData?.generated_at && (
                <p className="text-[10px] text-text-muted mt-3">
                  Generated: {forecastData.generated_at.slice(0, 19).replace('T', ' ')}
                  {forecastData.sample_norad_id ? ` · NORAD ${forecastData.sample_norad_id}` : ''}
                </p>
              )}
            </ChartCard>
          </div>

          {/* Forecast line chart */}
          {lineChartData.length > 0 && (
            <ChartCard title="Multi-Step Forecast Comparison" subtitle="ΔPOS_X (km) — actual vs. model forecasts">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={lineChartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="step" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} label={{ value: 'Step (5 min)', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} unit=" km" />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number | undefined, n: string | undefined) => [v != null ? `${v.toFixed(1)} km` : '—', n ?? '']} />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                  <Line dataKey="actual" stroke="#f1f5f9" strokeWidth={2} dot={false} name="Actual" strokeDasharray="4 2" />
                  {forecastLines.map(([model]) => (
                    <Line key={model} dataKey={model} stroke={MODEL_COLORS[model] ?? '#38bdf8'}
                      strokeWidth={1.5} dot={false} name={model} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          )}
        </>
      )}

      {!forecastData && (
        <ChartCard title="TSA Model Benchmarks">
          <EmptyState title="No forecast data" message="Run: make pipeline to train models and generate benchmarks" />
        </ChartCard>
      )}

      {/* ── Collision Analytics ────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mt-2 mb-1">
        <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">Collision Analytics</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk breakdown bar */}
        <ChartCard title="Collision Count by Risk Level" subtitle="Total pairs per risk category">
          {riskBar.some((r) => r.count > 0) ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={riskBar} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} barSize={40}>
                  {riskBar.map((r) => <Cell key={r.name} fill={r.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No risk data" message="Run: make pipeline" />
          )}
        </ChartCard>

        {/* Distance distribution */}
        <ChartCard title="Miss Distance Distribution" subtitle="Histogram of approach distances">
          {distanceBuckets.some((b) => b.count > 0) ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={distanceBuckets} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} barSize={40}>
                  {distanceBuckets.map((_, i) => (
                    <Cell key={i} fill={
                      i === 0 ? RISK_COLORS.CRITICAL
                      : i === 1 ? RISK_COLORS.HIGH
                      : i < 3  ? RISK_COLORS.MEDIUM
                      : RISK_COLORS.LOW
                    } />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No distance data" message="Run: make pipeline" />
          )}
        </ChartCard>

        {/* Distance vs Velocity scatter */}
        <ChartCard title="Distance vs. Relative Velocity" subtitle="Approach parameters, coloured by risk" className="lg:col-span-2">
          {scatter.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="distance" name="Distance" unit=" km" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <YAxis dataKey="velocity" name="Velocity" unit=" km/s" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={{ stroke: '#1e293b' }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v: number | undefined, n: string | undefined) => [v ?? '—', n ?? '']} />
                <Scatter data={scatter} shape="circle">
                  {scatter.map((s, i) => (
                    <Cell key={i} fill={RISK_COLORS[s.risk]} fillOpacity={0.7} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No scatter data" message="Run: make pipeline" />
          )}
          <div className="flex gap-4 mt-3">
            {(['CRITICAL','HIGH','MEDIUM','LOW'] as RiskLevel[]).map((r) => (
              <div key={r} className="flex items-center gap-1.5 text-xs text-text-muted">
                <span className="w-2 h-2 rounded-full" style={{ background: RISK_COLORS[r] }} />
                {r}
              </div>
            ))}
          </div>
        </ChartCard>
      </div>
    </div>
  );
}
