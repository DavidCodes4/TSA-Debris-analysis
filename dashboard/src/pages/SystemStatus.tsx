import {
  Server, CheckCircle2, XCircle, Satellite, Cpu,
  FlaskConical, FileJson, Database,
} from 'lucide-react';
import { useHealth, useStats, usePipelineStatus } from '../hooks/useQueries';
import StatCard from '../components/ui/StatCard';
import StatusDot from '../components/ui/StatusDot';
import ChartCard from '../components/ui/ChartCard';
import { StatCardSkeleton } from '../components/ui/Skeleton';
import { formatNumber, formatDistance } from '../utils/formatters';

export default function SystemStatus() {
  const { data: health, isLoading: healthLoading, isError } = useHealth();
  const { data: stats }    = useStats();
  const { data: pipeline } = usePipelineStatus();

  const apiOk         = health?.status === 'healthy' && !isError;
  const pipelineRan   = health?.pipeline_ran ?? false;
  const lastRun       = pipeline?.runs?.[0];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-xl font-bold text-text-primary">System Status</h2>
        <p className="text-sm text-text-muted mt-0.5">Pipeline health and output artefact overview</p>
      </div>

      {/* Health cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {healthLoading ? (
          Array.from({ length: 4 }).map((_, i) => <StatCardSkeleton key={i} />)
        ) : (
          <>
            {/* Dashboard API */}
            <div className="bg-bg-card border border-border-primary rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-accent-blue" />
                  <span className="text-sm font-semibold text-text-primary">Dashboard API</span>
                </div>
                <StatusDot status={apiOk ? 'online' : 'offline'} />
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                  {apiOk
                    ? <CheckCircle2 className="w-3.5 h-3.5 text-risk-low" />
                    : <XCircle className="w-3.5 h-3.5 text-risk-critical" />}
                  <span className="text-text-secondary">Flask API: {health?.status ?? 'unknown'}</span>
                </div>
                <div className="flex items-center gap-2 text-text-muted">
                  <span className="font-mono">{health?.timestamp?.slice(0,19).replace('T',' ') ?? '—'}</span>
                </div>
              </div>
            </div>

            {/* Pipeline state */}
            <div className="bg-bg-card border border-border-primary rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <FlaskConical className="w-4 h-4 text-accent-cyan" />
                  <span className="text-sm font-semibold text-text-primary">TSA Pipeline</span>
                </div>
                <StatusDot status={pipelineRan ? 'online' : 'warning'} pulse={pipelineRan} />
              </div>
              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                  {pipelineRan
                    ? <CheckCircle2 className="w-3.5 h-3.5 text-risk-low" />
                    : <XCircle className="w-3.5 h-3.5 text-risk-medium" />}
                  <span className="text-text-secondary">
                    {pipelineRan ? 'Output artefacts present' : 'Not yet run'}
                  </span>
                </div>
                {lastRun?.generated_at && (
                  <p className="text-text-muted font-mono">{lastRun.generated_at.slice(0, 19).replace('T', ' ')}</p>
                )}
              </div>
            </div>

            {/* Models */}
            <div className="bg-bg-card border border-border-primary rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-risk-medium" />
                  <span className="text-sm font-semibold text-text-primary">Trained Models</span>
                </div>
                <StatusDot status={pipelineRan ? 'online' : 'offline'} pulse={false} />
              </div>
              <div className="space-y-1.5 text-xs text-text-muted">
                <p>VAR  — statsmodels (pkl)</p>
                <p>LSTM — Keras stacked LSTM</p>
                <p>PatchTST — custom Transformer</p>
              </div>
            </div>

            {/* Debris catalog */}
            <div className="bg-bg-card border border-border-primary rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Satellite className="w-4 h-4 text-risk-low" />
                  <span className="text-sm font-semibold text-text-primary">Debris Catalog</span>
                </div>
                <StatusDot status={stats?.total_debris_objects ? 'online' : 'offline'} />
              </div>
              <div className="space-y-1.5 text-xs">
                <p className="text-text-secondary">{formatNumber(stats?.total_debris_objects ?? 0)} objects tracked</p>
                <p className="text-text-muted">SGP4-propagated, 5-min steps</p>
              </div>
            </div>
          </>
        )}
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard icon={Satellite}  label="Total Collision Pairs" value={formatNumber(stats?.total_collision_pairs ?? 0)} iconColor="text-accent-blue"    iconBg="bg-accent-blue/10" />
        <StatCard icon={Server}     label="Min Approach Distance" value={formatDistance(stats?.min_distance_km ?? null)}  iconColor="text-risk-critical"  iconBg="bg-risk-critical/10" />
        <StatCard icon={Database}   label="Avg Approach Distance" value={formatDistance(stats?.avg_distance_km ?? null)}  iconColor="text-risk-medium"    iconBg="bg-risk-medium/10" />
      </div>

      {/* Pipeline details from last run */}
      {lastRun && (
        <ChartCard title="Last Pipeline Run" subtitle="Metadata from dashboard_state.json">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            {[
              { label: 'Generated At',  value: lastRun.generated_at?.slice(0, 19).replace('T', ' ') ?? '—' },
              { label: 'Objects',        value: formatNumber(lastRun.n_objects) },
              { label: 'Timesteps',      value: formatNumber(lastRun.n_timesteps) },
              { label: 'TLE File',       value: lastRun.tle_file ?? '—' },
              { label: 'Best Model',     value: lastRun.best_model ?? '—' },
              { label: 'Best RMSE',      value: lastRun.best_rmse_km != null ? `${lastRun.best_rmse_km.toFixed(0)} km` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-bg-surface rounded-lg px-3 py-2.5">
                <p className="text-text-muted mb-1">{label}</p>
                <p className="font-mono text-text-primary">{value}</p>
              </div>
            ))}
          </div>
        </ChartCard>
      )}

      {/* Pipeline architecture */}
      <ChartCard title="Pipeline Architecture" subtitle="End-to-end data flow">
        <div className="font-mono text-xs text-text-secondary space-y-2 p-2">
          {[
            { step: '01_data_pipeline.py',       desc: 'Parse TLE archive → SGP4 propagate (5-min, 72h) → Output/ts_df.parquet' },
            { step: '02_train_models.py',         desc: 'Train VAR + LSTM + PatchTST → Output/models/' },
            { step: '03_evaluate_models.py',      desc: 'Benchmark 6 models (recursive forecast) → Output/forecast_results.json' },
            { step: '04_conjunction_assessment.py', desc: 'Altitude-bucket screening → Output/collision_alerts_*/part-00000.csv' },
            { step: '05_update_dashboard.py',     desc: 'Consolidate all outputs → Output/dashboard_state.json' },
            { step: 'dashboard_api.py',           desc: 'Flask API: reads Output/ → serves REST endpoints → this dashboard' },
          ].map(({ step, desc }) => (
            <div key={step} className="flex gap-4">
              <span className="text-accent-cyan w-56 flex-shrink-0">{step}</span>
              <span>{desc}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      {/* Quick commands */}
      <ChartCard title="Quick Commands" subtitle="Run these in your terminal">
        <div className="space-y-2 font-mono text-xs">
          {[
            { label: 'Full pipeline',              cmd: 'make pipeline' },
            { label: 'Dry-run (preview steps)',    cmd: 'make dry-run' },
            { label: 'Re-run data pipeline only',  cmd: 'make step1' },
            { label: 'Re-train models only',       cmd: 'make step2' },
            { label: 'Re-evaluate + refresh',      cmd: 'make refresh' },
            { label: 'Start this API (port 5050)', cmd: 'make api' },
            { label: 'Start React dashboard',      cmd: 'make dev' },
          ].map(({ label, cmd }) => (
            <div key={label} className="flex items-start gap-4 bg-bg-surface rounded-lg px-3 py-2">
              <span className="text-text-muted w-48 flex-shrink-0"># {label}</span>
              <span className="text-accent-cyan">{cmd}</span>
            </div>
          ))}
        </div>
      </ChartCard>

      {/* Output artefacts */}
      <ChartCard title="Output Artefacts" subtitle="Files produced by the pipeline under Output/">
        <div className="space-y-1.5 text-xs">
          {[
            { icon: FileJson,  file: 'ts_df.parquet',            desc: 'SGP4-densified time series for all objects' },
            { icon: Cpu,       file: 'models/var_model.pkl',     desc: 'Fitted VAR model (statsmodels)' },
            { icon: Cpu,       file: 'models/lstm_model.keras',  desc: 'Trained LSTM weights (Keras)' },
            { icon: Cpu,       file: 'models/patchtst_model.keras', desc: 'Trained PatchTST weights (Keras)' },
            { icon: FileJson,  file: 'forecast_results.json',   desc: 'Per-model RMSE / MAE / forecast series' },
            { icon: FileJson,  file: 'benchmark_table.csv',     desc: 'Benchmark table (all 6 models)' },
            { icon: FileJson,  file: 'conjunction_report.json', desc: 'All close-approach alerts with risk classification' },
            { icon: FileJson,  file: 'collision_alerts_*/part-00000.csv', desc: 'Raw CSV alerts consumed by the API' },
            { icon: Database,  file: 'space_debris_catalog.csv', desc: 'Debris object catalog for /api/debris' },
            { icon: FileJson,  file: 'dashboard_state.json',    desc: 'Consolidated state consumed by Flask API' },
          ].map(({ icon: Icon, file, desc }) => (
            <div key={file} className="flex items-start gap-3 bg-bg-surface rounded-lg px-3 py-2">
              <Icon className="w-3.5 h-3.5 text-accent-cyan mt-0.5 flex-shrink-0" />
              <span className="font-mono text-accent-cyan w-64 flex-shrink-0">{file}</span>
              <span className="text-text-muted">{desc}</span>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}
