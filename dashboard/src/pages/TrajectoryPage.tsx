import { useState, useEffect, useRef } from 'react';
import { Orbit, RefreshCw, RotateCcw, Maximize2, Minimize2, Info } from 'lucide-react';
import { useTrajectory, useAllTrajectories } from '../hooks/useQueries';
import type { OrbitalRegime, TrajectoryObjectSummary } from '../api/types';

// ── Constants ────────────────────────────────────────────────────────────────
const EARTH_RADIUS_KM = 6371;
const GM_EARTH        = 398600.4418;

const REGIME_COLORS: Record<OrbitalRegime, string> = {
  LEO: '#00e5ff',
  MEO: '#69f0ae',
  HEO: '#ffab40',
  GEO: '#ff5252',
};

const REGIME_LABELS: Record<OrbitalRegime, string> = {
  LEO: 'LEO < 2 000 km',
  MEO: 'MEO 2–8 k km',
  HEO: 'HEO 8–36.5 k km',
  GEO: 'GEO ≥ 36 500 km',
};

type RegimeFilter = 'ALL' | OrbitalRegime;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GlobeInstance = any;

// ── Helpers ──────────────────────────────────────────────────────────────────
function computeOrbitalPeriod(altKm: number): string {
  const r = EARTH_RADIUS_KM + Math.max(altKm, 160);
  const T = 2 * Math.PI * Math.sqrt((r * r * r) / GM_EARTH); // seconds
  const m = Math.round(T / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m} min`;
}

// ════════════════════════════════════════════════════════════════════════════
export default function TrajectoryPage() {
  const containerRef      = useRef<HTMLDivElement>(null);
  const globeRef          = useRef<GlobeInstance>(null);
  const [globeReady, setGlobeReady] = useState(false);

  const [regimeFilter, setRegimeFilter] = useState<RegimeFilter>('ALL');
  const [selectedId,   setSelectedId]   = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showPaths,    setShowPaths]    = useState(true);

  // ── Data ──────────────────────────────────────────────────────────────────
  const { data: allData, isLoading: allLoading, refetch } = useAllTrajectories({
    downsample: 10,
    regime: regimeFilter === 'ALL' ? undefined : regimeFilter,
  });

  const { data: singleData, isLoading: singleLoading } = useTrajectory(
    selectedId, 3,
  );

  // ── Counts per regime ─────────────────────────────────────────────────────
  const counts: Record<string, number> = { LEO: 0, MEO: 0, HEO: 0, GEO: 0 };
  (allData?.objects ?? []).forEach((o) => { counts[o.regime] = (counts[o.regime] ?? 0) + 1; });

  // ── Init globe.gl ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    let cancelled = false;
    (async () => {
      const mod = await import('globe.gl');
      if (cancelled || !containerRef.current) return;
      const Globe = (mod as any).default ?? mod;
      const { width, height } = containerRef.current.getBoundingClientRect();

      const g = new Globe(containerRef.current, { animateIn: true })
        .globeImageUrl('/textures/earth-blue-marble.jpg')
        .bumpImageUrl('/textures/earth-topology.png')
        .backgroundImageUrl('/textures/night-sky.png')
        .backgroundColor('#040714')
        .showAtmosphere(true)
        .atmosphereColor('#448aff')
        .atmosphereAltitude(0.15)
        .width(width || 800)
        .height(height || 600)
        // Paths layer — ground tracks
        .pathsData([])
        .pathPoints('points')
        .pathPointLat((p: [number, number]) => p[0])
        .pathPointLng((p: [number, number]) => p[1])
        .pathColor('color')
        .pathStroke(1.2)
        .pathDashLength(0.4)
        .pathDashGap(0.2)
        .pathDashAnimateTime(8000)
        // HTML markers — selected object dot
        .htmlElementsData([])
        .htmlLat('lat')
        .htmlLng('lng')
        .htmlAltitude(0)
        .htmlElement((d: any) => {
          const el = document.createElement('div');
          el.style.cssText = `
            width: 14px; height: 14px; border-radius: 50%;
            background: ${d.color};
            border: 2px solid white;
            box-shadow: 0 0 12px ${d.color}, 0 0 24px ${d.color}60;
            cursor: pointer;
          `;
          el.title = `NORAD ${d.norad_id} | ${d.regime} | ${d.mean_alt_km.toFixed(0)} km`;
          return el;
        });

      const ctrl = g.controls();
      if (ctrl) { ctrl.autoRotate = true; ctrl.autoRotateSpeed = 0.3; ctrl.enableDamping = true; }
      try { g.renderer().setPixelRatio(Math.min(window.devicePixelRatio, 2.5)); } catch (_) { /* */ }

      globeRef.current = g;
      setGlobeReady(true);
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Resize ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(() => {
      if (globeRef.current && containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        globeRef.current.width(width).height(height);
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [globeReady]);

  // ── Push path data to globe ───────────────────────────────────────────────
  useEffect(() => {
    const g = globeRef.current;
    if (!g || !globeReady || !allData) return;

    const paths = showPaths ? allData.objects.map((obj: TrajectoryObjectSummary) => ({
      points: obj.lats.map((la, i) => [la, obj.lons[i]] as [number, number]),
      color: REGIME_COLORS[obj.regime] + '99', // semi-transparent
      norad_id: obj.norad_id,
    })) : [];

    g.pathsData(paths);

    // Current position marker for each object (last lat/lon)
    const markers = allData.objects.map((obj: TrajectoryObjectSummary) => ({
      lat: obj.lats[obj.lats.length - 1],
      lon: obj.lons[obj.lons.length - 1],
      color: REGIME_COLORS[obj.regime],
      regime: obj.regime,
      norad_id: obj.norad_id,
      mean_alt_km: obj.mean_alt_km,
    }));
    g.htmlElementsData(markers);
  }, [globeReady, allData, showPaths]);

  // ── Highlight selected object ─────────────────────────────────────────────
  useEffect(() => {
    const g = globeRef.current;
    if (!g || !globeReady || !singleData) return;
    const pts = singleData.trajectory;
    if (!pts.length) return;
    const last = pts[pts.length - 1];
    g.pointOfView({ lat: last.lat, lng: last.lon, altitude: 2.0 }, 1200);
  }, [globeReady, singleData]);

  // ── Controls ──────────────────────────────────────────────────────────────
  const resetView = () => globeRef.current?.pointOfView({ lat: 20, lng: 0, altitude: 2.5 }, 1000);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.parentElement?.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  // ── Selected object detail ────────────────────────────────────────────────
  const selObj = allData?.objects.find((o) => String(o.norad_id) === selectedId);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="h-full flex flex-col gap-4">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Orbit className="w-7 h-7 text-cyan-400" />
          <div>
            <h1 className="text-2xl font-bold text-white">Orbital Trajectories</h1>
            <p className="text-sm text-slate-400">
              72-hour ground tracks for {allData?.count ?? '…'} debris objects
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowPaths((p) => !p)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
              showPaths
                ? 'border-cyan-500 text-cyan-400 bg-cyan-500/10'
                : 'border-slate-600 text-slate-400 hover:border-slate-400'
            }`}
          >
            {showPaths ? '⚡ Paths ON' : '⚡ Paths OFF'}
          </button>
          <button onClick={() => refetch()}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
            title="Refresh">
            <RefreshCw className="w-4 h-4" />
          </button>
          <button onClick={resetView}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
            title="Reset view">
            <RotateCcw className="w-4 h-4" />
          </button>
          <button onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition">
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Regime filter chips */}
      <div className="flex flex-wrap items-center gap-2">
        {(['ALL', 'LEO', 'MEO', 'HEO', 'GEO'] as const).map((r) => (
          <button
            key={r}
            onClick={() => setRegimeFilter(r)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition ${
              regimeFilter === r
                ? 'border-transparent text-white shadow-lg'
                : 'border-slate-600 text-slate-400 hover:border-slate-400 hover:text-slate-200'
            }`}
            style={regimeFilter === r
              ? { backgroundColor: r === 'ALL' ? '#3b82f6' : REGIME_COLORS[r as OrbitalRegime] }
              : {}}
          >
            {r === 'ALL' ? 'ALL REGIMES' : REGIME_LABELS[r as OrbitalRegime]}
            {r !== 'ALL' && <span className="ml-1 opacity-70">({counts[r] ?? 0})</span>}
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-4 min-h-0">

        {/* Globe */}
        <div className="relative flex-1 rounded-xl overflow-hidden bg-[#040714] border border-slate-800 min-h-[480px]">
          <div ref={containerRef} className="w-full h-full" />

          {(allLoading || !globeReady) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#040714]/90 z-10">
              <Orbit className="w-12 h-12 text-cyan-400 animate-spin mb-3" />
              <p className="text-slate-300 text-sm">
                {!globeReady ? 'Initialising globe…' : 'Loading trajectory data…'}
              </p>
            </div>
          )}

          {/* Legend */}
          <div className="absolute bottom-4 left-4 bg-[#0d1b2a]/90 backdrop-blur-sm border border-slate-700 rounded-lg px-3 py-2.5 z-20 text-xs space-y-1.5">
            <p className="text-slate-400 font-semibold uppercase tracking-wider mb-1">Orbital Regime</p>
            {(Object.entries(REGIME_COLORS) as [OrbitalRegime, string][]).map(([reg, col]) => (
              <div key={reg} className="flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: col }} />
                <span className="text-slate-300">{reg}</span>
                <span className="text-slate-500 ml-auto pl-4">({counts[reg] ?? 0})</span>
              </div>
            ))}
            <p className="text-slate-500 pt-1 border-t border-slate-700">
              Animated dashed paths = ground tracks
            </p>
          </div>

          {/* Stats */}
          <div className="absolute top-4 right-4 bg-[#0d1b2a]/90 backdrop-blur-sm border border-slate-700 rounded-lg px-3 py-2.5 z-20 text-xs text-right">
            <p className="text-slate-400">Objects shown</p>
            <p className="text-white text-2xl font-bold font-mono">{allData?.count ?? '—'}</p>
            <p className="text-slate-500">72h trajectories</p>
          </div>
        </div>

        {/* Right panel: object list + detail */}
        <div className="w-72 flex flex-col gap-3 overflow-y-auto">

          {/* Detail card for selected object */}
          {selectedId && (
            <div className="rounded-xl border border-cyan-500/40 bg-[#0d1b2a] p-4 text-xs space-y-2">
              <div className="flex items-center gap-2 mb-1">
                <Orbit className="w-4 h-4 text-cyan-400" />
                <span className="text-white font-semibold">NORAD {selectedId}</span>
                {selObj && (
                  <span className="ml-auto px-2 py-0.5 rounded-full text-white text-[10px]"
                    style={{ backgroundColor: REGIME_COLORS[selObj.regime] }}>
                    {selObj.regime}
                  </span>
                )}
              </div>
              {singleLoading && <p className="text-slate-400 animate-pulse">Loading…</p>}
              {singleData && (
                <>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-slate-300">
                    <span className="text-slate-500">Mean Alt</span>
                    <span className="text-right font-mono">{singleData.mean_alt_km.toFixed(0)} km</span>
                    <span className="text-slate-500">Data pts</span>
                    <span className="text-right font-mono">{singleData.n_points}</span>
                    <span className="text-slate-500">Period</span>
                    <span className="text-right font-mono">{computeOrbitalPeriod(singleData.mean_alt_km)}</span>
                    <span className="text-slate-500">Regime</span>
                    <span className="text-right font-mono">{singleData.regime}</span>
                  </div>
                  {singleData.trajectory.length > 0 && (() => {
                    const last = singleData.trajectory[singleData.trajectory.length - 1];
                    return (
                      <div className="pt-1 border-t border-slate-700 space-y-1 text-slate-300">
                        <p className="text-slate-500 uppercase tracking-wider text-[10px]">Last position</p>
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                          <span className="text-slate-500">Alt</span>
                          <span className="text-right font-mono">{last.altitude_km.toFixed(1)} km</span>
                          <span className="text-slate-500">Speed</span>
                          <span className="text-right font-mono">{last.speed_kms.toFixed(3)} km/s</span>
                          <span className="text-slate-500">Lat</span>
                          <span className="text-right font-mono">{last.lat.toFixed(2)}°</span>
                          <span className="text-slate-500">Lon</span>
                          <span className="text-right font-mono">{last.lon.toFixed(2)}°</span>
                        </div>
                      </div>
                    );
                  })()}
                </>
              )}
              <button
                onClick={() => setSelectedId(null)}
                className="w-full mt-1 py-1 rounded-lg text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 transition text-[11px]"
              >
                Clear selection
              </button>
            </div>
          )}

          {/* Object list */}
          <div className="rounded-xl border border-slate-800 bg-[#0d1b2a] overflow-hidden">
            <div className="px-3 py-2 border-b border-slate-800 flex items-center gap-2">
              <Info className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">
                Select object
              </span>
            </div>
            <div className="overflow-y-auto max-h-[520px]">
              {allLoading ? (
                <p className="text-xs text-slate-500 p-3 animate-pulse">Loading objects…</p>
              ) : (
                (allData?.objects ?? []).map((obj: TrajectoryObjectSummary) => (
                  <button
                    key={obj.norad_id}
                    onClick={() => setSelectedId(String(obj.norad_id))}
                    className={`w-full text-left px-3 py-2 flex items-center gap-2 border-b border-slate-800/50 hover:bg-slate-800/50 transition text-xs ${
                      String(obj.norad_id) === selectedId ? 'bg-slate-800/80' : ''
                    }`}
                  >
                    <span
                      className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: REGIME_COLORS[obj.regime] }}
                    />
                    <span className="text-slate-200 font-mono">{obj.norad_id}</span>
                    <span className="text-slate-500 ml-auto">{obj.mean_alt_km.toFixed(0)} km</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{ color: REGIME_COLORS[obj.regime], backgroundColor: REGIME_COLORS[obj.regime] + '20' }}>
                      {obj.regime}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-500 text-center pb-1">
        Ground tracks derived from ECI (x,y,z) → geodetic conversion · 5-min cadence · 72-hour window
        · Drag to rotate · scroll to zoom · click object for details
      </p>
    </div>
  );
}
