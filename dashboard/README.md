# Space Debris TSA — React Dashboard

A React + TypeScript + Vite dashboard for visualising space debris tracking, trajectory analysis, and collision risk assessment.

---

## Pages

| Route | Page | Description |
|---|---|---|
| `/` | **Globe** | 3D WebGL Earth (Three.js / globe.gl) showing all tracked objects as dots with collision-risk arcs |
| `/analytics` | **Analytics & TSA** | Model benchmark RMSE bar chart, forecast overlay, residual diagnostics summary |
| `/collision-alerts` | **Collision Alerts** | Sortable, filterable table of all conjunction events with risk badges (CRITICAL / HIGH / MEDIUM / LOW) |
| `/debris` | **Debris Objects** | Full catalog of 200 objects — NORAD ID, altitude, speed, ECI coordinates, orbital regime |
| `/trajectories` | **Trajectories** | Animated ground-track globe for all 200 objects. Regime filter chips (LEO/MEO/HEO/GEO). Click any object to zoom camera and show detail card |
| `/system-status` | **System Status** | Pipeline run history, health indicators, last run timestamp |

---

## API

All data is served by `dashboard_api.py` (Flask, `http://localhost:5050`). The dashboard communicates via `src/api/client.ts` using `axios` with React Query for caching.

| Endpoint | Hook | Used by |
|---|---|---|
| `GET /api/dashboard/stats` | `useDashboardStats` | Globe, Analytics |
| `GET /api/collisions` | `useCollisions` | Collision Alerts |
| `GET /api/collisions/globe` | `useGlobeCollisions` | Globe |
| `GET /api/debris` | `useDebris` | Debris Objects |
| `GET /api/tsa/state` | `useTsaState` | Analytics |
| `GET /api/tsa/forecast` | `useForecast` | Analytics |
| `GET /api/tsa/conjunctions` | `useConjunctions` | Collision Alerts |
| `GET /api/debris/trajectory` | `useTrajectory` | Trajectories (single object) |
| `GET /api/debris/trajectories/all` | `useAllTrajectories` | Trajectories (all objects) |

---

## Key Components

### `TrajectoryPage.tsx`

- Renders a `globe.gl` scene with animated dashed arcs for all 200 object ground tracks
- Fetches downsampled trajectories from `/api/debris/trajectories/all?downsample=12`
- Regime filter state controls which objects are shown
- On object click: pans globe camera to current lat/lon, opens detail sidebar

### `GlobePage.tsx`

- Three.js / `globe.gl` 3D earth
- Debris objects as coloured dots
- Collision pair arcs colour-coded by risk level

### `CollisionAlerts.tsx`

- Full sortable table of conjunction events from `/api/tsa/conjunctions`
- Risk badge component (`RiskBadge.tsx`) renders CRITICAL/HIGH/MEDIUM/LOW
- Filterable by risk level and NORAD ID search

---

## TypeScript Types (`src/api/types.ts`)

```typescript
// Trajectory types
interface TrajectoryPoint { timestamp: string; x: number; y: number; z: number;
  lat: number; lon: number; altitude_km: number; speed_kms: number; }

interface TrajectoryResponse { norad_id: number; object_name: string;
  altitude_regime: OrbitalRegime; points: TrajectoryPoint[]; }

interface TrajectoryObjectSummary { norad_id: number; altitude_regime: OrbitalRegime;
  track: Array<{ lat: number; lon: number; altitude_km: number }>; }

interface AllTrajectoriesResponse { objects: TrajectoryObjectSummary[]; total: number; }

type OrbitalRegime = 'LEO' | 'MEO' | 'GEO' | 'HEO';
```

---

## Development

```bash
# From the dashboard/ directory:
npm install
npm run dev        # Vite dev server on http://localhost:5173

# Requires Flask API running in a separate terminal:
cd .. && source .venv/bin/activate.fish
python dashboard_api.py   # http://localhost:5050
```

## Build

```bash
npm run build      # Production build → dist/
npm run preview    # Preview the production build locally
```

## Linting

```bash
npm run lint       # ESLint with TypeScript rules
```
