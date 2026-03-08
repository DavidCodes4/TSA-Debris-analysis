# Space Debris Risk Prediction — Complete Project Explanation

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Technology Stack](#2-technology-stack)
3. [Project Structure](#3-project-structure)
4. [Data Sources](#4-data-sources)
5. [Component 1 — TLE Batch Processor](#5-component-1--tle-batch-processor)
6. [Component 2 — Live Ingestion Pipeline](#6-component-2--live-ingestion-pipeline)
7. [Component 3 — ML Training (MLlibTraining.scala)](#7-component-3--ml-training-mllibtrainingscala)
8. [Component 4 — Collision Prediction Engine](#8-component-4--collision-prediction-engine)
9. [Component 5 — Pipeline Scheduler](#9-component-5--pipeline-scheduler)
10. [Component 6 — Supporting Scala Files](#10-component-6--supporting-scala-files)
11. [HDFS Storage Layout](#11-hdfs-storage-layout)
12. [Kafka Streaming](#12-kafka-streaming)
13. [End-to-End Data Flow](#13-end-to-end-data-flow)
14. [ML Models — Deep Dive](#14-ml-models--deep-dive)
15. [Live Pipeline Run — What Happens Every 5 Minutes](#15-live-pipeline-run--what-happens-every-5-minutes)
16. [Numbers At a Glance](#16-numbers-at-a-glance)

---

## 1. What Is This Project?

This is a **real-time space debris collision risk prediction system**. It tracks ~22,000 active objects in Earth orbit (satellites + debris + rocket bodies), propagates their positions to the current moment using the SGP4 orbital mechanics model, applies three MLlib machine learning models to classify and analyse each object, then detects and ranks all pairs of objects that are dangerously close to each other.

**The core problem it solves:**  
Space is getting crowded. There are ~30,000 tracked objects in orbit. When two objects come within 50 km of each other, it is called a *close approach*. If the distance drops below 1 km, it is a critical collision risk. This project automates finding those pairs in near-real time, every 5 minutes, for every tracked object.

**Key outputs per run:**
- 40–130 collision pairs detected, risk-ranked CRITICAL / HIGH / MEDIUM / LOW
- Every collision alert published to a Kafka topic for downstream consumers
- All results saved to HDFS as timestamped Parquet batches
- ML enrichment: orbit shell, predicted altitude, predicted speed, and anomaly flags on every object

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orbital Mechanics | **SGP4** (Python `sgp4` library) | Propagate TLE to current ECI position/velocity |
| Big Data Storage | **Apache Hadoop HDFS** | Store TLE archive, state vectors, ML models, outputs |
| Stream Processing | **Apache Kafka** | Publish collision alerts to real-time consumers |
| Batch ML | **Apache Spark 3.5 + Spark MLlib** | Train and run ML models on state vectors |
| ML Training | **Scala / Spark MLlib** | Random Forest, K-Means, Linear Regression |
| ML Inference | **Scala (CollisionPrediction)** | K-Means + LR inference on live data |
| ML Inference | **Python / PySpark** (RF) | RF inference inside live_ingest.py |
| Ingestion | **Python 3** | Download from HDFS, propagate, classify, write back |
| Scheduling | **Python 3** (pipeline_scheduler.py) | Replace Airflow — cron-style 3-task runner |
| Build System | **sbt 1.11** (Scala Build Tool) | Compile + run Scala/Spark jobs |
| Orbital Data | **Space-Track.org TLE files** | Historical TLE archive (710 parquet files, ~168M rows) |
| Orbital Library | **Orekit 12.0** (Java/Scala) | Batch TLE processing with rigorous propagation |

---

## 3. Project Structure

```
Space-Debris-Risk-Prediction/
│
├── src/main/scala/
│   ├── MLlibTraining.scala          ← One-time ML training (RF + K-Means + LR)
│   ├── CollisionPrediction.scala    ← Main live pipeline (Spark batch, runs every 5 min)
│   ├── TLEBatchProcessor.scala      ← Historical TLE → HDFS archive builder
│   ├── TLEProcessor.scala           ← Single-file TLE processor utility
│   ├── TLEStreamProcessor.scala     ← Kafka-based streaming TLE processor
│   ├── StreamingCollisionDetector.scala  ← Streaming collision detector (Kafka source)
│   └── CollisionDetector.scala      ← Early standalone collision detector
│
├── live_ingest.py                   ← Python: HDFS archive → SGP4 → RF → HDFS parquet
├── pipeline_scheduler.py            ← Python: Airflow replacement scheduler
├── run_pipeline.sh                  ← Shell: start/stop/status/logs helper
│
├── build.sbt                        ← Scala/Spark/Orekit/Kafka dependencies
├── orekit-data/                     ← Orekit earth orientation, ephemeris, weather data
│
└── Output/
    └── space_debris_catalog.csv     ← Catalog of all tracked objects (35,731 rows)
```

---

## 4. Data Sources

### 4.1 TLE Archive (HDFS)
- **Path:** `hdfs://localhost:9000/space-debris/state-vectors-archive`
- **Format:** Parquet, Snappy-compressed
- **Size:** 710 files, ~168 million rows
- **Content:** Historical TLE line pairs for every tracked NORAD object
- **Columns:** `NORAD_ID, EPOCH, TLE_LINE1, TLE_LINE2`
- **Origin:** Downloaded from Space-Track.org, processed via `TLEBatchProcessor.scala`

### 4.2 Object Catalog (HDFS)
- **Path:** `hdfs://localhost:9000/space-debris/catalog`
- **Format:** Parquet
- **Size:** 35,731 objects
- **Columns:** `NORAD_CAT_ID, OBJECT_NAME, OBJECT_TYPE, COUNTRY, PERIOD, INCLINATION, APOGEE, PERIGEE, RCS_SIZE`
- **Used for:** Joining to enrich collision results with object name + country, and training ML models

### 4.3 Live State Vectors (HDFS — written every run)
- **Path:** `hdfs://localhost:9000/space-debris/state-vectors/live_sv_*.parquet`
- **Format:** Parquet, Snappy-compressed, one file per pipeline run
- **Size:** ~1.5–1.8 MB per file, ~9,000–12,000 rows per file
- **Columns:** `NORAD_ID, OBJECT_TYPE, EPOCH, POS_X, POS_Y, POS_Z, VEL_X, VEL_Y, VEL_Z, ALTITUDE_KM, SPEED_KMS, TLE_LINE1, TLE_LINE2`
- **Accumulates** over time — every 5-minute run appends one new file

---

## 5. Component 1 — TLE Batch Processor

**File:** `src/main/scala/TLEBatchProcessor.scala`  
**Run once:** `sbt "runMain TLEBatchProcessor"`

### What It Does
This was the **first step** — a one-time historical data loader. It reads raw TLE text files (downloaded from Space-Track.org) from the local `data1/` directory, propagates each TLE using **Orekit** (the industry-standard Java orbital mechanics library), and writes the resulting state vectors as Parquet to HDFS.

### Why Orekit Here (Not SGP4)?
Orekit provides higher-precision force models (atmospheric drag, lunar/solar gravity, etc.) for batch historical processing. The live pipeline uses the faster `sgp4` Python library for real-time propagation.

### Steps Inside
```
1. Load Orekit Earth data (orekit-data/ folder)
2. Discover all .txt TLE files in data1/ directory
3. For each file:
   a. Parse TLE line pairs
   b. Propagate with OrekitTLE + TLEPropagator
   c. Extract ECI position (km) + velocity (km/s)
   d. Compute ALTITUDE_KM = |r| - 6371
4. Write all records to HDFS as Parquet
```

### Output Schema
```
NORAD_ID, TLE_LINE1, TLE_LINE2, EPOCH,
POS_X, POS_Y, POS_Z,     ← km, ECI TEME frame
VEL_X, VEL_Y, VEL_Z,     ← km/s
ALTITUDE_KM, OBJECT_TYPE
```

---

## 6. Component 2 — Live Ingestion Pipeline

**File:** `live_ingest.py`  
**Run by scheduler:** Every 5 minutes  
**Command:** `python3 live_ingest.py --n-sat 5000 --n-deb 10000 --sample-file 2`

### Purpose
Replaces any static snapshot with a **live propagation to right now**. Every run downloads the latest TLE for each NORAD object from the HDFS archive, propagates it to the current UTC epoch using SGP4, classifies each object using the RF model, then writes the resulting state vectors back to HDFS.

### Step-by-Step

#### Step 1 — Read TLE Archive from HDFS
```python
# Downloads parquet files from HDFS via WebHDFS REST API
files = hdfs_list("/space-debris/state-vectors-archive")
# Uses only 2 files by default (--sample-file 2) for speed
# Deduplicates: keeps latest TLE per NORAD_ID by EPOCH descending
combined.sort_values("EPOCH", ascending=False).drop_duplicates("NORAD_ID")
# Result: ~22,253 unique NORAD objects with their most recent TLE
```

#### Step 2 — SGP4 Propagation to NOW
```python
from sgp4.api import Satrec, jday
# For each TLE:
sat = Satrec.twoline2rv(TLE_LINE1, TLE_LINE2)
e, pos, vel = sat.sgp4(jd, jdfr)  # propagate to current epoch
# Filters: altitude < 150 km → skip (re-entry)
#          sgp4 error code != 0 → skip (invalid TLE)
# Result: ~20,737 valid state vectors
# ECI coordinates in TEME frame, km and km/s
```

#### Step 2b — RF Classification
```python
# Loads the trained Random Forest from HDFS via PySpark
rf_model = RandomForestClassificationModel.load(HDFS_RF_MODEL)
# Derives orbital features from ECI vectors:
#   PERIOD = 2π × √(r³/GM) / 60     (Kepler's 3rd law, minutes)
#   INCLINATION = arccos(hz / |h|)   (angular momentum vector)
#   APOGEE = PERIGEE = ALTITUDE_KM   (near-circular approximation)
# Batch-predicts OBJECT_TYPE for all records
# Overwrites the heuristic NORAD_ID % 5 label with the RF prediction
```

> ⚠️ **Current status:** PySpark is not installed in the venv, so the RF step falls back silently to the `NORAD_ID % 5` heuristic. Install with: `venv/bin/pip install pyspark`

#### Step 3 — Sampling
```python
# Caps the output to prevent massive cross-joins downstream
sats = random.sample(satellites, min(N_SAT, len(sats)))  # cap at 5,000
debs = random.sample(debris,     min(N_DEB, len(debs)))  # cap at 10,000
# Result: ~5,000 SATELLITES + ~4,000–8,000 DEBRIS = ~9,000–13,000 total
```

#### Step 4a — Direct HDFS Write (Kafka Disabled)
```python
# Writes Snappy-compressed Parquet directly via WebHDFS
# Filename: live_sv_YYYYMMDD_HHMMSS.parquet
# Size: ~1.5–1.8 MB
```

#### Step 4b — Kafka Path (When Kafka Enabled)
```python
# Produces each state vector as a JSON message to Kafka topic: state-vectors-live
# A parallel consumer thread (KafkaToHdfsConsumer) reads batches of 500 messages
# and writes them to HDFS in the same format
# Both paths produce identical HDFS output
```

---

## 7. Component 3 — ML Training (MLlibTraining.scala)

**File:** `src/main/scala/MLlibTraining.scala`  
**Run once:** `sbt "runMain MLlibTraining"`  
**Purpose:** Train all 4 models and save them permanently to HDFS

### Part 1 — Linear Regression: Altitude Prediction

**Training data:** 500,000 state vectors sampled from HDFS archive  
**Split:** 80% train, 20% test

```scala
// Features: full 6D state vector
val assembler = new VectorAssembler()
  .setInputCols(Array("POS_X", "POS_Y", "POS_Z", "VEL_X", "VEL_Y", "VEL_Z"))
  .setOutputCol("features")

// Label: orbital radius (sqrt(x²+y²+z²)) ← NOT altitude above surface
// Note: label is RADIUS = ALTITUDE + 6371 km
val lrAltitude = new LinearRegression()
  .setLabelCol("ALTITUDE")   // ALTITUDE here = sqrt(x²+y²+z²), radius in km
  .setFeaturesCol("features")
  .setMaxIter(100)
  .setRegParam(0.01)
  .setElasticNetParam(0.5)   // Elastic Net regularisation
```

**Saved to:** `hdfs://localhost:9000/space-debris/models/trajectory-altitude`  
**Purpose in pipeline:** Predicts expected orbital radius from state vector. A large delta (`ALTITUDE_DELTA_KM > 50`) means the TLE is stale or the object has manoeuvred.

---

### Part 2 — Linear Regression: Speed Prediction

**Training data:** Same 500,000 state vectors

```scala
// Features: position ONLY (no velocity)
val posAssembler = new VectorAssembler()
  .setInputCols(Array("POS_X", "POS_Y", "POS_Z"))
  .setOutputCol("pos_features")

// Label: orbital speed sqrt(vx²+vy²+vz²) in km/s
val lrSpeed = new LinearRegression()
  .setLabelCol("SPEED")
  .setFeaturesCol("pos_features")
  .setMaxIter(100)
  .setRegParam(0.01)
```

**Saved to:** `hdfs://localhost:9000/space-debris/models/trajectory-speed`  
**Purpose in pipeline:** Predicts expected speed from position alone (Kepler's law: objects at a given radius have a predictable speed). A large `SPEED_DELTA_KMS` flags a manoeuvring satellite (thruster fired).

---

### Part 3 — K-Means Clustering: Orbit Shell

**Training data:** 35,731 catalog objects with orbital parameters

```scala
val orbitAssembler = new VectorAssembler()
  .setInputCols(Array("PERIOD", "INCLINATION", "APOGEE", "PERIGEE"))
  .setOutputCol("orbit_features")

val kmeans = new KMeans()
  .setK(4)           // 4 natural orbit shells
  .setSeed(42)
  .setMaxIter(50)
```

**4 Clusters correspond to:**

| Cluster | Shell | Altitude Range | Examples |
|---|---|---|---|
| 0 | LEO | < 2,000 km | ISS, Starlink, most debris |
| 1 | MEO | 2,000 – 20,000 km | GPS, Galileo |
| 2 | GEO | ~35,786 km | Weather, comms satellites |
| 3 | HEO | > 40,000 km | Molniya, Tundra orbits |

**Saved to:** `hdfs://localhost:9000/space-debris/models/orbit-clustering`  
**Purpose in pipeline:** Tags every active object with its orbit shell. The shell label travels through to every Kafka alert and HDFS output record.

---

### Part 4 — Random Forest Classifier: Debris/Satellite

**Training data:** 35,731 labeled catalog objects

```scala
// Label preparation
val labelIndexer = new StringIndexer()
  .setInputCol("LABEL_NAME")     // "DEBRIS" or "SATELLITE"
  .setOutputCol("label")         // 0.0 or 1.0

// Same 4 orbital features as K-Means
val classifierAssembler = new VectorAssembler()
  .setInputCols(Array("PERIOD", "INCLINATION", "APOGEE", "PERIGEE"))
  .setOutputCol("features")

val rf = new RandomForestClassifier()
  .setNumTrees(20)
  .setMaxDepth(5)
  .setLabelCol("label")
  .setFeaturesCol("features")
```

**Accuracy achieved:** ~92% on 20% test split  
**Saved to:**
- `hdfs://localhost:9000/space-debris/models/debris-classifier`
- `hdfs://localhost:9000/space-debris/models/classifier-label-indexer`

**Purpose in pipeline:** Called from `live_ingest.py` (Step 2b) to classify every propagated object as SATELLITE or DEBRIS using its orbital characteristics — much more accurate than the `NORAD_ID % 5` heuristic.

---

## 8. Component 4 — Collision Prediction Engine

**File:** `src/main/scala/CollisionPrediction.scala`  
**Language:** Scala + Apache Spark 3.5 + Spark MLlib  
**Run by scheduler:** Every 5 minutes via `sbt "runMain CollisionPrediction"`

This is the **main analytical engine**. It reads from HDFS, runs all ML inferences, performs collision detection, classifies risk, and outputs to HDFS + Kafka.

### Step 1 — Initialize Spark Session
```scala
SparkSession.builder()
  .master("local[14]")            // use all 14 CPU cores
  .config("spark.driver.memory", "6g")
  .config("spark.sql.shuffle.partitions", "32")   // tuned for local mode
  .config("spark.sql.adaptive.enabled", "true")   // AQE for skew handling
```

### Step 2 — Load State Vectors from HDFS [1/7]
```scala
spark.read
  .option("mergeSchema", "false")
  .parquet("hdfs://.../state-vectors/live_sv_*.parquet")
  // Reads ALL accumulated live_sv_* files (grows every run)
  // Normalises NORAD_ID to IntegerType
  // Computes ALTITUDE_KM and VELOCITY_KMS if not present
```
**Output:** ~18,000–20,000 raw rows across all historical runs

### Step 3 — Load Catalog & Deduplicate [2/7]
```scala
// Dedup: keep only the LATEST epoch per NORAD_ID
Window.partitionBy("NORAD_ID").orderBy(col("EPOCH").desc)
// Left-join with catalog (broadcast join — catalog fits in memory)
// Adds: OBJECT_NAME, COUNTRY, RCS_SIZE
// Assigns CLASSIFICATION: SATELLITE | DEBRIS (from OBJECT_TYPE)
```

### Step 4 — Tracking Stop Filter [4/7]
```scala
// Objects are marked STOPPED and excluded from collision detection if:
//   ALTITUDE_KM < 150 km  → re-entering atmosphere
//   ALTITUDE_KM > 100,000 km → unrealistic data
//   ALTITUDE_KM is NULL/NaN → invalid state vector
// Stopped objects are saved to HDFS for audit trail
// Result: ~14,500–16,500 ACTIVE objects
```

### Step 4b — Orbit Shell Tagging via K-Means [4b/7]
```scala
KMeansModel.load(HDFS_KMEANS_MODEL)
// Derives orbital features from ECI vectors:
//   PERIOD_APPROX = 2π × √((alt + 6371)³ / GM) / 60
//   INCLINATION_APPROX = 0.0  (placeholder — not in ECI parquet)
val orbitAssembler = new VectorAssembler()
  .setInputCols(Array("PERIOD_APPROX", "INCLINATION_APPROX", "ALTITUDE_KM", "ALTITUDE_KM"))
  .setOutputCol("orbit_features")
// Cluster → Shell mapping uses altitude thresholds (not raw cluster index)
// because K-Means cluster ordering is non-deterministic across training runs
// Adds: ORBIT_SHELL (LEO/MEO/GEO/HEO), ORBIT_CLUSTER (0-3)
// Fallback: if model fails → same altitude-threshold logic, ORBIT_CLUSTER = -1
```

### Step 4c — LR Predicted Altitude & Speed [4c/7]
```scala
// Altitude model (6 features: pos + vel)
LinearRegressionModel.load(HDFS_LR_ALTITUDE_MODEL)
lrAssembler6 → "features" column
PREDICTED_ALTITUDE_KM = round(prediction - 6371.0, 3)  // subtract Earth radius
ALTITUDE_DELTA_KM = |ALTITUDE_KM - PREDICTED_ALTITUDE_KM|

// Speed model (3 features: pos only)
LinearRegressionModel.load(HDFS_LR_SPEED_MODEL)
lrAssembler3 → "pos_features" column
PREDICTED_SPEED_KMS = round(prediction, 6)
SPEED_DELTA_KMS = |VELOCITY_KMS - PREDICTED_SPEED_KMS|
```

### Step 5 — Collision Detection [5/7]

The naive approach (cross-join all pairs) is O(n²) — impossible for 15,000 objects (225M pairs). Instead, **altitude bucketing** is used:

```
Objects are assigned an altitude "bucket" of 50 km width.
SAT-1 at 550 km → bucket 11
SAT-2 is "exploded" into 3 buckets: [bucket-1, bucket, bucket+1]
Join on bucket number → only candidate pairs within ±50 km altitude are evaluated
Then compute exact 3D Euclidean distance for those candidates
```

```scala
// Bucket assignment
floor(col("alt_1") / BUCKET_SIZE)                   // single bucket for left side
explode(array(bucket, bucket-1, bucket+1))           // 3-bucket coverage for right side

// Exact distance (ECI 3D)
sqrt(pow(x2-x1,2) + pow(y2-y1,2) + pow(z2-z1,2))

// Filter: distance_km <= 50.0
// Types detected: SAT-SAT, SAT-DEB (DEB-DEB excluded — not actionable)
// Caps: max 10,000 satellites, max 15,000 debris
```

**Typical output:** 40–130 collision pairs per run

### Step 6 — Risk Classification [6/7]
```scala
when(distance_km <= 1.0,  "CRITICAL")  // imminent collision
when(distance_km <= 20.0, "HIGH")      // operator alert required
when(distance_km <= 35.0, "MEDIUM")    // monitoring required
otherwise                  "LOW"       // watch list

// Collision probability (inverse-distance model)
P = 1 / (1 + d²)    // d in km; P = 1.0 when d ≤ 0.01 km

// Relative velocity estimate
rel_vel = vel_1 + vel_2   // scalar approximation (km/s)
```

### Step 7 — Save Results [7/7]

| Output | Path | Format |
|---|---|---|
| Full batch | `hdfs://.../collision-predictions/batch_YYYYMMDD_HHmmss` | Parquet |
| HIGH+CRITICAL only | `hdfs://.../collision-predictions/high_risk_batch_*` | Parquet |
| Pipeline metrics | `hdfs://.../collision-predictions/pipeline_metrics` | Parquet (append) |
| Stopped tracking | `hdfs://.../stopped-tracking/batch_*` | Parquet |
| Kafka alerts | topic: `space_debris_collisions` | JSON (all pairs) |

---

## 9. Component 5 — Pipeline Scheduler

**Files:** `pipeline_scheduler.py`, `run_pipeline.sh`  
**Purpose:** Replaces Apache Airflow. Runs 3 tasks in sequence every N minutes.

### Architecture
```
pipeline_scheduler.py
  └── runs as a background daemon (nohup, PID saved to scheduler.pid)
  └── every 5 minutes, executes 3 tasks:

  Task 1: Health Check
    - HTTP GET to HDFS WebHDFS (port 9870)
    - TCP connect to Kafka broker (port 19092)
    - Stops pipeline run if either service is down

  Task 2: TLE Ingestion
    - Calls: venv/bin/python3 live_ingest.py --n-sat 5000 --n-deb 10000 --sample-file 2
    - Streams stdout/stderr live into scheduler.log

  Task 3: Collision Prediction
    - Calls: sbt "runMain CollisionPrediction"
    - Streams all sbt/Spark output into scheduler.log
```

### run_pipeline.sh Commands
```bash
./run_pipeline.sh start        # start background scheduler (every 5 min)
./run_pipeline.sh start 10     # start with 10-minute interval
./run_pipeline.sh stop         # kill the background process
./run_pipeline.sh restart      # stop + start
./run_pipeline.sh status       # show PID + last 20 log lines
./run_pipeline.sh logs         # tail -f scheduler.log (live output)
./run_pipeline.sh once         # run exactly once then exit
```

### Log Format
```
2026-03-05 10:30:09  INFO   Run #1 │ Pipeline started — 2026-03-05T05:00:09Z
2026-03-05 10:30:09  INFO   [Task 1/3] Health checks…
2026-03-05 10:30:09  INFO     ✅ HDFS WebHDFS — reachable
2026-03-05 10:30:09  INFO     ✅ Kafka broker localhost:19092 — reachable
2026-03-05 10:30:09  INFO   [Task 2/3] TLE ingestion (sgp4 → HDFS)…
2026-03-05 10:30:11  INFO     ✅ live_ingest.py — success
2026-03-05 10:30:11  INFO   [Task 3/3] Collision prediction (Spark / Scala)…
2026-03-05 10:30:31  INFO     ✅ CollisionPrediction.scala — success
2026-03-05 10:30:31  INFO   Run #1 complete in 22.3s
2026-03-05 10:30:31  INFO   health=True  ingest=True  collision=True
```

---

## 10. Component 6 — Supporting Scala Files

### `TLEProcessor.scala`
Single-file TLE processor. Reads one TLE text file, propagates with Orekit, writes to HDFS. Used for testing individual TLE files.

### `TLEStreamProcessor.scala`
Kafka-based streaming TLE processor. Consumes raw TLE messages from a Kafka topic, propagates with Orekit in micro-batches, writes state vectors to HDFS. Part of the original Kafka-first architecture.

### `StreamingCollisionDetector.scala`
A continuously-running streaming collision detector. Polls HDFS for new state vector files every N seconds (instead of batch-style). Uses the same bucketing collision detection algorithm as `CollisionPrediction.scala`. Publishes alerts to Kafka topic `collision-alerts`.

### `CollisionDetector.scala`
Early standalone version of the collision detector. Used local CSV catalog rather than HDFS. Predecessor to `CollisionPrediction.scala`.

---

## 11. HDFS Storage Layout

```
hdfs://localhost:9000/space-debris/
│
├── state-vectors-archive/          ← 710 parquet files, 168M rows (TLE history)
│   └── part-*.snappy.parquet
│
├── state-vectors/                  ← Live state vectors (one file per pipeline run)
│   └── live_sv_YYYYMMDD_HHMMSS.parquet
│
├── catalog/                        ← 35,731 object catalog
│   └── part-*.parquet
│
├── models/                         ← Trained ML models (saved by MLlibTraining.scala)
│   ├── debris-classifier/          ← Random Forest (SATELLITE/DEBRIS)
│   ├── classifier-label-indexer/   ← StringIndexer for RF labels
│   ├── orbit-clustering/           ← K-Means K=4 (LEO/MEO/GEO/HEO)
│   ├── trajectory-altitude/        ← LR model (6D → radius)
│   └── trajectory-speed/           ← LR model (3D position → speed)
│
├── collision-predictions/          ← Output from CollisionPrediction.scala
│   ├── batch_YYYYMMDD_HHmmss/      ← All collision pairs (Parquet)
│   ├── high_risk_batch_*/          ← CRITICAL+HIGH only (Parquet)
│   └── pipeline_metrics/           ← Appended metrics per run (Parquet)
│
├── stopped-tracking/               ← Objects that stopped being tracked
│   └── batch_YYYYMMDD_HHmmss/
│
└── ml-results/                     ← ML training outputs
    ├── clustered-debris/            ← K-Means predictions on catalog
    └── metrics/                    ← Training accuracy metrics (JSON)
```

---

## 12. Kafka Streaming

### Topics

| Topic | Producer | Consumer | Message Format | Purpose |
|---|---|---|---|---|
| `state-vectors-live` | `live_ingest.py` | `KafkaToHdfsConsumer` thread | JSON (one state vector) | Parallel HDFS write path |
| `space_debris_collisions` | `CollisionPrediction.scala` | Dashboard / downstream | JSON (collision record) | Real-time alerts |

### Collision Alert Message Structure
Every Kafka message on `space_debris_collisions` is a JSON record containing:
```json
{
  "norad_1": 37704,
  "name_1": "FENGYUN 1C DEB",
  "norad_2": 38168,
  "name_2": "FENGYUN 1C DEB",
  "collision_type": "SAT-DEB",
  "distance_km": 7.824884025,
  "risk_level": "HIGH",
  "collision_probability": 0.01606972,
  "relative_velocity_kms": 14.2,
  "orbit_shell_1": "LEO",
  "orbit_shell_2": "LEO",
  "orbit_cluster_1": 0,
  "orbit_cluster_2": 0,
  "alt_1": 845.2,
  "alt_2": 842.1,
  "detection_timestamp": "2026-03-05T10:30:30",
  "batch_id": "20260305_103015"
}
```

---

## 13. End-to-End Data Flow

```
Space-Track.org (historical TLE data)
        │
        ▼  [One-time, TLEBatchProcessor.scala]
HDFS /state-vectors-archive   (710 files, 168M rows)
        │
        ▼  [Every 5 min, live_ingest.py Step 1]
Latest TLE per NORAD_ID (~22,253 objects)
        │
        ▼  [Step 2: SGP4 propagation to NOW]
ECI State Vectors — position + velocity at this exact second
        │
        ▼  [Step 2b: RF Classifier (PySpark)]
OBJECT_TYPE = "SATELLITE" | "DEBRIS"  for each object
        │
        ▼  [Step 3: Sampling]
~5,000 SAT + ~4,000-8,000 DEB = ~9,000-13,000 objects selected
        │
        ├──────────────────────────────────────────────────────────►  Kafka: state-vectors-live
        ▼  [Step 4: Direct HDFS write]                                         │
HDFS live_sv_*.parquet  (~1.6 MB, ~10,000 rows)                     KafkaToHdfsConsumer
        │                                                                       │
        │◄──────────────────────────────────────────────────────────────────────┘
        │
        ▼  [CollisionPrediction.scala Step 1]
Read ALL live_sv_*.parquet  (~20,000 rows across all historical runs)
        │
        ▼  [Step 2: Dedup + Catalog join]
Latest position per NORAD_ID, enriched with name/country/RCS
        │
        ▼  [Step 3: Tracking stop filter]
~14,500-16,500 ACTIVE objects  (150 km < altitude < 100,000 km)
        │
        ▼  [Step 4b: K-Means inference]
ORBIT_SHELL = LEO | MEO | GEO | HEO  added to every row
        │
        ▼  [Step 4c: LR Altitude + Speed inference]
PREDICTED_ALTITUDE_KM, ALTITUDE_DELTA_KM,
PREDICTED_SPEED_KMS,   SPEED_DELTA_KMS     added to every row
        │
        ▼  [Step 5: Altitude-bucket collision detection]
Candidates: ~500K pairs evaluated  →  40-130 pairs within 50 km
        │
        ▼  [Step 6: Risk classification + probability]
risk_level = CRITICAL | HIGH | MEDIUM | LOW
collision_probability = 1/(1+d²)
        │
        ├──────────────────────────────────────────────────────────►  Kafka: space_debris_collisions
        ▼  [Step 7: HDFS save]                                          (~100 KB, 40-130 messages)
HDFS collision-predictions/batch_*  (all pairs)
HDFS high_risk_batch_*              (CRITICAL+HIGH only)
HDFS pipeline_metrics               (run statistics)
```

---

## 14. ML Models — Deep Dive

### Why These 3 Models?

| Model | Problem Solved | Why This Algorithm |
|---|---|---|
| **RF Classifier** | "Is this object a satellite or debris?" | RF handles non-linear decision boundaries well. Orbital parameters cluster differently for SATELLITE vs DEBRIS. High accuracy (~92%) on catalog labels. |
| **K-Means** | "Which orbit shell is this object in?" | Unsupervised — no labels needed. 4 clear natural clusters in orbital parameter space (LEO/MEO/GEO/HEO). Deterministic with seed=42. |
| **LR Altitude** | "Is this object where its TLE says it should be?" | LR learns the Keplerian relationship: given position+velocity, altitude is analytically predictable. Large deviation = stale TLE / manoeuvre. |
| **LR Speed** | "Is this object moving at the speed its position implies?" | Kepler's law: v ∝ 1/√r. LR learns this relationship from position alone. Deviation = thrust event. |

### Feature Engineering from ECI Vectors

The catalog has orbital elements (PERIOD, INCLINATION, APOGEE, PERIGEE) but live state vectors only have ECI Cartesian coordinates. The bridge:

```
Given: POS_X, POS_Y, POS_Z (km), VEL_X, VEL_Y, VEL_Z (km/s)

Orbital Radius:
  r = √(x² + y² + z²)
  ALTITUDE_KM = r - 6371

Orbital Period (Kepler's 3rd law):
  T = 2π × √(r³ / GM) / 60   where GM = 398600.4418 km³/s²

Inclination (angular momentum vector):
  h = r × v  (cross product)
  i = arccos(hz / |h|)

Apogee ≈ Perigee ≈ ALTITUDE_KM  (near-circular orbit assumption)
```

---

## 15. Live Pipeline Run — What Happens Every 5 Minutes

```
T+0:00   Scheduler wakes up
T+0:00   Task 1: Health check → HDFS reachable, Kafka reachable
T+0:02   Task 2: live_ingest.py starts
T+0:04     Read 475,975 TLE rows from 2 HDFS archive files
T+0:06     Deduplicated to 22,253 unique NORAD objects
T+0:14     SGP4 propagated: 20,737 OK, 1,516 errors
T+0:14     RF classifier (if available) — else heuristic fallback
T+0:16     Sampled: 5,000 SAT + 4,145 DEB = 9,145 total
T+0:18     Written to HDFS: live_sv_*.parquet  (1,590 KB)
T+0:22   Task 3: CollisionPrediction.scala starts (sbt JVM startup ~15s)
T+0:37     Spark Session initialized
T+0:38     Loaded 20,028 rows from all live_sv_*.parquet files
T+0:39     Deduped to ~17,000 latest positions
T+0:39     Joined with catalog (35,731 objects, broadcast)
T+0:40     Tracking filter: ~16,500 ACTIVE, ~163 STOPPED
T+0:41     K-Means: ORBIT_SHELL tagged (or altitude fallback)
T+0:41     LR Altitude: PREDICTED_ALTITUDE_KM computed
T+0:42     LR Speed: PREDICTED_SPEED_KMS computed
T+0:43     SAT-SAT detection: 41 pairs within 50 km
T+0:44     SAT-DEB detection: 85 pairs within 50 km
T+0:44     Risk classification: 2 CRITICAL, 5 HIGH, 39 MEDIUM, 80 LOW
T+0:45     Saved to HDFS: batch_20260305_103015
T+0:45     Published 126 alerts to Kafka: space_debris_collisions
T+0:45   Run #1 complete in 22.3s
T+5:00   Scheduler wakes up again → Run #2
```

---

## 16. Numbers At a Glance

| Metric | Value |
|---|---|
| Tracked NORAD objects | ~22,253 unique |
| Successful SGP4 propagations | ~20,737 per run |
| Active objects after filters | ~14,500–16,500 |
| State vector files in HDFS | grows by 1 per run (66+ files currently) |
| TLE archive size | 710 files, ~168 million rows |
| ML training dataset (LR/Speed) | 500,000 sampled state vectors |
| ML training dataset (RF/K-Means) | 35,731 catalog objects |
| RF classifier accuracy | ~92% |
| Collision pairs detected per run | 40–130 |
| HIGH or CRITICAL alerts per run | 2–10 |
| Kafka messages per run | 40–130 JSON records (~100 KB) |
| Pipeline duration | 22–45 seconds per run |
| Spark memory | 6 GB driver, 14 cores (local mode) |
| HDFS replication factor | 1 (single-node cluster) |
| Run interval | every 5 minutes |
