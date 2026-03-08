# ─────────────────────────────────────────────────────────────────────────────
#  Space Debris TSA — Project Makefile
#  Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────────────

PYTHON   := .venv/bin/python
PIPELINE := scripts/models/run_pipeline.py
API      := dashboard_api.py
DASH_DIR := dashboard

.PHONY: help install pipeline train eval conjunctions dashboard api dev \
        step1 step2 step3 step4 step5 dry-run clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Space Debris TSA — Available Targets"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install      Install Python dependencies"
	@echo "  make pipeline     Run the full 5-step TSA pipeline"
	@echo "  make dry-run      Preview pipeline steps without executing"
	@echo "  make step1        Step 1: TLE → SGP4 densification"
	@echo "  make step2        Step 2: Train VAR + LSTM + PatchTST"
	@echo "  make step3        Step 3: Evaluate & benchmark models"
	@echo "  make step4        Step 4: Conjunction / risk assessment"
	@echo "  make step5        Step 5: Update dashboard data files"
	@echo "  make api          Start the Flask dashboard API (port 5050)"
	@echo "  make dev          Start the React dashboard (port 5173)"
	@echo "  make clean        Remove generated Output/ artefacts"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name=tsa-venv \
	          --display-name "Python (.venv TSA)"
	cd $(DASH_DIR) && npm install
	@echo "✅ Dependencies installed."

# ── Pipeline steps ────────────────────────────────────────────────────────────
pipeline:
	$(PYTHON) $(PIPELINE)

dry-run:
	$(PYTHON) $(PIPELINE) --dry-run

step1:
	$(PYTHON) $(PIPELINE) --steps 1

step2:
	$(PYTHON) $(PIPELINE) --steps 2

step3:
	$(PYTHON) $(PIPELINE) --steps 3

step4:
	$(PYTHON) $(PIPELINE) --steps 4

step5:
	$(PYTHON) $(PIPELINE) --steps 5

# Skip training (use saved models), re-run evaluation + conjunction + dashboard
refresh:
	$(PYTHON) $(PIPELINE) --steps 3 4 5

# ── Servers ───────────────────────────────────────────────────────────────────
api:
	$(PYTHON) $(API)

dev:
	cd $(DASH_DIR) && npm run dev

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	rm -rf Output/ts_df.parquet \
	       Output/models/ \
	       Output/forecast_results.json \
	       Output/conjunction_report.json \
	       Output/dashboard_state.json \
	       Output/benchmark_table.csv \
	       Output/space_debris_catalog.csv \
	       Output/collision_alerts_*/
	@echo "✅ Output artefacts removed (raw data preserved)."
