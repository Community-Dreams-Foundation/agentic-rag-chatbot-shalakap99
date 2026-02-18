SHELL  := /bin/bash
PYTHON := python3

SAMPLE_PDF    := sample_docs/sample_paper.pdf
SAMPLE_PDF_URL := https://arxiv.org/pdf/1706.03762
SANITY_OUT    := artifacts/sanity_output.json

.PHONY: sanity download-sample run install clean

# ── Required by judges ─────────────────────────────────────────────────────
sanity: download-sample
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Agentic RAG — sanity check"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@mkdir -p artifacts data/chromadb_sanity
	$(PYTHON) scripts/run_sanity.py \
		--pdf    "$(SAMPLE_PDF)" \
		--output "$(SANITY_OUT)"
	@echo ""
	@echo "── Validating schema ────────────────────"
	$(PYTHON) scripts/verify_output.py "$(SANITY_OUT)"

# ── Download sample PDF if missing ────────────────────────────────────────
download-sample:
	@mkdir -p sample_docs
	@if [ ! -f "$(SAMPLE_PDF)" ]; then \
		echo "Downloading sample paper…"; \
		curl -L "$(SAMPLE_PDF_URL)" -o "$(SAMPLE_PDF)"; \
	fi

# ── Run Streamlit app ─────────────────────────────────────────────────────
run:
	streamlit run app/ui/streamlit_app.py --server.port 8501

# ── Install dependencies ──────────────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# ── Clean generated files ─────────────────────────────────────────────────
clean:
	rm -rf data/chromadb data/chromadb_sanity artifacts/sanity_output.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
