# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

- Setup (uv)
  - Install uv: https://docs.astral.sh/uv/
  - Create venv and install deps: `uv venv && uv sync`
- Setup (pip)
  - Create venv: `python -m venv .venv`
  - Activate (PowerShell): `.venv\Scripts\Activate.ps1`
  - Install: `pip install -e .`
- Environment
  - Set Gemini key (PowerShell): `$env:GEMINI_API_KEY = "<your_key>"`
- Run app
  - `streamlit run app.py` (server defaults from `.streamlit/config.toml`: port 5000, host 0.0.0.0)
- Tests / Lint
  - No tests or linters are configured in this repo.

## High-level Architecture

- Entry/UI: `app.py`
  - Streamlit UI with wide layout and styled components.
  - Sidebar handles file uploads (CSV/XLSX/XLS/JSON), validates and optimizes data (`utils.py`).
  - Session state keys: `chat_history`, `current_dataframe`, `data_analyzer`, `viz_generator`.
  - Tabs: AI Assistant (NL query -> analysis + viz), Analytics Dashboard (stats, quality, insights), Data Explorer (filter/export), Advanced Tools (outliers, correlations), plus a sidebar Visualization Center.
  - Visualization rendering via Plotly; image export via Kaleido.
- Analysis: `data_analyzer.py`
  - Wraps Google Generative AI (Gemini) client; constructs rich context from dataframe summary.
  - Core methods: `get_dataframe_summary`, `analyze_query_intent`, `suggest_multiple_visualizations`, `analyze_query`, `generate_automatic_insights`.
  - Defines `ChartType` enum and `VisualizationSuggestion`/`QueryAnalysis` dataclasses to structure recommendations.
  - Note: uses `os.getenv("GEMINI_API_KEY")`; ensure env var is set before running.
- Visualization: `visualization_generator.py`
  - Decides chart type from query/analysis; builds Plotly figures with a consistent blue palette.
  - Supports: histogram, bar, scatter, line, pie, box, correlation heatmap; plus `generate_automatic_visualizations`.
- Utilities: `utils.py`
  - `validate_dataframe`, `optimize_dataframe_types`, `safe_dataframe_display`, `chunk_dataframe`, `handle_missing_values`, etc.
  - Focus on memory reduction, display safety, and basic data hygiene.
- App config: `.streamlit/config.toml`
  - Server: headless, `address=0.0.0.0`, `port=5000`; theme colors/fonts.
- Packaging: `pyproject.toml`, `uv.lock`
  - Python 3.11+; deps include streamlit, pandas, plotly, numpy, openpyxl, kaleido, google-genai, sift-stack-py.

## Development Notes for Agents

- Start by ensuring `GEMINI_API_KEY` is available in the environment; the AI features rely on it.
- Preferred run flow: create/activate venv, install deps (uv or pip), then `streamlit run app.py`.
- Large CSVs are handled with chunked reads; the UI will optimize dtypes to reduce memory pressure.
- When generating code changes, preserve the Plotly color palette consistency and session state keys used across modules.
