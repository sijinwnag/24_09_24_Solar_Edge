# Repository Guidelines

## Project Structure & Module Organization
- `Code/` holds production Python scripts (e.g., `25_09_05_Mismatch_results_generator_multi.py`, `mismatch_analysis.py`) plus experiment notebooks; notebooks capture trials, scripts ship repeatable workflows.
- `Data/<site_id>/` stores raw inverter exports; treat these files as read-only inputs and never commit them.
- `Results/<run_date>/` contains generated plots, CSVs, and Excel summaries; keep runs isolated by date/site (e.g., `Results/v_from_i_combined/25_08_25_Results/...`).
- `Logbook/` records experiment notes, `Paper/` holds manuscripts, and `old_data/` archives historical references only.

## Build, Test, and Development Commands
- Create a virtual environment: `python -m venv .venv` then `.\\.venv\\Scripts\\Activate.ps1`.
- Install core dependencies: `pip install numpy pandas matplotlib pvlib scikit-learn imageio requests`.
- Run the multi-orientation workflow: `python Code/25_09_05_Mismatch_results_generator_multi.py`; adjust `run_multi_orientation_analysis()` near the file end to target specific sites or seasons.
- Launch notebooks with `jupyter lab` from the activated environment for exploratory analyses.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indents, type hints, and docstrings that state the physical assumptions for each calculation.
- Name notebooks, scripts, and result folders with the `YY_MM_DD_description` pattern.
- Prefer side-effect-free helper functions; reserve module-level constants for configuration dictionaries such as `MULTI_ORIENTATION_SITES`.

## Testing Guidelines
- No automated suite yet; validate by rerunning the pipeline on a representative site (e.g., `3794347`) and reviewing printed loss summaries plus generated plots.
- When modifying transforms, export intermediate DataFrames to a fresh `Results/` subfolder and diff them against a known-good run.
- Log anomalies or manual checks in `Logbook/` to keep the investigation trail visible.

## Commit & Pull Request Guidelines
- Use imperative commit titles (e.g., `feat: refactor multi-string grouping`) and keep each commit focused on one change.
- Mention touched datasets or regenerated outputs in commit bodies so reviewers know what to download or ignore.
- PRs should state motivation, list validation steps (commands and observed results), and link relevant figures or issues; avoid including raw telemetry and offer sampling instructions if reviewers need data.

## Security & Configuration Tips
- Treat site telemetry as sensitive; never push raw exports or identifiable screenshots to public branches.
- Load credentials from environment variables (e.g., `$env:SOLAREDGE_TOKEN`) using `os.getenv` when scripting.
- Keep `Data/` read-only; stage only derived outputs under `Results/`.
