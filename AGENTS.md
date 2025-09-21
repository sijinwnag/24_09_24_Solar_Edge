# Repository Guidelines

## Project Structure & Module Organization
- Production Python scripts (`25_09_05_Mismatch_results_generator_multi.py`, `mismatch_analysis.py`) and supporting notebooks sit in `Code/`; notebooks capture experiments, scripts ship repeatable workflows.
- Raw inverter exports stay in `Data/<site_id>/`; treat them as read-only source inputs.
- Store generated plots, CSVs, and Excel summaries under `Results/<run_date>/` so analyses stay reproducible.
- Use `Logbook/` for experiment notes, `Paper/` for manuscripts, and `old_data/` for archival references only.

## Build, Test, and Development Commands
- Start with a virtual environment: `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`.
- Install core dependencies: `pip install numpy pandas matplotlib pvlib scikit-learn imageio requests`.
- Run the multi-orientation workflow via `python Code/25_09_05_Mismatch_results_generator_multi.py`; adjust the `run_multi_orientation_analysis()` call near the file tail to target specific sites or seasons.
- Launch `jupyter lab` from the activated environment for notebook-driven iterations.

## Coding Style & Naming Conventions
- Follow PEP 8, four-space indents, and keep type hints plus docstrings that state the physical assumptions behind each calculation.
- Continue the `YY_MM_DD_description` pattern for notebooks, scripts, and result folders.
- Prefer side-effect free helper functions; reserve module-level constants for configuration dictionaries such as `MULTI_ORIENTATION_SITES`.

## Testing Guidelines
- With no automated suite yet, rerun the pipeline on a representative site (e.g., `3794347`) and review the printed loss summary plus generated plots before merging.
- When editing transforms, export intermediate DataFrames to a fresh `Results/` subfolder and diff them against a known-good run.
- Capture manual test notes or anomalies in `Logbook/` to keep the investigation trail visible.

## Commit & Pull Request Guidelines
- Write imperative commit titles (`feat: refactor multi-string grouping`) and keep each commit focused on one change.
- Mention touched datasets or regenerated outputs in the body so reviewers know what to download or ignore.
- Pull requests should explain motivation, list validation steps, and link relevant figures or issues; scrub large raw files from the diff and offer sampling instructions if needed.

## Data Handling & Configuration
- Treat site telemetry as sensitive; never push raw exports or identifiable screenshots to public branches.
- Keep credentials in environment variables (for example, `$env:SOLAREDGE_TOKEN`) and load them with `os.getenv` when scripting.
