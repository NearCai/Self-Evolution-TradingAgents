# Self-Evolution Skill TradingAgents

Self-Evolution Skill TradingAgents is a research-oriented multi-agent trading
framework for China A-share experiments. It extends the TradingAgents workflow
with a skill-evolution layer that converts completed trading episodes into
structured trading skills, verifies them, and injects accepted skills back into
future agent runs.

The repository is maintained as source code only. Generated artifacts and local
credential files are intentionally excluded from version control.

## Features

- Multi-agent trading workflow with analyst, researcher, trader, risk, and
  portfolio-management roles.
- China A-share data routing with market data validation and safe symbol
  handling.
- Continuous backtesting scripts for daily decision replay over a fixed window.
- Skill-evolution modules for experience extraction, candidate skill generation,
  and verifier-gated skill acceptance.
- Walk-forward and weekly online orchestration scripts for no-lookahead
  evaluation.
- CI coverage across Python 3.10, 3.11, 3.12, and 3.13, with strict Ruff linting.

## Repository Layout

| Path | Purpose |
|---|---|
| `tradingagents/` | Core package: agents, graph orchestration, dataflows, LLM clients, and skill evolution. |
| `tradingagents/evolution/` | Experience schema, skill synthesis, and verifier logic. |
| `scripts/` | Reproducible command-line workflows for A-share runs and skill evolution. |
| `cli/` | Interactive command-line application. |
| `tests/` | Unit and workflow tests used by CI. |
| `.github/workflows/` | Continuous integration configuration. |

## Installation

Use Python 3.10 or newer. Python 3.12 is recommended for local development.

```powershell
conda create -n tradingagents python=3.12
conda activate tradingagents
pip install -e ".[dev]"
```

For a clean runtime install without development tools:

```powershell
pip install .
```

## Configuration

Configure model and data-provider credentials through your shell, operating
system secret store, or CI secret manager. Do not commit local credential files.

For A-share workflows that should avoid noisy US-oriented sources, set:

```powershell
$env:TRADINGAGENTS_ENABLE_PREDICTION_MARKETS = "false"
$env:TRADINGAGENTS_ENABLE_US_SOCIAL_SOURCES = "false"
```

## Quick Start

Run the interactive CLI:

```powershell
tradingagents
```

Run an A-share continuous workflow:

```powershell
python scripts\run_continuous_backtest_ashare.py `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --start-date 2026-04-01 `
  --end-date 2026-06-30 `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --decision-source pm-rating `
  --memory-mode experiment `
  --memory-holding-days 5 `
  --output-dir artifacts\runs\ashare_q2_baseline `
  --disable-prediction-markets `
  --disable-us-social-sources
```

Run the weekly online skill-evolution controller:

```powershell
python scripts\run_weekly_online_skill_evolution.py `
  --baseline-dir artifacts\runs\ashare_q2_baseline `
  --output-dir artifacts\runs\weekly_skill_evolution `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --global-start-date 2026-04-01 `
  --online-start-date 2026-05-01 `
  --end-date 2026-06-30 `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --evolution-skill-types opportunity,promote `
  --disable-prediction-markets `
  --disable-us-social-sources
```

## Development

Run linting:

```powershell
ruff check .
```

Run the test suite:

```powershell
python -m pytest
```

Run the focused self-evolution checks:

```powershell
python -m pytest `
  tests\test_evolution_skills.py `
  tests\test_evolution_experience.py `
  tests\test_evolution_verifier.py `
  tests\test_continuous_ashare_backtest.py
```

## Security And Repository Hygiene

- Keep credentials in local shell configuration or a secret manager.
- Generated artifacts are ignored by Git.
- Cache folders, logs, archives, local databases, and credential files are not
  part of the public source tree.
- Before publishing, run `ruff check .`, `python -m pytest`, and a credential
  scan on tracked files.

## Disclaimer

This project is for research and engineering evaluation only. It is not
financial, investment, or trading advice.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
