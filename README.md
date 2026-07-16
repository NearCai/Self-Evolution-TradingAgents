# Self-Evolution Skill TradingAgents

Self-Evolution Skill TradingAgents is a research-oriented multi-agent trading
framework for China A-share experiments. It extends the TradingAgents workflow
with a skill-evolution layer that converts completed trading episodes into
structured skills, verifies them, and injects accepted skills into future agent
runs.

The public repository contains the runnable framework, experiment orchestration
scripts, tests, and project configuration needed to reproduce the workflows.

## Features

- Multi-agent trading workflow with analyst, researcher, trader, risk, and
  portfolio-management roles.
- China A-share data routing with market validation and safe symbol handling.
- Daily continuous backtesting for A-share portfolios.
- Skill extraction, synthesis, filtering, and verifier-gated acceptance.
- Fixed train/validation/test walk-forward workflow.
- Weekly online skill-evolution controller for no-lookahead iteration.

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

Use Python 3.10 or newer. Python 3.12 is recommended.

```powershell
conda create -n tradingagents python=3.12
conda activate tradingagents
pip install -e ".[dev]"
```

For a runtime-only install:

```powershell
pip install .
```

## Configuration

Set the credentials required by your selected model and data providers in the
local shell or runtime environment before launching an LLM workflow.

For the A-share experiment commands below:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:TRADINGAGENTS_ENABLE_PREDICTION_MARKETS = "false"
$env:TRADINGAGENTS_ENABLE_US_SOCIAL_SOURCES = "false"
```

## Interactive CLI

```powershell
tradingagents
```

## Experiment Reproduction

The commands below use three A-share tickers:

- `600519.SS`: Kweichow Moutai
- `000333.SZ`: Midea Group
- `600036.SS`: China Merchants Bank

### 1. Baseline Agent

This run evaluates the original multi-agent workflow without skill injection
over the 2026 Q2 window.

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
  --output-dir artifacts\ashare_q2_baseline `
  --disable-prediction-markets `
  --disable-us-social-sources

python scripts\evaluate_continuous_backtest_baselines.py `
  --output-dir artifacts\ashare_q2_baseline
```

### 2. Final Walk-Forward Skill Agent

This is the main self-evolution pipeline. It uses April decisions to build
experiences and candidate skills, validates the skill set in May, verifies
accepted skills, tests them in June, and stitches the full Q2 walk-forward
portfolio.

```powershell
python scripts\run_walkforward_skill_split.py `
  --baseline-dir artifacts\ashare_q2_baseline `
  --work-root artifacts\walkforward_q2 `
  --final-dir artifacts\walkforward_q2_best_skill_agent `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --decision-source pm-rating `
  --memory-mode experiment `
  --memory-holding-days 5 `
  --evolution-skill-max-skills 3 `
  --evolution-skill-max-chars 1800 `
  --evolution-skill-types opportunity,promote `
  --gate-preset research
```

### 3. Weekly Online Skill Evolution

This supplementary workflow updates or rejects the active skill library after
each completed weekly window.

```powershell
python scripts\run_weekly_online_skill_evolution.py `
  --baseline-dir artifacts\ashare_q2_baseline `
  --output-dir artifacts\weekly_online_skill_agent `
  --initial-skills-jsonl artifacts\walkforward_q2\train_2026_04_skills\candidate_skills.jsonl `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --global-start-date 2026-04-01 `
  --online-start-date 2026-05-01 `
  --end-date 2026-06-30 `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --decision-source pm-rating `
  --memory-mode experiment `
  --memory-holding-days 5 `
  --evolution-skill-max-skills 3 `
  --evolution-skill-max-chars 1800 `
  --evolution-skill-types opportunity,promote `
  --gate-preset research `
  --disable-prediction-markets `
  --disable-us-social-sources
```

## Notes

LLM-driven trading experiments are not fully deterministic. Outcomes may vary
with model sampling, provider behavior, data availability, and vendor updates.

This project is for research and engineering evaluation only. It is not
financial, investment, or trading advice.

## License

This project is distributed under the Apache License 2.0. See `LICENSE` for
details.
