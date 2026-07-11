# Experiment Plan and Status

This file is the running experiment ledger for the Agent-Based self-evolving
trading system project. Keep it updated after each verified run so the final
research report can trace which results are finished, which are exploratory,
and which commands reproduce them.

Last updated: 2026-07-11

## Status Legend

- `[x]` completed and checked.
- `[~]` completed but treated as exploratory/reference, not the final strict result.
- `[ ]` planned / not yet run.

## Completed Result Directories

| Status | Experiment | Result directory | Setting | Key result | Notes |
|---|---|---|---|---|---|
| `[~]` | Kimi full-analyst A-share exploratory run | `results/continuous_ashare_2026_06_kimi_full` | 5 stocks, 2026-06-01 to 2026-06-29, `market,news,fundamentals,social`, Kimi | 100/100 ok; portfolio CR `-2.67%`; Buy&Hold `-0.88%`; benchmark `+3.66%` | Useful as early full-analyst reference. Not final baseline because runtime/cost were high and news/social quality was unstable. |
| `[x]` | Final current A-share baseline | `results/continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm` | 3 main-board stocks, 2026-04-01 to 2026-06-29, `market,fundamentals`, DeepSeek flash, PM rating, long/cash | 177/177 ok; portfolio CR `-0.49%`; Buy&Hold `-8.37%`; benchmark `+8.40%` | Current canonical baseline. Paper-style rule evaluation and report figures were generated from this run. |
| `[x]` | Baseline-derived experience and candidate skills | `results/continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm/evolution` | Experiences distilled from current baseline; deterministic skill generator | 7 candidate skills: 5 caution, 2 opportunity | This is the first self-evolution artifact. It is useful for mechanism development, but strict walk-forward needs date-split generation. |
| `[x]` | First skill injection, caution-heavy | `results/continuous_ashare_2026_04_2w_deepseek_skill_main3_mf_pm` | 3 stocks, 2026-04-01 to 2026-04-14, `market,fundamentals`, DeepSeek flash, candidate skill injection | 27/27 ok; portfolio CR `-0.25%`; Buy&Hold `+0.54%`; benchmark `+3.25%` | Negative result. Skill injection made decisions too conservative and increased cash drag. Keep as ablation evidence. |
| `[x]` | Opportunity skill injection | `results/continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm` | 3 stocks, 2026-04-01 to 2026-04-14, `market,fundamentals`, DeepSeek flash, opportunity-aware candidate skill injection | 27/27 ok; portfolio CR `+1.65%`; Buy&Hold `+0.55%`; benchmark `+3.25%`; ZMR `+1.67%` | Positive mechanism result. Beats same-window baseline and most rule methods, but still below benchmark and slightly below ZMR. |
| `[x]` | Skill verification gate for opportunity run | `results/continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm/skill_verification` | Verifier compares evolved run with matched baseline window | PASSED; baseline CR `+0.36%`; evolved CR `+1.65%`; return delta `+1.28%`; all gates passed | First Hermes-style gate: completed runs, matched window, skill presence, return, drawdown, cash-drag, turnover. |
| `[x]` | Accepted-skill engineering validation on May window | `results/continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm` | 3 stocks, 2026-05-06 to 2026-05-14, `market,fundamentals`, DeepSeek flash, accepted skills from April two-week verifier | 21/21 ok; portfolio CR `0.00%`; Buy&Hold `-0.46%`; benchmark `-0.18%`; SMA `+0.88%`; default verifier PASSED but strict `--min-return-delta 0.001` FAILED | Engineering validation is inconclusive/weak. The strategy stayed fully in cash: avg position `0.00`, avg turnover `0.00`, no executed Buy. It avoided losses but did not prove active self-evolution. |

## Report Artifacts

| Status | Artifact | Path | Purpose |
|---|---|---|---|
| `[x]` | Baseline report | `reports/ashare_baseline_2026_04_06/ashare_baseline_report.pdf` | Baseline reproduction report with figures, tables, and rule comparisons. |
| `[x]` | Midterm design report | `reports/midterm_self_evolution_design_2026_07/midterm_self_evolution_design.pdf` | Midterm material combining baseline results and self-evolution skill design plan. |

## Current Experiment Interpretation

The current verified evidence supports the following claims:

1. The A-share-adapted TradingAgents baseline is reproducible on 3 main-board
   stocks for 2026-04 to 2026-06.
2. The baseline beats simple Buy&Hold and most technical-rule methods in the
   selected 3-stock portfolio, but it does not beat the board-aware benchmark.
3. Naive skill injection can hurt performance by making the agent too
   conservative.
4. Opportunity-aware skills reduce the worst effect of the conservative skill
   library and improve same-window performance on the first two-week window.
5. The accepted-skill May engineering validation did not produce active trading:
   the agent remained in cash and achieved `0.00%` return. This is not a useful
   improvement even though it did not underperform the matched baseline.
6. A verifier/gate is now available, but the currently accepted skills were
   generated from the full baseline window. For final research claims, a
   no-lookahead walk-forward split is still needed.

## Planned Experiment List

| Status | Experiment | Purpose | Planned output directory | Blocking dependency |
|---|---|---|---|---|
| `[x]` | Accepted-skill engineering validation on a later window | Check whether verifier-accepted skills remain useful outside the first two-week window. | `results/continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm` | Completed. Result was weak/inconclusive because the strategy stayed in cash. |
| `[ ]` | Strengthen verifier gates | Avoid accepting zero-activity runs as meaningful improvements. Add or use stricter gates: minimum return delta, position-utilization floor, cash-drag no-worsening, and active decision hit-rate. | Code/config update, then `skill_verification_strict` style outputs | Needed before treating skill acceptance as research-grade. |
| `[ ]` | No-lookahead April-only skill generation | Generate skills using only an earlier training window. | `results/walkforward_2026_q2/train_2026_04_skills` or equivalent | Need either a date-filter option for experience building or a filtered April-only baseline result folder. |
| `[ ]` | Walk-forward validation window | Use April-generated skills on May and select only skills that pass verifier. | `results/walkforward_2026_q2/val_2026_05_skill_selected` | Requires no-lookahead April skills. |
| `[ ]` | Walk-forward final test window | Evaluate verifier-selected skills on June without modifying them. | `results/walkforward_2026_q2/test_2026_06_skill_selected` | Requires validation-selected accepted skills. |
| `[ ]` | Skill-level hit-rate analysis | Attribute which injected skills changed decisions and whether those changes improved returns. | `results/walkforward_2026_q2/skill_hit_rate_analysis` | Needs decision-level skill trace or post-hoc matching logic. |
| `[ ]` | Optional GEPA-style skill rewrite | Let LLM rewrite candidate skill text based on verifier failures, then re-run gate. | `results/walkforward_2026_q2/gepa_style_skill_variants` | Should be attempted only after deterministic walk-forward is stable. |

## Commands

Run all commands from PowerShell.

### 0. Environment Setup

```powershell
conda activate tradingagents
cd E:\TradingAgents\Self-Evolution-TradingAgents
$env:PYTHONPATH = (Get-Location).Path
$env:TRADINGAGENTS_ENABLE_PREDICTION_MARKETS = "false"
$env:TRADINGAGENTS_ENABLE_US_SOCIAL_SOURCES = "false"
```

### 1. Current Baseline Reproduction

This has already been completed in
`results/continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm`.

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
  --output-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm `
  --disable-prediction-markets `
  --disable-us-social-sources
```

### 2. Build Experiences and Candidate Skills From Baseline

These artifacts already exist under the baseline folder's `evolution` directory.

```powershell
python scripts\build_trading_experiences.py `
  --result-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm

python scripts\generate_trading_skills.py `
  --experience-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm\evolution\experiences\trading_experiences.jsonl `
  --missed-upside-return 0.005

python scripts\render_trading_skill_context.py `
  --skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm\evolution\skills\candidate_skills.jsonl `
  --max-skills 3 `
  --max-chars 1600
```

### 3. Verify Completed Opportunity-Skill Run

This has already passed and generated `skill_verification`.

```powershell
python scripts\verify_trading_skills.py `
  --baseline-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm `
  --evolved-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm `
  --skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm\evolution\skills\candidate_skills.jsonl
```

### 4. Completed: Accepted-Skill Engineering Validation

This run used the verifier-accepted skill file and evaluated a later two-week
window. Because the current skill source was generated from the full 2026-04 to
2026-06 baseline, this run is an engineering validation, not a final
no-lookahead result.

Smoke command:

```powershell
python scripts\run_continuous_backtest_ashare.py `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --start-date 2026-05-01 `
  --end-date 2026-05-15 `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --decision-source pm-rating `
  --memory-mode experiment `
  --memory-holding-days 5 `
  --output-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm `
  --evolution-skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm\skill_verification\accepted_skills.jsonl `
  --evolution-skill-max-skills 3 `
  --evolution-skill-max-chars 1600 `
  --disable-prediction-markets `
  --disable-us-social-sources `
  --max-runs 3
```

If the smoke test is fine, resume the same command without `--max-runs 3`:

```powershell
python scripts\run_continuous_backtest_ashare.py `
  --tickers 600519.SS,000333.SZ,600036.SS `
  --start-date 2026-05-01 `
  --end-date 2026-05-15 `
  --analysts market,fundamentals `
  --llm-provider deepseek `
  --quick-model deepseek-v4-flash `
  --deep-model deepseek-v4-flash `
  --decision-source pm-rating `
  --memory-mode experiment `
  --memory-holding-days 5 `
  --output-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm `
  --evolution-skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm\skill_verification\accepted_skills.jsonl `
  --evolution-skill-max-skills 3 `
  --evolution-skill-max-chars 1600 `
  --disable-prediction-markets `
  --disable-us-social-sources
```

Then evaluate and verify:

```powershell
python scripts\evaluate_continuous_backtest_baselines.py `
  --output-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm

python scripts\verify_trading_skills.py `
  --baseline-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm `
  --evolved-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm `
  --skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm\skill_verification\accepted_skills.jsonl
```

Strict verifier check:

```powershell
python scripts\verify_trading_skills.py `
  --baseline-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_to_06_deepseek_flash_main3_mf_pm `
  --evolved-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm `
  --skills-jsonl E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_04_2w_deepseek_skill_opportunity_main3_mf_pm\skill_verification\accepted_skills.jsonl `
  --output-dir E:\TradingAgents\Self-Evolution-TradingAgents\results\continuous_ashare_2026_05_2w_deepseek_accepted_skill_main3_mf_pm\skill_verification_strict `
  --min-return-delta 0.001
```

Observed outcome:

- Default verifier: PASSED, because evolved CR `0.00%` was not worse than the
  matched baseline CR `0.00%`.
- Strict verifier: FAILED, because the return delta was `0.00%`, below the
  `0.10%` improvement requirement.
- Interpretation: not a meaningful improvement. The run stayed in cash.

### 5. Next Immediate Code Task

The May accepted-skill engineering validation shows that the current verifier is
too permissive for research claims. The next code task should update the
self-evolution harness before running more expensive experiments:

```text
1. Add date filters to build_trading_experiences.py so skills can be generated
   from 2026-04 only.
2. Strengthen verify_trading_skills.py defaults or add a --research-gate preset:
   minimum return delta, no increase in cash-drag, and a minimum active exposure
   or skill-hit signal.
3. Re-run a no-lookahead walk-forward:
   April train -> May validation -> June final test.
```

### 6. Strict Walk-Forward Plan

The final report should use this stricter design:

```text
2026-04 train: baseline only -> build experiences -> generate candidate skills
2026-05 validation: inject April skills -> verifier selects accepted skills
2026-06 test: inject accepted skills -> final comparison against baseline and rules
```

Current blocker: the experience builder needs date filtering, or we need a
separate April-only baseline result folder. This should be the next code task
before claiming final no-lookahead self-evolution results.
