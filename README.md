# Wind Turbine Predictive Maintenance: AutoGluon vs H2O AutoML

A comparative study of two AutoML frameworks — **AutoGluon** and **H2O AutoML** — for early fault detection in wind turbines, benchmarked against a logistic regression baseline. Built as my final year project (BSc Computer Science, Aston University).

## What this project does

Predicts an upcoming component failure in a wind turbine from SCADA sensor data (temperature, pitch angle, rotational speed, etc.), giving a 14-day early warning window before the fault occurs. Both AutoML frameworks were run close to their default/out-of-the-box settings — the goal was to test how well AutoML performs for a non-expert user without heavy manual tuning, not to find the absolute best possible model.

Data: the [CARE to Compare](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection) dataset — real-world SCADA data from six turbines, Wind Farm A.

## Results

| Metric | Logistic Regression | H2O AutoML | AutoGluon |
|---|---|---|---|
| Precision (pre-failure) | 0.570 | 0.951 | **0.960** |
| Recall (pre-failure) | 0.660 | 0.607 | 0.643 |
| F1 (pre-failure) | 0.614 | 0.740 | **0.770** |
| ROC-AUC | 0.872 | 0.945 | **0.960** |
| PR-AUC | 0.721 | 0.886 | **0.910** |
| Training time (s) | 12.9 | 604.4 | 642.1 |
| Total time (s) | 31.0 | 643.3 | 1,448.0 |
| Peak RAM (GB) | 0.99 | ~5.07 | 4.83 |
| Leader model | — | XGBoost | Stacked ensemble |

Both AutoML frameworks clearly beat the baseline. AutoGluon led on every metric, but the standard metrics alone made the gap between the two frameworks look fairly small.

### Cost-sensitive evaluation (the novel part)

Standard metrics treat a missed failure and a false alarm as equally bad, which isn't realistic — a missed gearbox failure can cost $200,000–$700,000, while a false alarm is typically just a routine inspection. I built a custom expected-cost evaluation (`cost = C_miss × FN + C_FA × FP`), swept across classification thresholds and cost ratios, across four deployment scenarios (onshore electrical, onshore gearbox, offshore typical, offshore worst-case).

Under this lens the gap widened a lot: in the onshore gearbox scenario, AutoGluon's cost-optimal threshold reached a minimum expected cost of **$3.24M vs $88.8M for H2O**. AutoGluon stayed the lowest-cost model across the full cost-ratio sweep (1 to 10,000).


## Setup

Developed in Google Colab, reading data from Google Drive — update the file paths at the top of each notebook to point to your own copy of the [CARE to Compare dataset](https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection) (Wind Farm A) to rerun. Run the baseline and two AutoML notebooks first, then `Model_Comparison.ipynb` and `Novelty_Cost_Senstive_Evaluation.ipynb`.

## Planned next steps

This was a controlled comparison of both frameworks close to their default settings, deliberately, to test what AutoML delivers with minimal tuning. Next steps I'm planning:
- Move to a higher-RAM environment to train on the full multi-farm dataset without the downsampling workaround Colab's free tier forced.
- Properly optimise one framework rather than running close to defaults, to see how much headroom is left.
- Extend to offshore turbine data, where the cost asymmetry between a missed failure and a false alarm is even more pronounced.
- Add time-aware cross-validation (both frameworks currently use random CV internally, which can introduce mild optimism in absolute scores).

---

## A closer look at the methodology

**Anti-leakage filtering.** An early AutoGluon run on unfiltered data returned an AUC of 1.0 — a strong signal something was wrong rather than a genuine result. Investigation showed several sensors (power output, RPM, phase currents) reflect the turbine's *reaction* to a fault rather than its early signs — pitch adjusts, RPM drops, power falls as a consequence of degradation, not a precursor to it. Including them let the model detect the fault after the fact rather than predict it. 30 sensors were removed on this basis (power/reactive power sensors, rotational speed sensors, wind speed sensors, electrical/phase sensors, energy counters), leaving mostly temperature and directional sensors that reflect slower, genuinely predictive degradation processes.

**Feature engineering.** Rolling mean/standard deviation were computed over 1h, 6h, and 24h windows for the retained sensors, calculated strictly backward-looking (`center=False` in pandas) to avoid forward leakage. This mattered because the 14-day pre-failure window is defined by gradual trends (e.g. thermal drift), not single-point readings — knowing whether a temperature sensor is trending upward over the past 24 hours is more informative than its current value alone.

**Class imbalance.** The natural class distribution was 88.4% normal vs 11.6% pre-failure (7.6:1). SMOTE was considered and rejected — generating synthetic interpolated samples in a temporally autocorrelated time series produces rows that don't correspond to any real physical state. Downsampling the majority class in training only (capped at 12,000 normal rows per turbine, ~2:1 ratio) was used instead, partly to fix this and partly because the full six-turbine dataset with rolling features exceeded Colab's free-tier RAM. The test set was left at its natural 7.6:1 distribution so evaluation reflects real operating conditions.

**Why AutoGluon outperformed H2O.** AutoGluon relies on stacking across a diverse set of model families; H2O trains and ranks candidate models via a leaderboard, and its final leader model here was XGBoost — effectively a single algorithm family. AutoGluon's broader ensemble was better placed to capture both the gradual (thermal drift) and shorter-term patterns present in the SCADA data. Looking at the predicted probability distributions explains the cost-sensitive gap too: H2O assigned near-zero probability to around 12% of genuine pre-failure cases, meaning those were unrecoverable no matter how the classification threshold was tuned. AutoGluon spread more of its uncertain cases into the mid-range of the probability scale, making them recoverable through threshold adjustment — which is why AutoGluon's advantage grew much larger under the cost-sensitive evaluation than the standard metrics alone suggested.

**Precision-recall trade-off.** Logistic regression had the highest recall (0.660) but lowest precision (0.570) — it catches more real pre-failure events but raises far more false alarms. Both AutoML models reverse this, pushing precision above 0.95 but dropping recall to ~0.62–0.64, meaning roughly one in three genuine pre-failure windows goes undetected at the default 0.50 threshold. Which trade-off is "better" depends on context: where a missed failure is catastrophic (e.g. offshore, where mobilisation costs are high), the higher-recall baseline might be preferable despite the false alarms; where false alarms are costly to act on, the AutoML models' precision wins.

**Computational cost.** Both AutoML frameworks used ~5GB RAM vs ~1GB for logistic regression. AutoGluon's total runtime (1,448s) was more than double H2O's (643s) under the same 600-second training budget, due to AutoGluon's `best_quality` preset triggering additional post-training stacking and feature-importance computation. For infrequent training this is a non-issue; for an operator retraining regularly on limited hardware, it's a real practical difference in H2O's favour.

**Scope note.** Both frameworks were evaluated close to their default configurations, deliberately, to test how much AutoML delivers with minimal expert tuning rather than finding a ceiling on performance — see Planned next steps above for where this goes from here.
