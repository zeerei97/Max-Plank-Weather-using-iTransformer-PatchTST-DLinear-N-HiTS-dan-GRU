## Experimental Design & Workflow
```text
+-----------------------------------------------------------------------+
|                    Max Planck Weather Dataset                         |
|                      (14 Weather Variables)                           |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                  Data Preprocessing & Scaling                         |
|          (Chronological Train-Test Split & Standardizer)              |
+-----------------------------------------------------------------------+
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
+-----------------------------------+ +-----------------------------------+
|   Experiment 1: Single-Target     | |   Experiment 2: Multi-Target /    |
|            Benchmark              | |        Exogenous Capacity         |
+-----------------------------------+ +-----------------------------------+
| Input : Temperature T (degC)      | | Input : All 14 Weather Variables  |
| Target: Temperature T (degC)      | | Target:                           |
|                                   | |  - 14 Targets (Multivariate):     |
| Models (Default Hyperparameters): | |    iTransformer, PatchTST         |
|  - iTransformer, PatchTST,        | |  - Target + Exogenous Features:   |
|    DLinear, NHITS, GRU            | |    DLinear, NHITS, GRU            |
+-----------------------------------+ +-----------------------------------+
                  |                                 |
                  +----------------+----------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                 Rolling-Window Cross Validation                       |
|             (Evaluated across distinct Time Cutoffs)                  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                      Performance Evaluation                           |
|             - Compute Absolute Metrics (MAE & RMSE)                   |
|             - Calculate Relative Improvement Delta (%)                |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                      Visual & Quantitative Output                     |
|           - Single-Cutoff Forecast Plots (Clean Overlay)              |
|           - Summary Delta Performance Table                           |
+-----------------------------------------------------------------------+

### Objective
This benchmark evaluates deep learning time-series models (**iTransformer, PatchTST, DLinear, NHITS, and GRU**) on the Max Planck Weather dataset. All models are trained under their default architectural configurations in `NeuralForecast` to establish a fair and standard baseline.

### Experimental Setup
To analyze model behavior under different input capabilities, the experiment is divided into two phases:

1. **Experiment 1: Single-Target Benchmark**
   * **Goal:** Establish a baseline prediction performance.
   * **Scope:** All models predict a single target variable—Temperature (`T (degC)`)—using only historical values of Temperature as input.

2. **Experiment 2: Multi-Target & Exogenous Capacity**
   * **Goal:** Evaluate how models leverage additional weather features.
   * **Multivariate Models (iTransformer, PatchTST):** Trained to predict all 14 weather variables simultaneously to capture inter-variable correlations.
   * **Exogenous-supported Models (DLinear, NHITS, GRU):** Predict Temperature (`T (degC)`) while incorporating historical exogenous features from the other 13 weather variables.

---

### Evaluation Strategy
* **Cross-Validation:** Evaluated across multiple rolling windows to prevent data leakage and capture seasonal variations.
* **Metrics:** Evaluated using Mean Absolute Error (**MAE**) and Root Mean Squared Error (**RMSE**).
* **Performance Delta (%):** Calculates the relative error reduction from Experiment 1 to Experiment 2:
  $$\text{Delta (\%)} = \frac{\text{Metric}_{\text{Exp1}} - \text{Metric}_{\text{Exp2}}}{\text{Metric}_{\text{Exp1}}} \times 100$$