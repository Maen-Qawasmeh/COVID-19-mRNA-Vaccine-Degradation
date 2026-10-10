# mRNA Reactivity and Degradation Prediction with Machine Learning

A multi-output regression project that predicts three RNA measurement targets from nucleotide, structure, and local context features. The notebook compares 20 model configurations, evaluates them with mean columnwise root mean squared error (MCRMSE), and analyzes prediction errors and feature importance.

The main workflow is implemented in **`COVID-19mRNAVaccineDegradation.ipynb`**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Modeling Workflow](#modeling-workflow)
- [Evaluation Metric](#evaluation-metric)
- [Models and Results](#models-and-results)
- [Outputs and Submission Export](#outputs-and-submission-export)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

This project explores tabular machine learning for RNA reactivity and degradation prediction in an mRNA vaccine degradation dataset. Each row represents an RNA sequence position, rather than a complete sequence.

The notebook predicts:

| Target | Measurement represented by the dataset column |
| --- | --- |
| `reactivity` | RNA reactivity. |
| `deg_Mg_pH10` | Degradation under the magnesium / pH 10 condition. |
| `deg_Mg_50C` | Degradation under the magnesium / 50°C condition. |

The main objectives are to:

- Encode nucleotide, structure, and predicted loop tokens.
- Explore model-based replacement of selected measurement values.
- Compare linear, distance-based, tree-based, and boosting regressors.
- Predict all three targets for each position.
- Rank models using MCRMSE.
- Examine boosting progress, prediction errors, and feature importance.

The notebook is an experimental prediction workflow. Its outputs are estimates of dataset measurements; it does not evaluate vaccine efficacy or clinical outcomes.

## Dataset

The notebook expects two CSV files in its working directory:

| File | Purpose |
| --- | --- |
| `train.csv` | Position-level features, target measurements, and measurement error columns. |
| `test.csv` | External position-level records and the auxiliary measurement columns required by the current preprocessing code. |

The supplied notebook does not include these CSV files or specify a dataset URL, competition link, or license. Obtain the matching, already prepared CSV files from their original provider.

**Input format matters:** the notebook expects expanded position-level tabular data with local context columns. It does not convert full RNA sequence strings or raw JSON records into this format.

### Input Features

| Field / Feature group | Role |
| --- | --- |
| `id` | RNA record identifier; removed before modeling. |
| `id_seqpos` | Position identifier; retained separately for submission, then removed from model inputs. |
| `sequence` | Nucleotide token such as `A`, `G`, `U`, or `C`. |
| `structure` | Structure token: `.`, `(`, or `)`. |
| `predicted_loop_type` | Loop token from `B`, `E`, `H`, `I`, `M`, `S`, or `X`. |
| Local context columns | Precomputed neighboring nucleotide, structure, and loop tokens, including columns such as `b4_structure`, `a4_structure`, `b5_sequence`, and `a5_sequence`. |
| Measurement columns | Three prediction targets plus seven auxiliary measurement / error columns. |

The notebook uses the following ten measurement columns during training cross-imputation:

```text
reactivity
reactivity_error
deg_Mg_pH10
deg_error_Mg_pH10
deg_pH10
deg_error_pH10
deg_Mg_50C
deg_error_Mg_50C
deg_50C
deg_error_50C
```

The saved training preview contains **130,560 rows and 45 columns**. After dropping two identifiers and ten measurement columns from the feature matrix, the model input contains **33 features**. The 67/33 split produces **87,475 training rows** and **43,085 validation rows**.

## Project Structure

```text
mrna-degradation-prediction/
├── README.md
├── COVID-19mRNAVaccineDegradation.ipynb
├── train.csv                    # Required input; obtain separately
├── test.csv                     # Required input; obtain separately
└── submission_xgb_tuned.csv     # Export attempted by the notebook
```

There is no standalone application or saved model artifact in the supplied project. Plotting and analysis happen inside the notebook.

## Requirements

The notebook metadata records **Python 3.12.12**. Exact package versions are not recorded.

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical calculations and MCRMSE. |
| `pandas` | CSV loading, preprocessing, results tables, and export. |
| `matplotlib` | Boosting progress and feature importance plots. |
| `seaborn` | Correlation heatmap. |
| `scikit-learn` | Data splitting, regressors, and multi-output wrappers. |
| `xgboost` | Cross-imputation, regression, and feature importance. |
| `catboost` | CatBoost regression. |
| `lightgbm` | LightGBM regression. |
| `jupyterlab` | Local notebook interface. |

The notebook fits XGBoost directly on a three-column target matrix. Use an XGBoost release that supports multi-output regression; the original installed version is not specified.

## Modeling Workflow

### 1. Load Position-Level Data

Read `train.csv` and `test.csv` with pandas. The preview shows nucleotide and context tokens alongside numerical measurements.

### 2. Cross-Impute Selected Measurements

For each measurement column, `impute_outliers_xgb`:

1. Treats values in the interval **[0, 1]** as valid.
2. Treats values outside that interval, including missing values, as candidates for replacement.
3. Fits an XGBoost regressor using the other selected measurement columns as predictors.
4. Replaces candidate values with model predictions.

Training preprocessing loops over all ten measurement columns, including the three final targets. Test preprocessing loops over the seven auxiliary columns. Each loop learns from its own dataset, and later imputations use earlier modifications.

This interval is an implementation assumption, not a validated measurement bound. The saved training preview includes an original reactivity value above one. Replacing such values changes the target data used for evaluation, and predicted replacements are not explicitly clipped to [0, 1].

### 3. Encode RNA Tokens

The same mapping is applied to object-valued columns in both datasets:

| Token group | Mapping |
| --- | --- |
| Nucleotide | `A → 0`, `G → 1`, `U → 2`, `C → 3`. |
| Structure | `. → 0`, `( → 1`, `) → 2`. |
| Predicted loop type | `B → 0`, `E → 1`, `H → 2`, `I → 3`, `M → 4`, `S → 5`, `X → 6`. |
| String sentinel | `'-1' → -1`. |

Identifier strings are not part of this token mapping and are dropped afterward. Other unrecognized strings are not automatically handled.

### 4. Separate Features and Targets

The target matrix contains `reactivity`, `deg_Mg_pH10`, and `deg_Mg_50C`.

The feature matrix excludes those targets and these seven columns:

```text
deg_pH10
deg_50C
reactivity_error
deg_error_Mg_pH10
deg_error_pH10
deg_error_Mg_50C
deg_error_50C
```

Removing those columns from the final inputs does not undo their earlier role in cross-imputation.

### 5. Split the Data

The notebook uses `train_test_split` with:

- **67% training / 33% validation**.
- **`random_state=42`**.
- A random split of rows, without grouping by RNA `id`.

### 6. Train Multiple Regressors

Most algorithms are wrapped in `MultiOutputRegressor`, which fits a separate estimator for each target. The two XGBoost configurations are fitted directly to the three-column target matrix.

The section titled “Hyperparameter Tuning” specifies one adjusted XGBoost configuration rather than performing an automated search:

```python
XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    colsample_bytree=0.9,
    reg_alpha=5,
    reg_lambda=6,
    min_child_weight=2,
    n_estimators=1500,
    random_state=42
)
```

### 7. Analyze Predictions

The notebook builds a sorted MCRMSE comparison table, refits the adjusted XGBoost model with training and validation evaluation sets, and plots the recorded metric against boosting rounds. This plot tracks boosting progress rather than performance against increasing training dataset sizes. Early stopping is not configured.

It also creates per-target absolute errors, a correlation heatmap of actual and predicted targets, and an XGBoost feature importance chart. The default importance shown by `plot_importance` is split-count weight; it is not a causal contribution measure or a classification F1 score.

## Evaluation Metric

The notebook uses **MCRMSE**, the mean of the root mean squared error values for the three target columns:

```text
MCRMSE = (RMSE(reactivity) + RMSE(deg_Mg_pH10) + RMSE(deg_Mg_50C)) / 3
```

**Lower is better.** This is a regression error metric, not an accuracy percentage.

The implemented function uses `N=3`, matching the three targets. Its validation scores are calculated against the **cross-imputed target values**, rather than an untouched copy of the original measurements.

## Models and Results

The following scores come from **saved notebook outputs**. They have not been independently reproduced and are local validation results, not verified competition leaderboard scores.

| Rank | Model / Configuration | MCRMSE |
| ---: | --- | ---: |
| 1 | **Random Forest** | **0.185218** |
| 2 | XGBoost, adjusted configuration | 0.188440 |
| 3 | CatBoost | 0.191070 |
| 4 | XGBoost, baseline | 0.191807 |
| 5 | LightGBM | 0.200458 |
| 6 | HistGradientBoosting | 0.200557 |
| 7 | Gradient Boosting | 0.220693 |
| 8 | K-Nearest Neighbors | 0.234092 |
| 9 | Decision Tree | 0.237459 |
| 10 | AdaBoost | 0.259921 |
| 11 | LassoCV | 0.261351 |
| 12 | RidgeCV | 0.261357 |
| 13 | Ridge | 0.261357 |
| 14 | Bayesian Ridge | 0.261358 |
| 15 | Huber | 0.263292 |
| 16 | Tweedie | 0.264476 |
| 17 | Orthogonal Matching Pursuit | 0.274227 |
| 18 | ElasticNet | 0.297833 |
| 19 | Lasso | 0.297833 |
| 20 | Passive Aggressive | 0.334821 |

**Random Forest** achieves the lowest saved error. The adjusted **XGBoost** configuration improves on baseline XGBoost and is used for the detailed error analysis and attempted submission export.

The validation design and altered target values limit what can be concluded from these rankings. They require confirmation with independent, sequence-grouped evaluation.

## Outputs and Submission Export

### Analysis Outputs

| Output | Description |
| --- | --- |
| `leaderboard` | In-memory table ranking all 20 configurations by MCRMSE. |
| XGBoost progress plot | Recorded training and validation metric over boosting rounds. |
| `error_analysis` | Actual values, predicted values, and absolute errors for three targets. |
| `error_analysis_full` | Error table combined with validation features. |
| Correlation heatmap | Correlations among actual and predicted target columns. |
| Feature importance plot | XGBoost split-count importance, with up to 40 displayed features. |
| `submission_xgb_tuned.csv` | CSV attempted by the final submission section. |

Charts are displayed in the notebook. The comparison table, error tables, and trained models are not explicitly saved as separate artifacts.

### Intended Submission Columns

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_Mg_50C
```

## Current Limitations

- **Targets are modified before evaluation:** values outside [0, 1] are replaced without a documented justification for that bound. Validation measures agreement with the modified targets.
- **Cross-imputation before splitting:** imputation models learn from all training rows, including rows later used for validation. The preprocessing is not isolated from the validation data.
- **Related target information in imputation:** measurement and error columns are used to reconstruct each other. Dropping them from the final feature matrix does not remove this dependency.
- **Row-level rather than sequence-level validation:** multiple positions from one RNA record can appear in both splits. This can make evaluation optimistic for prediction on entirely new RNA sequences.
- **Separate test imputation:** auxiliary test values are imputed using regressors fitted on test data rather than reusing fitted training transformations. The notebook later discards these auxiliary columns from the final inputs.
- **Incorrect original submission source:** the export uses validation predictions with external IDs. It must be replaced before submission.
- **Selected model is not refitted on all training rows:** the adjusted XGBoost model remains fitted on the 67% training subset.
- **Integer encoding implies an order:** numerical token codes introduce artificial distances for linear and distance-based models. The notebook does not compare one-hot encoding or feature scaling.
- **Limited schema checks:** unexpected tokens, missing context fields, and nonnumeric feature types are not systematically validated. Encoding produces a pandas downcasting warning in the saved run.
- **Incomplete reproducibility:** package versions and a full raw-data-to-feature preparation process are not provided.

## Future Improvements

- Preserve the original measurements and validate any proposed filtering bounds using the dataset documentation.
- Separate RNA records by `id` before fitting any learned preprocessing, using grouped validation.
- Avoid reconstructing validation targets from other measurements; evaluate against independent, original target values.
- Audit the inference-time availability of every feature and remove unnecessary test-data imputation.
- Add explicit schema, token, numeric dtype, and missing-value checks.
- Compare one-hot token encoding and scaling for linear and distance-based models.
- Report per-target RMSE alongside aggregate MCRMSE.
- Run grouped cross-validation and an actual hyperparameter search.
- Configure early stopping on a properly separated validation set for boosting experiments.
- Refit the selected model on the approved training data before predicting external rows.
- Save the fitted model, encoding rules, feature order, and tested package versions.
- Add dataset attribution, license information, and documentation for constructing the context columns.

## Project Status

**Experimental multi-output regression notebook.** Token encoding, 20 model configurations, MCRMSE ranking, and prediction analysis are implemented. Independent validation, justified measurement preprocessing, and corrected external prediction export are the main next steps.
