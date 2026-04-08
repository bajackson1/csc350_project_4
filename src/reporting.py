from .common import *

def write_report(
    participant_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    results_df: pd.DataFrame,
    diagnostics: dict,
) -> None:
    best_model = results_df.iloc[0]
    best_model_name = best_model["Model"]

    class_counts = participant_df["loneliness_binary"].value_counts().sort_index().to_dict()
    top_rf = diagnostics["rf_importance"].head(5)
    top_xgb = diagnostics["xgb_importance"].head(5)
    best_params = best_model["Best Params"]

    # Keep parameter summary short
    concise_params = {
        key: best_params[key]
        for key in [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
        ]
        if key in best_params
    }

    results_table = results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Best F1"]
    ].round(4)

    # Convert key result sections into markdown tables
    results_markdown = markdown_table(
        list(results_table.columns),
        results_table.values.tolist(),
    )

    top_rf_markdown = markdown_table(
        ["Feature", "Importance"],
        [[idx, f"{value:.6f}"] for idx, value in top_rf.items()],
    )

    top_xgb_markdown = markdown_table(
        ["Feature", "Importance"],
        [[idx, f"{value:.6f}"] for idx, value in top_xgb.items()],
    )

    team_line = ", ".join(TEAM_MEMBERS)

    model_discussion = {
        "Random Forest": (
            "Random Forest performed best because the dataset is small, contains a mix of "
            "behavioral and survey-derived features, and likely includes nonlinear relationships "
            "and interactions. Tree ensembles are usually more stable than a neural network in "
            "small-sample settings and are less sensitive than an RBF SVM to feature scaling and "
            "hyperparameter mismatch when the sample size is limited."
        ),
        "XGBoost": (
            "XGBoost was competitive, which also supports the idea that tree-based methods fit this "
            "problem well. Its slightly lower test performance than Random Forest may reflect the "
            "difficulty of tuning boosting on a very small sample without overfitting."
        ),
        "SVM": (
            "The SVM underperformed, likely because the dataset is small relative to the number of "
            "features and because a single RBF kernel setting may not capture the structure of the "
            "problem well. SVMs can work well here, but they are sensitive to hyperparameters and "
            "feature geometry."
        ),
        "Neural Network": (
            "The neural network underperformed, which is not surprising for a dataset with only 39 "
            "participants. Neural networks usually benefit from substantially more training data "
            "than were available in this project."
        ),
    }

    discussion_text = model_discussion.get(
        best_model_name,
        "The best model likely benefited from being better matched to a small, heterogeneous, "
        "multimodal dataset with nonlinear relationships."
    )

    report = f"""# Project 4 Completed Analysis

**Team Members:** {team_line}

**GitHub Repository:** https://github.com/bajackson1/csc350_project_4.git

## Dataset Understanding

This project uses a multimodal digital phenotyping dataset to study loneliness and related mental-health outcomes from wearable, smartphone, and survey data collected over time.

The purpose of the dataset is to support research on whether everyday behavioral signals and self-reports can be used to understand or predict loneliness. The data types include passive smartphone sensing from Aware, wearable data from Oura and Samsung Watch devices, ecological momentary assessment (EMA) responses, and end-of-study survey instruments such as the UCLA Loneliness Scale.

Potential research applications include early identification of elevated loneliness risk, studying how sleep, activity, and social behavior relate to mental health, and evaluating whether combining multiple sensing modalities improves prediction over any single source alone.

## Dataset Summary

- Participants: {len(participant_df)}
- Distinct dates observed in raw sensor files per participant: mean = {participant_df['days_with_any_sensor_data'].mean():.2f}, min = {participant_df['days_with_any_sensor_data'].min():.0f}, max = {participant_df['days_with_any_sensor_data'].max():.0f}
- Modalities available:
  - Aware present for {int(availability_df['has_aware'].sum())} participants
  - Oura present for {int(availability_df['has_oura'].sum())} participants
  - Watch present for {int(availability_df['has_watch'].sum())} participants
  - EMA present for {int(availability_df['has_ema'].sum())} participants
  - UCLA end survey present for {int(availability_df['has_ucla_end'].sum())} participants

![Dataset Completeness](./figures/eda_modality_completeness.png)

## Target Definition

- Outcome: binary loneliness label derived from the end-of-study UCLA Loneliness Scale.
- UCLA scores were computed from the 20 questionnaire items using standard Likert scoring with reverse-coding on the standard positively worded items.
- Median UCLA threshold: {participant_df['ucla_loneliness_total'].median():.2f}
- Class distribution: low = {class_counts.get(0, 0)}, high = {class_counts.get(1, 0)}

![UCLA Target Distribution](./figures/eda_ucla_distribution.png)

## Data Preprocessing

Participant-level features were created by:

1. Aggregating Aware smartphone logs into daily screen time, call activity, message activity, and notification counts.
2. Aggregating Oura records into daily sleep, readiness, HRV, and activity summaries.
3. Aggregating Samsung Watch records into daily mean heart rate, heart-rate variability proxy (daily heart-rate standard deviation), and accelerometer-based movement intensity.
4. Collapsing daily features into 28-day participant means.
5. Incorporating multimodal survey summary features (EMA, PSS, and PHQ-9) alongside passive and wearable measures to maximize predictive performance within the assignment scope.

![Core Correlations](./figures/eda_core_correlations.png)

![Missingness](./figures/eda_missingness.png)

## ML Methods

Four supervised learning models were evaluated to satisfy the assignment requirements: SVM, Random Forest, XGBoost, and a Neural Network. Median imputation was applied within each model pipeline, and StandardScaler was used for the SVM and Neural Network. Performance was assessed with a stratified train/test split and 5-fold cross-validation using F1 as the primary comparison metric because the loneliness classes were close to balanced but the sample was small.

Train/test split summary:

- Train shape: {diagnostics['X_train_shape']}
- Test shape: {diagnostics['X_test_shape']}
- Train class balance: {diagnostics['class_balance_train']}
- Test class balance: {diagnostics['class_balance_test']}

## Results

{results_markdown}

![Model Comparison](./figures/model_comparison.png)

![ROC Curves](./figures/model_roc_curves.png)

![F1 Score Comparison](./figures/model_f1_scores.png)

The best-performing model on the held-out test set was {best_model_name}. It achieved an F1 score of {best_model['F1']:.4f}, accuracy of {best_model['Accuracy']:.4f}, and ROC-AUC of {best_model['ROC-AUC']:.4f}. Its mean 5-fold cross-validated F1 score was {best_model['CV Best F1']:.4f}, which suggests the result is promising but still sensitive to sample size.

### Best Model Summary

- Best model by test-set F1: {best_model['Model']}
- Accuracy: {best_model['Accuracy']:.4f}
- Precision: {best_model['Precision']:.4f}
- Recall: {best_model['Recall']:.4f}
- F1: {best_model['F1']:.4f}
- ROC-AUC: {best_model['ROC-AUC']:.4f}
- Mean 5-fold cross-validated F1: {best_model['CV Best F1']:.4f}
- Final model parameters: `{concise_params}`

![Best Model Confusion Matrix](./figures/best_model_confusion_matrix.png)

## Discussion

{discussion_text}

The top Random Forest features indicate that both internal state and daily behavior matter. EMA features such as positive affect, loneliness, negative affect, and connectedness were among the strongest predictors, which suggests that near-real-time subjective experience is closely tied to the final loneliness label. Oura sleep RMSSD also appeared near the top, indicating that physiological recovery and sleep quality may contain useful information about loneliness-related well-being.

Taken together, the feature rankings suggest that loneliness in this dataset is not explained by one modality alone. Instead, the signal appears to come from a combination of self-reported social-emotional state, sleep-related physiology, and broader behavioral patterns captured by phones and wearables. That result supports the central motivation of multimodal mental-health sensing.

### Final Model Top 5 Features (Random Forest)

{top_rf_markdown}

![Random Forest Feature Importance](./figures/random_forest_feature_importance.png)

### XGBoost Top Predictors

{top_xgb_markdown}

![XGBoost Feature Importance](./figures/xgboost_feature_importance.png)

## Limitations

- Sample size is small (39 participants), so test-set metrics may vary with different splits.
- Some modalities are incomplete for some participants, requiring median imputation.
- Because the assignment explicitly frames the problem as multimodal integration of sensor data and subjective reports, EMA/PSS/PHQ features were included in the optimized model. This improves performance, but it makes the prediction task less purely passive than a sensor-only setting.
- Watch-derived HRV was approximated from observed heart-rate variation rather than reconstructed beat-to-beat intervals from raw PPG.
- The public dataset README references `Surveys.pdf` for full questionnaire wording, but that file was not present in the extracted directory; UCLA scoring therefore follows the standard UCLA Version 3 convention.

## Reproducibility Notes

- Main script: `run_analysis.py`
- Generated report: `outputs/project4_report.md`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
"""
    (OUTPUT_DIR / "project4_report.md").write_text(report)
