from src.data_processing import build_participant_dataset, save_eda_outputs
from src.modeling import build_model_inputs, fit_and_evaluate_models
from src.reporting import write_report

def main() -> None:
    participant_df, availability_df, daily_df = build_participant_dataset()
    save_eda_outputs(participant_df, availability_df, daily_df)
    X, y, feature_cols = build_model_inputs(participant_df)
    results_df, diagnostics = fit_and_evaluate_models(X, y, feature_cols)
    write_report(participant_df, availability_df, results_df, diagnostics)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'CV Best F1']].round(4))

if __name__ == '__main__':
    main()
