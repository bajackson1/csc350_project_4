from .common import *

def build_model_inputs(participant_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # Final multimodal feature set
    feature_cols = [
        "screen_on_minutes",
        "screen_event_count",
        "call_count",
        "call_duration_sec",
        "missed_call_count",
        "message_count",
        "messages_sent",
        "messages_received",
        "notification_count",

        "oura_sleep_duration",
        "oura_sleep_efficiency",
        "oura_sleep_score",
        "oura_steps",
        "oura_activity_score",
        "oura_readiness_score",
        "oura_hrv_balance",
        "oura_sleep_rmssd",
        "oura_sleep_hr_average",

        "watch_avg_heart_rate",
        "watch_hr_std",
        "watch_movement_intensity",
        "days_with_any_sensor_data",

        "ema_days",
        "ema_lonely_mean",
        "ema_connect_mean",
        "ema_isolate_mean",
        "ema_positive_mean",
        "ema_negative_mean",
        "pss_end_total",
        "phq9_latest_total",
    ]

    usable_cols = [col for col in feature_cols if col in participant_df.columns]

    X = participant_df[usable_cols].copy()
    y = participant_df["loneliness_binary"].astype(int)

    return X, y, usable_cols


def fit_and_evaluate_models(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    models = {
        "SVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1,
                        gamma=1,
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        # Best performing fixed settings from tuning sweep
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.01,
                        max_depth=2,
                        subsample=0.7,
                        colsample_bytree=1.0,
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        eval_metric="logloss",
                    ),
                ),
            ]
        ),

        "Neural Network": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        alpha=0.0001,
                        learning_rate_init=0.0005,
                        max_iter=5000,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = []
    fitted_models = {}
    best_params_map = {}

    # Use cross-validated F1 with held-out test set so comparison isn't as split-dependent
    for model_name, best_model in models.items():
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="f1", n_jobs=1)
        model_params = best_model.named_steps["model"].get_params()
        best_params_map[model_name] = model_params

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "CV Best F1": float(cv_scores.mean()),
                "Best Params": model_params,
            }
        )

        fitted_models[model_name] = {
            "estimator": best_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)

    # Save ROC curves for four required models
    plt.figure(figsize=(8, 6))

    for model_name, payload in fitted_models.items():
        fpr, tpr, _ = roc_curve(y_test, payload["y_proba"])
        auc = roc_auc_score(y_test, payload["y_proba"])

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))

    ordered = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    ordered.plot(kind="bar", ylim=(0, 1), rot=0, figsize=(10, 5), colormap="viridis")

    plt.title("Model Comparison Across Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=200)
    plt.close()

    best_model_name = results_df.iloc[0]["Model"]
    best_estimator = fitted_models[best_model_name]["estimator"]
    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_estimator(
        best_estimator,
        X_test,
        y_test,
        display_labels=["Low", "High"],
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix: {best_model_name}")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_model_confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.barplot(data=results_df, x="Model", y="F1", palette="rocket")
    plt.ylim(0, 1)
    plt.title("F1 Score by Model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_f1_scores.png", dpi=200)
    plt.close()

    # Use tree importances to rank strongest predictors
    rf_model = fitted_models["Random Forest"]["estimator"].named_steps["model"]
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    rf_importance.to_csv(
        TABLE_DIR / "random_forest_feature_importance.csv",
        header=["importance"],
        index_label="feature",
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x=rf_importance.head(12).values, y=rf_importance.head(12).index, palette="Blues_r")
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close()

    xgb_model = fitted_models["XGBoost"]["estimator"].named_steps["model"]
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    xgb_importance.to_csv(
        TABLE_DIR / "xgboost_feature_importance.csv",
        header=["importance"],
        index_label="feature",
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x=xgb_importance.head(12).values, y=xgb_importance.head(12).index, palette="Oranges_r")
    plt.title("Top XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "xgboost_feature_importance.png", dpi=200)
    plt.close()

    results_df.to_csv(TABLE_DIR / "model_results.csv", index=False)

    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            **{f"{name}_pred": payload["y_pred"] for name, payload in fitted_models.items()},
            **{f"{name}_proba": payload["y_proba"] for name, payload in fitted_models.items()},
        }
    ).to_csv(TABLE_DIR / "test_set_predictions.csv", index=False)

    return results_df, {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "class_balance_train": dict(Counter(y_train)),
        "class_balance_test": dict(Counter(y_test)),
        "best_model_name": best_model_name,
        "best_model_report": fitted_models[best_model_name]["report"],
        "rf_importance": rf_importance,
        "xgb_importance": xgb_importance,
    }
