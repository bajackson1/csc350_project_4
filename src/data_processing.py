from .common import *

def participant_sort_key(path: Path) -> int:
    return int(path.name.split("_")[-1])


def timestamp_to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="ms", errors="coerce").dt.date


def numeric_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else math.nan


def score_ucla(file_path: Path) -> float:
    df = pd.read_csv(file_path)
    row = df.iloc[0]
    total = 0

    # Score UCLA items
    for item in range(1, 21):
        score = UCLA_MAP.get(str(row[f"q{item}"]).strip())

        if score is None:
            raise ValueError(f"Unexpected UCLA response in {file_path}: {row[f'q{item}']}")
        
        # Reverse-code positively worded UCLA items
        if item in UCLA_REVERSED:
            score = 5 - score

        total += score

    return float(total)


def score_pss(file_path: Path) -> float:
    df = pd.read_csv(file_path)
    row = df.iloc[0]
    total = 0

    for item in range(1, 5):
        score = PSS_MAP.get(str(row[f"q{item}"]).strip())
        
        if score is None:
            return math.nan
        
        total += score

    return float(total)


def score_phq(file_path: Path) -> float:
    df = pd.read_csv(file_path)
    qcols = [f"q{i}" for i in range(1, 10) if f"q{i}" in df.columns]

    if not qcols:
        return math.nan
    
    return float(pd.to_numeric(df.loc[0, qcols], errors="coerce").sum())


def load_screen_features(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "screen_on_minutes", "screen_event_count"])

    df = pd.read_csv(file_path)

    if df.empty:
        return pd.DataFrame(columns=["date", "screen_on_minutes", "screen_event_count"])

    df = df.sort_values("timestamp").copy()
    df["date"] = timestamp_to_date(df["timestamp"])

    on_states = {1, 3}
    off_states = {0, 2}

    durations = []
    active_start = None
    active_date = None

    # Rebuild screen-on sessions from transitions
    for row in df.itertuples(index=False):
        state = getattr(row, "screen_status", math.nan)
        ts = getattr(row, "timestamp")
        current_date = getattr(row, "date")

        if pd.isna(state) or pd.isna(ts):
            continue

        state = int(state)

        if state in on_states and active_start is None:
            active_start = ts
            active_date = current_date
        elif state in off_states and active_start is not None:
            duration_sec = max((ts - active_start) / 1000, 0)

            # Drop unrealistic sessions from noisy sequences
            if duration_sec <= 6 * 60 * 60:
                durations.append({"date": active_date, "screen_on_minutes": duration_sec / 60})
            
            active_start = None
            active_date = None

    duration_df = pd.DataFrame(durations)
    
    duration_daily = (
        duration_df.groupby("date", as_index=False)["screen_on_minutes"].sum()
        if not duration_df.empty
        else pd.DataFrame(columns=["date", "screen_on_minutes"])
    )
    
    event_daily = (
        df.groupby("date", as_index=False).size().rename(columns={"size": "screen_event_count"})
    )
    
    return event_daily.merge(duration_daily, on="date", how="left")


def load_calls_features(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "call_count", "call_duration_sec"])
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        return pd.DataFrame(columns=["date", "call_count", "call_duration_sec"])
    
    df["date"] = timestamp_to_date(df["timestamp"])
    
    grouped = df.groupby("date", as_index=False).agg(
        call_count=("dur", "size"),
        call_duration_sec=("dur", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        missed_call_count=("type", lambda s: (pd.to_numeric(s, errors="coerce") == 3).sum()),
    )
    
    return grouped


def load_messages_features(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "message_count", "messages_sent", "messages_received"])
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        return pd.DataFrame(columns=["date", "message_count", "messages_sent", "messages_received"])
    
    df["date"] = timestamp_to_date(df["timestamp"])
    
    grouped = df.groupby("date", as_index=False).agg(
        message_count=("message_type", "size"),
        messages_sent=("message_type", lambda s: (pd.to_numeric(s, errors="coerce") == 2).sum()),
        messages_received=("message_type", lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum()),
    )
    
    return grouped


def load_notifications_features(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "notification_count"])
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        return pd.DataFrame(columns=["date", "notification_count"])
    
    df["date"] = timestamp_to_date(df["timestamp"])
    
    return df.groupby("date", as_index=False).size().rename(columns={"size": "notification_count"})


def load_aware_features(participant_dir: Path) -> pd.DataFrame:
    aware_dir = participant_dir / "Aware"
    
    daily_frames = [
        load_screen_features(aware_dir / "screen.csv"),
        load_calls_features(aware_dir / "calls.csv"),
        load_messages_features(aware_dir / "messages.csv"),
        load_notifications_features(aware_dir / "notifications.csv"),
    ]

    merged = None
    
    for frame in daily_frames:
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="date", how="outer")
    
    return merged if merged is not None else pd.DataFrame(columns=["date"])


def load_oura_features(participant_dir: Path) -> pd.DataFrame:
    files = list((participant_dir / "Oura").glob("*.csv"))
    
    if not files:
        return pd.DataFrame(
            columns=[
                "date",
                "oura_sleep_duration",
                "oura_sleep_efficiency",
                "oura_sleep_score",
                "oura_steps",
                "oura_activity_score",
                "oura_readiness_score",
                "oura_hrv_balance",
                "oura_sleep_rmssd",
                "oura_sleep_hr_average",
            ]
        )

    df = pd.read_csv(files[0])
    
    if df.empty:
        return pd.DataFrame(columns=["date"])
    
    df["date"] = timestamp_to_date(df["timestamp"])
   
    cols = {
        "OURA_sleep_duration": "oura_sleep_duration",
        "OURA_sleep_efficiency": "oura_sleep_efficiency",
        "OURA_sleep_score": "oura_sleep_score",
        "OURA_activity_steps": "oura_steps",
        "OURA_activity_score": "oura_activity_score",
        "OURA_readiness_score": "oura_readiness_score",
        "OURA_readiness_score_hrv_balance": "oura_hrv_balance",
        "OURA_sleep_rmssd": "oura_sleep_rmssd",
        "OURA_sleep_hr_average": "oura_sleep_hr_average",
    }
    
    # Keep Oura variables used in final model
    keep = ["date"] + [c for c in cols if c in df.columns]
    
    grouped = df[keep].groupby("date", as_index=False).agg(
        {col: "mean" for col in keep if col != "date"}
    )
    
    return grouped.rename(columns=cols)


def load_watch_features(participant_dir: Path) -> pd.DataFrame:
    watch_dir = participant_dir / "Watch"
    files = sorted(watch_dir.glob("*.csv"))
    
    if not files:
        return pd.DataFrame(
            columns=["date", "watch_avg_heart_rate", "watch_hr_std", "watch_movement_intensity"]
        )

    daily_rows = []
   
    # Aggregate each raw watch file before combining by day
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue
        
        if df.empty or "timestamp" not in df.columns:
            continue
        
        df["date"] = timestamp_to_date(df["timestamp"])
        
        hrm = pd.to_numeric(df.get("hrm"), errors="coerce")
        accx = pd.to_numeric(df.get("accx"), errors="coerce")
        accy = pd.to_numeric(df.get("accy"), errors="coerce")
        accz = pd.to_numeric(df.get("accz"), errors="coerce")

        # Use acceleration magnitude as movement proxy
        movement = np.sqrt(accx.pow(2) + accy.pow(2) + accz.pow(2))
        
        frame = pd.DataFrame(
            {
                "date": df["date"],
                "watch_avg_heart_rate": hrm,
                "watch_hr_std": hrm,
                "watch_movement_intensity": movement,
            }
        )
        
        grouped = frame.groupby("date", as_index=False).agg(
            watch_avg_heart_rate=("watch_avg_heart_rate", "mean"),
            watch_hr_std=("watch_hr_std", "std"),
            watch_movement_intensity=("watch_movement_intensity", "mean"),
        )
        
        daily_rows.append(grouped)

    if not daily_rows:
        return pd.DataFrame(
            columns=["date", "watch_avg_heart_rate", "watch_hr_std", "watch_movement_intensity"]
        )

    daily = pd.concat(daily_rows, ignore_index=True)
    
    # Merge repeated watch files from same day
    
    return daily.groupby("date", as_index=False).mean(numeric_only=True)


def load_ema_summary(participant_dir: Path) -> dict[str, float]:
    survey_dir = participant_dir / "Surveys"
    files = [f for f in survey_dir.glob("*ema*.csv")]
    
    if not files:
        return {
            "ema_days": math.nan,
            "ema_lonely_mean": math.nan,
            "ema_connect_mean": math.nan,
            "ema_isolate_mean": math.nan,
            "ema_positive_mean": math.nan,
            "ema_negative_mean": math.nan,
        }

    df = pd.read_csv(files[0])
    
    return {
        "ema_days": float(len(df)),
        "ema_lonely_mean": numeric_mean(df.get("lonely")),
        "ema_connect_mean": numeric_mean(df.get("connect")),
        "ema_isolate_mean": numeric_mean(df.get("isolate")),
        "ema_positive_mean": numeric_mean(df.get("positive")),
        "ema_negative_mean": numeric_mean(df.get("negative")),
    }


def merge_daily_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    
    for frame in frames:
        if frame is None or frame.empty:
            continue
        
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on="date", how="outer")
    
    return merged if merged is not None else pd.DataFrame(columns=["date"])


def build_participant_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    participants = sorted(DATA_DIR.glob("Participant_*"), key=participant_sort_key)
    participant_rows = []
    availability_rows = []
    daily_rows = []

    # Build model features and availability summaries in one pass
    for participant_dir in participants:
        participant = participant_dir.name
        survey_dir = participant_dir / "Surveys"

        aware_daily = load_aware_features(participant_dir)
        oura_daily = load_oura_features(participant_dir)
        watch_daily = load_watch_features(participant_dir)
        participant_daily = merge_daily_frames([aware_daily, oura_daily, watch_daily])
        participant_daily["participant"] = participant
        daily_rows.append(participant_daily)

        aggregated = (
            participant_daily.drop(columns=["date", "participant"], errors="ignore")
            .mean(numeric_only=True)
            .to_dict()
        )

        ucla_file = survey_dir / f"UCLA Loneliness at the END_{participant}.csv"
        pss_file = survey_dir / f"Perceived stress at the END_{participant}.csv"
        phq_files = sorted(survey_dir.glob(f"Patient Health Questionnaire every week_{participant}.csv"))
        phq_score = score_phq(phq_files[-1]) if phq_files else math.nan

        participant_row = {
            "participant": participant,
            **aggregated,
            **load_ema_summary(participant_dir),
            "ucla_loneliness_total": score_ucla(ucla_file) if ucla_file.exists() else math.nan,
            "pss_end_total": score_pss(pss_file) if pss_file.exists() else math.nan,
            "phq9_latest_total": phq_score,
            "days_with_any_sensor_data": float(participant_daily["date"].nunique())
            
            if "date" in participant_daily.columns
            else 0.0,
        }
        
        participant_rows.append(participant_row)

        availability_rows.append(
            {
                "participant": participant,
                "has_aware": int((participant_dir / "Aware").exists()),
                "has_screen": int((participant_dir / "Aware" / "screen.csv").exists()),
                "has_calls": int((participant_dir / "Aware" / "calls.csv").exists()),
                "has_messages": int((participant_dir / "Aware" / "messages.csv").exists()),
                "has_notifications": int((participant_dir / "Aware" / "notifications.csv").exists()),
                "has_oura": int(bool(list((participant_dir / "Oura").glob("*.csv")))),
                "has_watch": int(bool(list((participant_dir / "Watch").glob("*.csv")))),
                "has_ema": int(bool(list(survey_dir.glob("*ema*.csv")))),
                "has_ucla_end": int(ucla_file.exists()),
                "sensor_days": participant_row["days_with_any_sensor_data"],
            }
        )

    participant_df = pd.DataFrame(participant_rows).sort_values("participant")
    availability_df = pd.DataFrame(availability_rows).sort_values("participant")
    daily_df = pd.concat(daily_rows, ignore_index=True).sort_values(["participant", "date"])

    # Split target at participant-level median
    median_ucla = participant_df["ucla_loneliness_total"].median()
    
    participant_df["loneliness_binary"] = (
        participant_df["ucla_loneliness_total"] >= median_ucla
    ).astype(int)
    
    participant_df["ucla_median_threshold"] = median_ucla
    
    return participant_df, availability_df, daily_df


def save_eda_outputs(participant_df: pd.DataFrame, availability_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    modality_summary = pd.DataFrame(
        {
            "metric": [
                "participants",
                "participant_days_with_sensor_data_mean",
                "participant_days_with_sensor_data_min",
                "participant_days_with_sensor_data_max",
                "ucla_mean",
                "ucla_median",
                "ema_lonely_mean",
            ],
            "value": [
                len(participant_df),
                participant_df["days_with_any_sensor_data"].mean(),
                participant_df["days_with_any_sensor_data"].min(),
                participant_df["days_with_any_sensor_data"].max(),
                participant_df["ucla_loneliness_total"].mean(),
                participant_df["ucla_loneliness_total"].median(),
                participant_df["ema_lonely_mean"].mean(),
            ],
        }
    )
    modality_summary.to_csv(TABLE_DIR / "summary_statistics.csv", index=False)
    availability_df.to_csv(TABLE_DIR / "data_availability_by_participant.csv", index=False)
    participant_df.to_csv(TABLE_DIR / "participant_level_features.csv", index=False)
    daily_df.to_csv(TABLE_DIR / "daily_aggregated_features.csv", index=False)

    modality_counts = availability_df[
        ["has_screen", "has_calls", "has_messages", "has_notifications", "has_oura", "has_watch", "has_ema", "has_ucla_end"]
    ].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x=modality_counts.index, y=modality_counts.values, palette="crest")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Participants with data")
    plt.title("Dataset Completeness by Modality")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_modality_completeness.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(participant_df["ucla_loneliness_total"], bins=10, kde=True, color="#3a7ca5")
    
    plt.axvline(
        participant_df["ucla_loneliness_total"].median(),
        color="#c44536",
        linestyle="--",
        label="Median split",
    )
    
    plt.xlabel("UCLA Loneliness Total Score")
    plt.title("Target Distribution: UCLA Loneliness Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_ucla_distribution.png", dpi=200)
    plt.close()

    selected = participant_df[
        [
            "ucla_loneliness_total",
            "ema_lonely_mean",
            "screen_on_minutes",
            "oura_sleep_duration",
            "oura_steps",
            "watch_avg_heart_rate",
        ]
    ].copy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Core Variables")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_core_correlations.png", dpi=200)
    plt.close()

    missing = participant_df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if not missing.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.index, y=missing.values, palette="mako")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing rate")
        plt.title("Participant-Level Feature Missingness")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_missingness.png", dpi=200)
        plt.close()
