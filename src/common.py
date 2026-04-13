from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

TEAM_MEMBERS = [
    "Jacorian Adom",
    "Aiden Agas",
    "Brooks Jackson",
    "Thomas Morrissey",
]

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def resolve_data_dir() -> Path:
    candidates = [
        BASE_DIR.parent / "loneliness_data" / "dataset",
        BASE_DIR.parent,
    ]

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("Participant_*")):
            return candidate

    raise FileNotFoundError(
        "Could not locate dataset. Expected either ../loneliness_data/dataset "
        "or participant folders extracted directly beside the project directory."
    )


DATA_DIR = resolve_data_dir()

for path in (OUTPUT_DIR, FIG_DIR, TABLE_DIR):
    path.mkdir(parents=True, exist_ok=True)


UCLA_MAP = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Always": 4}
# Standard reverse-keyed items for the UCLA Loneliness Scale Version 3.
UCLA_REVERSED = {1, 5, 6, 9, 10, 15, 16, 19, 20}

PSS_MAP = {
    "Never": 0,
    "Almost never": 1,
    "Sometimes": 2,
    "Fairly often": 3,
    "Very often": 4,
}


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered_rows = [[str(value) for value in row] for row in rows]

    widths = [
        max(len(str(header)), max((len(row[i]) for row in rendered_rows), default=0))
        for i, header in enumerate(headers)
    ]

    header_line = (
        "| "
        + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
        + " |"
    )

    divider_line = (
        "| "
        + " | ".join("-" * widths[i] for i in range(len(headers)))
        + " |"
    )

    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rendered_rows
    ]

    return "\n".join([header_line, divider_line, *body_lines])
