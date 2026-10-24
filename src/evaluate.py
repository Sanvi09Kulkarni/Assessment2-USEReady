"""Model definitions, comparison and tuning helpers."""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42


def get_models():
    """Fresh, untuned instances of every candidate model."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def make_pipeline(name, model):
    """Impute missing values (train median); scale only for Logistic Regression."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", model)])


def score_on_test(pipe, X_test, y_test):
    """Accuracy and macro precision / recall / F1 on the test set."""
    pred = pipe.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": p,
        "Recall": r,
        "Macro F1": f,
    }


def compare_models(X_train, X_test, y_train, y_test, with_cv=True, n_splits=4):
    """Fit every model; return a results table and the fitted pipelines.

    CV Macro F1 is computed on the training data only (TimeSeriesSplit), so it can
    be used for model selection without touching the test month.
    """
    rows, fitted = [], {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for name, model in get_models().items():
        pipe = make_pipeline(name, model)
        row = {"Model": name}
        if with_cv:
            row["CV Macro F1"] = cross_val_score(
                pipe, X_train, y_train, cv=tscv, scoring="f1_macro", n_jobs=1
            ).mean()
        pipe.fit(X_train, y_train)
        row.update(score_on_test(pipe, X_test, y_test))
        rows.append(row)
        fitted[name] = pipe
    return pd.DataFrame(rows).set_index("Model").round(3), fitted


PARAM_GRIDS = {
    "XGBoost": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
        "model__min_child_weight": [1, 5, 10],
    },
    "LightGBM": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_samples": [20, 50, 100],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    },
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5, 0.8],
    },
}
