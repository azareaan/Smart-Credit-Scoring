"""Preprocessing pipeline for anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_anomaly.config import FEATURES_CATEGORICAL, FEATURES_NUMERIC, RATIO_FEATURES


@dataclass
class PreprocessOutput:
    scaled: np.ndarray
    feature_names: list[str]
    raw_frame: pd.DataFrame


class Preprocessor:
    """Create ratio features, preserve missingness signals, and scale inputs."""

    def __init__(self, numeric_features: Iterable[str] | None = None) -> None:
        self.base_numeric_features = list(numeric_features or FEATURES_NUMERIC)
        self.numeric_features = list(self.base_numeric_features)
        self.categorical_features = list(FEATURES_CATEGORICAL)
        self.ratio_features = dict(RATIO_FEATURES)
        self.column_transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] | None = None

    def _add_ratio_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for ratio_name, (numerator, denominator) in self.ratio_features.items():
            frame[ratio_name] = frame[numerator] / frame[denominator].replace(0, np.nan)
        return frame

    def _add_missing_flags(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        for column in self.base_numeric_features:
            flag_name = f"IS_MISSING_{column}"
            frame[flag_name] = frame[column].isna().astype(int)
        return frame

    def _build_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

    def fit_transform(self, frame: pd.DataFrame) -> PreprocessOutput:
        prepared = self._add_ratio_features(frame)
        prepared = self._add_missing_flags(prepared)
        self.numeric_features = [
            *self.base_numeric_features,
            *list(self.ratio_features.keys()),
            *[f"IS_MISSING_{col}" for col in FEATURES_NUMERIC],
        ]
        self.column_transformer = self._build_transformer()
        scaled = self.column_transformer.fit_transform(prepared)
        self.feature_names_ = self._get_feature_names()
        return PreprocessOutput(scaled=scaled, feature_names=self.feature_names_, raw_frame=prepared)

    def transform(self, frame: pd.DataFrame) -> PreprocessOutput:
        if self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted.")
        prepared = self._add_ratio_features(frame)
        prepared = self._add_missing_flags(prepared)
        scaled = self.column_transformer.transform(prepared)
        feature_names = self.feature_names_ or self._get_feature_names()
        return PreprocessOutput(scaled=scaled, feature_names=feature_names, raw_frame=prepared)

    def inverse_transform(self, scaled: np.ndarray) -> pd.DataFrame:
        if self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted.")
        inverse_parts = []
        num_columns = []
        for name, _, columns in self.column_transformer.transformers_:
            if name == "num":
                num_columns = list(columns)
                break
        num_width = len(num_columns)
        for name, transformer, columns in self.column_transformer.transformers_:
            if name == "num":
                numeric = transformer.named_steps["scaler"].inverse_transform(
                    transformer.named_steps["imputer"].inverse_transform(scaled[:, :num_width])
                )
                inverse_parts.append(pd.DataFrame(numeric, columns=columns))
            elif name == "cat":
                encoded = scaled[:, num_width:]
                decoded = transformer.named_steps["encoder"].inverse_transform(encoded)
                inverse_parts.append(pd.DataFrame(decoded, columns=columns))
        return pd.concat(inverse_parts, axis=1)

    def _get_feature_names(self) -> list[str]:
        if self.column_transformer is None:
            return []
        feature_names: list[str] = []
        for name, transformer, columns in self.column_transformer.transformers_:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                encoder = transformer.named_steps["encoder"]
                feature_names.extend(encoder.get_feature_names_out(columns).tolist())
        return feature_names
