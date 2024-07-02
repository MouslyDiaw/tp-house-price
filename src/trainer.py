"""Training module"""
from typing import Callable, Union, Dict, Optional, List

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from settings.params import SEED


class Trainer:
    def __init__(self,
                 data: pd.DataFrame,
                 numerical_transformer: list,
                 categorical_transformer: list,
                 estimator: Callable,
                 target,
                 features: Optional[List[str]] = None,
                 test_size: Optional[float] = None,
                 cv: Optional[int] = None
                 ):
        logger.info(f"Test size: {test_size} | cross validation: {cv}")
        self.test_size = test_size
        self.cv = cv
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer
        self.estimator = estimator

        # Split the data into training and test sets. (0.75, 0.25) split.
        data_train, data_test = train_test_split(data, random_state=SEED)
        logger.info(f"Train size: {len(data_train)} | Test size: {len(data_test)}")
        # The predicted column is target
        self.y_train = data_train[target]
        self.y_test = data_test[target]
        # get features data
        if not features:
            self.x_train = data_train.drop([target], axis=1)
            self.x_test = data_test.drop([target], axis=1)
        else:
            self.x_train = data_train.loc[:, features]
            self.x_test = data_test.loc[:, features]

    def define_pipeline(self, **kwargs: dict) -> Pipeline:
        """ Define pipeline for modeling

        Args:
            **kwargs:

        Returns:
            Pipeline: sklearn pipeline
        """
        numerical_transformer = make_pipeline(*self.numerical_transformer)

        categorical_transformer = make_pipeline(*self.categorical_transformer)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, make_column_selector(dtype_include=["number"])),
                ("cat", categorical_transformer, make_column_selector(dtype_include=["object", "bool"])),
            ],
            remainder="drop",  # non-specified columns are dropped
            verbose_feature_names_out=False,  # will not prefix any feature names with the name of the transformer
        )

        # Append regressor to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        model_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", self.estimator)])
        logger.info(f"{model_pipe}")
        return model_pipe

    @staticmethod
    def eval_metrics(y_actual: Union[pd.DataFrame, pd.Series, np.ndarray],
                     y_pred: Union[pd.DataFrame, pd.Series, np.ndarray]
                     ) -> Dict[str, float]:
        """ Compute evaluation metrics

        Args:
            y_actual: Ground truth (correct) target values
            y_pred: Estimated target values.

        Returns:
            Dict[str, float]: dictionary of evaluation metrics.
                Expected keys are: "rmse", "mae", "r2", "max_error"

        """
        # Root mean squared error
        rmse = mean_squared_error(y_actual, y_pred, squared=False)
        # mean absolute error
        mae = mean_absolute_error(y_actual, y_pred)
        # R-squared: coefficient of determination
        r2 = r2_score(y_actual, y_pred)
        # max error: maximum value of absolute error (y_actual - y_pred)
        maxerror = max_error(y_actual, y_pred)
        return {"rmse": rmse,
                "mae": mae,
                "r2": r2,
                "max_error": maxerror
                }

    def tune_hyperparams(self):
        return None

    def train(self):
        # Useful for multiple runs (only doing one run in this sample notebook)
        # with mlflow.start_run():
        # define model
        sk_model = self.define_pipeline()
        sk_model.fit(self.x_train, self.y_train)
        # Evaluate Metrics
        y_train_pred = sk_model.predict(self.x_train)
        y_test_pred = sk_model.predict(self.x_test)
        train_metrics = Trainer.eval_metrics(self.y_train, y_train_pred)
        test_metrics = Trainer.eval_metrics(self.y_test, y_test_pred)

        # log out metrics
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Test metrics: {test_metrics}")
