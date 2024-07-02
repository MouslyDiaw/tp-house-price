"""Module for loading data from OpenData."""

from typing import Optional

import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_openml


def load_data(dataset_name: str, columns_to_lower: Optional[bool] = False) -> pd.DataFrame:
    """Fetch dataset from openml by name.

    Args:
        dataset_name (str): dataset name to load
        columns_to_lower (Optional[bool]): default is False
            flag to know if we should transform column names to lower

    Returns:
        pd.DataFrame: feature and target data

    """
    data = pd.DataFrame()
    logger.info(f"Dataset lo load: {dataset_name}")
    if not dataset_name:
        raise ValueError("Dataset name, like ``dataset_name``, must be defined!")
    if dataset_name == "house_prices":
        dframe_house = fetch_openml(dataset_name, return_X_y=False, parser='auto', target_column=None)
        data = dframe_house.data
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data description: {dframe_house.DESCR}")
    if columns_to_lower:
        logger.info("Columns will be transformed to lower!")
        data.columns = data.columns.str.lower()
    return data
