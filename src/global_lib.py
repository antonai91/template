#!/usr/bin/python

import pandas as pd
import numpy as np

def standardization(df_: pd.DataFrame, list_cols: list):
    df = df_.copy()

    for col in cols:
        df[col + " orig"] = df[col]
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df