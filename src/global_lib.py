#!/usr/bin/python

import pandas as pd
import numpy as np
import re

def standardization(df_: pd.DataFrame, list_cols: list):
    df = df_.copy()

    for col in cols:
        df[col + " orig"] = df[col]
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df

def nameCleaning(df):
    # Custom cleaning
    df.columns = [re.sub("[\\. \\(\\)\\/]+", "_", elem) for elem in df.columns]
    df.columns = [re.sub("-", "_", elem) for elem in df.columns]
    df.columns = [re.sub("'", "", elem) for elem in df.columns]
    df.columns = [re.sub(",", "_", elem) for elem in df.columns]
    df.columns = [re.sub(":", "_", elem) for elem in df.columns]
    df.columns = [re.sub("<", "MIN", elem) for elem in df.columns]
    df.columns = [re.sub(">", "MAG", elem) for elem in df.columns]
    df.columns = [re.sub("&", "E", elem) for elem in df.columns]
    df.columns = [re.sub("Â°", "", elem) for elem in df.columns]
    df.columns = [re.sub("%", "PERC", elem) for elem in df.columns]
    df.columns = [re.sub("\\+", "_", elem) for elem in df.columns]
    # String upper
    df.columns = [elem.lower() for elem in df.columns]
    # Trim
    df.columns = [elem.strip() for elem in df.columns]
    # Cut recurring underscore
    df.columns = [re.sub("_+", "_", elem) for elem in df.columns]
    return(df)