#!/usr/bin/python

import pandas as pd
import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import KFold, LeaveOneOut

from tqdm import tqdm


def sklearn_cross_validation_prediction(X, y, model, n_splits, flag_loo):

    X.fillna(-999, inplace=True)

    if flag_loo:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    y_hat = np.zeros(shape=y.shape)
    
    for indexes in tqdm(kf.split(X)):
    
        X_ = X.iloc[indexes[0]]
        y_ = y.iloc[indexes[0]]
        
        model.fit(X_, y_)

        y_hat[indexes[1]] = model.predict(X.iloc[indexes[1]])

    return y_hat

def ebm_cross_validation_prediction(X, y, n_splits, dict_tickers, flag_loo):

    X.fillna(-999, inplace=True)
    
    df_local_explainer = pd.DataFrame()

    if flag_loo:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_hat = np.zeros(shape=y.shape)

    for indexes in tqdm(kf.split(X)):
        
        model = ExplainableBoostingRegressor(min_samples_leaf=5, outer_bags=12, inner_bags=12, random_state=SEED)

        X_ = X.iloc[indexes[0]]
        y_ = y.iloc[indexes[0]]

        model.fit(X_, y_)

        ebm_local = model.explain_local(X.iloc[indexes[1]], y.iloc[indexes[1]])

        for index in range(len(indexes[1])):
            local_data = ebm_local.data(index)
            df_tmp = pd.DataFrame({"name": local_data["names"], "score": local_data["scores"]})
            df_tmp["pandas_index"] =  indexes[1][index]
            df_tmp["symbol"] = df_tmp["pandas_index"].map(lambda x: dict_tickers[x])

            df_local_explainer = pd.concat([df_local_explainer, df_tmp], ignore_index=True)

        y_hat[indexes[1]] = model.predict(X.iloc[indexes[1]])

    return y_hat, df_local_explainer