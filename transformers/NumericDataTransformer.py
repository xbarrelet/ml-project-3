import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NumericDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_energy_proportions_data=False, data_transformation_mode="None"):
        self.add_energy_proportions_data = add_energy_proportions_data
        self.data_transformation_mode = data_transformation_mode

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        start = np.count_nonzero(DataFrame(x).isnull().values)

        if self.data_transformation_mode == "normalization":
            x = DataFrame(MinMaxScaler().fit_transform(x, y), columns=x.columns)

        elif self.data_transformation_mode == "standardization":
            # TODO: add with_mean and with_std parameters? Both true by default.
            #  Could be added in new transformation mode.
            x = DataFrame(StandardScaler().fit_transform(x, y), columns=x.columns)

        if not self.add_energy_proportions_data:
            x.drop(columns=["StreamProportion(kBtu)", "NaturalGasProportion(kBtu)"], axis=1, inplace=True)

        end = np.count_nonzero(DataFrame(x).isnull().values)
        if end - start != 0:
            print("numeric mismatch")
        return x

    def get_params(self, deep=True):
        return {
            "add_energy_proportions_data": self.add_energy_proportions_data,
            "data_transformation_mode": self.data_transformation_mode
        }

    def set_params(self, **parameters):
        self.add_energy_proportions_data = parameters["add_energy_proportions_data"]
        self.data_transformation_mode = parameters["data_transformation_mode"]
        return self
