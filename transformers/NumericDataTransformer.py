import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NumericDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_energy_proportions_data=False, data_transformation_mode="None"):
        self.add_energy_proportions_data = add_energy_proportions_data
        self.data_transformation_mode = data_transformation_mode
        self.min_max_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()

    def fit(self, x, y=None):
        if self.data_transformation_mode == "normalization":
            self.min_max_scaler.fit(x)

        elif self.data_transformation_mode == "standardization":
            self.standard_scaler.fit(x)

        return self

    def transform(self, x, y=None):
        columns = x.columns
        if self.data_transformation_mode == "normalization":
            x = self.min_max_scaler.transform(x)

        elif self.data_transformation_mode == "standardization":
            x = self.standard_scaler.transform(x)

        x = DataFrame(x, columns=columns)

        if not self.add_energy_proportions_data:
            x.drop(columns=["SteamProportion", "NaturalGasProportion"], axis=1, inplace=True)

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
