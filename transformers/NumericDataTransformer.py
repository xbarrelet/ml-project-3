from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def add_energy_proportion_column(row, energy_name):
    try:
        if (row[f'{energy_name}(kBtu)']) < 1:
            return 0.0

        return round(float(row[f'SiteEnergyUseWN(kBtu)']) / row[f'{energy_name}(kBtu)'], 2)

    except ValueError:
        return 0.0


class NumericDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_energy_proportions_data=False, data_transformation_mode="None"):
        self.add_energy_proportions_data = add_energy_proportions_data
        self.data_transformation_mode = data_transformation_mode

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if self.data_transformation_mode == "normalization":
            x = DataFrame(MinMaxScaler().fit_transform(x, y), columns=x.columns)

        elif self.data_transformation_mode == "standardization":
            # TODO: add with_mean and with_std parameters? Both true by default.
            #  Could be added in new transformation mode.
            x = DataFrame(StandardScaler().fit_transform(x, y), columns=x.columns)

        if self.add_energy_proportions_data:
            x['steam_proportion'] = x.apply(lambda row:
                                            add_energy_proportion_column(row, 'SteamUse'),
                                            axis=1)
            x['natural_gas_proportion'] = x.apply(lambda row:
                                                  add_energy_proportion_column(row, 'NaturalGas'),
                                                  axis=1)
        return x

    def get_params(self, deep=True):
        return {"add_energy_proportions_data": self.add_energy_proportions_data,
                "data_transformation_mode": self.data_transformation_mode}

    def set_params(self, **parameters):
        self.add_energy_proportions_data = parameters["add_energy_proportions_data"]
        self.data_transformation_mode = parameters["data_transformation_mode"]
        return self
