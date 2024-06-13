import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

STRING_COLUMNS_NAME = ["BuildingType",
                       # "PrimaryPropertyType", "Neighborhood", "LargestPropertyUseType"
                       ]
ENERGY_STAR_SCORE_COLUMN = "ENERGYSTARScore"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class StringDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, consider_string_values=False, encoding_mode="None", keep_energy_star_score=False):
        self.consider_string_values = consider_string_values
        self.encoding_mode = encoding_mode
        self.keep_energy_star_score = keep_energy_star_score

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if not self.consider_string_values:
            return x.drop(columns=STRING_COLUMNS_NAME, axis=1)

        x = DataFrame(x, columns=x.columns)

        if self.keep_energy_star_score:
            x = x[(x[ENERGY_STAR_SCORE_COLUMN] != -1)]
        else:
            x.drop(columns=[ENERGY_STAR_SCORE_COLUMN], axis=1)

        if self.encoding_mode == "OneHotEncoding":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            one_hot_encoded = encoder.fit_transform(x[STRING_COLUMNS_NAME], y)
            # print(encoder.get_feature_names_out(STRING_COLUMNS_NAME))
            one_hot_df = DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(STRING_COLUMNS_NAME))

            x = DataFrame(x, columns=x.columns)
            df_encoded = pd.concat([x, one_hot_df], axis=1)
            x = df_encoded.drop(STRING_COLUMNS_NAME, axis=1)

        if self.encoding_mode == "TargetEncoding":
            x = DataFrame(TargetEncoder(target_type="continuous").fit_transform(x, y), columns=x.columns)

        return x

    def get_params(self, deep=True):
        return {
            "consider_string_values": self.consider_string_values,
            "encoding_mode": self.encoding_mode,
            "keep_energy_star_score": self.keep_energy_star_score
        }

    def set_params(self, **parameters):
        self.consider_string_values = parameters["consider_string_values"]
        self.encoding_mode = parameters["encoding_mode"]
        self.keep_energy_star_score = parameters["keep_energy_star_score"]
        return self
