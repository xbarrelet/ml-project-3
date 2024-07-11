import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

STRING_COLUMNS_NAMES = ["BuildingType", "PrimaryPropertyType", "Neighborhood"]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def transform_data(x, transformer):
    transformed_x = transformer.transform(x[STRING_COLUMNS_NAMES])
    transformed_x_df = DataFrame(transformed_x, columns=transformer.get_feature_names_out(STRING_COLUMNS_NAMES))
    transformed_x_df.index = x.index

    encoded_df = pd.concat([x, transformed_x_df], axis=1)
    x = encoded_df.drop(STRING_COLUMNS_NAMES, axis=1)

    return x


class StringDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, consider_string_values=False, encoding_mode="None", keep_energy_star_score=False):
        self.consider_string_values = consider_string_values
        self.encoding_mode = encoding_mode
        self.keep_energy_star_score = keep_energy_star_score
        self.te = TargetEncoder(target_type="continuous", )
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, x, y=None):
        x = DataFrame(x, columns=x.columns)

        if self.encoding_mode == "OneHotEncoding":
            self.ohe.fit(x[STRING_COLUMNS_NAMES])

        if self.encoding_mode == "TargetEncoding":
            self.te.fit(x[STRING_COLUMNS_NAMES], y)

        return self

    def transform(self, x, y=None):
        if not self.consider_string_values:
            return x.drop(columns=STRING_COLUMNS_NAMES, axis=1)

        if self.encoding_mode == "OneHotEncoding":
            x = transform_data(x, self.ohe)

        if self.encoding_mode == "TargetEncoding":
            x = transform_data(x, self.te)

        return x

    def get_params(self, deep=True):
        return {
            "consider_string_values": self.consider_string_values,
            "encoding_mode": self.encoding_mode
        }

    def set_params(self, **parameters):
        self.consider_string_values = parameters["consider_string_values"]
        self.encoding_mode = parameters["encoding_mode"]
        return self
