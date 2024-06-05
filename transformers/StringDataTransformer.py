from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder

STRING_COLUMNS_NAME = ["BuildingType"]


class StringDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, consider_string_values=False, encoding_mode="None"):
        self.consider_string_values = consider_string_values
        self.encoding_mode = encoding_mode

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        if not self.consider_string_values:
            return x.drop(columns=STRING_COLUMNS_NAME, axis=1)

        if self.encoding_mode == "OrdinalEncoding":
            x = DataFrame(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(x, y),
                          columns=x.columns)

        if self.encoding_mode == "OneHotEncoding":
            x = DataFrame(OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False).fit_transform(x, y),
                          columns=x.columns)

        if self.encoding_mode == "TargetEncoding":
            x = DataFrame(TargetEncoder(target_type="continuous").fit_transform(x, y), columns=x.columns)

        return x

    def get_params(self, deep=True):
        return {"consider_string_values": self.consider_string_values,
                "encoding_mode": self.encoding_mode}

    def set_params(self, **parameters):
        self.consider_string_values = parameters["consider_string_values"]
        self.encoding_mode = parameters["encoding_mode"]
        return self
