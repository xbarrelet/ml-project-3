import os
import shutil

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from transformers.NumericDataTransformer import NumericDataTransformer
from transformers.StringDataTransformer import StringDataTransformer

# Target columns: TotalGHGEmissions, Electricity(kWh), ENERGYSTARScore
DATA_FILEPATH = "resources/2016_Building_Energy_Benchmarking.csv"

# TARGET_COLUMN = "TotalGHGEmissions"  #GHGEmissionsIntensity est mieux car normalisée par surface, essaie une ou l'autre
TARGET_COLUMN = "GHGEmissionsIntensity"
# TARGET_COLUMN2 = "Electricity(kWh)"  #SiteEnergyUseWN (kBtu) est mieux
TARGET_COLUMN2 = "SiteEnergyUseWN(kBtu)"

# STRUCTURAL_DATA_COLUMNS
CONSIDERED_COLUMNS = ["BuildingType", "PrimaryPropertyType", "ZipCode", "CouncilDistrictCode", "Neighborhood",
                      "Latitude", "Longitude", "YearBuilt", "NumberofBuildings", "NumberofFloors",
                      "PropertyGFAParking", "PropertyGFABuilding(s)", "ListOfAllPropertyUseTypes",
                      "LargestPropertyUseType", "SteamUse(kBtu)", "NaturalGas(kBtu)",
                      # "ENERGYSTARScore",
                      TARGET_COLUMN,
                      TARGET_COLUMN2
                      ]

NUMERIC_CONSIDERED_COLUMNS = ["ZipCode", "CouncilDistrictCode", "Latitude", "Longitude", "YearBuilt",
                              "NumberofBuildings", "NumberofFloors", "PropertyGFAParking", "PropertyGFABuilding(s)",
                              "SteamUse(kBtu)", "NaturalGas(kBtu)",
                              # "ENERGYSTARScore",
                              TARGET_COLUMN,
                              TARGET_COLUMN2
                              ]


# Sois sur ne garder que celles qui font du sens mais ajoute des nouvelles colonnes avec le
# pourcentage steam/autre pour voir la provenance de lelectricite comme dans la donnee:
# par exemple la nature et proportions des sources d’énergie utilisées..

# Tu peux supprimer les valeurs vides et fais tes predictions. Tu peux comparer vers la fin les modeles avec suppressions
# des valeurs manquantes ou remplissage.


def remove_last_run_plots():
    shutil.rmtree('plots', ignore_errors=True)
    os.mkdir('plots')


def save_plot(plot, filename: str, prefix: str) -> None:
    os.makedirs(f"plots/{prefix}", exist_ok=True)

    fig = plot.get_figure()
    fig.savefig(f"plots/{prefix}/{filename}.png")
    plt.close()


def show_missing_value(dataframe: DataFrame) -> None:
    present_data_percentages = dataframe.notna().mean().sort_values(ascending=False)

    print("Listing present data percentages for each column:")
    print(present_data_percentages)
    print("\n")


def remove_duplicates(dataframe: DataFrame) -> None:
    initial_count = len(dataframe)
    pd.DataFrame.drop_duplicates(dataframe, subset=['PropertyName'], inplace=True)
    duplicates_number = initial_count - len(dataframe)
    print(f"{duplicates_number} duplicates were removed based on the PropertyName.\n")


def load_and_filter_data() -> DataFrame:
    df: DataFrame = pd.read_csv(DATA_FILEPATH, header=0, sep=",")
    remove_duplicates(df)
    return df[NUMERIC_CONSIDERED_COLUMNS]


def display_information_missing_values_and_produces_plot(df: DataFrame, filename: str) -> None:
    # present_data_percentages = df.notna().mean().sort_values(ascending=False)
    #
    # print("Listing present data percentages for each column:")
    # print(present_data_percentages)
    # print("\n")

    plot = msno.bar(df, figsize=(15, 18))
    save_plot(plot, filename, "missing_values")


def clean_dataset(df: DataFrame) -> DataFrame:
    return df.dropna()


def clean_data(df: DataFrame) -> DataFrame:
    # print(df.info())
    display_information_missing_values_and_produces_plot(df, "missing_values_after_loading")
    df = clean_dataset(df)
    display_information_missing_values_and_produces_plot(df, "missing_values_after_cleaning")
    return df


def display_all_results(grid_search):
    print("Displaying now each combination of hyperparameters tested with their mean scores:")
    cvres = grid_search.cv_results_

    for mean_score, params in sorted(zip(cvres["mean_test_score"], cvres["params"]), reverse=True):
        print(np.sqrt(-mean_score), params)
    print("\n")


def display_features_and_their_score():
    print("Displaying the most important features:")
    feature_importances = grid_search.best_estimator_._final_estimator.feature_importances_

    # The feature_importances don't include the proportion columns... Only 14 values, the original columns count
    for result in sorted(zip(feature_importances, NUMERIC_CONSIDERED_COLUMNS + ["steam_proportion", "natural_gas_proportion"]), reverse=True):
        print(result)
    print("\n")


def comparing_results_to_test_and_validation_sets():
    final_model = grid_search.best_estimator_

    print("Evaluating test set:")
    final_predictions = final_model.predict(x_test)
    # print(classification_report(final_predictions, y_test))
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"final_rmse:{final_rmse}\n")

    print("Evaluating validation set:")
    final_predictions2 = final_model.predict(x_validation)
    # print(classification_report(final_predictions2, y_validation))
    final_mse = mean_squared_error(y_validation, final_predictions2)
    final_rmse = np.sqrt(final_mse)
    print(f"final_rmse:{final_rmse}\n")


if __name__ == '__main__':
    print("Welcome to this new project!")
    remove_last_run_plots()

    dataframe: DataFrame = load_and_filter_data()
    print("The dataset has been loaded and filtered. Let's clean the data.\n")

    print(f"Dataset size before cleaning:{len(dataframe)}")
    # TODO: Do I want a transformer with multiple ways of cleaning data? dropna() is not the best...
    dataframe = clean_data(dataframe)
    print(f"Dataset size after cleaning:{len(dataframe)}\n")

    set_without_target_column_values = dataframe.drop(TARGET_COLUMN, axis=1)
    target_column_values = dataframe[TARGET_COLUMN]
    x, x_test, y, y_test = train_test_split(set_without_target_column_values, target_column_values, test_size=0.2,
                                            random_state=42)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25, random_state=42)

    print(f"training set size:{len(x_train)}, validation set size:{len(x_validation)}, test set size:{len(x_test)}\n")

    first_pipeline = Pipeline(steps=[
        # ("string-data-transformer", StringDataTransformer()),
        ("numeric-data-transformer", NumericDataTransformer()),
        ("model", RandomForestRegressor())
    ])

    transformer_param_grid = [{
        # Custom estimators
        'numeric-data-transformer__add_energy_proportions_data': [False, True],
        'numeric-data-transformer__data_transformation_mode': ["None", "normalization", "standardization"],
        # 'string-data-transformer__consider_string_values': [True],
        # 'string-data-transformer__encoding_mode': ["None", "OrdinalEncoding", "OneHotEncoding", "TargetEncoding"],

        # Model
        'model__n_estimators': [10, 30, 100, 300], 'model__max_features': [2, 4, 6, 8, 10, 12]
    }]

    grid_search = GridSearchCV(first_pipeline, transformer_param_grid, cv=5, scoring='neg_mean_squared_error',
                               refit=True, n_jobs=-1)
    grid_search.fit(x, y)

    print(f"Best score:{-1 * grid_search.best_score_} with params:{grid_search.best_params_}\n")

    display_all_results(grid_search)
    display_features_and_their_score()

    comparing_results_to_test_and_validation_sets()
