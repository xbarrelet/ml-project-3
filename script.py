import os
import shutil
import warnings

import lightgbm
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import shap
import xgboost
from mpmath import rf
from pandas.core.frame import DataFrame
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from transformers.NumericDataTransformer import NumericDataTransformer
from transformers.StringDataTransformer import StringDataTransformer

# Target columns: TotalGHGEmissions, Electricity(kWh), ENERGYSTARScore
DATA_FILEPATH = "resources/2016_Building_Energy_Benchmarking.csv"

# TARGET_COLUMN = "TotalGHGEmissions"  #GHGEmissionsIntensity est mieux car normalisée par surface, essaie une ou l'autre
TARGET_COLUMN = "GHGEmissionsIntensity"
# TARGET_COLUMN2 = "Electricity(kWh)"  #SiteEnergyUseWN (kBtu) est mieux
TARGET_COLUMN2 = "SiteEnergyUseWN(kBtu)"
# Tu vois avoir 2 notebooks, un avec chaque valeur cible mais meme code
ENERGY_STAR_SCORE_COLUMN = "ENERGYSTARScore"

# STRUCTURAL_DATA_COLUMNS
CONSIDERED_COLUMNS = ["BuildingType",
                      "PrimaryPropertyType", "Neighborhood", "LargestPropertyUseType",
                      "ZipCode", "CouncilDistrictCode", "Latitude", "Longitude", "YearBuilt",
                      "NumberofBuildings", "NumberofFloors", "PropertyGFAParking", "PropertyGFABuilding(s)",
                      "SteamUse(kBtu)", "NaturalGas(kBtu)", TARGET_COLUMN, TARGET_COLUMN2,
                      ENERGY_STAR_SCORE_COLUMN]


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
    return df[CONSIDERED_COLUMNS]


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


def clean_non_habitation_buildings(df):
    initial_size = len(df)
    non_residential_buildings_df = df[~df.BuildingType.str.contains("Multifamily")]
    print(f"Removing {initial_size - len(non_residential_buildings_df)} non-residential buildings")
    return non_residential_buildings_df


def fill_missing_values_for_energy_star_score(df):
    df.fillna({ENERGY_STAR_SCORE_COLUMN: -1}, inplace=True)
    df[ENERGY_STAR_SCORE_COLUMN] = df[ENERGY_STAR_SCORE_COLUMN].replace("NULL", -1)
    return df


def prepare_data(df: DataFrame) -> DataFrame:
    # print(df.info())
    df = fill_missing_values_for_energy_star_score(df)
    display_information_missing_values_and_produces_plot(df, "missing_values_after_loading")
    df = clean_dataset(df)
    display_information_missing_values_and_produces_plot(df, "missing_values_after_cleaning")

    df = add_energy_proportions_columns(df)
    df = clean_non_habitation_buildings(df)

    return df


def add_energy_proportions_columns(df):
    steam_column = df.apply(lambda row: add_energy_proportion_column(row, 'SteamUse'), axis=1)
    df = df.assign(**{'SteamProportion(kBtu)': steam_column.values})

    natural_gas_column = df.apply(lambda row: add_energy_proportion_column(row, 'NaturalGas'), axis=1)
    df = df.assign(**{'NaturalGasProportion(kBtu)': natural_gas_column.values})

    return df


def display_all_results(grid_search_cv):
    # DOESNT WORK WITH MULTIPLE MODELS
    print("Displaying now each combination of hyperparameters tested with their mean test scores:")
    cvres = grid_search_cv.cv_results_

    for mean_score, params in sorted(zip(cvres["mean_test_score"], cvres["params"]), reverse=True):
        print(np.sqrt(-mean_score), params)
    print("\n")


def display_features_and_their_score(grid_search_cv: GridSearchCV, df: DataFrame):
    print("Displaying the most important features:")
    try:
        feature_importances = grid_search_cv.best_estimator_._final_estimator.regressor_.feature_importances_

        # The feature_importances don't include the proportion columns... Only 14 values, the original columns count
        for result in sorted(zip(feature_importances, CONSIDERED_COLUMNS), reverse=True):
            print(result)
        print("\n")

    except Exception:
        try:
            importance = grid_search_cv.best_estimator_._final_estimator.regressor_.coef_
            keys = list(df.keys())
            for i, v in enumerate(importance):
                print(f'Feature: {keys[i]}, Score: {v}')
        except Exception as e:
            print(f"No method to check the features and their score for the model was found: {e}")


def add_energy_proportion_column(row, energy_name):
    try:
        if (row[f'{energy_name}(kBtu)']) < 1:
            return 0.0

        return round(float(row[f'SiteEnergyUseWN(kBtu)']) / row[f'{energy_name}(kBtu)'], 2)

    except ValueError:
        return 0.0


def comparing_results_to_test_sets(grid_search_cv, x_test, y_test, x_train, y_train):
    final_model = grid_search_cv.best_estimator_

    print("Evaluating test set:")
    train_accuracy = grid_search_cv.score(x_train, y_train)
    test_accuracy = grid_search_cv.score(x_test, y_test)
    print(f"train_accuracy:{-train_accuracy}, test_accuracy:{-test_accuracy}")

    final_predictions = final_model.predict(x_test)
    # TODO: Tu peux comparer ton y_test avec final_predictions pour voir l-ecart par valeur, voir si certaines lignes sont trop differentes.
    # Tu peux utiliser ca pour essayer de comprendre ce qui se passe et si tu veux eliminer quelques lignes.
    # Le probleme ici est ton manque de donnees, ca implique trop de variance
    final_mse = mean_squared_error(y_test, final_predictions)
    # Root Mean Squared Error: average difference between values predicted by a model and the actual values
    final_rmse = np.sqrt(final_mse)
    print(f"Final Root Mean Squared Error:{final_rmse}\n")

z

def get_models_and_their_hyperparameters():
    estimators_params_grid = {
        'string-data-transformer__keep_energy_star_score': [False, True],
        'string-data-transformer__consider_string_values': [False],
        'string-data-transformer__encoding_mode': ["None", "OneHotEncoding"],
        # 'string-data-transformer__encoding_mode': ["OneHotEncoding", "TargetEncoding"],

        'numeric-data-transformer__add_energy_proportions_data': [False, True],
        'numeric-data-transformer__data_transformation_mode': ["None", "normalization", "standardization"]
    }

    return [
        # {
        #     **estimators_params_grid,
        #
        #     # Model
        #     'model__regressor': [LinearRegression()]
        # },
        # {
        #     **estimators_params_grid,
        #
        #     # The Least Absolute Shrinkage and Selection Operator is abbreviated as “LASSO.” Lasso regression is a type
        #     # of regularisation. It is preferred over regression methods for more precise prediction. This model makes
        #     # use of shrinkage which is the process by which data values are shrunk towards a central point known as
        #     # the mean. L1 regularisation is used in Lasso Regression. It is used when there are many features because
        #     # it performs feature selection automatically. The main purpose of Lasso Regression is to find the
        #     # coefficients that minimize the error sum of squares by applying a penalty to these coefficients.
        #     'model__regressor': [Lasso()],
        #     'model__regressor__alpha': np.arange(0.01, 1.0, 0.01),
        # },
        # {
        #     **estimators_params_grid,
        #
        #     # Similar to the LASSO regression, ridge regression puts a similar constraint on the coefficients by
        #     # introducing a penalty factor. However, while lasso regression takes the magnitude of the coefficients,
        #     # ridge regression takes the square.
        #     'model__regressor': [Ridge()],
        #     'model__regressor__alpha': np.arange(0.01, 1.0, 0.01),
        # },
        # {
        #     **estimators_params_grid,
        #
        #     # SVM works by finding a hyperplane in a high-dimensional space that best separates data into different
        #     # classes. It aims to maximize the margin (the distance between the hyperplane and the nearest data points
        #     # of each class) while minimizing classification errors.
        #     # SVR extends Support Vector Machines (SVM) into regression problems, allowing for the prediction of
        #     # continuous outcomes rather than classifying data into discrete categories as with a classifier.
        #     'model__regressor': [SVR(cache_size=500)],
        #     'model__regressor__C': [0.1, 1, 10, 100, 1000], 'model__regressor__gamma': [0.001, 0.0001],
        #     'model__regressor__epsilon': [0.001, 0.01, 0.1, 1]
        # },
        {
            **estimators_params_grid,

            # The decision tree uses a tree structure. Starting from tree root, branching according to the conditions
            # and heading toward the leaves, the goal leaf is the prediction result. This decision tree has the
            # disadvantage of overfitting test data if the hierarchy is too deep. As a means to prevent this
            # overfitting, the idea of the ensemble method is used for decision trees. This technique uses a
            # combination of multiple decision trees rather than simply a single decision tree.
            #
            # Random forests create multiple decision trees by splitting a dataset based on random numbers.
            # It prevents overfitting by making predictions for all individual decision trees and
            # averaging the regression results.
            'model__regressor': [RandomForestRegressor(n_estimators=300)],
            'model__regressor__max_depth': [7], 'model__regressor__max_features': [9],
            # 'model__regressor__max_depth': range(2, 8), 'model__regressor__max_features': range(2, 10)
        },
        # {
        #     **estimators_params_grid,
        #
        #     # Gradient boosting, on the other hand, is a technique for repeatedly adding decision trees so that the
        #     # next decision tree corrects the previous decision tree error. Compared to Random forest, the results are
        #     # more sensitive to parameter settings during training. However, with the correct parameter settings,
        #     # you will get better test results than random forest.
        #     'model__regressor': [GradientBoostingRegressor(n_estimators=300)],
        #     'model__regressor__max_depth': range(2, 8), 'model__regressor__max_features': range(2, 10)
        # },
        # {
        #     **estimators_params_grid,
        #
        #     # XGBoost ('eXtreme Gradient Boosting') and sklearn's GradientBoost are fundamentally the same as they are
        #     # both gradient boosting implementations. XGBoost is a lot faster than sklearn's. XGBoost is quite
        #     # memory-efficient and can be parallelized. # Having used both, XGBoost's speed is quite impressive and
        #     # its performance is superior to sklearn's GradientBoosting.
        #     # https://xgboost.readthedocs.io/en/stable/tutorials/model.html
        #     'model__regressor': [xgboost.XGBRegressor(tree_method="hist", n_estimators=300)],
        #     'model__regressor__max_depth': range(2, 11)
        # },
        # {
        #     **estimators_params_grid,
        #     # LightGBM's unique leaf-wise split algorithm produces simpler models that use significantly less memory
        #     # compared to XGBoost during training. XGBoost implements disk-based tree learning and in-memory prediction
        #     # for better memory management. But LightGBM has the edge for lower memory usage overall.
        #     'model__regressor': [lightgbm.LGBMRegressor(tree_method="hist", n_estimators=300, verbose_eval=-1,
        #                                                 early_stopping_rounds=100)],
        #     'model__regressor__max_depth': range(2, 11)
        # },
    ]


def explain_with_lime():
    categorical_features = np.argwhere(
        np.array([len(set(x_train.data[:, x])) for x in range(x_train.data.shape[1])]) <= 10).flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train, feature_names=y_train, class_names=[TARGET_COLUMN],
                                                       categorical_features=categorical_features, verbose=True,
                                                       mode='regression')
    i = 25
    exp = explainer.explain_instance(y_train[i], rf.predict, num_features=5)
    exp.show_in_notebook(show_table=True)


def explain_with_shap():
    # SHAP ONLY FLOAT VALUES?
    explainer = shap.TreeExplainer(grid_search_cv.best_estimator_._final_estimator.regressor_)
    explanation = explainer(x_train)
    shap_values = explanation.values
    print(shap_values)


if __name__ == '__main__':
    print("Welcome to this new project!")
    remove_last_run_plots()

    dataframe: DataFrame = load_and_filter_data()
    print("The dataset has been loaded and filtered. Let's clean the data.\n")

    print(f"Dataset size before cleaning and preparation:{len(dataframe)}")
    dataframe = prepare_data(dataframe)
    print(f"Dataset size after cleaning and preparation:{len(dataframe)}\n")
    # Pour la prez tu peux aussi faire des bivariees avec la target, heatmap et ACP.

    set_without_target_column_values = dataframe.drop(TARGET_COLUMN, axis=1)
    target_column_values = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(set_without_target_column_values, target_column_values,
                                                        test_size=0.2, random_state=42)
    print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

    first_pipeline = Pipeline(steps=[
        ("string-data-transformer", StringDataTransformer()),
        ("numeric-data-transformer", NumericDataTransformer()),
        ("model", TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1))
    ])

    for model_parameters in get_models_and_their_hyperparameters():
        model_name = model_parameters['model__regressor'][0].__class__.__name__
        print(f"Evaluating now the model:{model_name}")

        grid_search_cv = GridSearchCV(first_pipeline, model_parameters,
                                      cv=KFold(5, shuffle=True), scoring='neg_mean_squared_error', refit=True,
                                      n_jobs=-1)
        grid_search_cv.fit(x_train, y_train)

        print(f"Best score:{grid_search_cv.best_score_} with params:{grid_search_cv.best_params_}\n")

        # display_all_results(grid_search_cv)
        display_features_and_their_score(grid_search_cv, dataframe)

        comparing_results_to_test_sets(grid_search_cv, x_test, y_test, x_train, y_train)

        # TODO: Check https://openclassrooms.com/fr/paths/794/projects/1509/resources
        # explain_with_lime()
        # explain_with_shap()


