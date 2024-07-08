import os
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost
from lime.lime_tabular import LimeTabularExplainer
from pandas.core.frame import DataFrame
from scipy.linalg import LinAlgWarning
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from transformers.NumericDataTransformer import NumericDataTransformer
from transformers.StringDataTransformer import StringDataTransformer

warnings.filterwarnings("ignore", category=LinAlgWarning, module='sklearn')

DATA_FILEPATH = "resources/2016_Building_Energy_Benchmarking.csv"

TARGET_COLUMN = "GHGEmissionsIntensity"
TARGET_COLUMN2 = "SiteEnergyUseWN(kBtu)"
ENERGY_STAR_SCORE_COLUMN = "ENERGYSTARScore"

CONSIDERED_COLUMNS = ["BuildingType", "PrimaryPropertyType", "Neighborhood", "ZipCode", "CouncilDistrictCode",
                      "ComplianceStatus", "Latitude", "Longitude", "YearBuilt", "NumberofBuildings", "NumberofFloors",
                      "PropertyGFABuilding(s)", "SteamUse(kBtu)", "NaturalGas(kBtu)", TARGET_COLUMN, TARGET_COLUMN2,
                      ENERGY_STAR_SCORE_COLUMN]

STRING_COLUMNS_NAMES = ["BuildingType", "PrimaryPropertyType", "Neighborhood"]

TREE_REGRESSORS = ["RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor"]


def remove_last_run_plots():
    shutil.rmtree('plots', ignore_errors=True)

    os.mkdir('plots')
    os.mkdir('plots/shap_results')


def save_plot(plot, filename: str, prefix: str) -> None:
    os.makedirs(f"plots/{prefix}", exist_ok=True)

    fig = plot.get_figure()
    fig.savefig(f"plots/{prefix}/{filename}.png")
    plt.close()


def remove_duplicates(dataframe: DataFrame) -> None:
    initial_count = len(dataframe)
    pd.DataFrame.drop_duplicates(dataframe, subset=['PropertyName'], inplace=True)
    duplicates_number = initial_count - len(dataframe)
    print(f"{duplicates_number} duplicates were removed based on the PropertyName.\n")


def load_and_filter_data() -> DataFrame:
    df: DataFrame = pd.read_csv(DATA_FILEPATH, header=0, sep=",")
    remove_duplicates(df)
    return df[CONSIDERED_COLUMNS]


def clean_dataset(df: DataFrame) -> DataFrame:
    df = df.drop(df[df['NumberofBuildings'] == 0].index)
    df = df.drop(df[(df['NumberofFloors'] == 0) | (df['NumberofFloors'] > 80)].index)
    df = df.drop(df[df["SiteEnergyUseWN(kBtu)"] == 0].index)
    df = df.drop(df[df["GHGEmissionsIntensity"] == 0].index)

    df = df.drop(df[df["ComplianceStatus"] != "Compliant"].index)
    df = df.drop(columns=["ComplianceStatus"], axis=1)

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
    df = clean_dataset(df)
    df = add_energy_proportions_columns(df)
    df = clean_non_habitation_buildings(df)

    return df


def add_energy_proportions_columns(df):
    steam_column = df.apply(lambda row: add_energy_proportion_column(row, 'SteamUse'), axis=1)
    df = df.assign(**{'SteamProportion': steam_column.values})

    natural_gas_column = df.apply(lambda row: add_energy_proportion_column(row, 'NaturalGas'), axis=1)
    df = df.assign(**{'NaturalGasProportion': natural_gas_column.values})

    return df


def create_feature_importance_plots(grid_search_cv: GridSearchCV, keys, model_name: str):
    features = []

    if hasattr(grid_search_cv.best_estimator_._final_estimator.regressor_, 'feature_importances_'):
        sns.set_theme(rc={'figure.figsize': (17, 12)})
        feature_importances = grid_search_cv.best_estimator_._final_estimator.regressor_.feature_importances_

        for result in sorted(zip(feature_importances, CONSIDERED_COLUMNS), reverse=True):
            score = result[0]
            if score > 0.02:
                features.append({"name": result[1], "score": score})

    elif hasattr(grid_search_cv.best_estimator_._final_estimator.regressor_, 'coef_'):
        sns.set_theme(rc={'figure.figsize': (25, 20)})
        importance = grid_search_cv.best_estimator_._final_estimator.regressor_.coef_

        for i, v in enumerate(importance):
            if v != 0:
                features.append({"name": keys[i], "score": v})
    else:
        return

    barplot = sns.barplot(DataFrame(features), x="score", y="name", hue="name", legend=False)
    barplot.set(xlabel=None)
    barplot.set(ylabel=None)
    save_plot(barplot, f"feature_importance_{model_name}", "feature_importance")


def add_energy_proportion_column(row, energy_name):
    try:
        if (row[f'{energy_name}(kBtu)']) < 1:
            return 0.0

        return round(float(row[f'{energy_name}(kBtu)'] / row[f'SiteEnergyUseWN(kBtu)']), 2)

    except ValueError:
        return 0.0


def get_accuracy_of_predictions(grid_search_cv, x_test, y_test, x_train, y_train):
    # Root Mean Squared Error: average difference between values
    train_score = grid_search_cv.score(x_train, y_train)
    test_score = grid_search_cv.score(x_test, y_test)
    print(f"train_score:{train_score}, test_score:{test_score}")
    rmse_train_accuracy = np.sqrt(-1 * grid_search_cv.score(x_train, y_train))
    rmse_test_accuracy = np.sqrt(-1 * grid_search_cv.score(x_test, y_test))

    # Another way to verify the prediction of a set, typically used if you have a third validation set
    # final_model = grid_search_cv.best_estimator_
    # final_predictions = final_model.predict(x_test)
    # final_mse = mean_squared_error(y_test, final_predictions)
    # final_rmse = np.sqrt(final_mse)

    return rmse_train_accuracy, rmse_test_accuracy


def get_models_and_their_hyperparameters():
    data_transformers_parameters_grid = {
        'string-data-transformer__keep_energy_star_score': [False, True],
        'string-data-transformer__consider_string_values': [False, True],
        'string-data-transformer__encoding_mode': ["OneHotEncoding", "TargetEncoding"],

        'numeric-data-transformer__add_energy_proportions_data': [True, False],
        'numeric-data-transformer__data_transformation_mode': ["None", "normalization", "standardization"]
    }

    return [
        {
            **data_transformers_parameters_grid,

            # The Least Absolute Shrinkage and Selection Operator is abbreviated as “LASSO.” Lasso regression is a type
            # of regularisation. It is preferred over regression methods for more precise prediction. This model makes
            # use of shrinkage which is the process by which data values are shrunk towards a central point known as
            # the mean. L1 regularisation is used in Lasso Regression. It is used when there are many features because
            # it performs feature selection automatically. The main purpose of Lasso Regression is to find the
            # coefficients that minimize the error sum of squares by applying a penalty to these coefficients.
            'model__regressor': [Lasso()],
            # 'model__regressor__alpha': [0.2],
            'model__regressor__alpha': np.arange(0.01, 1.0, 0.01),
        },
        {
            **data_transformers_parameters_grid,

            # Similar to the LASSO regression, ridge regression puts a similar constraint on the coefficients by
            # introducing a penalty factor. However, while lasso regression takes the magnitude of the coefficients,
            # ridge regression takes the square.
            'model__regressor': [Ridge()],
            'model__regressor__alpha': np.arange(1, 100.0, 1),
        },
        {
            **data_transformers_parameters_grid,

            # The elastic net is a regularized regression method that linearly combines the L1 and L2 penalties
            # of the lasso and ridge methods.
            'model__regressor': [ElasticNet()],
            'model__regressor__alpha': [1e-2, 1e-1, 1.0, 10.0],
            'model__regressor__l1_ratio': np.arange(0.1, 1, 0.1)
        },
        {
            **data_transformers_parameters_grid,

            # SVM works by finding a hyperplane in a high-dimensional space that best separates data into different
            # classes. It aims to maximize the margin (the distance between the hyperplane and the nearest data points
            # of each class) while minimizing classification errors.
            # SVR extends Support Vector Machines (SVM) into regression problems, allowing for the prediction of
            # continuous outcomes rather than classifying data into discrete categories as with a classifier.
            'model__regressor': [SVR(cache_size=500)],
            'model__regressor__C': [0.1, 1, 10, 100, 1000], 'model__regressor__gamma': [0.001, 0.0001],
            'model__regressor__epsilon': [0.001, 0.01, 0.1, 1]
        },
        {
            **data_transformers_parameters_grid,

            # The decision tree uses a tree structure. Starting from tree root, branching according to the conditions
            # and heading toward the leaves, the goal leaf is the prediction result. This decision tree has the
            # disadvantage of overfitting test data if the hierarchy is too deep. As a means to prevent this
            # overfitting, the idea of the ensemble method is used for decision trees. This technique uses a
            # combination of multiple decision trees rather than simply a single decision tree.
            #
            # Random forests create multiple decision trees by splitting a dataset based on random numbers.
            # It prevents overfitting by making predictions for all individual decision trees and
            # averaging the regression results.
            # 'model__regressor': [RandomForestRegressor(n_estimators=10)],
            # 'model__regressor__max_depth': [7], 'model__regressor__max_features': [9],
            'model__regressor': [RandomForestRegressor(n_estimators=300)],
            'model__regressor__max_depth': range(2, 8), 'model__regressor__max_features': range(2, 10)
        },
        {
            **data_transformers_parameters_grid,

            # Gradient boosting, on the other hand, is a technique for repeatedly adding decision trees so that the
            # next decision tree corrects the previous decision tree error. Compared to Random forest, the results are
            # more sensitive to parameter settings during training. However, with the correct parameter settings,
            # you will get better test results than random forest.
            # 'model__regressor': [GradientBoostingRegressor(n_estimators=10)],
            # 'model__regressor__max_depth': [3], 'model__regressor__max_features': [9]
            'model__regressor': [GradientBoostingRegressor(n_estimators=300)],
            'model__regressor__max_depth': range(2, 8), 'model__regressor__max_features': range(2, 11)
        },
        {
            **data_transformers_parameters_grid,

            # XGBoost ('eXtreme Gradient Boosting') and sklearn's GradientBoost are fundamentally the same as they are
            # both gradient boosting implementations. XGBoost is a lot faster than sklearn's. XGBoost is quite
            # memory-efficient and can be parallelized. # Having used both, XGBoost's speed is quite impressive and
            # its performance is superior to sklearn's GradientBoosting.
            # https://xgboost.readthedocs.io/en/stable/tutorials/model.html
            'model__regressor': [xgboost.XGBRegressor(tree_method="hist", n_estimators=300)],
            'model__regressor__max_depth': range(2, 11)
        }
    ]


def create_comparison_plots(results, scoring):
    results_df = pd.DataFrame(results)

    for score in scoring:
        results_df.sort_values(f"{score}_test_score", inplace=True)
        performance_plot = (results_df[[f"{score}_train_score", f"{score}_test_score", "model_name"]]
                            .plot(kind="bar", x="model_name", figsize=(15, 8), rot=0,
                                  title="Models Performance Sorted by Test Accuracy"))
        performance_plot.legend([f"{score} Train Accuracy", f"{score} Test Accuracy"])
        performance_plot.title.set_size(20)
        performance_plot.set(xlabel=None)

        save_plot(performance_plot, f"{score}_performance_plot", "comparison")

    results_df.sort_values("fit_time", inplace=True)
    cv_time_plot = (results_df[["fit_time", "model_name"]]
                    .plot(kind="bar", x="model_name", figsize=(15, 5), rot=0,
                          title="Models Gridsearch CV RMSE time (seconds)"))
    cv_time_plot.title.set_size(20)
    cv_time_plot.legend(["Gridsearch CV Time"])
    cv_time_plot.set(xlabel=None)
    save_plot(cv_time_plot, "cv_time_plot", "comparison")


def explain_results_using_shap(model, model_name, x_test):
    try:
        if model_name in TREE_REGRESSORS:
            shap_values = shap.TreeExplainer(model).shap_values(x_test)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(x_test, 5))
            shap_values = explainer(x_test)

        shap.summary_plot(shap_values, x_test)
        plt.savefig(f'plots/shap_results/summary_plot_{model_name}.png')
        plt.close()
    except Exception as e:
        print(f"Shap analysis failed for model:{model_name} because:", e)


def explain_results_using_lime(x_train, x_test, model_name, model, feat_names):
    explainer = LimeTabularExplainer(x_train.to_numpy(), feature_names=feat_names,
                                     class_names=[TARGET_COLUMN], verbose=True, mode='regression')
    i = 1
    try:
        explanation = explainer.explain_instance(x_test.to_numpy()[i], model.predict,
                                                 num_features=5)
        lime_results = []
        for feat_index, ex in explanation.as_map()[1]:
            lime_results.append({"name": feat_names[feat_index], "score": ex})
        barplot = sns.barplot(DataFrame(lime_results), x="score", y="name", hue="name", legend=False)
        barplot.set(xlabel=None)
        barplot.set(ylabel=None)
        save_plot(barplot, f"lime_results_{model_name}", "lime_results")
    except Exception as e:
        print(f"Lime analysis for model:{model_name} failed because of:", e)


def transform_sets_to_current_best_parameters(x_train, x_test, y_train, y_test, best_parameters):
    transformed_x_train = Pipeline(steps=[
        ("string-data-transformer", StringDataTransformer(
            consider_string_values=best_parameters['string-data-transformer__consider_string_values'],
            encoding_mode=best_parameters['string-data-transformer__encoding_mode'],
            keep_energy_star_score=best_parameters['string-data-transformer__keep_energy_star_score'])),
        ("numeric-data-transformer", NumericDataTransformer(
            add_energy_proportions_data=best_parameters['numeric-data-transformer__add_energy_proportions_data'],
            data_transformation_mode=best_parameters['numeric-data-transformer__data_transformation_mode']))
    ]).fit_transform(x_train, y_train)

    transformed_x_test = Pipeline(steps=[
        ("string-data-transformer", StringDataTransformer(
            consider_string_values=best_parameters['string-data-transformer__consider_string_values'],
            encoding_mode=best_parameters['string-data-transformer__encoding_mode'],
            keep_energy_star_score=best_parameters['string-data-transformer__keep_energy_star_score'])),
        ("numeric-data-transformer", NumericDataTransformer(
            add_energy_proportions_data=best_parameters['numeric-data-transformer__add_energy_proportions_data'],
            data_transformation_mode=best_parameters['numeric-data-transformer__data_transformation_mode']))
    ]).fit_transform(x_test, y_test)

    return transformed_x_train, transformed_x_test


def add_scores_to_result(result, cv_results, scoring):
    for scorer in scoring:
        best_train_score = max(cv_results["mean_train_" + scorer])
        best_test_score = max(cv_results["mean_test_" + scorer])

        result[f"{scorer}_train_score"] = best_train_score
        result[f"{scorer}_test_score"] = best_test_score


if __name__ == '__main__':
    print("Welcome to this new project!")
    remove_last_run_plots()

    dataframe: DataFrame = load_and_filter_data()
    print("The dataset has been loaded and filtered. Let's clean the data.\n")

    print(f"Dataset size before cleaning and preparation:{len(dataframe)}")
    dataframe = prepare_data(dataframe)
    print(f"Dataset size after cleaning and preparation:{len(dataframe)}\n")

    set_without_target_column_values = dataframe.drop(TARGET_COLUMN, axis=1)
    target_column_values = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(set_without_target_column_values, target_column_values,
                                                        test_size=0.2, random_state=42)
    print(f"training set size:{len(x_train)}, test set size:{len(x_test)}\n")

    results = []
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    for model_parameters in get_models_and_their_hyperparameters():
        pipeline = Pipeline(steps=[
            ("string-data-transformer", StringDataTransformer()),
            ("numeric-data-transformer", NumericDataTransformer()),
            ("model", TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1))
        ])

        model_name = model_parameters['model__regressor'][0].__class__.__name__
        print(f"Evaluation started for model:{model_name}")

        grid_search_cv = GridSearchCV(pipeline, model_parameters, cv=KFold(10, shuffle=True),
                                      scoring=scoring, n_jobs=-1, refit='neg_root_mean_squared_error',
                                      return_train_score=True)
        start_time = time.time()
        grid_search_cv.fit(x_train, y_train)
        fit_time = time.time() - start_time

        best_parameters = grid_search_cv.best_params_
        print(f"Best mean squared score:{grid_search_cv.best_score_} with params:{best_parameters}\n")

        transformed_x_train, transformed_x_test = transform_sets_to_current_best_parameters(x_train, x_test, y_train,
                                                                                            y_test, best_parameters)
        create_feature_importance_plots(grid_search_cv, transformed_x_train.keys(), model_name)

        model = grid_search_cv.best_estimator_._final_estimator.regressor_

        result = {
            "best_parameters": best_parameters,
            "model": model,
            "model_name": model_name,
            "fit_time": fit_time
        }

        add_scores_to_result(result, grid_search_cv.cv_results_, scoring)
        results.append(result)

    create_comparison_plots(results, scoring)

    print("\nDisplaying now Lime and Shap results for the best model\n")
    best_result = sorted(results, key=lambda x: x['neg_root_mean_squared_error_test_score'], reverse=True)[0]

    transformed_x_train, transformed_x_test = transform_sets_to_current_best_parameters(x_train, x_test, y_train,
                                                                                        y_test,
                                                                                        best_result['best_parameters'])
    explain_results_using_lime(transformed_x_train, transformed_x_test, best_result['model_name'], best_result['model'],
                               transformed_x_train.columns)
    explain_results_using_shap(best_result['model'], best_result['model_name'], transformed_x_test)
