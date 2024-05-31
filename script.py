import os
import shutil
from pyexpat import model

import pandas as pd
import sklearn
import torch
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import missingno as msno

# Target columns: TotalGHGEmissions, Electricity(kWh), ENERGYSTARScore
DATA_FILEPATH = "resources/2016_Building_Energy_Benchmarking.csv"

TARGET_COLUMN = "TotalGHGEmissions"  #GHGEmissionsIntensity est mieux car normalisée par surface, essaie une ou l'autre
TARGET_COLUMN2 = "Electricity(kWh)"  #SiteEnergyUseWN (kBtu) est mieux

# STRUCTURAL_DATA_COLUMNS
CONSIDERED_COLUMNS = ["BuildingType", "PrimaryPropertyType", "ZipCode", "CouncilDistrictCode", "Neighborhood",
                      "Latitude", "Longitude", "YearBuilt", "NumberofBuildings", "NumberofFloors",
                      "PropertyGFAParking", "PropertyGFABuilding(s)", "ListOfAllPropertyUseTypes",
                      "LargestPropertyUseType", "ENERGYSTARScore", TARGET_COLUMN, TARGET_COLUMN2]
# Sois sur ne garder que celles qui font du sens mais ajoute des nouvelles colonnes avec le
# pourcentage steam/autre pour voir la provenance de lelectricite comme dans la donnee:
#par exemple la nature et proportions des sources d’énergie utilisées..
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
    for column_name in dataframe.columns:
        if column_name == TARGET_COLUMN:
            continue

        df = df[(df[column_name].isna())]

    return df


def prepare_data(df: DataFrame) -> DataFrame:
    # print(df.info())
    display_information_missing_values_and_produces_plot(df, "missing_values_after_loading")
    df = clean_dataset(df)
    display_information_missing_values_and_produces_plot(df, "missing_values_after_cleaning")
    return df



if __name__ == '__main__':
    print("Welcome to this new project!")
    remove_last_run_plots()

    dataframe: DataFrame = load_and_filter_data()
    print("The dataset has been loaded and filtered. Let's clean the data.\n")

    dataframe = prepare_data(dataframe)

    # COMMENCE PAR SUIVRE CE COURS: https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning/8296611-tirez-un-maximum-de-ce-cours
    # This helps: https://medium.com/swlh/my-first-work-with-pytorch-eea3bc82068

    X = dataframe.drop(TARGET_COLUMN, axis=1).values
    y = dataframe[TARGET_COLUMN].values
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Choisir un bon modele: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

