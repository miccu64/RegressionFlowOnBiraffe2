import os
import random
from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats
import torch


__y_labels = ["VALENCE", "AROUSAL"]


def biraffe2_prepare_data(data_folder: str):
    test_path = os.path.join(data_folder, "test_data")
    train_path = os.path.join(data_folder, "train_data")
    [os.makedirs(p, exist_ok=True) for p in [test_path, train_path]]

    data_info = pd.read_csv(os.path.join(data_folder, "BIRAFFE2-metadata.csv"), delimiter=";")

    dataframes = {}
    for _, data in data_info.iterrows():
        subject_id = data["ID"]
        print(f"Loading subject: {subject_id}")

        values = list(set([data["PHOTOS"], data["GAMEPAD"], data["GAME-1"]]))
        if len(values) != 1 or values[0] != "Y":
            continue

        full_dataframe = __load_subject_files(data_folder, subject_id)
        if len(full_dataframe.values) < 5:
            continue

        dataframes[subject_id] = full_dataframe

    print("Saving data to files...")

    train_dataframes = {}
    test_dataframes = {}
    for subject_id, dataframe in dataframes.items():
        if random.choices([True, False], weights=[0.2, 0.8], k=1)[0]:
            test_dataframes[subject_id] = dataframe
        else:
            train_dataframes[subject_id] = dataframe

    __prepare_and_save_dataset(train_dataframes, train_path)
    __prepare_and_save_dataset(test_dataframes, test_path)

    print("Done!")


def __prepare_and_save_dataset(
    dataframes: Dict[int, pd.DataFrame], save_path: str
) -> Dict[int, pd.DataFrame]:
    dataframes = __remove_outliers(dataframes)
    min, max = __find_min_max(dataframes)

    for subject_id, dataframe in dataframes.items():
        if len(dataframe.values) < 5:
            continue

        dataframe = __normalize_dataframe(dataframe, min, max)
        filepath = os.path.join(save_path, f"SUB{subject_id}-data.csv")
        dataframe.to_csv(filepath, index=False)


def __load_subject_files(data_folder: str, subject_id: int) -> pd.DataFrame:
    face_dataframe = __prepare_csv_dataframe(
        os.path.join(data_folder, "BIRAFFE2-photo", f"SUB{subject_id}-Face.csv"), "GAME-TIMESTAMP"
    )
    gamepad_dataframe = __prepare_csv_dataframe(
        os.path.join(data_folder, "BIRAFFE2-gamepad", f"SUB{subject_id}-Gamepad.csv"), "TIMESTAMP"
    )
    log_dataframe = __prepare_json_log_dataframe(
        os.path.join(
            data_folder, "BIRAFFE2-games", f"SUB{subject_id}", f"SUB{subject_id}-Level01_Log.json"
        )
    )

    all_dataframes = [gamepad_dataframe, face_dataframe, log_dataframe]
    if any(df.empty for df in all_dataframes):
        return pd.DataFrame([])

    full_dataframe = pd.concat(all_dataframes, axis=1, join="inner")
    emotion_dataframe = __emotion_to_arousal_valence(
        full_dataframe[face_dataframe.columns.tolist()]
    )
    if emotion_dataframe.empty:
        return pd.DataFrame([])

    y_columns = emotion_dataframe.columns.tolist()
    y_tensor = torch.tensor(emotion_dataframe.values)

    x_columns = [
        col
        for col in full_dataframe.columns.tolist()
        if col not in y_columns + face_dataframe.columns.tolist()
    ]
    x_tensor = torch.tensor(full_dataframe[x_columns].values)
    result_dataframe = pd.DataFrame(
        torch.cat((x_tensor, y_tensor), dim=1), columns=x_columns + y_columns
    )
    return __remove_correlated_columns(result_dataframe)


def __emotion_to_arousal_valence(dataframe: pd.DataFrame) -> pd.DataFrame:
    x_valence_mapping = {
        "FEAR": -0.2,
        "SURPRISE": 0.3,
        "ANGER": -0.8,
        "DISGUST": -1.0,
        "HAPPINESS": 1.0,
        "NEUTRAL": 0.0,
        "SADNESS": -0.8,
        "CONTEMPT": -0.9,
    }
    y_arousal_mapping = {
        "FEAR": 1.0,
        "SURPRISE": 1.0,
        "ANGER": 0.7,
        "DISGUST": 0.2,
        "HAPPINESS": 0.1,
        "NEUTRAL": 0.0,
        "SADNESS": -0.4,
        "CONTEMPT": 0.3,
    }

    results = []
    for _, data in dataframe.iterrows():
        valence = sum(x_valence_mapping[emotion] * value for emotion, value in data.items())
        arousal = sum(y_arousal_mapping[emotion] * value for emotion, value in data.items())
        results.append([valence, arousal])

    return pd.DataFrame(results, columns=__y_labels)


def __nans_delete(dataframe: pd.DataFrame, timestamp_column_name: str) -> pd.DataFrame:
    # delete rows having NaN timestamp
    dataframe = dataframe.dropna(subset=[timestamp_column_name])
    # drop NaN-fully columns
    dataframe = dataframe.dropna(axis=1, how="all")
    # delete rows containing at least one NaN
    dataframe = dataframe.dropna()
    return dataframe


def __timestamp_to_10_digits(timestamp: float) -> int:
    derived_timestamp = float(timestamp)
    max_value = 10**10
    while derived_timestamp > max_value:
        derived_timestamp = derived_timestamp / 10
    return int(derived_timestamp)


def __round_timestamps(dataframe: pd.DataFrame, timestamp_column_name: str):
    dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(
        __timestamp_to_10_digits
    )
    return dataframe.groupby(timestamp_column_name).mean().reset_index()


def __prepare_csv_dataframe(file_path: str, timestamp_column_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(file_path, delimiter=";")
    dataframe = __nans_delete(dataframe, timestamp_column_name)
    dataframe = __round_timestamps(dataframe, timestamp_column_name)
    return dataframe.set_index(timestamp_column_name)


def __prepare_json_log_dataframe(file_path: str) -> pd.DataFrame:
    log_dataframe = pd.read_json(file_path)
    log_dataframe = log_dataframe.drop(columns=["idOfSound", "timestampOfSound"])

    timestamp_column_name = "timestamp"
    log_dataframe = __nans_delete(log_dataframe, timestamp_column_name)
    log_dataframe[timestamp_column_name] = log_dataframe[timestamp_column_name].apply(
        lambda x: float(x.timestamp())
    )
    log_dataframe = __round_timestamps(log_dataframe, timestamp_column_name)

    log_dataframe = log_dataframe.rename(columns=str.upper)
    log_dataframe = log_dataframe.set_index(timestamp_column_name.upper())

    return log_dataframe


def __remove_correlated_columns(dataframe: pd.DataFrame):
    correlated_cols = ["COLLECTEDMONEY", "COLLECTEDHEALTH", "XMAX", "YMAX", "GYR-Z"]
    return dataframe[[col for col in dataframe.columns.tolist() if col not in correlated_cols]]


def __find_min_max(dataframes: Dict[int, pd.DataFrame]) -> tuple[torch.Tensor, torch.Tensor]:
    all_data = pd.concat(dataframes.values(), ignore_index=True)
    return all_data.min(), all_data.max()


def __remove_outliers(dataframes: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    all_data = pd.concat(dataframes.values(), ignore_index=True)
    outlier_mask = (np.abs(stats.zscore(all_data)) < 3).all(axis=1)

    result_dataframes = {}
    for id, dataframe in dataframes.items():
        lenght = dataframe.shape[0]
        result_dataframes[id] = dataframe[outlier_mask[:lenght]]
        outlier_mask = outlier_mask[lenght:].reset_index(drop=True)

    return result_dataframes


def __normalize_dataframe(
    dataframe: pd.DataFrame, min_values: pd.DataFrame, max_values: pd.DataFrame
) -> pd.DataFrame:
    # scale to the range [-1, 1]
    return -1 + 2 * (dataframe - min_values) / (max_values - min_values)


if __name__ == "__main__":
    data_path = "data/BIRAFFE2"
    biraffe2_prepare_data(data_path)
