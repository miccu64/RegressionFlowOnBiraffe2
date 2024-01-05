import os
from typing import Dict
import numpy as np
import pandas as pd
import torch


def biraffe2_prepare_data(data_folder: str):
    path = os.path.join(data_folder, "prepared_data")
    if not os.path.exists(path):
        os.makedirs(path)

    data_info = pd.read_csv(os.path.join(data_folder, "BIRAFFE2-metadata.csv"), delimiter=";")

    dataframes = {}
    for _, data in data_info.iterrows():
        subject_id = data["ID"]
        print(f"Loading subject: {subject_id}")

        values = list(set([data["PHOTOS"], data["GAMEPAD"], data["GAME-1"]]))
        if len(values) != 1 or values[0] != "Y":
            continue

        full_dataframe = __load_subject_files(data_folder, subject_id)
        if len(full_dataframe.values) < 20:
            continue

        dataframes[subject_id] = full_dataframe

    min, max = __find_min_max(dataframes)

    print("Saving data to files...")
    for subject_id, dataframe in dataframes.items():
        __save_normalized_subject(path, subject_id, dataframe, min, max)

    print("Done!")


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
        "SADNESS": -0.6,
        "CONTEMPT": 0.4,
    }

    results = []
    for _, data in dataframe.iterrows():
        valence = sum(x_valence_mapping[emotion] * value for emotion, value in data.items())
        arousal = sum(y_arousal_mapping[emotion] * value for emotion, value in data.items())
        results.append([valence, arousal])

    return pd.DataFrame(results, columns=["VALENCE", "AROUSAL"])


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
    all_data = []
    for dataframe in dataframes.values():
        all_data.append(dataframe.values)

    all_data = torch.tensor(np.concatenate(all_data, axis=0))

    min, _ = torch.min(all_data, dim=0, keepdim=True)
    max, _ = torch.max(all_data, dim=0, keepdim=True)

    # TODO: delete commented code when won't be anymore necessary
    # normalized_all_data = -1 + 2 * (all_data - min) / (max - min)
    # normalized_dataframe = pd.DataFrame(normalized_all_data, columns=dataframe.columns)
    # correlation_matrix = normalized_dataframe.corr().abs()
    # np.savetxt("aaa.txt", correlation_matrix.values)

    # # Create a heatmap using seaborn
    # sns.heatmap(correlation_matrix)
    # plt.title("Correlation Matrix Heatmap")
    # plt.show()
    return min, max


def __save_normalized_subject(
    path: str, subject_id: int, dataframe: pd.DataFrame, min: torch.Tensor, max: torch.Tensor
) -> None:
    # scale and shift to the range [-1, 1]
    normalized_tensor = -1 + 2 * (torch.tensor(dataframe.values) - min) / (max - min)

    normalized_dataframe = pd.DataFrame(normalized_tensor, columns=dataframe.columns)
    normalized_dataframe.to_csv(os.path.join(path, f"SUB{subject_id}-data.csv"), index=False)


if __name__ == "__main__":
    biraffe2_prepare_data("data/BIRAFFE2")
