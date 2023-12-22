import os
import pandas as pd
import torch


def biraffe2_prepare_data(data_folder: str):
    paths = [os.path.join(data_folder, "test"), os.path.join(data_folder, "train")]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    data_info = pd.read_csv(os.path.join(data_folder, "BIRAFFE2-metadata.csv"), delimiter=";")
    test_data_interval = len(data_info) // (len(data_info) * 0.2)

    for index, data in data_info.iterrows():
        subject_id = data["ID"]
        print(f"Preparing subject: {subject_id}")

        values = list(set([data["PHOTOS"], data["GAMEPAD"], data["GAME-1"]]))
        if len(values) != 1 or values[0] != "Y":
            continue

        send_to_test_set = index % test_data_interval == 0
        __save_subject_as_csv(data_folder, subject_id, send_to_test_set)


def __save_subject_as_csv(data_folder: str, subject_id: int, is_test: bool) -> None:
    face_dataframe = __prepare_csv_dataframe(
        os.path.join(data_folder, "BIRAFFE2-photo", f"SUB{subject_id}-Face.csv"), "GAME-TIMESTAMP"
    )
    gamepad_dataframe = __prepare_csv_dataframe(
        os.path.join(data_folder, "BIRAFFE2-gamepad", f"SUB{subject_id}-Gamepad.csv"), "TIMESTAMP"
    )
    log_dataframe = __prepare_json_log_dataframe(
        os.path.join(data_folder, "BIRAFFE2-games", f"SUB{subject_id}", f"SUB{subject_id}-Level01_Log.json")
    )

    full_dataframe = pd.concat([gamepad_dataframe, face_dataframe, log_dataframe], axis=1, join="inner")
    if len(full_dataframe) < 100:
        return

    emotion_dataframe = __emotion_to_arousal_valence(full_dataframe[face_dataframe.columns.tolist()])

    y_columns = emotion_dataframe.columns.tolist()
    y_tensor = __normalize_dataframe_to_tensor(emotion_dataframe)

    x_columns = [
        col for col in full_dataframe.columns.tolist() if col not in y_columns + face_dataframe.columns.tolist()
    ]
    correlated_cols = [
        "XMAX",
        "YMAX",
        "GYR-X",
        "SHOOTSCOUNTER",
        "HITCOUNTER",
        "MONEY",
        "COLLECTEDMONEY",
        "COLLECTEDHEALTH",
    ]
    x_columns = [col for col in x_columns if col not in correlated_cols]
    x_tensor = __normalize_dataframe_to_tensor(full_dataframe[x_columns])

    normalized_dataframe = pd.DataFrame(torch.cat((x_tensor, y_tensor), dim=1), columns=x_columns + y_columns)

    directory = "test" if is_test else "train"
    normalized_dataframe.to_csv(
        os.path.join(data_folder, directory, f"sample-SUB{subject_id}-normalized.csv"), index=False
    )


def __emotion_to_arousal_valence(dataframe: pd.DataFrame) -> pd.DataFrame:
    arousal_mapping = {
        "FEAR": 0.7,
        "SURPRISE": 0.45,
        "ANGER": 0.25,
        "DISGUST": -0.45,
        "HAPPINESS": 0.5,
        "NEUTRAL": 0.0,
        "SADNESS": -0.3,
        "CONTEMPT": -0.7,
    }
    valence_mapping = {
        "FEAR": -0.2,
        "SURPRISE": 0.75,
        "ANGER": -0.7,
        "DISGUST": -0.2,
        "HAPPINESS": 0.7,
        "NEUTRAL": 0.0,
        "SADNESS": -0.7,
        "CONTEMPT": -0.2,
    }

    results = []
    for _, data in dataframe.iterrows():
        valence = sum(valence_mapping[emotion] * value for emotion, value in data.items())
        arousal = sum(arousal_mapping[emotion] * value for emotion, value in data.items())
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
    dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(__timestamp_to_10_digits)
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
    log_dataframe[timestamp_column_name] = log_dataframe[timestamp_column_name].apply(lambda x: float(x.timestamp()))
    log_dataframe = __round_timestamps(log_dataframe, timestamp_column_name)

    log_dataframe = log_dataframe.rename(columns=str.upper)
    log_dataframe = log_dataframe.set_index(timestamp_column_name.upper())

    return log_dataframe


def __normalize_dataframe_to_tensor(dataframe: pd.DataFrame) -> torch.Tensor:
    tensor = torch.tensor(dataframe.values)

    min_vals, _ = torch.min(tensor, dim=0, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=0, keepdim=True)
    # scale and shift to the range [-1, 1]
    normalized_tensor = -1 + 2 * (tensor - min_vals) / (max_vals - min_vals)

    return normalized_tensor
