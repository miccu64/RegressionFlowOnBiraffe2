from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Biraffe2Dataset(Dataset):
    def __init__(self, load_ready_dataset: bool = False):
        if load_ready_dataset:
            normalized_dataframe = pd.read_csv("sample-SUB211-normalized.csv")
            self.y_columns = ["VALENCE", "AROUSAL"]
            self.x_columns = [col for col in normalized_dataframe.columns.tolist() if col not in self.y_columns]
            self.y_tensor = self.__normalize_dataframe_to_tensor(normalized_dataframe[self.y_columns])
            self.x_tensor = self.__normalize_dataframe_to_tensor(normalized_dataframe[self.x_columns])
            return

        data_folder = "data/BIRAFFE2/allData/singleSample/"

        face_dataframe = self.__prepare_csv_dataframe(data_folder + "sample-SUB211-Face.csv", "GAME-TIMESTAMP")
        gamepad_dataframe = self.__prepare_csv_dataframe(data_folder + "sample-SUB211-Gamepad.csv", "TIMESTAMP")
        log_dataframe = self.__prepare_json_log_dataframe(data_folder + "sample-SUB211-Level01_Log.json")

        full_dataframe = pd.concat([gamepad_dataframe, face_dataframe, log_dataframe], axis=1, join="inner")

        emotion_dataframe = self.__emotion_to_arousal_valence(full_dataframe[face_dataframe.columns.tolist()])
        self.y_columns = emotion_dataframe.columns.tolist()
        self.y_tensor = self.__normalize_dataframe_to_tensor(emotion_dataframe)

        self.x_columns = [
            col
            for col in full_dataframe.columns.tolist()
            if col not in self.y_columns + face_dataframe.columns.tolist()
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
        self.x_columns = [col for col in self.x_columns if col not in correlated_cols]

        self.x_tensor = self.__normalize_dataframe_to_tensor(full_dataframe[self.x_columns])

        normalized_dataframe = pd.DataFrame(
            torch.cat((self.x_tensor, self.y_tensor), dim=1), columns=self.x_columns + self.y_columns
        )
        normalized_dataframe.to_csv("sample-SUB211-normalized.csv", index=False)

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

    # happindess
    def __emotion_to_arousal_valence(self, dataframe: pd.DataFrame) -> pd.DataFrame:
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

    def __nans_delete(self, dataframe: pd.DataFrame, timestamp_column_name: str) -> pd.DataFrame:
        # delete rows having NaN timestamp
        dataframe = dataframe.dropna(subset=[timestamp_column_name])
        # drop NaN-fully columns
        dataframe = dataframe.dropna(axis=1, how="all")
        # delete rows containing at least one NaN
        dataframe = dataframe.dropna()
        return dataframe

    def __timestamp_to_10_digits(self, timestamp: float) -> int:
        derived_timestamp = float(timestamp)
        max_value = 10**10
        while derived_timestamp > max_value:
            derived_timestamp = derived_timestamp / 10
        return int(derived_timestamp)

    def __round_timestamps(self, dataframe: pd.DataFrame, timestamp_column_name: str):
        dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(self.__timestamp_to_10_digits)
        return dataframe.groupby(timestamp_column_name).mean().reset_index()

    def __prepare_csv_dataframe(self, file_path: str, timestamp_column_name: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path, delimiter=";")
        dataframe = self.__nans_delete(dataframe, timestamp_column_name)
        dataframe = self.__round_timestamps(dataframe, timestamp_column_name)
        return dataframe.set_index(timestamp_column_name)

    def __prepare_json_log_dataframe(self, file_path: str) -> pd.DataFrame:
        log_dataframe = pd.read_json(file_path)
        log_dataframe = log_dataframe.drop(columns=["idOfSound", "timestampOfSound"])

        timestamp_column_name = "timestamp"
        log_dataframe = self.__nans_delete(log_dataframe, timestamp_column_name)
        log_dataframe[timestamp_column_name] = log_dataframe[timestamp_column_name].apply(
            lambda x: float(x.timestamp())
        )
        log_dataframe = self.__round_timestamps(log_dataframe, timestamp_column_name)

        log_dataframe = log_dataframe.rename(columns=str.upper)
        log_dataframe = log_dataframe.set_index(timestamp_column_name.upper())

        return log_dataframe

    def __normalize_dataframe_to_tensor(self, dataframe: pd.DataFrame) -> torch.Tensor:
        tensor = torch.tensor(dataframe.values)

        min_vals, _ = torch.min(tensor, dim=0, keepdim=True)
        max_vals, _ = torch.max(tensor, dim=0, keepdim=True)
        # scale and shift to the range [-1, 1]
        normalized_tensor = -1 + 2 * (tensor - min_vals) / (max_vals - min_vals)

        return normalized_tensor

    def decode_obj(x_valence: float, y_arousal: float):
        obj_size = 0.001
        class_id = 1
        object = np.array([x_valence, y_arousal, x_valence + obj_size, y_arousal + obj_size, class_id])
        object = np.expand_dims(np.expand_dims(np.expand_dims(object, 0), 0), 3).astype(np.float32)

        # tl = top-left corner of object, tr = top-right, bl = bottom-left
        x_tl = object[:, :, 0:1, :]
        y_tl = object[:, :, 1:2, :]
        x_br = object[:, :, 2:3, :]
        y_br = object[:, :, 3:4, :]
        object = np.concatenate((x_tl, y_tl, x_br, y_br, object[:, :, 4:5, :]), axis=2)
        return object

    def draw_sdd_heatmap(img: np.ndarray, log_px_pred, X, Y, save_path):
        def transparent_cmap(cmap, N=255):
            "Copy colormap and set alpha values"
            mycmap = cmap
            mycmap._init()
            mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
            return mycmap

        Z = log_px_pred.reshape(-1)
        Z = np.exp(Z)
        vmax = np.max(Z)
        vmin = np.min(Z)
        h, w, _ = img.shape
        plt.figure(
            figsize=(
                w // 25,
                h // 25,
            )
        )
        plt.imshow(img)
        plt.contourf(X, Y, Z.reshape(X.shape), vmin=vmin, vmax=vmax, cmap=transparent_cmap(plt.cm.jet), levels=20)

        plt.axis("off")
        plt.savefig(save_path, format="png", bbox_inches="tight", pad_inches=0)
