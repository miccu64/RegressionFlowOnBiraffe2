import pandas as pd
import torch
from torch.utils.data import Dataset


class Biraffe2Dataset(Dataset):
    def __init__(self, load_ready_dataset: bool = False):
        if load_ready_dataset:
            normalized_dataframe = pd.read_csv('sample-SUB211-normalized.csv')
            self.y_columns = ['ANGER', 'CONTEMPT', 'DISGUST', 'FEAR', 'HAPPINESS', 'NEUTRAL', 'SADNESS', 'SURPRISE']
            self.x_columns = [col for col in normalized_dataframe.columns.tolist() if col not in self.y_columns]
            self.y_tensor = self.__normalize_dataframe_to_tensor(normalized_dataframe[self.y_columns])
            self.x_tensor = self.__normalize_dataframe_to_tensor(normalized_dataframe[self.x_columns])
            return

        data_folder = 'data/BIRAFFE2/allData/singleSample/'
        
        face_dataframe = self.__prepare_csv_dataframe(data_folder + 'sample-SUB211-Face.csv', 'GAME-TIMESTAMP')
        gamepad_dataframe = self.__prepare_csv_dataframe(data_folder + 'sample-SUB211-Gamepad.csv', 'TIMESTAMP')
        log_dataframe = self.__prepare_json_log_dataframe(data_folder + 'sample-SUB211-Level01_Log.json')

        full_dataframe = pd.concat([gamepad_dataframe, face_dataframe, log_dataframe], axis=1, join='inner')

        self.y_columns = face_dataframe.columns.tolist()
        self.x_columns = [col for col in full_dataframe.columns.tolist() if col not in self.y_columns]
        correlated_cols = ['XMAX', 'YMAX', 'GYR-X', 'SHOOTSCOUNTER', 'HITCOUNTER', 'MONEY', 'COLLECTEDMONEY',
                           'COLLECTEDHEALTH']
        self.x_columns = [col for col in self.x_columns if col not in correlated_cols]

        self.x_tensor = self.__normalize_dataframe_to_tensor(full_dataframe[self.x_columns])
        self.y_tensor = self.__normalize_dataframe_to_tensor(full_dataframe[self.y_columns])

        normalized_dataframe = pd.DataFrame(torch.cat((self.x_tensor, self.y_tensor), dim=1),
                                            columns=self.x_columns + self.y_columns)
        normalized_dataframe.to_csv('sample-SUB211-normalized.csv')

    def __len__(self):
        return len(self.x_tensor)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

    def __nans_delete(self, dataframe: pd.DataFrame, timestamp_column_name: str) -> pd.DataFrame:
        # delete rows having NaN timestamp
        dataframe = dataframe.dropna(subset=[timestamp_column_name])
        # drop NaN-fully columns
        dataframe = dataframe.dropna(axis=1, how='all')
        # delete rows containing at least one NaN
        dataframe = dataframe.dropna()
        return dataframe

    def __timestamp_to_10_digits(self, timestamp: float) -> int:
        derived_timestamp = float(timestamp)
        max_value = 10 ** 10
        while derived_timestamp > max_value:
            derived_timestamp = derived_timestamp / 10
        return int(derived_timestamp)

    def __round_timestamps(self, dataframe: pd.DataFrame, timestamp_column_name: str):
        dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(self.__timestamp_to_10_digits)
        return dataframe.groupby(timestamp_column_name).mean().reset_index()

    def __prepare_csv_dataframe(self, file_path: str, timestamp_column_name: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path, delimiter=';')
        dataframe = self.__nans_delete(dataframe, timestamp_column_name)
        dataframe = self.__round_timestamps(dataframe, timestamp_column_name)
        return dataframe.set_index(timestamp_column_name)

    def __prepare_json_log_dataframe(self, file_path: str) -> pd.DataFrame:
        log_dataframe = pd.read_json(file_path)
        log_dataframe = log_dataframe.drop(columns=['idOfSound', 'timestampOfSound'])

        timestamp_column_name = 'timestamp'
        log_dataframe = self.__nans_delete(log_dataframe, timestamp_column_name)
        log_dataframe[timestamp_column_name] = log_dataframe[timestamp_column_name].apply(
            lambda x: float(x.timestamp()))
        log_dataframe = self.__round_timestamps(log_dataframe, timestamp_column_name)

        log_dataframe = log_dataframe.rename(columns=str.upper)
        log_dataframe = log_dataframe.set_index(timestamp_column_name.upper())

        return log_dataframe

    def __normalize_dataframe_to_tensor(self, dataframe: pd.DataFrame) -> torch.Tensor:
        tensor = torch.tensor(dataframe.values)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        return (tensor - mean) / std
