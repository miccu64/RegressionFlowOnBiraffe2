import pandas as pd
import torch
from torch.utils.data import Dataset


class Biraffe2Dataset(Dataset):
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

    def __init__(self):
        data_folder = '../../data/BIRAFFE2/allData/singleSample/'

        face_dataframe = self.__prepare_csv_dataframe(data_folder + 'sample-SUB211-Face.csv', 'GAME-TIMESTAMP')
        gamepad_dataframe = self.__prepare_csv_dataframe(data_folder + 'sample-SUB211-Gamepad.csv', 'TIMESTAMP')
        log_dataframe = self.__prepare_json_log_dataframe(data_folder + 'sample-SUB211-Level01_Log.json')

        full_dataframe = pd.concat([gamepad_dataframe, face_dataframe, log_dataframe], axis=1, join='inner')

        tensor = torch.tensor(full_dataframe.values)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        normalized_tensor = (tensor - mean) / std

        y_columns = face_dataframe.columns.tolist()
        x_columns = full_dataframe.columns.tolist()
        x_columns = [element for element in x_columns if element not in y_columns]

        import seaborn as sns
        import matplotlib.pyplot as plt
        # %matplotlib inline
        corr = abs(pd.DataFrame(normalized_tensor).corr())

        # plot the heatmap
        sns.heatmap(corr)
        plt.show()

        a = 0

    # def __len__(self):
    #     return len(self.train_points)
    #
    # def __getitem__(self, idx):
    #     return self.train_points[idx, 0], self.train_points[idx, 1]


biraffe = Biraffe2Dataset()
