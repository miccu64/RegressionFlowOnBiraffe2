import pandas as pd


class Biraffe2Dataset:
    def nans_delete(self, dataframe: pd.DataFrame, timestamp_column_name: str) -> pd.DataFrame:
        # delete rows having NaN timestamp
        dataframe = dataframe.dropna(subset=[timestamp_column_name])
        # drop NaN-fully columns
        dataframe = dataframe.dropna(axis=1, how='all')
        # delete rows containing at least one NaN
        dataframe = dataframe.dropna()
        return dataframe

    def round_timestamps(self, dataframe: pd.DataFrame, timestamp_column_name: str):
        dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(lambda x: int(x * 10))
        return dataframe.groupby(timestamp_column_name).mean().reset_index()

    def prepare_csv_dataframe(self, file_path: str, timestamp_column_name: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path, delimiter=';')
        dataframe = self.nans_delete(dataframe, timestamp_column_name)
        dataframe = self.round_timestamps(dataframe, timestamp_column_name)
        return dataframe.set_index(timestamp_column_name)


    def __init__(self):
        data_folder = '../../data/BIRAFFE2/allData/singleSample/'

        face_dataframe = self.prepare_csv_dataframe(data_folder + 'sample-SUB211-Face.csv', 'GAME-TIMESTAMP')

        gamepad_dataframe = self.prepare_csv_dataframe(data_folder + 'sample-SUB211-Gamepad.csv', 'TIMESTAMP')

        log_dataframe = pd.read_json(data_folder + 'sample-SUB211-Level01_Log.json')
        log_dataframe = log_dataframe.drop(columns=['idOfSound', 'timestampOfSound'])
        timestamp_column_name = 'timestamp'
        log_dataframe = self.nans_delete(log_dataframe, timestamp_column_name)
        log_dataframe[timestamp_column_name] = log_dataframe[timestamp_column_name].apply(lambda x: int(round(x.timestamp())))
        log_dataframe = self.round_timestamps(log_dataframe, timestamp_column_name)
        log_dataframe = log_dataframe.set_index(timestamp_column_name)

        full_dataframe = pd.concat([gamepad_dataframe, face_dataframe, log_dataframe], axis=1)

        a = 0


biraffe = Biraffe2Dataset()
