import pandas as pd


class Biraffe2Dataset:
    def prepare_csv_dataframe(self, file_path: str, timestamp_column_name: str) -> pd.DataFrame:
        dataframe = pd.read_csv(file_path, delimiter=';')
        # delete rows having NaN timestamp
        dataframe = dataframe.dropna(subset=[timestamp_column_name])
        # drop NaN-fully columns
        dataframe = dataframe.dropna(axis=1, how='all')
        # delete rows containing at least one NaN
        dataframe = dataframe.dropna()
        # round timestamps
        dataframe[timestamp_column_name] = dataframe[timestamp_column_name].apply(lambda x: int(x * 10))
        return dataframe

    def __init__(self):
        data_folder = '../../data/BIRAFFE2/allData/singleSample/'

        face_dataframe = self.prepare_csv_dataframe(data_folder + 'sample-SUB211-Face.csv', 'GAME-TIMESTAMP')

        gamepad_dataframe = self.prepare_csv_dataframe(data_folder + 'sample-SUB211-Gamepad.csv', 'TIMESTAMP')
        gamepad_dataframe = gamepad_dataframe.groupby('TIMESTAMP').mean().reset_index()

        a = 0


biraffe = BIRAFFE2()
