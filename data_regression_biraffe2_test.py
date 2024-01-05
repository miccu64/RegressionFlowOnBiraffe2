import os
import pandas as pd

from data_regression_biraffe2 import Biraffe2Dataset


class Biraffe2DatasetTest(Biraffe2Dataset):
    def __init__(self, data_path: str):
        self.X = []
        self.Y = []
        self.y_labels = ["VALENCE", "AROUSAL"]

        files = self.get_files(os.path.join(data_path, "test_data"))
        self.subjects = []
        for file in files:
            position = file.find("SUB")
            self.subjects.append(file[position : (position + 6)])

        for file in files:
            dataframe = pd.read_csv(file)
            self.x_labels = [col for col in dataframe.columns.tolist() if col not in self.y_labels]
            self.X.append(dataframe[self.x_labels].values)
            self.Y.append(dataframe[self.y_labels].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.subjects[idx]
