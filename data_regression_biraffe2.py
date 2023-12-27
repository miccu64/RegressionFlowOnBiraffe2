import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt


class Biraffe2Dataset(Dataset):
    def __init__(self, data_path: str):
        self.X = []
        self.Y = []
        self.y_labels = ["VALENCE", "AROUSAL"]

        files = self.get_files(False, data_path)
        for file in files:
            dataframe = pd.read_csv(file)
            self.x_labels = [col for col in dataframe.columns.tolist() if col not in self.y_labels]
            self.X.append(dataframe[self.x_labels].values)
            self.Y.append(dataframe[self.y_labels].values)
            break

        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_files(self, is_test: bool, data_path: str) -> list[str]:
        files = glob.glob(os.path.join(data_path, "prepared_data", "*.csv"))
        count = int(len(files) * 0.2)
        return files[-count:] if is_test else files[:-count]

    # TODO: delete decode_obj() and draw_sdd_heatmap() if not necessary
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

    def draw_biraffe2_heatmap(img: np.ndarray, log_px_pred, X, Y, save_path):
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
