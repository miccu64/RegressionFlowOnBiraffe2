from matplotlib import pyplot as plt
import numpy as np


def draw_biraffe2_heatmap(
    x_valence: [float], y_arousal: [float], log_px_pred, log_py_pred, X, Y, save_path: str
):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap

    Z_diff = np.exp(log_py_pred.reshape(-1))
    vmax_diff = np.max(Z_diff)
    vmin_diff = np.min(Z_diff)

    figure, ax = plt.subplots()
    values_range = [-1, 1]
    plt.xlim(values_range)
    plt.ylim(values_range)
    plt.scatter(x_valence, y_arousal)

    cmap_diff = transparent_cmap(plt.cm.viridis)
    ax.contourf(
        X,
        Y,
        Z_diff.reshape(X.shape),
        vmin=vmin_diff,
        vmax=vmax_diff,
        cmap=cmap_diff,
        levels=20
    )

    figure.savefig(save_path, format="png", bbox_inches="tight")
    plt.close()


