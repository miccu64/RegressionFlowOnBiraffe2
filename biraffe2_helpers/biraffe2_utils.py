from matplotlib import pyplot as plt
import numpy as np


def draw_biraffe2_heatmap(x_valence: [float], y_arousal: [float], log_pred, X, Y, save_path: str, title: str):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.clip(np.linspace(0, 1.0, N + 4), 0, 1.0)
        return mycmap

    Z_diff = np.exp(log_pred.reshape(-1))
    vmax_diff = np.max(Z_diff)
    vmin_diff = np.min(Z_diff)

    figure, ax = plt.subplots()
    plt.title(title)
    values_range = [-1.1, 1.1]
    plt.xlim(values_range)
    plt.ylim(values_range)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.scatter(x_valence, y_arousal, color=(1.0, 0.5, 0.0))

    cmap_diff = transparent_cmap(plt.cm.viridis)
    ax.contourf(
        X, Y, Z_diff.reshape(X.shape), vmin=vmin_diff, vmax=vmax_diff, cmap=cmap_diff, levels=20
    )

    figure.savefig(save_path, format="png", bbox_inches="tight")
    plt.close()


def ratio_points_out_of_range(points: np.ndarray) -> float:
    wrong_points = 0
    for point in points:
        if point[0] > 1 or point[1] > 1 or point[0] < -1 or point[1] < -1:
            wrong_points += 1

    return wrong_points / points.shape[0] * 100
