import json
import os
import numpy as np
import torch
from biraffe2_helpers.biraffe2_utils import draw_biraffe2_heatmap
from data_regression_biraffe2_test import Biraffe2DatasetTest

import mmfp_utils
from args import get_args
from models.networks_regression_biraffe2 import HyperRegression


def get_grid_logprob(x, model):
    x_sp = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_sp, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    _, _, (log_py_grid, log_px_grid) = model.get_logprob(
        x, torch.tensor(XX).unsqueeze(0).to(args.gpu)
    )
    return (X, Y), (log_px_grid.detach().cpu().numpy(), log_py_grid.detach().cpu().numpy())


def main(args):
    args.gpu = 0
    args.num_blocks = 1
    args.resume_checkpoint = "checkpoints/biraffe2_v2/checkpoint-latest.pt"
    args.data_dir = "data/BIRAFFE2"
    args.dims = '16-16-16'
    args.hyper_dims = '64-16'

    test_data = Biraffe2DatasetTest(args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    args.input_size = len(test_data.x_labels)
    args.output_size = len(test_data.y_labels)
    args.input_dim = 2

    model = HyperRegression(args)
    model = model.cuda()
    resume_checkpoint = args.resume_checkpoint
    print(f"Resume Path: {resume_checkpoint}")
    checkpoint = torch.load(resume_checkpoint, map_location="cpu")
    model_serialize = checkpoint["model"]
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], "results")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    nll_px_sum = 0
    nll_py_sum = 0

    multimod_emd_sum = 0

    counter = 0.0

    results = []
    for _, data in enumerate(test_loader):
        x_all, y_gt_all, subject = data
        x_all = x_all.float().to(args.gpu).squeeze()
        y_gt_all = y_gt_all.float().to(args.gpu).squeeze()
        subject = subject[0]

        for row in range(x_all.shape[0]):
            if row % 50 != 0:
                continue
            
            x = x_all[row, :].unsqueeze(0)
            y_gt = y_gt_all[row, :].unsqueeze(0).unsqueeze(1)
            _, y_pred = model.decode(x, 1000)

            log_py, log_px, _ = model.get_logprob(x, y_gt)

            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()

            print(f"Subject: {subject}, row: {row}")
            print("nll_x", str(-1.0 * log_px))
            print("nll_y", str(-1.0 * log_py))
            print("nll_(x+y)", str(-1.0 * log_px + log_py))

            nll_px_sum = nll_px_sum + -1.0 * log_px
            nll_py_sum = nll_py_sum + -1.0 * log_py
            counter = counter + 1.0
            y_pred = y_pred.cpu().detach().numpy().squeeze()

            multimod_emd = mmfp_utils.wemd_from_pred_samples(y_pred)
            multimod_emd_sum += multimod_emd
            print("multimod_emd", multimod_emd)

            valence = y_pred[:, 0]
            arousal = y_pred[:, 1]

            (X, Y), (log_px_grid, log_py_grid) = get_grid_logprob(x, model)

            draw_biraffe2_heatmap(
                x_valence=valence,
                y_arousal=arousal,
                log_px_pred=log_px_grid,
                log_py_pred=log_py_grid,
                X=X,
                Y=Y,
                save_path=os.path.join(save_path, f"{subject}-{row}-heatmap.png"),
            )

            result_row = {
                "subject": subject,
                "row": row,
                "nll_x": float(-1.0 * log_px),
                "nll_y": float(-1.0 * log_py),
                "multimod_emd": float(multimod_emd),
            }
            results.append(result_row)

    print("Mean log_p_x: ", nll_px_sum / counter)
    print("Mean log_p_y: ", nll_py_sum / counter)
    print("Mean multimod_emd:", multimod_emd_sum / counter)

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    args = get_args()
    main(args)
