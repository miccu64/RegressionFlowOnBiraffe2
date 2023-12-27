import io
import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from biraffe2_helpers.biraffe2_utils import draw_biraffe2_heatmap
from data_regression_biraffe2_test import Biraffe2DatasetTest

import mmfp_utils
from args import get_args
from models.networks_regression_biraffe2 import HyperRegression
from utils import draw_hyps, draw_sdd_heatmap


def get_grid_logprob(
        height, width, x, model
):
    x_sp = np.linspace(0, width - 1, width // 1)
    y = np.linspace(0, height - 1, height // 1)
    X, Y = np.meshgrid(x_sp, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    _, _, (log_py_grid, log_px_grid) = model.get_logprob(
        x,
        torch.tensor(XX).unsqueeze(0).to(args.gpu)
    )
    return (X, Y), (log_px_grid.detach().cpu().numpy(), log_py_grid.detach().cpu().numpy())


def main(args):
    args.gpu = 0
    args.num_blocks = 1
    args.resume_checkpoint = 'checkpoints/biraffe2_v2/checkpoint-latest.pt'
    args.data_dir = "data/BIRAFFE2"
    args.dims = '32-32-32'

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
    model_serialize = checkpoint['model']
    model.load_state_dict(model_serialize)
    model.eval()
    save_path = os.path.join(os.path.split(resume_checkpoint)[0], 'results')
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
            x = x_all[row, :].unsqueeze(0)
            y_gt = y_gt_all[row, :].unsqueeze(0).unsqueeze(1)
            _, y_pred = model.decode(x, 100)

            log_py, log_px, _ = model.get_logprob(x, y_gt)

            log_py = log_py.cpu().detach().numpy().squeeze()
            log_px = log_px.cpu().detach().numpy().squeeze()

            hyps_name = f"{subject}-{row}-hyps.jpg"
            print(hyps_name)
            print("nll_x", str(-1.0 * log_px))
            print("nll_y", str(-1.0 * log_py))
            print("nll_(x+y)", str(-1.0 * log_px + log_py))

            nll_px_sum = nll_px_sum + -1.0 * log_px
            nll_py_sum = nll_py_sum + -1.0 * log_py
            counter = counter + 1.0
            y_pred = y_pred.cpu().detach().numpy().squeeze()

            valence = y_pred[:, 0]
            arousal = y_pred[:, 1]

            values_range = [-1, 1]
            plt.xlim(values_range)
            plt.ylim(values_range)
            plt.scatter(valence, arousal)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_matlike = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)
            plt.close()

            cv2.imwrite(os.path.join(save_path, hyps_name), img_matlike)

            multimod_emd = mmfp_utils.wemd_from_pred_samples(y_pred)
            multimod_emd_sum += multimod_emd
            print("multimod_emd", multimod_emd)

            (X, Y), (log_px_grid, log_py_grid) = get_grid_logprob(img_matlike.shape[0], img_matlike.shape[1], x, model)

            draw_biraffe2_heatmap(
                img=img_matlike,
                log_px_pred=log_px_grid,
                X=X, Y=Y,
                save_path=os.path.join(save_path, f"{subject}-{row}-heatmap.png")
            )

            result_row = {
                "subject": subject,
                "row": row,
                "nll_x": float(-1.0 * log_px),
                "nll_y": float(-1.0 * log_py),
                "multimod_emd": float(multimod_emd)
            }
            results.append(result_row)

    print("Mean log_p_x: ", nll_px_sum / counter)
    print("Mean log_p_y: ", nll_py_sum / counter)
    print("Mean multimod_emd:", multimod_emd_sum / counter)

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    args = get_args()
    main(args)
