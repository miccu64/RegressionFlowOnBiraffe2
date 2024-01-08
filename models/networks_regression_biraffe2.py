import torch
from torch import optim
from torch import nn
from models.flow import get_hyper_cnf
from utils import standard_laplace_logprob, truncated_normal, standard_normal_logprob
from torch.distributions.laplace import Laplace


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class HyperRegression(nn.Module):
    def __init__(self, args):
        super(HyperRegression, self).__init__()
        self.input_dim = args.input_dim
        self.hyper = HyperFlowNetwork(args)
        self.args = args
        self.point_cnf = get_hyper_cnf(self.args)
        self.gpu = args.gpu
        self.logprob_type = args.logprob_type

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == "adam":
                optimizer = optim.Adam(
                    params,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                )
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer

        opt = _get_opt_(list(self.hyper.parameters()) + list(self.point_cnf.parameters()))
        return opt

    def forward(self, x, y, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        target_networks_weights = self.hyper(x)

        # Loss
        y, delta_log_py = self.point_cnf(
            y, target_networks_weights, torch.zeros(batch_size, y.size(1), 1).to(y)
        )
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        loss = -log_px.mean()

        loss.backward()
        opt.step()
        recon = -log_px.sum()
        recon_nats = recon / float(y.size(0))
        return recon_nats

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def sample_laplace(size, gpu=None):
        m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        y = m.sample(sample_shape=torch.Size([size[0], size[1], size[2]])).float().squeeze(3)
        y = y if gpu is None else y.cuda(gpu)
        return y

    def decode(self, z, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code
        target_networks_weights = self.hyper(z)
        if self.logprob_type == "Laplace":
            y = self.sample_laplace((z.size(0), num_points, self.input_dim), self.gpu)
        if self.logprob_type == "Normal":
            y = self.sample_gaussian((z.size(0), num_points, self.input_dim), None, self.gpu)
        x = self.point_cnf(y, target_networks_weights, reverse=True).view(*y.size())
        return y, x

    def get_logprob(self, x, y_in):
        batch_size = x.size(0)
        target_networks_weights = self.hyper(x)

        # Loss
        y, delta_log_py = self.point_cnf(
            y_in, target_networks_weights, torch.zeros(batch_size, y_in.size(1), 1).to(y_in)
        )
        if self.logprob_type == "Laplace":
            log_py = standard_laplace_logprob(y)
        if self.logprob_type == "Normal":
            log_py = standard_normal_logprob(y)

        batch_log_py = log_py.sum(dim=2)
        batch_log_px = batch_log_py - delta_log_py.sum(dim=2)
        log_py = log_py.view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, y.size(1), 1).sum(1)
        log_px = log_py - delta_log_py

        return log_py, log_px, (batch_log_py, batch_log_px)


class HyperFlowNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.z_size = args.zdim
        self.use_bias = True
        self.relu_slope = 0.2

        input_size = args.input_size
        output_size = args.output_size

        output = []

        dims = tuple(map(int, args.hyper_dims.split("-")))
        self.n_out = dims[-1]
        model = []
        for k in range(len(dims)):
            if k == 0:
                model.append(
                    nn.Linear(in_features=input_size, out_features=dims[k], bias=self.use_bias)
                )
            else:
                model.append(
                    nn.Linear(in_features=dims[k - 1], out_features=dims[k], bias=self.use_bias)
                )
            if k < len(dims) - 1:
                model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)

        dims = tuple(map(int, args.dims.split("-")))
        for k in range(len(dims)):
            if k == 0:
                output.append(nn.Linear(self.n_out, args.input_dim * dims[k], bias=True))
            else:
                output.append(nn.Linear(self.n_out, dims[k - 1] * dims[k], bias=True))
            # bias
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            # scaling
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            output.append(nn.Linear(self.n_out, dims[k], bias=True))
            # shift
            output.append(nn.Linear(self.n_out, dims[k], bias=True))

        output.append(nn.Linear(self.n_out, dims[-1] * output_size, bias=True))
        # bias
        output.append(nn.Linear(self.n_out, output_size, bias=True))
        # scaling
        output.append(nn.Linear(self.n_out, output_size, bias=True))
        output.append(nn.Linear(self.n_out, output_size, bias=True))
        # shift
        output.append(nn.Linear(self.n_out, output_size, bias=True))

        self.output = ListModule(*output)

    def forward(self, x):
        output = self.model(x)
        multi_outputs = []
        for j, target_network_layer in enumerate(self.output):
            multi_outputs.append(target_network_layer(output))
        multi_outputs = torch.cat(multi_outputs, dim=1)
        return multi_outputs
