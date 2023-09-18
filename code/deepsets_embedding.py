import enum
import torch
from torch import nn


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Deep sets layer from https://arxiv.org/abs/1703.06114
        """
        super(DeepSetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Linear(in_features, out_features, bias=True)
        self.layer2 = nn.Linear(in_features, out_features, bias=True)
        # self.layernorm_layer = torch.nn.LayerNorm(out_features)
        # self.layernorm_layer = torch.nn.BatchNorm1d(out_features)
        self.relu_layer = nn.ReLU(True)

    def forward(self, x):
        return self.relu_layer(
            (self.layer1(x) + self.layer2(x - x.mean(dim=1, keepdim=True)))
        )


class FCResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        features,
        depth=1,
        dropout_rate=0.01,
        num_outputs=None,
        activation="relu",
    ):
        super().__init__()
        """
        Fully connected resnet type architecture
        """
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "elu":
            self.activation = torch.nn.functional.elu
        else:
            raise ValueError("That acivation is unknown")

    def forward(self, x):
        x = self.first(x)

        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x)

        return x


def SequentialMLP(
    dims,
    activation="relu",
    final=False,
    input_layer=False,
    bias=True,
    layer_norm=False,
    batch_norm=False,
    **tkwargs,
):
    layers = []
    for idx, n in enumerate(dims):
        if final and idx == len(dims) - 1:
            layers.append(nn.Linear(n, 1, bias=bias, **tkwargs))
        elif input_layer is True and idx == 0:
            layers.append(nn.Linear(1, n, bias=bias, **tkwargs))
        else:
            layers.append(nn.Linear(n, n, bias=bias, **tkwargs))
        if not final:
            # if input_layer is False:
            #     # if layer_norm:
            #     layers.append(torch.nn.LayerNorm(n))
            # else:
            #     print(f"Batch norm {n}")
            # if batch_norm:
            #     layers.append(torch.nn.BatchNorm1d(n)),
            # layers.append(nn.Tanh())
            layers.append(Swish())
            # layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class DeepEmbeddingHetSpaces(nn.Module):
    def __init__(self, network_dims, dropout_ratio1=0.1, dropout_ratio2=0.1, **tkwargs):
        super(DeepEmbeddingHetSpaces, self).__init__()
        self.dense_v = FCResNet(input_dim=1, features=network_dims[0], **tkwargs)
        self.dense_c = FCResNet(
            input_dim=network_dims[0], features=network_dims[0], **tkwargs
        )
        # equivariant layer
        self.deepset_layer = DeepSetLayer(
            in_features=network_dims[0], out_features=network_dims[0]
        )
        self.dense_fz = FCResNet(
            input_dim=network_dims[0] + 1, features=network_dims[0] + 1, **tkwargs
        )
        self.dense_gz = FCResNet(
            input_dim=network_dims[0] + 1, features=network_dims[0] + 1, **tkwargs
        )
        self.dense_fy = FCResNet(
            input_dim=2 * (network_dims[0] + 1) - 1,
            features=network_dims[0] + 1,
            **tkwargs,
        )
        self.dropout_layer1 = nn.Dropout(p=dropout_ratio1)
        self.dropout_layer2 = nn.Dropout(p=dropout_ratio2)

    def forward(self, sup_x, sup_y, que_x, training_mode=False):
        """
        Args:
            que_x [N_q x d]
            sup_x [N_s x d]
            sup_y [N_s x 1]
        """
        vs_bar = self.dense_v(sup_x.unsqueeze(-1))
        vs_bar = torch.mean(vs_bar, dim=0)
        vs_bar = self.dense_c(vs_bar)

        cs_bar = self.dense_v(sup_y.unsqueeze(-1))
        cs_bar = torch.mean(cs_bar, dim=0)
        cs_bar = self.dense_c(cs_bar)

        concat_set_inputs = torch.cat((vs_bar, cs_bar), dim=0)
        concat_set_inputs = self.deepset_layer(concat_set_inputs)

        p_xs = concat_set_inputs[: (sup_x.shape[-1]), ...]
        p_ys = concat_set_inputs[(sup_x.shape[-1]) :, ...]

        p_xs = p_xs.unsqueeze(0).repeat(que_x.shape[0], 1, 1)
        p_ys = p_ys.repeat(que_x.shape[0], 1, 1)

        z = torch.cat((que_x.unsqueeze(-1), p_xs), dim=-1)

        z = self.dense_fz(z)
        z = torch.mean(z, dim=1)
        z = self.dense_gz(z)

        y = torch.cat((z.unsqueeze(-2), p_ys), dim=-1)
        y = self.dense_fy(y)
        return y.squeeze(1)
