import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1.0, bias_fill=0.0):
    if not isinstance(module_list, (list, tuple)):
        module_list = [module_list]

    for module in module_list:
        for layer in module.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu",
                )
                layer.weight.data *= scale

                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)

            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="linear",
                )
                layer.weight.data *= scale

                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)

            elif isinstance(layer, _BatchNorm):
                init.constant_(layer.weight, 1)

                if layer.bias is not None:
                    layer.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwargs):
    layers = []

    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))

    return nn.Sequential(*layers)


def default_conv(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        strides,
        padding=kernel_size // 2,
        bias=bias,
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        conv=default_conv,
        n_feats=64,
        kernel_size=3,
        bias=True,
        bn=False,
        act=None,
        res_scale=1,
    ):
        super(ResBlock, self).__init__()

        if act is None:
            act = nn.ReLU(inplace=True)

        layers = []

        for i in range(2):
            layers.append(
                conv(
                    n_feats,
                    n_feats,
                    kernel_size,
                    bias=bias,
                )
            )

            if bn:
                layers.append(nn.BatchNorm2d(n_feats))

            if i == 0:
                layers.append(act)

        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        residual = self.body(x).mul(self.res_scale)
        output = residual + x

        return output


class DomainEncoder(nn.Module):
    def __init__(
        self,
        num_in_ch=1,
        num_feat=64,
        ds=64,
        num_block=4,
        bn=True,
    ):
        super(DomainEncoder, self).__init__()

        self.conv_first = nn.Conv2d(
            num_in_ch,
            num_feat,
            3,
            1,
            1,
        )

        self.body = make_layer(
            ResBlock,
            num_block,
            n_feats=num_feat,
            bn=bn,
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp_mu = nn.Linear(
            num_feat,
            ds,
        )

        self.mlp_logvar = nn.Linear(
            num_feat,
            ds,
        )

        default_init_weights(
            [
                self.conv_first,
                self.mlp_mu,
                self.mlp_logvar,
            ],
            scale=0.1,
        )

    def forward(self, x):
        feature = self.lrelu(
            self.conv_first(x)
        )

        feature = self.body(feature)

        feature = self.gap(feature).flatten(1)

        mu_s = self.mlp_mu(feature)
        logvar_s = self.mlp_logvar(feature)

        return mu_s, logvar_s


class FiLM_Layer(nn.Module):
    def __init__(self, ds, num_feat):
        super(FiLM_Layer, self).__init__()

        self.mlp_gamma = nn.Linear(
            ds,
            num_feat,
        )

        self.mlp_beta = nn.Linear(
            ds,
            num_feat,
        )

        nn.init.normal_(
            self.mlp_gamma.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.mlp_gamma.bias,
            1,
        )

        nn.init.normal_(
            self.mlp_beta.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.constant_(
            self.mlp_beta.bias,
            0,
        )

    def forward(self, feature, s):
        gamma = self.mlp_gamma(s).unsqueeze(-1).unsqueeze(-1)
        beta = self.mlp_beta(s).unsqueeze(-1).unsqueeze(-1)

        output = feature * gamma + beta

        return output


class ResNet_shape_FiLM(nn.Module):
    def __init__(
        self,
        num_in_ch=1,
        num_out_ch=1,
        num_feat=64,
        num_block=10,
        ds=64,
        bn=False,
    ):
        super(ResNet_shape_FiLM, self).__init__()

        self.conv_first = nn.Conv2d(
            num_in_ch,
            num_feat,
            3,
            1,
            1,
        )

        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    n_feats=num_feat,
                    bn=bn,
                )
                for _ in range(num_block)
            ]
        )

        self.films = nn.ModuleList(
            [
                FiLM_Layer(
                    ds,
                    num_feat,
                )
                for _ in range(num_block)
            ]
        )

        self.conv_last = nn.Conv2d(
            num_feat,
            num_out_ch,
            3,
            1,
            1,
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

        default_init_weights(
            [
                self.conv_first,
                self.conv_last,
            ],
            scale=0.1,
        )

    def forward(self, x, s):
        feature = self.lrelu(
            self.conv_first(x)
        )

        for block, film in zip(self.blocks, self.films):
            feature = block(feature)
            feature = film(feature, s)

        output = self.conv_last(
            self.lrelu(feature)
        )

        return output