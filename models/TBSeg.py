import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientunet import get_efficientunet_b2

from .Basic_module import Criterion, Visualization
from .ResNet import ResNet_shape_FiLM, DomainEncoder


class TBSeg(nn.Module):
    def __init__(self, args):
        super(TBSeg, self).__init__()

        self.args = args
        self.num_classes = args.num_classes
        self.ds = getattr(args, "ds", 64)
        self.nu = getattr(args, "nu", 4.0)
        self.gamma_grad = getattr(args, "gamma_grad", 10.0)
        self.num_feat = getattr(args, "num_feat", 64)
        self.shape_blocks = getattr(args, "shape_blocks", 10)

        self.domain_enc = DomainEncoder(
            num_in_ch=1,
            num_feat=self.num_feat,
            ds=self.ds,
        )

        self.res_shape = ResNet_shape_FiLM(
            num_in_ch=1,
            num_out_ch=2,
            num_feat=self.num_feat,
            num_block=self.shape_blocks,
            ds=self.ds,
            bn=False,
        )

        self.segmenter = get_efficientunet_b2(
            out_channels=2 * self.num_classes,
            pretrained=False,
        )

        self.softmax = nn.Softmax(dim=1)

        Dx = torch.zeros([1, 1, 3, 3], dtype=torch.float)
        Dx[:, :, 1, 1] = 1.0
        Dx[:, :, 1, 0] = -0.25
        Dx[:, :, 1, 2] = -0.25
        Dx[:, :, 0, 1] = -0.25
        Dx[:, :, 2, 1] = -0.25
        self.Dx = nn.Parameter(data=Dx, requires_grad=False)

    @staticmethod
    def sample_normal_jit(mu, log_var):
        sigma = torch.exp(log_var / 2)
        eps = mu.mul(0).normal_()
        z = eps.mul_(sigma).add_(mu)
        return z, eps

    @staticmethod
    def pad_to_multiple(x, multiple=32):
        height, width = x.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple

        x = F.pad(
            x,
            pad=(0, pad_width, 0, pad_height),
            mode="constant",
            value=0.0,
        )

        return x, height, width

    def generate_s(self, samples, sample=True):
        mu_s, log_var_s = self.domain_enc(samples)
        log_var_s = torch.clamp(log_var_s, -20, 0)

        if sample:
            s, _ = self.sample_normal_jit(mu_s, log_var_s)
        else:
            s = mu_s

        mu_sq_sum = torch.sum(mu_s ** 2, dim=1, keepdim=True)
        var_sum = torch.sum(torch.exp(log_var_s), dim=1, keepdim=True)

        eta_s = (self.nu + self.ds) / (
            self.nu + mu_sq_sum + var_sum
        )

        return s, mu_s, log_var_s, eta_s

    def generate_x(self, samples, s, sample=True):
        feature = self.res_shape(samples, s)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)

        if sample:
            x, _ = self.sample_normal_jit(mu_x, log_var_x)
        else:
            x = mu_x

        return x, mu_x, log_var_x

    def generate_z(self, x, sample=True):
        padded_x, original_h, original_w = self.pad_to_multiple(
            x,
            multiple=32,
        )

        padded_x = padded_x.repeat(1, 3, 1, 1)

        feature = self.segmenter(padded_x)
        feature = feature[..., :original_h, :original_w]

        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)

        if sample:
            z, _ = self.sample_normal_jit(mu_z, log_var_z)
            z = self.softmax(z)
        else:
            z = self.softmax(mu_z)

        mu_z = self.softmax(mu_z)

        return z, mu_z, log_var_z

    def forward(self, samples, return_auxiliary=None):
        if return_auxiliary is None:
            return_auxiliary = self.training

        sample_latents = self.training

        s, mu_s, log_var_s, eta_s = self.generate_s(
            samples,
            sample=sample_latents,
        )

        x, mu_x, log_var_x = self.generate_x(
            samples,
            s,
            sample=sample_latents,
        )

        z_input = x if self.training else mu_x

        z, mu_z, log_var_z = self.generate_z(
            z_input,
            sample=sample_latents,
        )

        pred = z if self.training else mu_z

        out = {
            "pred_masks": pred,
        }

        if not return_auxiliary:
            return out

        K = self.num_classes
        _, _, W, H = samples.shape

        appearance = samples - mu_x

        mu_rho_hat = (2.0 * self.args.gamma_rho + 1.0) / (
            appearance * appearance + 2.0 * self.args.phi_rho
        )
        mu_rho_hat = torch.clamp(mu_rho_hat, 1e4, 1e8)

        normalization = torch.sum(mu_rho_hat).detach()

        difference_y = F.conv2d(samples, self.Dx, padding=1)
        grad_y_sq = difference_y * difference_y

        eta_s_spatial = eta_s.view(samples.shape[0], 1, 1, 1)

        exponent = self.gamma_grad * eta_s_spatial.detach() * grad_y_sq
        exponent = torch.clamp(exponent, max=9.0)

        modulation_term = torch.exp(exponent)
        dynamic_phi_upsilon = self.args.phi_upsilon * modulation_term

        alpha_upsilon_hat = 2.0 * self.args.gamma_upsilon + K

        difference_x = F.conv2d(mu_x, self.Dx, padding=1)

        beta_upsilon_hat = (
            torch.sum(
                mu_z
                * (
                    difference_x * difference_x
                    + 2.0 * torch.exp(log_var_x)
                ),
                dim=1,
                keepdim=True,
            )
            + 2.0 * dynamic_phi_upsilon
        )

        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
        mu_upsilon_hat = torch.clamp(mu_upsilon_hat, 1e6, 1e10)

        difference_z = F.conv2d(
            mu_z,
            self.Dx.expand(K, 1, 3, 3),
            padding=1,
            groups=K,
        )

        alpha_omega_hat = 2.0 * self.args.gamma_omega + 1.0

        pseudo_pi = torch.mean(
            mu_z,
            dim=(2, 3),
            keepdim=True,
        )

        beta_omega_hat = (
            pseudo_pi
            * (
                difference_z * difference_z
                + 2.0 * torch.exp(log_var_z)
            )
            + 2.0 * self.args.phi_omega
        )

        mu_omega_hat = alpha_omega_hat / beta_omega_hat
        mu_omega_hat = torch.clamp(mu_omega_hat, 1e2, 1e6)

        alpha_pi_hat = self.args.alpha_pi + W * H / 2.0

        beta_pi_hat = (
            torch.sum(
                mu_omega_hat
                * (
                    difference_z * difference_z
                    + 2.0 * torch.exp(log_var_z)
                ),
                dim=(2, 3),
                keepdim=True,
            )
            / 2.0
            + self.args.beta_pi
        )

        digamma_pi = torch.special.digamma(
            alpha_pi_hat + beta_pi_hat
        ) - torch.special.digamma(beta_pi_hat)

        kl_y = appearance * mu_rho_hat.detach() * appearance

        kl_mu_z = torch.sum(
            digamma_pi.detach()
            * difference_z
            * mu_omega_hat.detach()
            * difference_z,
            dim=1,
        )

        kl_sigma_z = torch.sum(
            digamma_pi.detach()
            * (
                2.0 * torch.exp(log_var_z) * mu_omega_hat.detach()
                - log_var_z
            ),
            dim=1,
        )

        kl_mu_x = torch.sum(
            difference_x
            * difference_x
            * mu_upsilon_hat.detach()
            * mu_z.detach(),
            dim=1,
        )

        kl_sigma_x = (
            torch.sum(
                2.0
                * torch.exp(log_var_x)
                * mu_upsilon_hat.detach()
                * mu_z.detach(),
                dim=1,
            )
            - log_var_x.squeeze(1)
        )

        kl_mu_s = torch.sum(
            eta_s.detach() * mu_s * mu_s,
            dim=1,
        )

        kl_sigma_s = torch.sum(
            eta_s.detach() * torch.exp(log_var_s) - log_var_s,
            dim=1,
        )

        visualize = {
            "anatomy": mu_x,
            "appearance": appearance,
            "posterior": mu_z[:, 1:2, ...],
            "v_shape": mu_upsilon_hat,
            "v_seg": mu_omega_hat[:, 1:2, ...],
        }

        out.update(
            {
                "kl_y": kl_y,
                "kl_mu_z": kl_mu_z,
                "kl_sigma_z": kl_sigma_z,
                "kl_mu_x": kl_mu_x,
                "kl_sigma_x": kl_sigma_x,
                "kl_mu_S": kl_mu_s,
                "kl_sigma_S": kl_sigma_s,
                "normalization": normalization,
                "visualize": visualize,
            }
        )

        return out


class TBSeg_Criterion(Criterion):
    def __init__(self, args):
        super(TBSeg_Criterion, self).__init__(args)
        self.bayes_loss_coef = args.bayes_loss_coef

    def loss_Bayes(self, outputs):
        N = outputs["normalization"]

        loss_y = torch.sum(outputs["kl_y"]) / N
        loss_mu_x = torch.sum(outputs["kl_mu_x"]) / N
        loss_sigma_x = torch.sum(outputs["kl_sigma_x"]) / N
        loss_mu_z = torch.sum(outputs["kl_mu_z"]) / N
        loss_sigma_z = torch.sum(outputs["kl_sigma_z"]) / N

        W = outputs["pred_masks"].shape[2]
        H = outputs["pred_masks"].shape[3]
        spatial_size = W * H

        loss_mu_S = torch.mean(outputs["kl_mu_S"]) / spatial_size
        loss_sigma_S = torch.mean(outputs["kl_sigma_S"]) / spatial_size

        loss_Bayes = (
            loss_y
            + loss_mu_x
            + loss_sigma_x
            + loss_mu_z
            + loss_sigma_z
            + loss_mu_S
            + loss_sigma_S
        )

        return loss_Bayes

    def loss_CrossEntropy(self, src_masks, targets, eps=1e-12):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        targets_onehot = torch.zeros_like(src_masks)
        targets_onehot.scatter_(1, targets.long(), 1).float()

        cross_entropy = -torch.sum(
            targets_onehot * torch.log(src_masks + eps),
            dim=1,
        )

        return cross_entropy.mean()

    def forward(self, pred, grnd):
        loss_ce = self.loss_CrossEntropy(
            pred["pred_masks"],
            grnd,
        )

        loss_bayes = self.loss_Bayes(pred)

        losses = loss_ce + self.bayes_loss_coef * loss_bayes

        loss_dict = {
            "loss_CE": loss_ce,
            "loss_Bayes": loss_bayes,
            "Dice": self.compute_dice(pred["pred_masks"], grnd),
        }

        return losses, loss_dict


class TBSegVis(Visualization):
    def __init__(self):
        super(TBSegVis, self).__init__()

    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs, "inputs", epoch, writer)
        self.save_image(outputs.float(), "outputs", epoch, writer)
        self.save_image(labels.float(), "labels", epoch, writer)

        for key, value in others.items():
            self.save_image(value.float(), key, epoch, writer)


def build(args):
    model = TBSeg(args)
    criterion = TBSeg_Criterion(args)
    visualizer = TBSegVis()
    return model, criterion, visualizer