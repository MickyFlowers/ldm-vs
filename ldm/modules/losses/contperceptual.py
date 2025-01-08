import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


def compute_kernel(x, y):
    x_size = x.shape[2]
    y_size = y.shape[2]
    dim = x.shape[1]

    tiled_x = x.view(x.shape[0], dim, x_size, 1).repeat(1, 1, 1, y_size)
    tiled_y = y.view(y.shape[0], dim, 1, y_size).repeat(1, 1, x_size, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=1) / dim * 1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return (
        torch.sum(x_kernel) / x_kernel.shape[0]
        + torch.sum(y_kernel) / y_kernel.shape[0]
        - 2 * torch.sum(xy_kernel) / xy_kernel.shape[0]
    )


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        # mmd_weight=0.1,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        # self.mmd_weight = mmd_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # mmd loss
        # s = torch.randn_like(posteriors.mode()).to(kl_loss.device)
        # loss_mmd = compute_mmd(s.flatten(2), posteriors.mode().flatten(2))
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                # + self.mmd_weight * loss_mmd
                + d_weight * disc_factor * g_loss
            )

            log = {
                "{}/weighted_nll_loss".format(split): weighted_nll_loss.clone()
                .detach()
                .mean(),
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                # "{}/mmd_loss".format(split): loss_mmd.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class LPIPSImagesWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar1 = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.logvar2 = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator1 = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator2 = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs1,
        inputs2,
        reconstructions1,
        reconstructions2,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss1 = torch.abs(inputs1.contiguous() - reconstructions1.contiguous())
        rec_loss2 = torch.abs(inputs2.contiguous() - reconstructions2.contiguous())
        if self.perceptual_weight > 0:
            p_loss1 = self.perceptual_loss(
                inputs1.contiguous(), reconstructions1.contiguous()
            )
            p_loss2 = self.perceptual_loss(
                inputs2.contiguous(), reconstructions2.contiguous()
            )
            rec_loss1 = rec_loss1 + self.perceptual_weight * p_loss1
            rec_loss2 = rec_loss2 + self.perceptual_weight * p_loss2

        nll_loss1 = rec_loss1 / torch.exp(self.logvar1) + self.logvar1
        nll_loss2 = rec_loss2 / torch.exp(self.logvar2) + self.logvar2
        weighted_nll_loss1 = nll_loss1
        weighted_nll_loss2 = nll_loss2
        if weights is not None:
            weighted_nll_loss1 = weights * nll_loss1
            weighted_nll_loss2 = weights * nll_loss2
        weighted_nll_loss1 = torch.sum(weighted_nll_loss1) / weighted_nll_loss1.shape[0]
        weighted_nll_loss2 = torch.sum(weighted_nll_loss2) / weighted_nll_loss2.shape[0]
        nll_loss1 = torch.sum(nll_loss1) / nll_loss1.shape[0]
        nll_loss2 = torch.sum(nll_loss2) / nll_loss2.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake1 = self.discriminator1(reconstructions1.contiguous())
                logits_fake2 = self.discriminator2(reconstructions2.contiguous())
            else:
                assert self.disc_conditional
                logits_fake1 = self.discriminator1(
                    torch.cat((reconstructions1.contiguous(), cond), dim=1)
                )
                logits_fake2 = self.discriminator2(
                    torch.cat((reconstructions2.contiguous(), cond), dim=1)
                )
            g_loss1 = -torch.mean(logits_fake1)
            g_loss2 = -torch.mean(logits_fake2)

            if self.disc_factor > 0.0:
                try:
                    d_weight1 = self.calculate_adaptive_weight(
                        nll_loss1, g_loss1, last_layer=last_layer
                    )
                    d_weight2 = self.calculate_adaptive_weight(
                        nll_loss2, g_loss2, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight1 = torch.tensor(0.0)
                    d_weight2 = torch.tensor(0.0)
            else:
                d_weight1 = torch.tensor(0.0)
                d_weight2 = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss1 = weighted_nll_loss1 + d_weight1 * disc_factor * g_loss1
            loss2 = weighted_nll_loss2 + d_weight2 * disc_factor * g_loss2

            loss = loss1 + loss2 + self.kl_weight * kl_loss
            weighted_nll_loss = weighted_nll_loss1 + weighted_nll_loss2
            logvar = self.logvar1 + self.logvar2
            nll_loss = nll_loss1 + nll_loss2
            rec_loss = rec_loss1 + rec_loss2
            g_loss = g_loss1 + g_loss2
            d_weight = (d_weight1 + d_weight2) / 2
            log = {
                "{}/weighted_nll_loss".format(split): weighted_nll_loss.clone()
                .detach()
                .mean(),
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real1 = self.discriminator1(inputs1.contiguous().detach())
                logits_real2 = self.discriminator2(inputs2.contiguous().detach())
                logits_fake1 = self.discriminator1(
                    reconstructions1.contiguous().detach()
                )
                logits_fake2 = self.discriminator2(
                    reconstructions2.contiguous().detach()
                )
            else:
                logits_real1 = self.discriminator1(
                    torch.cat((inputs1.contiguous().detach(), cond), dim=1)
                )
                logits_fake1 = self.discriminator1(
                    torch.cat((reconstructions1.contiguous().detach(), cond), dim=1)
                )
                logits_real2 = self.discriminator2(
                    torch.cat((inputs2.contiguous().detach(), cond), dim=1)
                )
                logits_fake2 = self.discriminator2(
                    torch.cat((reconstructions2.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss1 = disc_factor * self.disc_loss(logits_real1, logits_fake1)
            d_loss2 = disc_factor * self.disc_loss(logits_real2, logits_fake2)
            d_loss = d_loss1 + d_loss2
            logits_real = logits_real1 + logits_real2
            logits_fake = logits_fake1 + logits_fake2

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class NormLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        mse_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        # self.mmd_weight = mmd_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors1,
        posteriors2,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors1.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        mse_loss = torch.nn.functional.mse_loss(
            posteriors1.mode(), posteriors2.mode(), reduction="mean"
        )

        # mmd loss
        # s = torch.randn_like(posteriors.mode()).to(kl_loss.device)
        # loss_mmd = compute_mmd(s.flatten(2), posteriors.mode().flatten(2))
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + self.mse_weight * mse_loss
                + d_weight * disc_factor * g_loss
            )

            log = {
                "{}/weighted_nll_loss".format(split): weighted_nll_loss.clone()
                .detach()
                .mean(),
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
                "{}/mse_loss".format(split): mse_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
