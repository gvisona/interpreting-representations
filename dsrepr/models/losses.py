import torch
import torch.nn as nn

class VAE_loss(nn.Module):
    def __init__(self, reconstruction_loss=None):
        super(VAE_loss, self).__init__()
        if reconstruction_loss is None:
            self.reconstruction_loss = nn.MSELoss()
        self.reconstruction_loss = reconstruction_loss

    def forward(self, predict, target, mu, log_var, kld_weight = 1):
        reconstruction_loss = self.reconstruction_loss(predict, target)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kld_weight*kld_loss

class BetaVAE_Loss(nn.Module):
    def __init__(self, beta=1):
        super(BetaVAE_Loss, self).__init__()
        # assert beta >= 1
        self.beta = beta
        self.MSE = nn.MSELoss()

    def forward(self, predict, target, mu, log_var):
        assert predict.shape[0] == target.shape[0]
        reconstruction_loss = self.MSE(predict, target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_weight = 1
        # Original is kld_weight = self.params['batch_size']/ self.num_train_imgs, but I don't really understand it
        # In the github repo there is also an alternative definition of the beta loss https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        return reconstruction_loss + self.beta * kld_weight * kld_loss

def reparameterize_ada(mu1, mu2, log_var1, log_var2, method="GVAE"):
    """[summary]

    Args:
        mu1 (Tensor): mean generated by the encoding of image 1
        mu2 (Tensor): mean generated by the encoding of image 2
        log_var1 (Tensor): logarithm of the variances generated by the encoding of image 1
        log_var2 (Tensor): logarithm of the variances generated by the encoding of image 2
        method (str, optional): Averaging function for the common FOVs. Defaults to "GVAE".
            For details check section "Relation to prior work" from https://arxiv.org/pdf/2002.02886.pdf

    Raises:
        ValueError: When an invalid averaging method is provided the function will not default to one.

    Returns:
        (Tensor, Tensor): The two sampled values from the reparametrization
    """
    common_fovs = get_common_FOVs(mu1, mu2, log_var1, log_var2)
    if method == "GVAE":
        avg_mu = 0.5 * (mu1[common_fovs] + mu2[common_fovs])
        avg_var = 0.5 * (log_var1[common_fovs].exp() + log_var2[common_fovs].exp())
        avg_logvar = avg_var.log()
    elif method == "ML-VAE":
        avg_var = 1/(1/log_var1[common_fovs] + 1/log_var2[common_fovs])
        avg_mu = (mu1[common_fovs]/log_var1[common_fovs] + mu2[common_fovs]/log_var2[common_fovs]) * avg_var
        avg_logvar = avg_var.log()
    else:
        raise ValueError("Select a valid averaging method")

    mu1[common_fovs] = avg_mu
    mu2[common_fovs] = avg_mu
    log_var1[common_fovs] = avg_logvar
    log_var2[common_fovs] = avg_logvar

    std1 = torch.exp(0.5*log_var1)
    eps1 = torch.randn_like(std1)
    z1 = eps1.mul(std1).add_(mu1)

    std2 = torch.exp(0.5*log_var2)
    eps2 = torch.randn_like(std2)
    z2 = eps2.mul(std2).add_(mu2)

    return z1, z2

def get_common_FOVs(mu1, mu2, log_var1, log_var2):
    """
    Find the common factors of variation using an empirical threshold

    Args:
        mu1 (Tensor): mean generated by the encoding of image 1
        mu2 (Tensor): mean generated by the encoding of image 2
        log_var1 (Tensor): logarithm of the variances generated by the encoding of image 1
        log_var2 (Tensor): logarithm of the variances generated by the encoding of image 2

    Returns:
        Tensor: boolean list of the components to average
    """
    kl_vec = 0.5 * (log_var2 - log_var1 + (log_var1.exp() + (mu1-mu2)**2)/ log_var2.exp() - 1)
    max_vals, max_idxs = kl_vec.max(-1)
    min_vals, min_idxs = kl_vec.min(-1)
    threshold = 0.5 * (max_vals + min_vals)
    common_FOVs = kl_vec < threshold.view(-1, 1)
    if common_FOVs.shape[0] == 1:
        common_FOVs = common_FOVs.squeeze(0)
    return common_FOVs

class AdaVAE_Loss(nn.Module):
    def __init__(self, beta=1):
        super(AdaVAE_Loss, self).__init__()
        #assert beta >= 1
        self.beta = beta
        self.MSE = nn.MSELoss()

    def forward(self, reconstr_x1, reconstr_x2, x1, x2, mu1, mu2, log_var1, log_var2):

        mse1 = self.MSE(reconstr_x1, x1)
        mse2 = self.MSE(reconstr_x2, x2)

        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim = 1), dim = 0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim = 1), dim = 0)

        return mse1 + mse2 + self.beta*(kld_loss1 + kld_loss2)

