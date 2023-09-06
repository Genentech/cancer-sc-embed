import torch
import torch.nn as nn
from scvi.distributions import NegativeBinomial
from scvi.nn import one_hot

from ._utils import ResnetFC


class ExpActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class DecoderZX(nn.Module):
    """Parameterizes the counts likelihood for the data given the latent variables."""

    def __init__(
        self,
        n_in,
        n_out,
        n_nuisance,
        linear_decoder,
        n_hidden=128,
        activation="softmax",
    ):
        super().__init__()
        if activation == "softmax":
            activation_ = nn.Softmax(-1)
        elif activation == "softplus":
            activation_ = nn.Softplus()
        elif activation == "exp":
            activation_ = ExpActivation()
        elif activation == "sigmoid":
            activation_ = nn.Sigmoid()
        else:
            raise ValueError("activation must be one of 'softmax' or 'softplus'")
        self.linear_decoder = linear_decoder
        self.n_nuisance = n_nuisance
        self.n_latent = n_in - n_nuisance
        if linear_decoder:
            self.amat = nn.Linear(self.n_latent, n_out, bias=False)
            self.amat_site = nn.Parameter(
                torch.randn(self.n_nuisance, self.n_latent, n_out)
            )
            self.offsets = nn.Parameter(torch.randn(self.n_nuisance, n_out))
            self.dropout_ = nn.Dropout(0.1)
            self.activation_ = activation_

        else:
            self.px_mean = ResnetFC(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                activation=activation_,
            )
        self.px_r = nn.Parameter(torch.randn(n_out))

    def forward(self, z, size_factor):
        if self.linear_decoder:
            nuisance_oh = z[..., -self.n_nuisance :]
            z0 = z[..., : -self.n_nuisance]
            x1 = self.amat(z0)

            nuisance_ids = torch.argmax(nuisance_oh, -1)
            As = self.amat_site[nuisance_ids]
            z0_detach = self.dropout_(z0.detach())[..., None]
            x2 = (As * z0_detach).sum(-2)
            offsets = self.offsets[nuisance_ids]
            mu = x1 + x2 + offsets
            mu = self.activation_(mu)
        else:
            mu = self.px_mean(z)
        mu = mu * size_factor
        return NegativeBinomial(mu=mu, theta=self.px_r.exp())


class LinearDecoderUZ(nn.Module):
    def __init__(
        self,
        n_latent,
        #n_sample,
        n_vars_cnv,
        n_out,
        scaler=False,
        scaler_n_hidden=32,
    ):
        super().__init__()
        self.n_latent = n_latent
        #self.n_sample = n_sample
        self.n_out = n_out
        self.n_vars_cnv = n_vars_cnv

        #original mrvi code
        #self.amat_sample = nn.Parameter(torch.randn(n_sample, self.n_latent, n_out)) # A matrix. set of weights for each sample that map from latent dim to "n_out," which in practice is set to latent dim
        #self.offsets = nn.Parameter(torch.randn(n_sample, n_out))

        #CNV code (causes everything to become a blob, even in the case of cnv data = 1)
        self.cnv_to_amat = nn.Sequential(
            nn.Linear(n_vars_cnv, n_latent*n_out), # a map from cnv embedding to the weights of the A matrix in eqn 1
            #nn.Softplus() #previously tried ReLu
        )
        """the initialization in this block did not improve performance
        # initialize the weights so that the A matrix values will be approx normal for a CNV input of all ones.
        nn.init.normal_(self.cnv_to_amat[0].weight.data)
        with torch.no_grad():
            self.cnv_to_amat[0].weight.data = self.cnv_to_amat[0].weight.data / n_vars_cnv 
        """

        """ the initialization in this block did not improve performance
        # initialize the weights of the cnv_to_amat layer such that Amat will be close to identity matrix
        with torch.no_grad():
            self.cnv_to_amat.weight.data = (torch.eye(n_latent, n_out) * 1/n_latent).reshape(-1, 1).expand(-1, n_vars_cnv) #should be shape out_features x in_features
        """
        self.cnv_to_offsets = nn.Linear(n_vars_cnv, n_out) # a map from cnv embedding to the offsets in eqn 1

        #code that I used when debugging the case when no cnv data is passed in (will not handle cnv data properly)
        #fixed the issue of everything becoming one blob when no cnv data passed in...
        #self.cnv_to_amat = nn.Parameter(torch.randn(n_latent, n_out)) # a map from cnv embedding to the weights of the A matrix in eqn 1
        #self.cnv_to_offsets = nn.Parameter(torch.randn(n_out)) # a map from cnv embedding to the offsets in eqn 1

        self.scaler = None
        if scaler:
            self.scaler = nn.Sequential(
                nn.Linear(n_latent + n_vars_cnv, scaler_n_hidden),
                nn.LayerNorm(scaler_n_hidden),
                nn.ReLU(),
                nn.Linear(scaler_n_hidden, 1),
                nn.Sigmoid(),
            )

    def forward(self, u, cnv, mc_samples):
        #original code that caused everything to become a blob in the case of no cnv data 
        if mc_samples == 1:
            As = self.cnv_to_amat(cnv).reshape((-1, self.n_latent, self.n_out))
            #As = torch.tril(As)
        else:
            As = self.cnv_to_amat(cnv).reshape((mc_samples, -1, self.n_latent, self.n_out))
            #As = torch.tril(As)
        offsets = self.cnv_to_offsets(cnv)
        #these are updating: print(self.cnv_to_amat.weight.data[:4, :4])

        #code that I used when debugging the case when no cnv data is passed in (will not handle cnv data properly)
        #fixed the issue of everything becoming one blob when no cnv data passed in...
        """if mc_samples == 1:
            As = self.cnv_to_amat[None, :, :].expand(u.shape[0], -1, -1)
            offsets = self.cnv_to_offsets[None, :].expand(u.shape[0], -1)
        else:
            As = self.cnv_to_amat[None, None, :, :].expand(mc_samples, u.shape[1], -1, -1)
            offsets = self.cnv_to_offsets[None, None, :].expand(mc_samples, u.shape[1], -1)"""


        u_detach = u.detach()[..., None]
        z2 = (As * u_detach).sum(-2)
        delta = z2 + offsets
        if self.scaler is not None:
            print("expected linear_decoder_uz_scaler to be False; you have entered a code block that has not been updated!")
            #sample_oh = one_hot(sample_id, self.n_sample)
            #if u.ndim != sample_oh.ndim:
            #    sample_oh = sample_oh[None].expand(u.shape[0], *sample_oh.shape)
            inputs = torch.cat([u.detach(), cnv], -1)
            delta = delta * self.scaler(inputs)
        return u + delta


class DecoderUZ(nn.Module):
    def __init__(
        self,
        n_latent,
        n_vars_cnv,
        n_out,
        dropout_rate=0.0,
        n_layers=1,
        n_hidden=128,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_vars_cnv = n_vars_cnv
        self.n_in = n_latent + n_vars_cnv
        self.n_out = n_out

        arch_mod = self.construct_arch(self.n_in, n_hidden, n_layers, dropout_rate) + [
            nn.Linear(n_hidden, self.n_out, bias=False)
        ]
        self.mod = nn.Sequential(*arch_mod)

        arch_scaler = self.construct_arch(
            self.n_latent, n_hidden, n_layers, dropout_rate
        ) + [nn.Linear(n_hidden, 1)]
        self.scaler = nn.Sequential(*arch_scaler)
        self.scaler.append(nn.Sigmoid())

    @staticmethod
    def construct_arch(n_inputs, n_hidden, n_layers, dropout_rate):
        """Initializes MLP architecture"""

        block_inputs = [
            nn.Linear(n_inputs, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
        ]

        block_inner = n_layers * [
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        ]
        return block_inputs + block_inner

    def forward(self, u):
        u_ = u.clone()
        if u_.dim() == 3:
            print("This code block hasn't been updated for CNV data")
            n_samples, n_cells, n_features = u_.shape
            u0_ = u_[:, :, : self.n_latent].reshape(-1, self.n_latent)
            u_ = u_.reshape(-1, n_features)
            pred_ = self.mod(u_).reshape(n_samples, n_cells, -1)
            scaler_ = self.scaler(u0_).reshape(n_samples, n_cells, -1)
        else:
            pred_ = self.mod(u)
            scaler_ = self.scaler(u[:, : self.n_latent])
        mean = u[..., : self.n_latent] + scaler_ * pred_
        return mean
