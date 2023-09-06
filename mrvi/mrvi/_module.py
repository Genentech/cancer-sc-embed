import torch
import torch.distributions as db
import torch.nn as nn
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import one_hot
from torch.distributions import kl_divergence as kl

from ._components import DecoderUZ, DecoderZX, LinearDecoderUZ
from ._utils import ConditionalBatchNorm1d, NormalNN
from ._constants import MRVI_REGISTRY_KEYS

DEFAULT_PX_HIDDEN = 32
DEFAULT_PZ_LAYERS = 1
DEFAULT_PZ_HIDDEN = 32


class MrVAE(BaseModuleClass):
    def __init__(
        self,
        n_input,
        #n_sample,
        #n_obs_per_sample,
        n_cats_per_nuisance_keys,
        n_vars_cnv, # will=0 if no CNV data
        n_latent=10,
        #n_latent_sample=2,
        n_latent_cnv = 50,
        linear_decoder_zx=True,
        linear_decoder_uz=True,
        linear_decoder_uz_scaler=False,
        linear_decoder_uz_scaler_n_hidden=32,
        px_kwargs=None,
        pz_kwargs=None,
        
    ):
        super().__init__()
        px_kwargs = dict(n_hidden=DEFAULT_PX_HIDDEN)
        if px_kwargs is not None:
            px_kwargs.update(px_kwargs)
        pz_kwargs = dict(n_layers=DEFAULT_PZ_LAYERS, n_hidden=DEFAULT_PZ_HIDDEN)
        if pz_kwargs is not None:
            pz_kwargs.update(pz_kwargs)

        self.n_cats_per_nuisance_keys = n_cats_per_nuisance_keys
        #self.n_sample = n_sample
        #assert n_latent_sample != 0
        #self.sample_embeddings = nn.Embedding(n_sample, n_latent_sample)
        self.n_latent_cnv = n_latent_cnv
        self.cnv_fc = nn.Linear(n_vars_cnv, self.n_latent_cnv) #fully connected layer to embed CNV data

        n_nuisance = sum(self.n_cats_per_nuisance_keys)
        if n_nuisance == 0:
            n_nuisance = 1 # will create dummy nuisance if none are passed in
        # Generative model
        self.px = DecoderZX(
            n_latent + n_nuisance,
            n_input,
            n_nuisance=n_nuisance,
            linear_decoder=linear_decoder_zx,
            **px_kwargs,
        )
        self.qu = NormalNN(128 + n_latent_cnv, n_latent, n_categories=1)

        self.linear_decoder_uz = linear_decoder_uz
        if linear_decoder_uz:
            self.pz = LinearDecoderUZ(
                n_latent,
                n_latent_cnv,
                n_latent,
                scaler=linear_decoder_uz_scaler,
                scaler_n_hidden=linear_decoder_uz_scaler_n_hidden,
            )
        else:
            self.pz = DecoderUZ(
                n_latent,
                n_latent_cnv,
                n_latent,
                **pz_kwargs,
            )
        #self.n_obs_per_sample = nn.Parameter(n_obs_per_sample, requires_grad=False)

        self.x_featurizer = nn.Sequential(nn.Linear(n_input, 128), nn.ReLU())
        self.x_featurizer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.bnn = ConditionalBatchNorm1d(128, n_latent_cnv)
        self.bnn2 = ConditionalBatchNorm1d(128, n_latent_cnv)

    def _get_inference_input(self, tensors, **kwargs):
        x = tensors[REGISTRY_KEYS.X_KEY]
        #sample_key = MRVI_REGISTRY_KEYS.SAMPLE_KEY
        #sample_index = tensors[sample_key] if sample_key in tensors.keys() else None
        cnv_key = REGISTRY_KEYS.CNV_KEY
        cnv = tensors[cnv_key] if cnv_key in tensors.keys() else None
        
        categorical_nuisance_keys = tensors[
            MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS
        ] if MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS in tensors.keys() else None
        return dict(
            x=x,
            #sample_index=sample_index,
            cnv=cnv,
            categorical_nuisance_keys=categorical_nuisance_keys,
        )

    @auto_move_data
    def inference(
        self,
        x,
        #sample_index,
        cnv,
        categorical_nuisance_keys,
        mc_samples=1,
        #cf_sample=None,
        use_mean=False,
    ):
        x_ = torch.log1p(x)

        #sample_index_cf = sample_index if cf_sample is None else cf_sample
        #zsample = self.sample_embeddings(sample_index_cf.long().squeeze(-1))
        if cnv is not None:
            cnv_embed = self.cnv_fc(cnv)
        else:
            cnv_embed = torch.ones(x.shape[0], self.n_latent_cnv).to(x.device)
        cnv_embed_ = cnv_embed
        if mc_samples >= 2:
            cnv_embed_ = cnv_embed[None].expand(mc_samples, *cnv_embed.shape)

        if categorical_nuisance_keys is not None:
            nuisance_oh = []
            for dim in range(categorical_nuisance_keys.shape[-1]):
                nuisance_oh.append(
                    one_hot(
                        categorical_nuisance_keys[:, [dim]],
                        self.n_cats_per_nuisance_keys[dim],
                    )
                )
            nuisance_oh = torch.cat(nuisance_oh, dim=-1)
        else:
            nuisance_oh = torch.zeros(x.shape[0], 1).to(x.device)

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, cnv_embed)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, cnv_embed)
        if x_.ndim != cnv_embed_.ndim:
            x_feat_ = x_feat[None].expand(mc_samples, *x_feat.shape)
            nuisance_oh = nuisance_oh[None].expand(mc_samples, *nuisance_oh.shape)
        else:
            x_feat_ = x_feat

        inputs = torch.cat([x_feat_, cnv_embed_], -1)
        qu = self.qu(inputs)
        if use_mean:
            u = qu.loc
        else:
            u = qu.rsample()

        if self.linear_decoder_uz:
            z = self.pz(u, cnv_embed_, mc_samples)
        else:
            inputs = torch.cat([u, cnv_embed_], -1)
            z = self.pz(inputs)
        library = torch.log(x.sum(1)).unsqueeze(1)

        return dict(
            qu=qu,
            u=u,
            z=z,
            #zsample=zsample,
            library=library,
            nuisance_oh=nuisance_oh,
        )

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        res = dict(
            z=inference_outputs["z"],
            library=inference_outputs["library"],
            nuisance_oh=inference_outputs["nuisance_oh"],
        )

        return res

    @auto_move_data
    def generative(
        self,
        z,
        library,
        nuisance_oh,
    ):

        inputs = torch.concat([z, nuisance_oh], dim=-1)
        px = self.px(inputs, size_factor=library.exp())
        h = px.mu / library.exp()

        pu = db.Normal(0, 1)
        return dict(px=px, pu=pu, h=h)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):

        reconstruction_loss = (
            -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        )
        kl_u = kl(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)
        kl_local_for_warmup = kl_u

        weighted_kl_local = kl_weight * kl_local_for_warmup
        loss = torch.mean(reconstruction_loss + weighted_kl_local)

        kl_local = torch.tensor(0.0)
        kl_global = torch.tensor(0.0)
        return LossOutput(
            loss = loss,
            reconstruction_loss = reconstruction_loss,
            kl_local = kl_local,
            kl_global = kl_global
        )
