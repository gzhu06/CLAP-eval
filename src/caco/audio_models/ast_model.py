import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax import struct

from typing import Any, Optional, Sequence, Tuple, Union

import os

from einops import rearrange, reduce, repeat

PyTreeDef = type(jax.tree_util.tree_structure(None))

@struct.dataclass
class ASTConfig:

    hidden_size: int
    num_heads: int
    num_layers: int
    mlp_size: int

    num_freq_patches: int
    dropout_rate: float

    dtype: jnp.dtype

    # use_dynamic_positional_bias: bool
    # bias_mlp_size: int
    # time_delta_div_scale: float # divide value for time delta in dynamic bias
    # pos_delta_div_scale: float # divide value for position delta in dynamic bias

    use_fixed_time_positional_embedding: bool
    max_time_ind: Optional[int]
    use_fixed_freq_positional_embedding: bool

    use_dist_token: bool # some pretrained models may have distillation token


class SelfAttention(nn.Module):
    config: ASTConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time_inds: jnp.ndarray,
        freq_inds: jnp.ndarray,
        mask: jnp.ndarray,
        is_train: bool = False
    ) -> jnp.ndarray:

        seq_len = time_inds.shape[-1]
        batch_size = time_inds.shape[0]

        qkv = nn.Dense(
            3*self.config.hidden_size,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)

        q, k, v = rearrange(qkv, '... s (q m d) -> q ... s m d', q=3, m=self.config.num_heads)

        pad_width = 2 if self.config.use_dist_token else 1

        # def _get_delta(y: jnp.ndarray, div_scale: jnp.ndarray) -> jnp.ndarray:
        #     y = (y[..., None, :] - y[..., :, None])
        #     y = y.astype(jnp.float32) / div_scale
        #     y = jnp.log1p(jnp.abs(y)) * jnp.sign(y)
        #     # cls token at start of sequence
        #     y = jnp.pad(y, pad_width=[[0, 0], [pad_width, 0], [pad_width, 0]])
        #     return y

        # if self.config.use_dynamic_positional_bias: # TODO(jd) test...
        #     def _bias_mlp(time_delta: jnp.ndarray, freq_onehot: jnp.ndarray, pos_delta: jnp.ndarray) -> jnp.ndarray:
        #         bias_input = jnp.concatenate(
        #             [time_delta[..., None], 
        #                 repeat(freq_onehot, 'b s o -> b s a o', a=seq_len+1), 
        #                 repeat(freq_onehot, 'b s o -> b a s o', a=seq_len+1),  
        #                 repeat(pos_delta, '1 s a -> b s a 1', b=batch_size)], 
        #             axis=-1
        #         )
        #         h = nn.Dense(
        #             self.config.bias_mlp_size,
        #             dtype=self.config.dtype,
        #             kernel_init=nn.initializers.xavier_uniform(),
        #             name='bias_dense_1'
        #         )(bias_input)
        #         h = nn.relu(h)
        #         h = nn.Dense(
        #             self.config.num_heads,
        #             dtype=self.config.dtype,
        #             kernel_init=nn.initializers.xavier_uniform(), # TODO(jd) zeros??
        #             name='bias_dense_2'
        #         )(h)
        #         return h

        #     time_delta = _get_delta(time_inds, div_scale=self.config.time_delta_div_scale)
        #     freq_onehot = jax.nn.one_hot(freq_inds+1, num_classes=self.config.num_freq_patches+1)
        #     freq_onehot = jnp.concatenate(
        #     [jnp.array(batch_size*[[1]+[0]*self.config.num_freq_patches])[:, None], freq_onehot], axis=-2)

        #     pos_ids = jnp.arange(time_inds.shape[-1])[None]
        #     pos_delta = _get_delta(pos_ids, div_scale=self.config.pos_delta_div_scale)
        #     bias = _bias_mlp(time_delta, freq_onehot, pos_delta)
        #     bias = rearrange(bias, 'b s a h -> b h s a')
        # else:
        #     bias = None

        bias = None

        attention_mask = rearrange(jnp.pad(mask, [[0,0], [pad_width, 0]], constant_values=1), 'b s -> b 1 s 1')

        x = nn.attention.dot_product_attention(
            q, k, v, 
            bias=bias, 
            mask=attention_mask, 
            broadcast_dropout=False, 
            dropout_rate=self.config.dropout_rate, 
            dropout_rng=self.make_rng('dropout') if is_train else None,
            deterministic=not is_train
        )
        x = nn.DenseGeneral(
            self.config.hidden_size,
            axis=(-2, -1),
            dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)

        return x


class MLP(nn.Module):
    config: ASTConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        is_train: bool = True
    ) -> jnp.ndarray:

        x = nn.Dense(
            self.config.mlp_size,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=(not is_train))
        x = nn.Dense(
            self.config.hidden_size,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=(not is_train))

        return x


class EncoderLayer(nn.Module):
    config: ASTConfig
    scan: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time_inds: jnp.ndarray,
        freq_inds: jnp.ndarray,
        mask: jnp.ndarray,
        is_train: bool = False
    ) -> jnp.ndarray:

        h = nn.LayerNorm(dtype=self.config.dtype)(x)

        h = SelfAttention(self.config)(
            x=h,
            time_inds=time_inds,
            freq_inds=freq_inds,
            mask=mask,
            is_train=is_train
        )
        x = x + h

        h = nn.LayerNorm(dtype=self.config.dtype)(x)
        h = MLP(config=self.config)(h, is_train=is_train)

        x = x + h

        if self.scan: 
            return x, None

        return x




class ASTModel(nn.Module):
    config: ASTConfig
    scan: bool = True

    @nn.compact
    def __call__(
        self,
        audio_patches: jnp.ndarray,
        audio_time_inds: jnp.ndarray,
        audio_freq_inds: jnp.ndarray,
        audio_mask: jnp.ndarray,
        is_train: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        x = nn.Dense(
            self.config.hidden_size,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name='in_proj'
        )(audio_patches)

        if self.config.use_fixed_time_positional_embedding:
            assert self.config.max_time_ind is not None
            time_pos_emb = self.param('time_positional_embedding', 
                                      nn.initializers.normal(stddev=0.02), 
                                      (self.config.max_time_ind, x.shape[-1]))
            x = x + jnp.take_along_axis(time_pos_emb[None], audio_time_inds[..., None], axis=-2)[0]

        if self.config.use_fixed_freq_positional_embedding:
            freq_pos_emb = self.param('freq_positional_embedding', 
                                      nn.initializers.normal(stddev=0.02), 
                                      (self.config.num_freq_patches, x.shape[-1]))
            x = x + jnp.take_along_axis(freq_pos_emb[None], audio_freq_inds[..., None], axis=-2)[0]


        if self.config.use_dist_token:
            dist_emb_shape = (1, x.shape[-1])
            dist_embed = self.param('dist_embedding', nn.initializers.normal(stddev=0.02), dist_emb_shape)
            x = jnp.concatenate([repeat(dist_embed, '1 h -> b 1 h', b=x.shape[0]), x], axis=-2)

        cls_emb_shape = (1, x.shape[-1])
        cls_embed = self.param('cls_embedding', nn.initializers.normal(stddev=0.02), cls_emb_shape)
        x = jnp.concatenate([repeat(cls_embed, '1 h -> b 1 h', b=x.shape[0]), x], axis=-2)

        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=(not is_train))

        if self.scan:
            x, _ = nn.scan(
                EncoderLayer,
                variable_axes={'params': 0},
                split_rngs={'params': True, 'dropout': True},
                in_axes=nn.broadcast,
                length=self.config.num_layers,
            )(self.config, scan=True)(
                x,
                audio_time_inds,
                audio_freq_inds,
                audio_mask,
                is_train
            )
        else:
            for _ in range(self.config.num_layers):
                x = EncoderLayer(self.config, scan=False)(
                    x=x,
                    time_inds=audio_time_inds,
                    freq_inds=audio_freq_inds,
                    mask=audio_mask,
                    is_train=is_train
                )
        
        x = nn.LayerNorm(dtype=self.config.dtype, name='out_norm')(x)

        return (x[:, 0], x)



def ast_update_pretrained_parameters(
    params: PyTreeDef,
    pretrained_path: Union[str, os.PathLike],
    prefix: Optional[Tuple[str]] = None,
) -> PyTreeDef:

    from flax.core import freeze, unfreeze
    from flax.traverse_util import flatten_dict, unflatten_dict

    from utils import load_params

    pretrained_weights = load_params(pretrained_path)
    pretrained_weight_names = list(pretrained_weights.keys())

    num_layers = max([int(n.split('.')[1]) for n in pretrained_weight_names if 'blocks' in n])+1

    params = flatten_dict(unfreeze(params))


    if prefix is None:
        prefix = tuple()

    def _update_params(params: PyTreeDef, key: Tuple, weight: Any) -> PyTreeDef:
        key = prefix+key
        if key not in params.keys():
            print(f'Custom model is missing pretrained_key: {key}')
        else:
            if weight.shape != params[key].shape:
                print(weight.shape, params[key].shape)
                raise ValueError(f'Shape mismatch between params for key {key}')
            # print(f'Updated: {key}')
            params[key] = jnp.asarray(weight)

        return params


    if any([prefix + ('ScanEncoderLayer_0',) == k[:1+len(prefix)] for k in params.keys()]): # check if scan

        # self-attention
        params = _update_params(
            params,
            (f'ScanEncoderLayer_0', f'SelfAttention_0', f'Dense_0', 'kernel'),
            jnp.stack([pretrained_weights[f'blocks.{i}.attn.qkv.weight'].T for i in range(num_layers)], axis=0)
        )
        params = _update_params(
            params,
            (f'ScanEncoderLayer_0', f'SelfAttention_0', f'Dense_0', 'bias'),
            jnp.stack([pretrained_weights[f'blocks.{i}.attn.qkv.bias'] for i in range(num_layers)], axis=0)
        )
        dg_key = (f'ScanEncoderLayer_0', f'SelfAttention_0', f'DenseGeneral_0', 'kernel')
        params = _update_params(
            params,
            dg_key,
            jnp.stack([
                rearrange(pretrained_weights[f'blocks.{i}.attn.proj.weight'], 'o (h d) -> h d o', h=params[prefix+dg_key].shape[0])
                for i in range(num_layers)
            ])
        )
        params = _update_params(
            params,
            (f'ScanEncoderLayer_0', f'SelfAttention_0', f'DenseGeneral_0', 'bias'),
            jnp.stack([pretrained_weights[f'blocks.{i}.attn.proj.bias'] for i in range(num_layers)], axis=0)   
        )

        # mlp
        for j in (0, 1):
            params = _update_params(
                params,
                (f'ScanEncoderLayer_0', f'MLP_0', f'Dense_{j}', 'kernel'),
                jnp.stack([pretrained_weights[f'blocks.{i}.mlp.fc{j+1}.weight'].T for i in range(num_layers)], axis=0)
            )
            params = _update_params(
                params,
                (f'ScanEncoderLayer_0', f'MLP_0', f'Dense_{j}', 'bias'),
                jnp.stack([pretrained_weights[f'blocks.{i}.mlp.fc{j+1}.bias'] for i in range(num_layers)], axis=0)                
            )

        # layer norms
        for j in (0, 1):
            params = _update_params(
                params,
                (f'ScanEncoderLayer_0', f'LayerNorm_{j}', 'scale'),
                jnp.stack([pretrained_weights[f'blocks.{i}.norm{j+1}.weight'] for i in range(num_layers)], axis=0)
            )
            params = _update_params(
                params,
                (f'ScanEncoderLayer_0', f'LayerNorm_{j}', 'bias'),
                jnp.stack([pretrained_weights[f'blocks.{i}.norm{j+1}.bias'] for i in range(num_layers)], axis=0)
            )
        
    
    else: # NO SCAN
        # encoder blocks
        for i in range(num_layers):
            # self-attention
            params = _update_params(
                params,
                (f'EncoderLayer_{i}', f'SelfAttention_0', f'Dense_0', 'kernel'),
                pretrained_weights[f'blocks.{i}.attn.qkv.weight'].T
            )
            params = _update_params(
                params,
                (f'EncoderLayer_{i}', f'SelfAttention_0', f'Dense_0', 'bias'),
                pretrained_weights[f'blocks.{i}.attn.qkv.bias']
            )
            dg_key = (f'EncoderLayer_{i}', f'SelfAttention_0', f'DenseGeneral_0', 'kernel')
            params = _update_params(
                params,
                dg_key,
                rearrange(pretrained_weights[f'blocks.{i}.attn.proj.weight'], 'o (h d) -> h d o', h=params[prefix+dg_key].shape[0])
            )
            params = _update_params(
                params,
                (f'EncoderLayer_{i}', f'SelfAttention_0', f'DenseGeneral_0', 'bias'),
                pretrained_weights[f'blocks.{i}.attn.proj.bias']
            )

            # mlp
            for j in (0, 1):
                params = _update_params(
                    params,
                    (f'EncoderLayer_{i}', f'MLP_0', f'Dense_{j}', 'kernel'),
                    pretrained_weights[f'blocks.{i}.mlp.fc{j+1}.weight'].T
                )
                params = _update_params(
                    params,
                    (f'EncoderLayer_{i}', f'MLP_0', f'Dense_{j}', 'bias'),
                    pretrained_weights[f'blocks.{i}.mlp.fc{j+1}.bias']
                )

            # layer norms
            for j in (0, 1):
                params = _update_params(
                    params,
                    (f'EncoderLayer_{i}', f'LayerNorm_{j}', 'scale'),
                    pretrained_weights[f'blocks.{i}.norm{j+1}.weight']
                )
                params = _update_params(
                    params,
                    (f'EncoderLayer_{i}', f'LayerNorm_{j}', 'bias'),
                    pretrained_weights[f'blocks.{i}.norm{j+1}.bias']
                )

        
    # in proj
    params = _update_params(
        params,
        ('in_proj', 'kernel'),
        rearrange(pretrained_weights['patch_embed.proj.weight'], 'd 1 h w -> (w h) d')  # TODO(jd) CHECK FLIP!
    )
    params = _update_params(
        params,
        ('in_proj', 'bias'),
        pretrained_weights['patch_embed.proj.bias']
    )

    # time positional embedding
    time_pos_emb = pretrained_weights['time_new_pos_embed'][0, :, 0].T
    time_diff = params[prefix+('time_positional_embedding',)].shape[0] - time_pos_emb.shape[0]
    if time_diff > 0:
        time_pos_emb_remainder = repeat(time_pos_emb[-1], 'd -> r d', r=time_diff) + params[prefix+('time_positional_embedding',)][time_pos_emb.shape[0]:]
        time_pos_emb = jnp.concatenate([time_pos_emb, time_pos_emb_remainder], axis=0)
    elif time_diff < 0:
        time_pos_emb = time_pos_emb[:params[prefix+('time_positional_embedding',)].shape[0]]
    params = _update_params(
        params,
        ('time_positional_embedding',),
        time_pos_emb
    )

    # freq positional embedding #TODO(jd) why 12 freq dim
    params = _update_params(
        params,
        ('freq_positional_embedding',),
        pretrained_weights['freq_new_pos_embed'][0, :, :params[prefix+('freq_positional_embedding',)].shape[0], 0].T
    )

    # out norm
    params = _update_params(
        params,
        ('out_norm', 'scale'),
        pretrained_weights['norm.weight']
    )
    params = _update_params(
        params,
        ('out_norm', 'bias'),
        pretrained_weights['norm.bias']
    )

    # cls embed
    params = _update_params(
        params,
        ('cls_embedding', ),
        pretrained_weights['cls_token'][0] + pretrained_weights['new_pos_embed'][:, 0]
    )
    
    params = _update_params(
        params,
        ('dist_embedding', ),
        pretrained_weights['dist_token'][0] + pretrained_weights['new_pos_embed'][:, 1]
    )

    params = freeze(unflatten_dict(params))

    return params