import jax.numpy as jnp
from flax.training import checkpoints
from .text_models.roberta_text_model import RobertaConfig, RobertaModel, RobertaDecoder
from transformers import RobertaTokenizerFast

def load_caco(ckpt_path, use_decoder=True):
    from .caco import CACO, CACOConfig, LossConfig
    from .audio_models.mae import AudioEncoder, AudioTransformerConfig

    # load caco state dict
    caco_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
    caco_params = caco_state_dict['0']['params']

    # text model configs
    text_module = RobertaModel(RobertaConfig())
    decoder_module = RobertaDecoder(RobertaConfig(num_hidden_layers=4))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # audio model configs
    encoder_config = AudioTransformerConfig(
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=1024,
        patch_size=(16 * 16), # (width * height)
        max_time_ind=1000,
        num_freq_patches=8,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        dtype=jnp.float32,
    )
    audio_module = AudioEncoder(encoder_config)

    # CACO config
    caco_config = CACOConfig(
        dtype=jnp.float32,
        logit_scale_init_value=2.,        
        num_attention_pool_heads=8, 
        use_decoder=use_decoder,
        projection_size=768,
    )

    # loss module config
    loss_config = LossConfig(
        decoder_weight=1.0,
        decoder_label_smoothing=0.0,
    )

    # caco model config
    caco_model = CACO(
        caco_config=caco_config,
        loss_config=loss_config,
        audio_module=audio_module,
        text_module=text_module,
        decoder_module=decoder_module,
    )

    caco_model_dict = {'tokenizer':tokenizer, 
                       'caco_model':caco_model, 
                       'caco_params':caco_params}

    return caco_model_dict


def load_audiomae(ckpt_path):
    from .audio_models.mae import AudioEncoder, AudioTransformerConfig

    # load audiomae state dict
    audiomae_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
    audiomae_params = audiomae_state_dict['0']['params']['AudioEncoder_0']
    
    encoder_config = AudioTransformerConfig(
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=1024,
        patch_size=(16 * 16), # (width * height)
        max_time_ind=1000,
        num_freq_patches=8,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        dtype=jnp.float32,
    )
    
    audiomae_model = AudioEncoder(encoder_config)
    
    audiomae_model_dict = {'audiomae_model':audiomae_model, 
                           'audiomae_params':audiomae_params}
    
    return audiomae_model_dict

def load_caco_ast(ckpt_path, use_decoder=True):
    from .caco_ast import CACO, CACOConfig, LossConfig
    from .audio_models.ast_model import ASTConfig, ASTModel
    
    # load caco state dict
    caco_state_dict = checkpoints.restore_checkpoint(ckpt_path, target=None)
    caco_params = caco_state_dict['0']['params']

    # text model configs
    text_module = RobertaModel(RobertaConfig())
    decoder_module = RobertaDecoder(RobertaConfig(num_hidden_layers=4))
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # audio model configs
    ast_config = ASTConfig(
        dtype=jnp.float32,
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        # bias_mlp_size=128,
        mlp_size=768*4,
        num_freq_patches=8,
        dropout_rate=0.1,
        # time_delta_div_scale=1000/3,
        # pos_delta_div_scale=1000/3,
        # use_dynamic_positional_bias=False,
        use_fixed_freq_positional_embedding=True,
        use_fixed_time_positional_embedding=True,
        max_time_ind=384,
        use_dist_token=True
    )

    audio_module = ASTModel(ast_config)

    # CACO config
    caco_config = CACOConfig(
        dtype=jnp.float32,
        logit_scale_init_value=2.,
        use_decoder=use_decoder,
        projection_size=768
        )

    # loss module config
    loss_config = LossConfig(
        decoder_weight=1.0,
        decoder_label_smoothing=0.0,
    )

    # caco model config
    caco_model = CACO(
        caco_config=caco_config,
        loss_config=loss_config,
        audio_module=audio_module,
        text_module=text_module,
        decoder_module=decoder_module,
    )

    caco_model_dict = {'tokenizer':tokenizer, 
                       'caco_model':caco_model, 
                       'caco_params':caco_params}

    return caco_model_dict
