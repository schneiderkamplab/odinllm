import click
import copy
import torch
from transformers import MixtralConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralDecoderLayer, MixtralRMSNorm, MixtralRotaryEmbedding, MixtralSparseMoeBlock

from ..shared import load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from ..utils import args_config, end, start, status, trainable_parameters

@click.group()
def _expand():
    pass
@_expand.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("expanded-model", type=click.Path(exists=False))
@click.option("--every", "-e", default=None, type=int)
@click.option("--layers", "-t", default=None, type=str)
@click.option("--zero", "-t", default=None, type=str, multiple=True)
@click.option("--add-experts", "-a", default=None, type=int)
@click.option("--gate", "-g", default=None, type=str)
@click.option("--experts", "-x", default=None, type=str)
@args_config
def expand(args, config):
    return __expand(args, config)

def __expand(args, config):
    model = load_model(
        args.pretrained_model,
        config=config,
    )
    layers = model.get_submodule(args.layers)
    if args.every is None and args.add_experts is None:
        raise ValueError("either --every or --add-experts needs to be specified")
    if args.every is not None:
        start(f"Expanding model from {len(layers)} layers")
        inserts = reversed(range(args.every-1, len(layers), args.every))
        for i in inserts:
            layers.insert(i, copy.deepcopy(layers[i]))
            model.config.num_hidden_layers += 1
            for zero in args.zero:
                try:
                    module = layers[i].get_submodule(zero)
                    module.weight.data = torch.zeros_like(module.weight.data)
                except AttributeError as e:
                    status(f"WARNING could not find {args.layers}.{i}.{zero}")
        status(f"#layers: {len(layers)}", end='')
        end()
    if args.add_experts is not None and args.add_experts > 0:
        start(f"Adding {args.add_experts} experts to each layer")
        for layer in layers:
            gate = layer.get_submodule(args.gate)
            experts = layer.get_submodule(args.experts)
            for j in range(args.add_experts):
                gate.weight.data = torch.cat((gate.weight.data, gate.weight.data[j:j+1,:]), 0)
                gate.out_features += 1
                model.config.num_local_experts += 1
                experts.append(copy.deepcopy(experts[j % len(experts)]))
        status(f"#experts: {len(experts)}", end='')
        status(f"%gate: {gate.weight.data.shape}", end='')
        end()
    if args.add_experts is not None and args.add_experts == 0:
        start(f"Wrapping model into Mixtral MoE")
        _, all_params = trainable_parameters(model)
        status(f"#params: {all_params}", end='')
        mixtral_config = MixtralConfig(
            vocab_size=model.config.vocab_size,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_hidden_layers=model.config.num_hidden_layers,
            num_attention_heads=model.config.num_attention_heads,
            num_key_value_heads=model.config.num_key_value_heads,
            hidden_act=model.config.hidden_act,
            max_position_embeddings=model.config.max_position_embeddings,
            initializer_range=model.config.initializer_range,
            rms_norm_eps=model.config.rms_norm_eps,
            use_cache=model.config.use_cache,
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            tie_word_embeddings=model.config.tie_word_embeddings,
            rope_theta=model.config.rope_theta,
            sliding_window=model.config.sliding_window,
            attention_dropout=model.config.attention_dropout,
            num_experts_per_tok=1, num_local_experts=1, output_router_logits=False, router_aux_loss_coef=0.001,
        )
        mixtral = MixtralForCausalLM(config=mixtral_config)
        status("created", end='')
        mixtral.to(model.dtype)
        mixtral.to(model.device)
        mixtral.metadata = model.metadata
        mixtral.model.embed_tokens = model.model.embed_tokens
        mixtral.model.norm = MixtralRMSNorm(hidden_size=mixtral_config.hidden_size, eps=mixtral_config.rms_norm_eps)
        mixtral.model.norm.weight = model.model.norm.weight
        mixtral.lm_head = model.lm_head
        for i in range(mixtral_config.num_hidden_layers):
            status(f"layer {i}", end='')
            old_layer = model.model.layers[i]
            old_self_attn = old_layer.self_attn
            old_rotary_emb = old_self_attn.rotary_emb
            self_attn = MixtralAttention(config=mixtral_config, layer_idx=i)
            self_attn.q_proj = old_self_attn.q_proj
            self_attn.k_proj = old_self_attn.k_proj
            self_attn.v_proj = old_self_attn.v_proj
            self_attn.o_proj = old_self_attn.o_proj
            self_attn.rotary_emb = MixtralRotaryEmbedding(dim=old_rotary_emb.dim, max_position_embeddings=old_rotary_emb.max_position_embeddings, base=old_rotary_emb.base, device=model.device)
            moe = MixtralSparseMoeBlock(config=mixtral_config)
            moe.gate.weight.data = torch.ones(1, mixtral_config.hidden_size)
            expert = moe.experts[0]
            expert.w1 = old_layer.mlp.gate_proj
            expert.w2 = old_layer.mlp.down_proj
            expert.w3 = old_layer.mlp.up_proj
            expert.act_fn = old_layer.mlp.act_fn
            layer = MixtralDecoderLayer(config=mixtral_config, layer_idx=i)
            layer.self_attn = self_attn
            layer.block_sparse_moe = moe
            layer.input_layernorm = MixtralRMSNorm(hidden_size=mixtral_config.hidden_size, eps=mixtral_config.rms_norm_eps)
            layer.input_layernorm.weight = old_layer.input_layernorm.weight
            layer.post_attention_layernorm = MixtralRMSNorm(hidden_size=mixtral_config.hidden_size, eps=mixtral_config.rms_norm_eps)
            layer.post_attention_layernorm.weight = old_layer.post_attention_layernorm.weight
            mixtral.model.layers[i] = layer
        model = mixtral
        _, all_params = trainable_parameters(model)
        status(f"#params: {all_params}", end='')
        end()
    save_model(model, args.expanded_model)
    tokenizer = load_tokenizer(args.pretrained_model, config)
    save_tokenizer(tokenizer, args.expanded_model)
    save_metadata(model.metadata, config, args.expanded_model)
    
