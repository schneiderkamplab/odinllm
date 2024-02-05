import click
from collections import defaultdict
import copy
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MixtralConfig, MixtralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralDecoderLayer, MixtralRMSNorm, MixtralRotaryEmbedding, MixtralSparseMoeBlock

from ..shared import load_model, load_tokenizer, save_metadata, save_model, save_tokenizer
from ..utils import args_config, end, format_text, start, status, trainable_parameters

def tprint(tokens, decode):
    pass#print([(k, decode[k]) for k in tokens])

def nprint(ngrams, decode):
    print({f"{decode[a]} {decode[b]}": c for (a, b), c in ngrams.items()})

@click.group()
def _edit():
    pass
@_edit.command()
@click.argument("config", type=click.Path(exists=False), nargs=-1)
@click.argument("pretrained-model", type=click.Path(exists=True))
@click.argument("edited-model", type=click.Path(exists=False))
@click.option("--info", default=None, type=bool)
@click.option("--add-every", default=None, type=int)
@click.option("--layers", default=None, type=str)
@click.option("--zero", default=None, type=str, multiple=True)
@click.option("--add-experts", default=None, type=int)
@click.option("--gate", default=None, type=str)
@click.option("--experts", default=None, type=str)
@click.option("--adjust-layers", default=None, type=int)
@click.option("--extend-vocab", default=None, type=int)
@click.option("--vocab-files", default=None, type=click.Path(exists=True), multiple=True)

@args_config
def edit(args, config):
    return __edit(args, config)

def __edit(args, config):
    if args.add_every is None and args.add_experts is None and args.adjust_layers is None and args.extend_vocab is None:
        raise ValueError("either --add-every, --adjust-layers, or --add-experts needs to be specified")
    tokenizer = load_tokenizer(
        args.pretrained_model,
        config=config,
    )
    model = load_model(
        args.pretrained_model,
        config=config,
    )
    if args.info:
        start("Printing model information")
        status(model)
    layers = model.get_submodule(args.layers)
    if args.add_every is not None:
        start(f"Expanding model from {len(layers)} layers")
        inserts = reversed(range(args.add_every-1, len(layers), args.add_every))
        for i in inserts:
            layers.insert(i, copy.deepcopy(layers[i]))
            model.config.num_hidden_layers += 1
            for zero in args.zero:
                try:
                    module = layers[i].get_submodule(zero)
                    module.weight.data = torch.zeros_like(module.weight.data)
                    status(f"zeroed {args.layers}.{len(layers)-1}.{zero}")
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
                experts.append(copy.deepcopy(experts[j % len(experts)]))
        model.config.num_experts_per_tok = 2
        model.config.num_local_experts = len(experts)
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
    if args.adjust_layers is not None:
        start(f"Adjusting number of layers from {len(layers)} layers")
        adjust_layers = args.adjust_layers if args.adjust_layers >= 0 else -args.adjust_layers
        for _ in range(len(layers)-adjust_layers):
            del layers[-1 if args.adjust_layers >= 0 else 0]
            model.config.num_hidden_layers -= 1
        for _ in range(adjust_layers-len(layers)):
            layers.append(copy.deepcopy(layers[-1]))
            model.config.num_hidden_layers += 1
            for zero in args.zero:
                try:
                    module = layers[-1].get_submodule(zero)
                    module.weight.data = torch.zeros_like(module.weight.data)
                    status(f"zeroed {args.layers}.{len(layers)-1}.{zero}")
                except AttributeError as e:
                    status(f"WARNING could not find {args.layers}.{len(layers)-1}.{zero}")
        status(f"#layers: {len(layers)}", end='')
        end()
    if args.extend_vocab is not None:
        start("Loading and tokenizing corpora")
        corpora = []
        for vocab_file in args.vocab_files:
            corpus = []
            with open(vocab_file, "rt") as f:
                for line in f:
                    corpus.append({"text": format_text(json.loads(line), tokenizer=tokenizer)})
            corpora.append(corpus)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        vocab = tokenizer.get_vocab()
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))
        orig_vocab_len = len(vocab)
        orig_max_id = max(vocab.values())
        max_vocab_len = orig_vocab_len+args.extend_vocab
        decode = {v: k for k, v in vocab.items()}
        next_id = orig_max_id + 1
        tokenized = []
        for samples in tqdm(corpora, desc="Tokenizing corpora"):
            for sample in tqdm(samples, desc="Tokenizing samples"):
                tokenized.extend(tokenizer(sample["text"])["input_ids"])
        tprint(tokenized, decode)
        end()
        start(f"Extending vocabulary from {model.config.vocab_size} to {model.config.vocab_size+args.extend_vocab}")
        ngrams = defaultdict(int)
        token2indices = defaultdict(list)
        num2ngrams = defaultdict(set)
        max_ngram = 0
        max_in_word_ngram = 0
        for i in tqdm(range(len(tokenized)-1), desc="Creating ngram statistics"):
            ngram = (tokenized[i], tokenized[i+1])
            num = ngrams[ngram]+1
            ngrams[ngram] = num
            num2ngrams[num].add(ngram)
            if num > max_ngram:
                max_ngram = num
            if num > max_in_word_ngram and decode[ngram[1]][0].isalpha():
                max_in_word_ngram = num
            if num-1:
                num2ngrams[num-1].remove(ngram)
            token2indices[tokenized[i]].append(i)
        #nprint(ngrams, decode)
        extra_merges = []
        pbar = tqdm(total=args.extend_vocab, desc="Extending vocabulary")
        while len(vocab) < max_vocab_len:
            #status(f"{next_id}", end='')
            best_pair = None
            for num in range(max_in_word_ngram,1,-1):
                for ngram in num2ngrams[num]:
                    if decode[ngram[1]][0].isalpha():
                        best_pair = ngram
                        max_in_word_ngram = num
                        break
                else:
                    continue
                break
            #end(end='')
            if best_pair is None:
                for num in range(max_ngram,1,-1):
                    for ngram in num2ngrams[num]:
                        best_pair = ngram
                        max_ngram = num
                        break
                    else:
                        continue
                    break
            if best_pair is None:
                break
            #end(end='')
            #status(f"{num} out of {max_in_word_ngram} out of {max_ngram}", end='')
            #nprint({best_pair: ngrams[best_pair]}, decode)
            #end(end='')
            num = ngrams[best_pair]
            del ngrams[best_pair]
            num2ngrams[num].remove(best_pair)
            a, b = decode[best_pair[0]], decode[best_pair[1]]
            extra_merges.append(f"{a} {b}")
            next_token = f"{a}{b}"
            vocab[next_token] = next_id
            decode[next_id] = next_token
            #end(end='')
            len_tokenized = len(tokenized)
            for i in token2indices[a].copy():
                _a = tokenized[i]
                if _a == -1:
                    continue
                assert _a == a
                _b = -1
                j = i+1
                while j < len_tokenized:
                    _b = tokenized[j]
                    if _b != -1:
                        break
                    j += 1
                if _b != b:
                    continue
                tokenized[i] = next_id
                tokenized[j] = -1
                token2indices[a].remove(i)
                token2indices[b].remove(j)
                token2indices[next_id].append(i)
                _i = i-1
                while _i >= 0:
                    if tokenized[_i] != -1:
                        ngram = (tokenized[_i], next_id)
                        num = ngrams[ngram]+1
                        ngrams[ngram] = num
                        num2ngrams[num].add(ngram)
                        if num-1:
                            num2ngrams[num-1].remove(ngram)
                        break
                    _i -= 1
                _j = j+1
                while _j < len_tokenized:
                    if tokenized[_j] != -1:
                        ngram = (next_id, tokenized[_j])
                        num = ngrams[ngram]+1
                        ngrams[ngram] = num
                        num2ngrams[num].add(ngram)
                        if num-1:
                            num2ngrams[num-1].remove(ngram)
                        break
                    _j += 1
            #end(end='')
            if not next_id % 10000:
                tokenized = [t for t in tokenized if t != -1]
            next_id += 1
            #tprint(tokenized, decode)
            #end(end='')
            pbar.update(1)
        #print({k: vocab[k] for k in list(vocab.keys())[orig_vocab_len:]})
        #print(extra_merges)
        tokenizer_json = json.load(open(os.path.join(args.pretrained_model, "tokenizer.json"), "rt"))
        tokenizer_json["model"]["vocab"] = vocab
        tokenizer_json["model"]["merges"].extend(extra_merges)
        os.makedirs(args.edited_model, exist_ok=True)
        json.dump(tokenizer_json, open(os.path.join(args.edited_model, "tokenizer.json"), "wt"), indent=2, ensure_ascii=False)
        json.dump(json.load(open(os.path.join(args.pretrained_model, "special_tokens_map.json"), "rt")), open(os.path.join(args.edited_model, "special_tokens_map.json"), "wt"), indent=2, ensure_ascii=False)
        json.dump(json.load(open(os.path.join(args.pretrained_model, "tokenizer_config.json"), "rt")), open(os.path.join(args.edited_model, "tokenizer_config.json"), "wt"), indent=2, ensure_ascii=False)
        status(f"#extended: {len(vocab)-orig_vocab_len}", end='')
        end()
    else:
        save_model(model, args.edited_model)
        tokenizer = load_tokenizer(args.pretrained_model, config)
        save_tokenizer(tokenizer, args.edited_model)
    save_metadata(model.metadata, config, args.edited_model)
    
