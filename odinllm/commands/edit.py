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

def add_ngram(ngrams, num2ngrams, a, b):
    ngram = (a, b)
    num = ngrams[ngram]+1
    ngrams[ngram] = num
    num2ngrams[num][ngram] = None
    if num-1:
        del num2ngrams[num-1][ngram]
    return num

def del_ngram(ngrams, num2ngrams, a, b):
    ngram = (a, b)
    num = ngrams[ngram]-1
    if num:
        ngrams[ngram] = num
        num2ngrams[num][ngram] = None
    else:
        del ngrams[ngram]
    del num2ngrams[num+1][ngram]
    return num

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
@click.option("--min-in-word-bigrams", default=None, type=int)
@click.option("--min-bigrams", default=None, type=int)
@click.option("--batch-size", default=None, type=int)
@click.option("--embeddings", default=None, type=str)
@click.option("--lm-head", default=None, type=str)
@click.option("--combine-embeddings", default=None, type=str)
@args_config
def edit(args, config):
    return __edit(args, config)

def __edit(args, config):
    if args.add_every is None and args.add_experts is None and args.adjust_layers is None and args.extend_vocab is None:
        raise ValueError("either --add-every, --adjust-layers, or --add-experts needs to be specified")
    if args.extend_vocab is not None and args.combine_embeddings not in ("average", "combine"):
            raise ValueError(f"unknown value for combine_embeddings: {args.combine_embeddings} (options: 'average', 'combine')")
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
        corpora = []
        for vocab_file in tqdm(args.vocab_files, desc="Loading corpora"):
            corpus = []
            with open(vocab_file, "rt") as f:
                for line in f:
                    corpus.append(format_text(json.loads(line), tokenizer=tokenizer))
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
        batch_size = 100000 if args.batch_size is None else args.batch_size
        for i, samples in enumerate(tqdm(corpora, desc="Tokenizing corpora")):
            pbar = tqdm(total=(len(samples)+batch_size-1)//batch_size, desc=f"Tokenizing corpus in batches of {batch_size}", disable=False)
            while samples:
                batch, samples, corpora[i] = samples[:batch_size], samples[batch_size:], corpora[i][batch_size:]
                for tokenized_sample in tokenizer(batch, batch)["input_ids"]:
                    tokenized.extend(tokenized_sample)
                pbar.update(1)
        pbar.close()
        #tprint(tokenized, decode)
        start(f"Extending vocabulary from {model.config.vocab_size} to {model.config.vocab_size+args.extend_vocab}")
        tokenizer_json = json.load(open(os.path.join(args.pretrained_model, "tokenizer.json"), "rt"))
        special_tokens = {t["id"]: t["content"] for t in tokenizer_json["added_tokens"]}
        ngrams = defaultdict(int)
        token2indices = defaultdict(dict)
        num2ngrams = defaultdict(dict)
        max_ngram = 0
        max_in_word_ngram = 0
        len_tokenized = len(tokenized)
        for i in tqdm(range(len_tokenized), desc="Creating ngram statistics"):
            if tokenized[i] in special_tokens:
                continue
            if i+1 < len_tokenized and not tokenized[i+1] in special_tokens:
                num = add_ngram(ngrams, num2ngrams, tokenized[i], tokenized[i+1])
                if num > max_ngram:
                    max_ngram = num
                if num > max_in_word_ngram and decode[tokenized[i+1]][0].isalpha():
                    max_in_word_ngram = num
            token2indices[tokenized[i]][i] = None
        min_in_word_bigrams = 0 if args.min_in_word_bigrams is None else args.min_in_word_bigrams if args.min_in_word_bigrams >= 0 else max_in_word_ngram+1
        min_bigrams = 0 if args.min_bigrams is None else args.min_bigrams if args.min_bigrams >= 0 else max_ngram+1
        extra_merges = []
        embeddings = model.get_submodule(args.embeddings)
        lm_head = model.get_submodule(args.lm_head)
        if args.combine_embeddings == "average":
            average_embedding = embeddings.weight.data.mean(dim=0)
            average_lm_head = lm_head.weight.data.mean(dim=0)
        pbar = tqdm(total=args.extend_vocab, desc="Extending vocabulary", disable=False)
        while len(vocab) < max_vocab_len:
            best_pair = None
            for num in range(max_in_word_ngram, min_in_word_bigrams-1, -1):
                for ngram in num2ngrams[num]:
                    if decode[ngram[1]][0].isalpha():
                        best_pair = ngram
                        max_in_word_ngram = num
                        break
                else:
                    continue
                break
            if best_pair is None:
                for num in range(max_ngram, min_bigrams-1, -1):
                    for ngram in num2ngrams[num]:
                        best_pair = ngram
                        max_ngram = num
                        break
                    else:
                        continue
                    break
            if best_pair is None:
                break
            num = ngrams[best_pair]
            a, b = decode[best_pair[0]], decode[best_pair[1]]
            if args.combine_embeddings == "average":
                new_embedding = average_embedding
                new_lm_head = average_lm_head
            elif args.combine_embeddings == "combine":
                new_embedding = (
                    embeddings.weight.data[best_pair[0]]*len(a)+
                    embeddings.weight.data[best_pair[1]]*len(b)
                    ) / (len(a)+len(b))
                new_lm_head = (
                    lm_head.weight.data[best_pair[0]]*len(a)+
                    lm_head.weight.data[best_pair[1]]*len(b)
                    ) / (len(a)+len(b))
            else:
                assert(False)
            extra_merges.append(f"{a} {b}")
            embeddings.weight.data = torch.nn.parameter.Parameter(torch.cat((embeddings.weight.data, new_embedding.unsqueeze(0)), 0))
            lm_head.weight.data = torch.nn.parameter.Parameter(torch.cat((lm_head.weight.data, new_lm_head.unsqueeze(0)), 0))
            next_token = f"{a}{b}"
            vocab[next_token] = next_id
            decode[next_id] = next_token
            len_tokenized = len(tokenized)
            a, b = best_pair
            for i in list(token2indices[a].keys()):
                _a = tokenized[i]
                if _a == -1:
                    continue
                assert _a == a
                _b = -1
                j = i+1
                while j < len_tokenized:
                    _b = tokenized[j]
                    if _b >= 2:
                        break
                    j += 1
                if _b != b:
                    continue
                tokenized[i] = next_id
                tokenized[j] = -1
                del_ngram(ngrams, num2ngrams, a, b)
                del token2indices[a][i]
                del token2indices[b][j]
                token2indices[next_id][i] = None
                _i = i-1
                while _i >= 0:
                    if tokenized[_i] != -1:
                        if tokenized[_i] in special_tokens:
                            break
                        add_ngram(ngrams, num2ngrams, tokenized[_i], next_id)
                        del_ngram(ngrams, num2ngrams, tokenized[_i], a)
                        break
                    _i -= 1
                _j = j+1
                while _j < len_tokenized:
                    if tokenized[_j] != -1:
                        if tokenized[_j] in special_tokens:
                            break
                        add_ngram(ngrams, num2ngrams, next_id, tokenized[_j])
                        del_ngram(ngrams, num2ngrams, b, tokenized[_j])
                        break
                    _j += 1
            assert(not ngrams[best_pair])
            assert(best_pair not in num2ngrams[num])
            next_id += 1
            if False and not next_id % 1000:
                tokenized = [t for t in tqdm(tokenized, desc="Compacting tokenized corpus") if t != -1]
                token2indices = defaultdict(dict)
                for i, t in enumerate(tqdm(tokenized, desc="Rebuilding token indices")):
                    if not t in special_tokens:
                        token2indices[t][i] = None
            pbar.update(1)
        pbar.close()
        end()
        start("Saving extended tokenizer")
        tokenizer_json["model"]["vocab"] = vocab
        tokenizer_json["model"]["merges"].extend(extra_merges)
        os.makedirs(args.edited_model, exist_ok=True)
        json.dump(tokenizer_json, open(os.path.join(args.edited_model, "tokenizer.json"), "wt"), indent=2, ensure_ascii=False)
        json.dump(json.load(open(os.path.join(args.pretrained_model, "special_tokens_map.json"), "rt")), open(os.path.join(args.edited_model, "special_tokens_map.json"), "wt"), indent=2, ensure_ascii=False)
        json.dump(json.load(open(os.path.join(args.pretrained_model, "tokenizer_config.json"), "rt")), open(os.path.join(args.edited_model, "tokenizer_config.json"), "wt"), indent=2, ensure_ascii=False)
        status(f"#extended: {len(vocab)-orig_vocab_len}", end='')
        end()
        tokenizer = load_tokenizer(args.edited_model, config)
        model.config.vocab_size = len(vocab)
    save_model(model, args.edited_model)
    save_tokenizer(tokenizer, args.edited_model)
    save_metadata(model.metadata, config, args.edited_model)
    
