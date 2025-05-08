import os, re, json

import torch, numpy as np
import pandas as pd
from baukit import TraceDict

# Include prompt creation helper functions
from .prompt_utils import *
from .intervention_utils import *
from .model_utils import *
from .eval_utils import *


# Attention Activations
def gather_attn_activations(prompt_data, layers, dummy_labels, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt 

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    layers: layer names to get activatons from
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    idx_map: map of token indices to respective averaged token indices
    idx_avg: dict containing token indices of multi-token words
    """   
    
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config['prepend_bos'])
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)

    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, idx_map, idx_avg

def get_mean_head_activations(dataset, model, model_config, tokenizer, 
        n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False, 
        prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt,
        if n_icl_examples = 0, source prompt instuction prompt)
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    n_test_examples = 1
    if prefixes is not None and separators is not None:
        if type(prefixes) == dict and type(separators) == dict:
            prefix = prefixes
            separator = separators
        elif type(prefixes) == list and type(separators) == list:
            rand_idx = np.random.choice(len(prefixes))
            prefix = prefixes[rand_idx]
            sep_rand_idx = np.random.choice(len(separators))
            separator = separators[sep_rand_idx]
        else:
            raise ValueError("prefixes and separators should be either list or dict")
        
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, 
            prefixes=prefix, separators=separator, model_config=model_config)
    else:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, 
            model_config=model_config)
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], 
        model_config['n_heads'], len(dummy_labels), model_config['resid_dim']//model_config['n_heads'])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos =  False if model_config['prepend_bos'] else True

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        
        if prefix is not None and separator is not None:
            if type(prefixes) == dict and type(separators) == dict:
                prefix = prefixes
                separator = separators
            elif type(prefixes) == list and type(separators) == list:
                prefix = prefixes[rand_idx]
                separator = separators[sep_rand_idx]
            else:
                raise ValueError("prefixes and separators should be either both list or dict")
            
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                shuffle_labels=shuffle_labels, prefixes=prefix, separators=separator)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                shuffle_labels=shuffle_labels)
        sentence = [create_prompt(prompt_data)]
        if n == 0: 
            print(f"Prompt for getting mean head act :\n{sentence[0]}")
            
        activations_td,idx_map,idx_avg = gather_attn_activations(prompt_data=prompt_data, 
            layers = model_config['attn_hook_names'], 
            dummy_labels=dummy_labels, 
            model=model, 
            tokenizer=tokenizer, 
            model_config=model_config)
        

        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) 
            for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        stack_filtered = stack_initial[:,:,list(idx_map.keys())]
        for (i,j) in idx_avg.values():
            stack_filtered[:,:,idx_map[i]] = stack_initial[:,:,i:j+1].mean(axis=2) # Average activations of multi-token words across all its tokens
        
        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations

# Layer Activations
def gather_layer_activations(prompt_data, layers, model, tokenizer, model_config):
    """
    Collects activations for an ICL prompt 

    Parameters:
    prompt_data: dict containing
    layers: layer names to get activatons from
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    """   
    
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config['prepend_bos'])
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)

    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td

def get_mean_layer_activations(dataset, model, model_config, tokenizer, n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each layer in the model, at the final predictive token.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each layer hidden state of the model taken across n_trials ICL prompts
    """
    n_test_examples = 1
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    # If the model already prepends a bos token by default, we don't want to add one
    prepend_bos =  False if model_config['prepend_bos'] else True

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
        activations_td = gather_layer_activations(prompt_data=prompt_data, 
                                                  layers = model_config['layer_hook_names'], 
                                                  model=model, 
                                                  tokenizer=tokenizer, 
                                                  model_config=model_config)
        
        stack_initial = torch.vstack([activations_td[layer].output[0] for layer in model_config['layer_hook_names']])
        stack_filtered = stack_initial[:,-1,:] #Last token 
        
        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations

# Attention Weights
def get_value_weighted_attention(sentence, model, model_config, tokenizer):
    """

    Parameters:
    sentence: sentence to extract attention patterns for
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer

    Returns:
    attentions: attention heatmaps
    value_weighted_attn: value-weighted attention heatmaps
    """    
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    output = model(**inputs, output_attentions=True) # batch_size x n_tokens x vocab_size, only want last token prediction
    attentions = torch.vstack(output.attentions) # (layers, heads, tokens, tokens)
    values = torch.vstack([output.past_key_values[i][1] for i in range(model_config['n_layers'])]) # (layers, heads, tokens, head_dim)
    value_weighted_attn = torch.einsum("abcd,abd->abcd", attentions, values.norm(dim=-1))
    return attentions, value_weighted_attn

def get_token_averaged_attention(dataset, model, model_config, tokenizer, n_shots=10, storage_max=100, filter_set=None):
    """

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_shots: number of ICL example pairs to use for each prompt
    storage_max: max number of sentences to average attention pattern over
    filter_set: list of ints to filter to desired dataset instances

    Returns:
    attn_storage: attention heatmaps
    vw_attn_storage: value-weighted attention heatmaps
    token_labels: sample token labels for an n-shot prompt
    """
    if filter_set is not None:
        storage_size = min(len(filter_set), storage_max)
        storage_inds = [int(x) for x in filter_set[:storage_size]]
    else:
        storage_size = min(len(dataset['valid']), storage_max)
        storage_inds = [int(x) for x in np.arange(storage_size)]

    dummy_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, model_config=model_config)
    attn_storage = torch.zeros(storage_size, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), len(dummy_labels))
    vw_attn_storage = torch.zeros(storage_size, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), len(dummy_labels))

    for ind,s in enumerate(storage_inds):
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        
        # If the model already prepends a bos token by default, we don't want to add one
        add_bos =  False if model_config['prepend_bos'] else True

        word_pairs_test = dataset['valid'][s]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token = add_bos)
        
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target'].values()

        token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, prepend_bos=model_config['prepend_bos'])
        #TODO: delete 
        print("prompt_string:\n", prompt_string)

        idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
        
        sentence = [prompt_string]     

        attentions, value_weighted_attn = get_value_weighted_attention(sentence, model,model_config,tokenizer)

        attentionsv2 = attentions.clone()
        vw_v2 = value_weighted_attn.clone()

        # sum attention of multi-token phrases into single token (for avging purposes)
        for (i,j) in idx_avg.values():
            attentionsv2[:,:,:,i] = attentions[:,:,:,i:j+1].sum(axis=-1)
            attentionsv2[:,:,i+1:, i+1:j+1] = 0 

            vw_v2[:,:,:,i] = value_weighted_attn[:,:,:,i:j+1].sum(axis=-1)
            vw_v2[:,:,i+1:, i+1:j+1] = 0 
        
        del attentions
        del value_weighted_attn

        token_avgd_attention = attentionsv2[:,:,list(idx_map.keys())][:,:,:,list(idx_map.keys())]
        token_avgd_vw_attention = vw_v2[:,:,list(idx_map.keys())][:,:,:,list(idx_map.keys())]

        attn_storage[ind] = token_avgd_attention
        vw_attn_storage[ind] = token_avgd_vw_attention

    return attn_storage, vw_attn_storage, token_labels

def prefix_matching_score(model, model_config, min_token_idx=1000, max_token_idx=10000, seq_len=100, batch_size=4):
    """
    Computes the prefix matching score - part of checking whether an attention head is a traditional "induction head"
    
    Parameters:
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    min_token_idx: vocabulary token index lower bound
    max_token_idx: vocabulary token index upper bound
    seq_len: length of sequence to be duplicated
    batch_size: number of sequences to test
    
    Returns:
    score_per_head: prefix-matching score for each head in the model of size (n_layers, n_heads)
    """
    rand_tokens = torch.randint(min_token_idx, max_token_idx, (batch_size, seq_len))
    rand_tokens_repeat =  torch.concat((rand_tokens, rand_tokens), dim=1).to(model.device)

    output = model(rand_tokens_repeat, output_attentions = True)
    attentions = torch.vstack(output.attentions) # (n_layers, n_heads, tokens, tokens)

    score = attentions.diagonal(1-seq_len, dim1=-2, dim2=-1)
    score_per_head = score.reshape(model_config['n_layers'],batch_size,model_config['n_heads'],-1).mean(axis=-1).mean(axis=1).cpu().T
    
    return score_per_head

def compute_function_vector(mean_activations, indirect_effect, model, model_config, n_top_heads = 10, token_class_idx=-1):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention.
        
        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim)  or (Layers, Heads, head_dim)
            containing the average activation of each head for a particular task
         
        indirect_effect: tensor of size (N, Layers, Heads, class(optional)) containing the indirect_effect of each head across N trials
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the summed function vector
        token_class_idx: int indicating which token class to use, -1 is default for last token computations

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)         
    """
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device

    li_dims = len(indirect_effect.shape)
    
    if li_dims == 3 and token_class_idx == -1:
        mean_indirect_effect = indirect_effect.mean(dim=0)
    else:
        assert(li_dims == 4)
        mean_indirect_effect = indirect_effect[:,:,:,token_class_idx].mean(dim=0) # Subset to token class of interest

    # Compute Top Influential Heads (L,H)
    h_shape = mean_indirect_effect.shape 
    topk_vals, topk_inds  = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
    top_heads = top_lh[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)
    T = -1 # Intervention & values taken from last token

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path'] or 'gemma' in model_config['name_or_path'] or 'olmo' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = torch.zeros(model_resid_dim)
        if mean_activations.dim() == 4:
            x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        else: #'n_layers n_heads head_dim'
            x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
    
    function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads

def compute_universal_function_vector_instructions(mean_activations, model, model_config, n_top_heads=10):
    """
        Computes a "function vector" vector that communicates the task observed in instructions used for downstream intervention
        using the set of heads with universally highest causal effect computed across a set of ICL tasks
        
        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the function vector

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)         
    """
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device

    # Universal Set of Heads
    if 'OLMo-2-1124-7B-Instruct' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    elif 'Llama-3.1-8B-Instruct' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(6, 7, 0.0057), (2, 31, 0.0045), (5, 29, 0.004), (9, 19, 0.0036), (4, 4, 0.0034), 
                     (4, 5, 0.0033), (7, 11, 0.0031), (1, 16, 0.0031), (8, 24, 0.003), (13, 31, 0.0027), 
                     (3, 11, 0.0027), (0, 10, 0.0027), (31, 2, 0.0026), (1, 30, 0.0025), (6, 4, 0.0024), 
                     (16, 30, 0.0024), (5, 16, 0.0023), (1, 10, 0.0023), (4, 9, 0.0021), (6, 6, 0.0021), 
                     (5, 26, 0.0021), (7, 17, 0.0021), (2, 5, 0.0021), (5, 13, 0.002), (9, 29, 0.002), 
                     (13, 16, 0.002), (9, 13, 0.0018), (9, 31, 0.0018), (5, 31, 0.0017), (7, 5, 0.0017), 
                     (8, 31, 0.0017), (13, 13, 0.0017), (5, 18, 0.0017), (31, 5, 0.0016), (5, 17, 0.0015), 
                     (15, 10, 0.0015), (9, 8, 0.0015), (2, 9, 0.0015), (2, 7, 0.0015), (2, 16, 0.0015),] 
                 
    elif 'Llama-2-7b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    elif 'Llama-2-13b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    elif 'Llama-2-70b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    elif 'gpt-neox' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    elif 'gpt-j' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        raise NotImplementedError("Universal set not implemented for instructions")
    
    else:
        raise ValueError("Model not supported")
    
    top_heads = top_heads[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)
    T = -1 # Intervention & values taken from last token

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path']:
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense
        elif 'olmo' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        else: 
            raise ValueError("Model not supported")

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads

def compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention
        using the set of heads with universally highest causal effect computed across a set of ICL tasks
        
        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the function vector

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)         
    """
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device

    # Universal Set of Heads
    if 'OLMo-2-1124-7B-Instruct' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(15, 0, 0.1623), (15, 11, 0.1563), (17, 18, 0.0722), (15, 18, 0.0573), (10, 12, 0.0424), (16, 23, 0.0382), (12, 24, 0.0356), (12, 31, 0.0243), (13, 28, 0.0242), (20, 1, 0.0208), 
                     (17, 31, 0.0197), (12, 25, 0.0164), (16, 2, 0.0139), (13, 31, 0.0135), (21, 24, 0.0134), (26, 22, 0.0132), (15, 19, 0.0128), (14, 0, 0.0124), (19, 18, 0.0114), (24, 5, 0.0106), 
                     (24, 22, 0.0104), (16, 6, 0.0102), (21, 23, 0.0101), (14, 2, 0.009), (15, 10, 0.0088), (17, 23, 0.0081), (10, 29, 0.0072), (12, 21, 0.007), (12, 13, 0.0068), (16, 24, 0.0065), 
                     (14, 14, 0.0064), (26, 23, 0.0061), (15, 8, 0.006), (24, 15, 0.0053), (11, 0, 0.0052), (9, 25, 0.0051), (10, 21, 0.005), (17, 17, 0.0049), (16, 5, 0.0049), (9, 2, 0.0048),
                     (19, 21, 0.0048), (19, 19, 0.0046), (31, 30, 0.0046), (16, 21, 0.0046), (26, 14, 0.0045), (15, 1, 0.0045), (16, 25, 0.0044), (13, 24, 0.0044), (13, 22, 0.0041), (25, 19, 0.0041), 
                     (14, 17, 0.0041), (17, 28, 0.0038), (31, 20, 0.0038), (12, 30, 0.0037), (10, 19, 0.0035), (12, 16, 0.0035), (31, 23, 0.0032), (15, 30, 0.0032), (27, 15, 0.0032), (25, 5, 0.0031), 
                     (20, 27, 0.0031), (16, 29, 0.0031), (11, 4, 0.003), (27, 31, 0.0029), (31, 0, 0.0029), (24, 6, 0.0028), (14, 24, 0.0028), (22, 29, 0.0028), (30, 9, 0.0028), (15, 14, 0.0028), 
                     (30, 12, 0.0028), (25, 26, 0.0027), (13, 6, 0.0026), (12, 22, 0.0025), (20, 16, 0.0025), (31, 29, 0.0025), (12, 7, 0.0024), (13, 1, 0.0024), (15, 7, 0.0024), (19, 16, 0.0024), 
                     (16, 20, 0.0024), (22, 24, 0.0023), (24, 21, 0.0023), (30, 8, 0.0022), (10, 15, 0.0022), (26, 15, 0.0022), (19, 20, 0.0021), (27, 0, 0.0021), (18, 21, 0.0021), (16, 3, 0.0021),
                     (16, 18, 0.0019), (21, 8, 0.0019), (19, 29, 0.0019), (17, 5, 0.0019), (11, 21, 0.0019), (14, 13, 0.0019), (15, 2, 0.0018), (23, 14, 0.0018), (14, 21, 0.0018), (22, 17, 0.0018)]
    
    elif 'Llama-3.1-8B-Instruct' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(13, 27, 0.1256), (16, 29, 0.0297), (15, 17, 0.0292), (10, 5, 0.0221), (15, 1, 0.0205), (15, 16, 0.02), (10, 7, 0.0139), (30, 16, 0.0105), (15, 28, 0.0103), (11, 30, 0.0095), (11, 29, 0.0091), 
                     (14, 11, 0.0079), (11, 28, 0.0074), (15, 2, 0.0071), (14, 6, 0.0069), (17, 5, 0.0069), (10, 12, 0.0069), (31, 31, 0.0065), (13, 23, 0.006), (31, 14, 0.0056), (16, 28, 0.0055), (13, 19, 0.0053), 
                     (14, 19, 0.0049), (14, 29, 0.0049), (17, 8, 0.0048), (13, 22, 0.0044), (31, 5, 0.0044), (21, 10, 0.0044), (13, 31, 0.0042), (13, 6, 0.0041), (11, 16, 0.0041), (21, 2, 0.004), (15, 24, 0.0038), 
                     (15, 21, 0.0038), (15, 19, 0.0038), (12, 23, 0.0037), (15, 5, 0.0037), (12, 8, 0.0036), (19, 23, 0.0033), (13, 8, 0.0032), (13, 18, 0.0032), (11, 0, 0.0031), (14, 27, 0.0031), (15, 7, 0.003), 
                     (12, 29, 0.003), (13, 1, 0.0029), (19, 0, 0.0029), (22, 19, 0.0028), (9, 25, 0.0027), (13, 2, 0.0027), (18, 22, 0.0026), (30, 27, 0.0025), (30, 13, 0.0025), (27, 28, 0.0025), (12, 3, 0.0024), 
                     (16, 17, 0.0024), (26, 15, 0.0024), (9, 27, 0.0024), (13, 24, 0.0024), (30, 19, 0.0023), (31, 24, 0.0022), (14, 12, 0.0021), (16, 19, 0.0021), (28, 15, 0.0021), (31, 11, 0.0019), (30, 11, 0.0019),
                     (31, 15, 0.0019), (14, 21, 0.0019), (12, 0, 0.0018), (18, 29, 0.0018), (10, 8, 0.0017), (25, 23, 0.0017), (14, 8, 0.0017), (11, 6, 0.0016), (13, 3, 0.0015), (19, 2, 0.0014), (13, 29, 0.0014), 
                     (19, 3, 0.0014), (10, 14, 0.0014), (20, 27, 0.0014), (6, 31, 0.0013), (15, 30, 0.0013), (12, 11, 0.0013), (30, 12, 0.0013), (8, 1, 0.0013), (11, 2, 0.0012), (27, 29, 0.0012), (12, 7, 0.0012), (21, 9, 0.0011), 
                     (12, 17, 0.0011), (15, 20, 0.0011), (16, 22, 0.0011), (12, 25, 0.0011), (8, 16, 0.001), (20, 8, 0.001), (19, 13, 0.001), (24, 2, 0.0009), (28, 10, 0.0009), (12, 27, 0.0009), (30, 26, 0.0009), (14, 26, 0.0008), 
                     (7, 8, 0.0008), (8, 3, 0.0008), (11, 26, 0.0008), (6, 2, 0.0008), (14, 10, 0.0008), (22, 30, 0.0008), (31, 12, 0.0008), (17, 12, 0.0008), (27, 1, 0.0008), (17, 18, 0.0008), (12, 15, 0.0008), (12, 18, 0.0007),
                     (12, 2, 0.0007), (9, 29, 0.0007), (26, 0, 0.0007), (12, 22, 0.0007), (9, 30, 0.0007), (16, 31, 0.0007), (11, 22, 0.0007), (21, 1, 0.0007), (7, 22, 0.0006), (10, 26, 0.0006), (17, 4, 0.0006), (12, 19, 0.0006), 
                     (14, 15, 0.0006), (30, 22, 0.0006), (30, 8, 0.0006), (22, 9, 0.0006), (15, 10, 0.0006), (16, 7, 0.0006), (6, 21, 0.0006), (31, 9, 0.0006)]
    elif 'Llama-2-7b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(14, 1, 0.0391), (11, 2, 0.0225), (9, 25, 0.02), (12, 15, 0.0196), (12, 28, 0.0191), (13, 7, 0.0171), (11, 18, 0.0152), (12, 18, 0.0113), (16, 10, 0.007), (14, 16, 0.007),
                     (14, 14, 0.0048), (16, 1, 0.0042), (18, 1, 0.0042), (19, 16, 0.0041), (13, 30, 0.0034), (18, 26, 0.0032), (14, 7, 0.0032), (16, 0, 0.0031), (16, 29, 0.003), (29, 30, 0.003),
                     (16, 6, 0.0029), (15, 11, 0.0027), (12, 11, 0.0026), (11, 22, 0.0023), (16, 19, 0.0021), (15, 23, 0.002), (16, 20, 0.0019), (15, 9, 0.0019), (17, 28, 0.0019), (14, 18, 0.0018),
                     (8, 26, 0.0018), (29, 26, 0.0018), (15, 8, 0.0018), (13, 13, 0.0017), (30, 9, 0.0017), (13, 23, 0.0017), (13, 10, 0.0016), (11, 30, 0.0016), (12, 26, 0.0015), (19, 27, 0.0015),
                     (14, 9, 0.0014), (14, 10, 0.0013), (31, 17, 0.0013), (31, 4, 0.0013), (15, 17, 0.0013), (10, 5, 0.0012), (14, 11, 0.0012), (19, 12, 0.0012), (16, 7, 0.0012), (15, 24, 0.0011),
                     (26, 28, 0.0011), (11, 15, 0.0011), (15, 25, 0.0011), (17, 12, 0.0011), (13, 2, 0.0011), (14, 5, 0.0011), (14, 3, 0.001), (26, 30, 0.001), (27, 29, 0.001), (25, 12, 0.0009),
                     (15, 13, 0.0009), (10, 14, 0.0009), (28, 13, 0.0009), (17, 19, 0.0008), (19, 2, 0.0008), (12, 23, 0.0008), (15, 26, 0.0008), (28, 21, 0.0008), (15, 10, 0.0008), (12, 0, 0.0007),
                     (6, 16, 0.0007), (7, 28, 0.0007), (27, 7, 0.0007), (11, 28, 0.0007), (29, 15, 0.0006), (13, 8, 0.0006), (13, 17, 0.0006), (8, 0, 0.0006), (22, 17, 0.0006), (22, 20, 0.0006), 
                     (12, 2, 0.0006), (26, 9, 0.0006), (31, 26, 0.0006), (22, 27, 0.0005), (16, 26, 0.0005), (13, 1, 0.0005), (26, 2, 0.0005), (30, 10, 0.0005), (11, 25, 0.0005), (29, 20, 0.0005),
                     (19, 15, 0.0005), (12, 10, 0.0005), (12, 3, 0.0005), (30, 5, 0.0004), (6, 9, 0.0004), (15, 16, 0.0004), (23, 28, 0.0004), (22, 5, 0.0004), (31, 19, 0.0004), (26, 14, 0.0004)]
    elif 'Llama-2-13b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(13, 13, 0.0402), (12, 17, 0.0332), (15, 38, 0.0269), (14, 34, 0.0209), (19, 2, 0.0116), (19, 36, 0.0106), (13, 4, 0.0106), (18, 11, 0.01), (10, 15, 0.0087), (13, 23, 0.0077),
                     (14, 7, 0.0074), (15, 36, 0.0046), (12, 8, 0.0046), (17, 7, 0.0044), (38, 29, 0.0043), (15, 32, 0.0037), (17, 18, 0.0034), (16, 9, 0.0033), (14, 23, 0.0032), (39, 13, 0.0029),
                     (39, 14, 0.0027), (18, 22, 0.0026), (21, 32, 0.0026), (15, 18, 0.0026), (13, 14, 0.0026), (11, 31, 0.0025), (14, 39, 0.0024), (19, 14, 0.0023), (36, 23, 0.0021), (21, 7, 0.0021),
                     (8, 23, 0.002), (18, 18, 0.002), (17, 28, 0.002), (17, 9, 0.0019), (13, 27, 0.0017), (13, 34, 0.0017), (13, 12, 0.0016), (21, 2, 0.0016), (16, 16, 0.0015), (15, 31, 0.0015),
                     (26, 35, 0.0015), (10, 18, 0.0014), (11, 27, 0.0014), (13, 25, 0.0014), (15, 26, 0.0013), (5, 32, 0.0013), (20, 12, 0.0013), (18, 15, 0.0013), (16, 23, 0.0013), (25, 5, 0.0013),
                     (34, 6, 0.0012), (15, 2, 0.0012), (15, 27, 0.0012), (18, 20, 0.0012), (16, 19, 0.0011), (37, 4, 0.001), (19, 7, 0.001), (19, 3, 0.0009), (38, 14, 0.0009), (20, 21, 0.0009),
                     (21, 30, 0.0009), (16, 11, 0.0009), (13, 24, 0.0009), (9, 31, 0.0008), (14, 13, 0.0008), (16, 29, 0.0008), (15, 17, 0.0008), (19, 6, 0.0008), (23, 36, 0.0008), (18, 17, 0.0007),
                     (15, 34, 0.0007), (14, 29, 0.0007), (15, 7, 0.0007), (13, 17, 0.0007), (20, 11, 0.0007), (35, 16, 0.0007), (39, 27, 0.0007), (29, 27, 0.0006), (30, 24, 0.0006), (19, 37, 0.0006),
                     (39, 21, 0.0006), (13, 36, 0.0006), (37, 30, 0.0006), (16, 36, 0.0006), (15, 3, 0.0006), (19, 13, 0.0006), (13, 10, 0.0006), (14, 19, 0.0005), (36, 3, 0.0005), (15, 25, 0.0005),
                     (16, 0, 0.0005), (16, 10, 0.0005), (20, 29, 0.0005), (25, 13, 0.0005), (14, 36, 0.0005), (36, 7, 0.0005), (17, 0, 0.0005), (11, 37, 0.0005), (23, 18, 0.0005), (35, 10, 0.0005)]
    elif 'Llama-2-70b' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(33, 63, 0.0315), (36, 3, 0.0313), (29, 7, 0.0193), (40, 50, 0.0147), (26, 57, 0.0136), (40, 57, 0.0134), (40, 54, 0.0127), (36, 0, 0.011), (29, 3, 0.0109), (39, 61, 0.0085),
                     (77, 8, 0.0082), (14, 29, 0.0079), (39, 26, 0.0074), (37, 17, 0.0069), (40, 55, 0.0066), (34, 40, 0.0064), (39, 56, 0.0063), (34, 41, 0.0061), (36, 54, 0.0058), (29, 1, 0.0058),
                     (38, 20, 0.0053), (40, 48, 0.0051), (39, 30, 0.005), (34, 60, 0.0048), (34, 42, 0.0045), (26, 62, 0.0044), (77, 15, 0.0044), (77, 14, 0.0042), (43, 63, 0.0041), (31, 27, 0.004),
                     (31, 20, 0.004), (40, 6, 0.0038), (44, 63, 0.0036), (36, 41, 0.0034), (79, 34, 0.0033), (46, 31, 0.0033), (29, 4, 0.0033), (39, 36, 0.0032), (42, 10, 0.0031), (14, 30, 0.003),
                     (26, 25, 0.0029), (40, 61, 0.0028), (40, 39, 0.0028), (34, 25, 0.0028), (39, 59, 0.0027), (34, 56, 0.0025), (26, 31, 0.0025), (43, 4, 0.0025), (11, 21, 0.0024), (47, 44, 0.0023),
                     (76, 44, 0.0022), (38, 18, 0.0022), (75, 62, 0.0022), (21, 32, 0.0021), (51, 41, 0.002), (36, 32, 0.002), (44, 59, 0.0019), (43, 27, 0.0019), (40, 51, 0.0019), (32, 3, 0.0019),
                     (38, 11, 0.0018), (32, 11, 0.0018), (35, 2, 0.0018), (25, 13, 0.0018), (42, 12, 0.0017), (25, 3, 0.0017), (24, 0, 0.0017), (38, 3, 0.0017), (34, 46, 0.0016), (31, 5, 0.0016),
                     (38, 55, 0.0016), (40, 21, 0.0016), (40, 33, 0.0016), (77, 25, 0.0015), (42, 18, 0.0015), (35, 34, 0.0015), (7, 63, 0.0014), (24, 45, 0.0014), (39, 34, 0.0014), (27, 35, 0.0014),
                     (38, 34, 0.0014), (38, 19, 0.0013), (41, 33, 0.0013), (18, 61, 0.0013), (22, 36, 0.0013), (38, 51, 0.0013), (25, 7, 0.0013), (29, 17, 0.0012), (28, 45, 0.0012), (35, 8, 0.0012),
                     (69, 17, 0.0012), (72, 26, 0.0012), (44, 18, 0.0012), (43, 7, 0.0012), (76, 34, 0.0011), (10, 62, 0.0011), (14, 31, 0.0011), (45, 57, 0.0011), (25, 14, 0.0011), (30, 15, 0.0011),
                     (47, 1, 0.0011), (15, 46, 0.0011), (27, 57, 0.001), (37, 37, 0.001), (30, 9, 0.001), (16, 28, 0.001), (28, 7, 0.001), (29, 18, 0.001), (35, 5, 0.001), (14, 28, 0.001), (72, 24, 0.001),
                     (37, 10, 0.001), (26, 63, 0.001), (72, 29, 0.001), (39, 13, 0.001), (77, 59, 0.0009), (76, 36, 0.0009), (23, 59, 0.0009), (39, 35, 0.0009), (43, 16, 0.0009), (33, 49, 0.0009),
                     (33, 31, 0.0009), (29, 19, 0.0009), (43, 2, 0.0009), (40, 45, 0.0009), (76, 50, 0.0009), (38, 35, 0.0009), (39, 28, 0.0009), (20, 4, 0.0009), (36, 2, 0.0008), (38, 12, 0.0008),
                     (20, 47, 0.0008), (78, 44, 0.0008), (39, 57, 0.0008), (30, 26, 0.0008), (63, 52, 0.0008), (7, 62, 0.0008), (30, 6, 0.0008), (25, 10, 0.0008), (76, 32, 0.0008), (36, 45, 0.0008),
                     (27, 44, 0.0008), (38, 58, 0.0008), (38, 6, 0.0008), (36, 46, 0.0008), (31, 21, 0.0008), (22, 38, 0.0007), (36, 44, 0.0007), (71, 61, 0.0007), (37, 15, 0.0007), (39, 31, 0.0007),
                     (27, 48, 0.0007), (24, 41, 0.0007), (43, 49, 0.0007), (40, 26, 0.0007), (13, 31, 0.0007), (21, 34, 0.0007), (26, 61, 0.0007), (36, 11, 0.0007), (28, 34, 0.0007), (22, 18, 0.0007),
                     (34, 3, 0.0007), (40, 52, 0.0007), (32, 37, 0.0006), (76, 13, 0.0006), (74, 58, 0.0006), (43, 24, 0.0006), (17, 2, 0.0006), (21, 4, 0.0006), (59, 50, 0.0006), (37, 44, 0.0006),
                     (27, 46, 0.0006), (69, 28, 0.0006), (29, 11, 0.0006), (31, 25, 0.0006), (20, 18, 0.0006), (40, 63, 0.0006), (37, 19, 0.0006), (36, 23, 0.0006), (34, 13, 0.0006), (69, 19, 0.0006),
                     (44, 17, 0.0006), (44, 32, 0.0005), (26, 23, 0.0005), (42, 13, 0.0005), (34, 18, 0.0005), (75, 56, 0.0005), (37, 14, 0.0005), (25, 50, 0.0005), (42, 61, 0.0005), (43, 1, 0.0005),
                     (77, 27, 0.0005), (40, 24, 0.0005), (63, 50, 0.0005), (24, 25, 0.0005), (43, 30, 0.0005), (79, 23, 0.0005), (38, 62, 0.0005), (23, 9, 0.0005), (35, 30, 0.0005), (32, 34, 0.0005),
                     (39, 60, 0.0005), (29, 63, 0.0005), (55, 8, 0.0005), (6, 12, 0.0005), (39, 47, 0.0005), (44, 14, 0.0005), (36, 47, 0.0005), (6, 34, 0.0005), (41, 8, 0.0005), (36, 1, 0.0005),
                     (30, 22, 0.0005), (52, 20, 0.0005), (52, 56, 0.0004), (64, 23, 0.0004), (74, 5, 0.0004), (41, 63, 0.0004), (67, 23, 0.0004), (17, 23, 0.0004), (49, 23, 0.0004), (76, 39, 0.0004),
                     (49, 59, 0.0004), (18, 30, 0.0004), (37, 8, 0.0004), (23, 27, 0.0004), (36, 43, 0.0004), (57, 3, 0.0004), (39, 37, 0.0004), (37, 61, 0.0004), (39, 25, 0.0004), (25, 25, 0.0004),
                     (23, 38, 0.0004), (38, 49, 0.0004), (35, 27, 0.0004), (32, 9, 0.0004), (69, 30, 0.0004), (25, 9, 0.0004), (39, 32, 0.0004), (34, 57, 0.0004), (40, 47, 0.0004), (19, 51, 0.0004),
                     (16, 0, 0.0004), (20, 19, 0.0004), (44, 57, 0.0004), (40, 34, 0.0004), (79, 25, 0.0004), (69, 27, 0.0004), (76, 26, 0.0004), (26, 30, 0.0004), (72, 31, 0.0004), (26, 29, 0.0004),
                     (55, 15, 0.0004), (33, 58, 0.0004), (18, 25, 0.0004), (25, 2, 0.0004), (33, 27, 0.0004), (20, 40, 0.0004), (24, 27, 0.0004), (17, 3, 0.0004), (18, 62, 0.0004), (47, 7, 0.0004),
                     (33, 28, 0.0004), (31, 11, 0.0004), (24, 28, 0.0004), (37, 7, 0.0004), (40, 7, 0.0004), (32, 61, 0.0004)]
    elif 'gpt-neox' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(9, 42, 0.0293), (12, 4, 0.0224), (9, 28, 0.019), (11, 57, 0.0079), (10, 43, 0.0073), (12, 14, 0.0069), (14, 31, 0.0065), (9, 23, 0.0057), (11, 21, 0.0054), (11, 4, 0.0052),
                     (9, 21, 0.0052), (18, 23, 0.005), (13, 9, 0.0048), (14, 49, 0.0048), (12, 20, 0.0047), (8, 30, 0.0045), (12, 59, 0.0043), (16, 42, 0.0039), (11, 34, 0.0038), (9, 33, 0.0038),
                     (9, 3, 0.0036), (11, 48, 0.0035), (14, 63, 0.0032), (18, 11, 0.0032), (13, 7, 0.003), (9, 27, 0.0029), (11, 23, 0.0029), (16, 30, 0.0027), (10, 17, 0.0026), (9, 55, 0.0024),
                     (11, 38, 0.0024), (11, 59, 0.0024), (20, 8, 0.0024), (15, 42, 0.0023), (11, 47, 0.0023), (9, 15, 0.0023), (8, 47, 0.0023), (10, 40, 0.0023), (18, 18, 0.0022), (9, 1, 0.0021),
                     (13, 12, 0.0021), (14, 5, 0.002), (16, 18, 0.0019), (13, 63, 0.0019), (9, 20, 0.0018), (26, 38, 0.0018), (21, 60, 0.0017), (17, 55, 0.0016), (17, 30, 0.0016), (10, 56, 0.0015),
                     (12, 3, 0.0015), (10, 16, 0.0014), (10, 0, 0.0013), (15, 62, 0.0013), (12, 15, 0.0013), (9, 34, 0.0013), (12, 18, 0.0013), (23, 46, 0.0012), (16, 53, 0.0012), (11, 1, 0.0011),
                     (9, 2, 0.0011), (10, 27, 0.0011), (23, 54, 0.0011), (16, 54, 0.0011), (12, 30, 0.0011), (11, 14, 0.0011), (16, 44, 0.001), (14, 27, 0.001), (26, 31, 0.001), (15, 0, 0.001),
                     (13, 46, 0.001), (15, 57, 0.001), (15, 17, 0.001), (19, 12, 0.0009), (9, 49, 0.0009), (10, 7, 0.0009), (19, 46, 0.0009), (8, 21, 0.0009), (25, 24, 0.0008), (19, 29, 0.0008),
                     (12, 21, 0.0008), (8, 18, 0.0008), (12, 35, 0.0008), (9, 10, 0.0008), (19, 40, 0.0008), (38, 5, 0.0008), (13, 31, 0.0007), (10, 38, 0.0007), (10, 12, 0.0007), (11, 31, 0.0007),
                     (10, 1, 0.0007), (23, 15, 0.0007), (13, 40, 0.0007), (9, 5, 0.0007), (22, 33, 0.0007), (13, 36, 0.0006), (8, 32, 0.0006), (16, 21, 0.0006), (14, 11, 0.0006), (13, 61, 0.0006)]
    elif 'gpt-j' in model_config['name_or_path']:
        print(f"loading top heads for {model_config['name_or_path']}")
        top_heads = [(15, 5, 0.0587), (9, 14, 0.0584), (12, 10, 0.0526), (8, 1, 0.0445), (11, 0, 0.0445), (13, 13, 0.019), (8, 0, 0.0184), (14, 9, 0.016), (9, 2, 0.0127), (24, 6, 0.0113), (15, 11, 0.0092),
                     (6, 6, 0.0069), (14, 0, 0.0068), (17, 8, 0.0068), (21, 2, 0.0067), (10, 11, 0.0066), (11, 2, 0.0057), (17, 0, 0.0054), (20, 11, 0.0051), (23, 0, 0.0047), (20, 0, 0.0046), (15, 7, 0.0045),
                     (27, 2, 0.0045), (21, 15, 0.0044), (11, 4, 0.0044), (18, 6, 0.0043), (9, 6, 0.0042), (4, 12, 0.004), (11, 15, 0.004), (20, 2, 0.0036), (10, 0, 0.0035), (16, 9, 0.0031), (11, 14, 0.0031),
                     (12, 4, 0.003), (9, 7, 0.003), (18, 3, 0.003), (19, 5, 0.003), (22, 5, 0.0027), (25, 3, 0.0026), (18, 9, 0.0025)]
    else:
        raise ValueError("Model not supported")
    
    top_heads = top_heads[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)
    T = -1 # Intervention & values taken from last token

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path']:
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense
        elif 'olmo' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        else: 
            raise ValueError("Model not supported")

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads