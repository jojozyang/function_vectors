import torch
import nnsight

def path_patch(source, base, sender_head:tuple, receiver_heads:list, model, sender_index:int=-1, receiver_indices:list=[-1], composition:str='q'):
    """
    """

    composition = composition.lower()
    assert composition in ['q', 'k', 'v'], "Need to choose one of ['q','k','v']"

    tensor_1 = task.tensor_from_expression([source]) # tensorize string
    tensor_2 = task.tensor_from_expression([base]) # tensorize string 

    inputs_1, targets_1 = tensor_1[:, :-1], tensor_1[:, 1:]
    inputs_2, targets_2 = tensor_2[:, :-1], tensor_2[:, 1:]

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    hidden_size = model.config.n_embd
    head_dim = hidden_size // n_head
    batch_size = tensor_1.shape[0]  

    path_patching_score = torch.zeros(n_layer, n_head)
    
    source_cache = []
    base_cache = []

    # 1. Store attn head outputs from source   
    with model.trace(inputs_1):
        for layer_index in range(n_layer):
            attn_in = model.transformer.h[layer_index].attn.c_proj.input
            source_cache.append(attn_in.reshape(batch_size, -1, n_head, head_dim).save())
    source_cache = torch.stack(source_cache)

    # 2. Store attn head outputs from base
    with model.trace(inputs_2):
        for layer_index in range(n_layer):
            attn_in = model.transformer.h[layer_index].attn.c_proj.input
            base_cache.append(attn_in.reshape(batch_size, -1, n_head, head_dim).save())
        base_logits = model.lm_head.output.save()
    base_cache = torch.stack(base_cache)


    # 3. Store input of receiver head, altered by patching sender head output
    receiver_head_cache = {}
    with model.trace(inputs_2):
        for layer_index in range(n_layer):
            attn_in = model.transformer.h[layer_index].attn.c_proj.input
            attn_out = attn_in.reshape(batch_size, -1, n_head, head_dim)

            for head_index in range(n_head):
                if layer_index == sender_head[0] and head_index == sender_head[1]: # Patch Sender Head
                    attn_out[:, sender_index, head_index, :] = source_cache[layer_index, :, sender_index, head_index, :]
                else: # Remove chained effects through other heads
                    attn_out[:, sender_index, head_index, :] = base_cache[layer_index, :, sender_index, head_index, :]
            
            # Cache specified input to receiver head
            for i, (receiver_L, receiver_H) in enumerate(receiver_heads):

                if receiver_L == layer_index:
                    qkv = model.transformer.h[receiver_L].attn.c_attn.output.view(batch_size, -1, 3, hidden_size)
                    if composition == 'q':
                        receiver_act = qkv[:,receiver_indices[i],0].view(batch_size, n_head, head_dim)
                    if composition == 'k':
                        receiver_act = qkv[:,receiver_indices[i],1].view(batch_size, n_head, head_dim)
                    if composition == 'v':
                        receiver_act = qkv[:,receiver_indices[i],2].view(batch_size, n_head, head_dim)
                    
                    receiver_head_cache[(receiver_L,receiver_H)] = receiver_act[:,receiver_H,:].save()

    del source_cache, base_cache
            
    # 4. Patch the inputs to the receiver nodes (heads) and store the intervention logits         
    with model.trace(inputs_2):
        for layer_index in range(n_layer):
            qkv = model.transformer.h[layer_index].attn.c_attn.output.view(batch_size, -1, 3, hidden_size)
            for i, (receiver_L, receiver_H) in enumerate(receiver_heads):
                if layer_index == receiver_L:
                    if composition == 'q':
                        q = qkv[:,receiver_indices[i],0].view(batch_size, n_head, head_dim)
                        q[:,receiver_H,:] = receiver_head_cache[(receiver_L, receiver_H)]
                    elif composition == 'k':
                        k = qkv[:,receiver_indices[i],1].view(batch_size, n_head, head_dim)
                        k[:,receiver_H,:] = receiver_head_cache[(receiver_L, receiver_H)]
                    elif composition == 'v':
                        v = qkv[:,receiver_indices[i],2].view(batch_size, n_head, head_dim)
                        v[:,receiver_H,:] = receiver_head_cache[(receiver_L, receiver_H)]
                
        intervention_logits = model.lm_head.output.save()

    
    del receiver_head_cache


    return base_logits[:,-1], intervention_logits[:,-1], targets_1[:,-1], targets_2[:,-1]