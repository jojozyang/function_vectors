import os, json
import torch, numpy as np
import argparse

# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *

def eval_patch_mlp(mlp_O, mlp_O_name, results_dir, 
    dataset, model, model_config, tokenizer, 
    filter_set, generate_str, metric, prefixes, separators, seed, 
    mlp_layer, n_shots
):  
    """
    Patch MLP Outputs + Evaluation
    """
    zs_results = {}
    fs_shuffled_results = {}

    # Patch to individual layer 
    print("patching to individual layer")
    eval_edit_layer = [mlp_layer, model_config['n_layers']] 
    for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
        set_seed(seed)
        # zero shot 
        if generate_str:
            zs_results[edit_layer] = n_shot_eval(dataset=dataset,
                intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=0, 
                model=model, model_config=model_config, tokenizer=tokenizer, 
                filter_set=filter_set, generate_str=generate_str, metric=metric, 
                prefixes=prefixes, separators=separators,
                intervention_type='mlp_O',
            )
        else:
            zs_results[edit_layer] = n_shot_eval(dataset=dataset,
                intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=0, 
                model=model, model_config=model_config, tokenizer=tokenizer, 
                filter_set=filter_set,
                prefixes=prefixes, separators=separators,
                intervention_type='mlp_O',
            )
        set_seed(seed)
        # few shots with shuffled labels 
        if generate_str:
            fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, 
                intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=n_shots, 
                model=model, model_config=model_config, tokenizer=tokenizer, 
                filter_set = filter_set, generate_str=generate_str, metric=metric, 
                shuffle_labels=True, prefixes=prefixes, separators=separators,
                intervention_type='mlp_O',
            )
        else:
            fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, 
                intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=n_shots, 
                model=model, model_config=model_config, tokenizer=tokenizer, 
                filter_set = filter_set, 
                shuffle_labels=True, prefixes=prefixes, separators=separators,
                intervention_type='mlp_O',
            )

    # patch to multiple layers 
    print("patching to multiple layers")
    edit_layer = np.arange(mlp_layer, model_config['n_layers'])
    dict_key = f"{edit_layer[0]}-{edit_layer[-1]}"
    set_seed(seed)
    ## zero shot 
    if generate_str:
        zs_results[dict_key] = n_shot_eval(dataset=dataset,
            intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=0, 
            model=model, model_config=model_config, tokenizer=tokenizer, 
            filter_set=filter_set, generate_str=generate_str, metric=metric, 
            prefixes=prefixes, separators=separators,
            intervention_type='mlp_O',
        )
    else:
        zs_results[dict_key] = n_shot_eval(dataset=dataset,
            intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=0, 
            model=model, model_config=model_config, tokenizer=tokenizer, 
            filter_set=filter_set,
            prefixes=prefixes, separators=separators,
            intervention_type='mlp_O',
        )
    set_seed(seed)
    ## few shots with shuffled labels 
    if generate_str:
        fs_shuffled_results[dict_key] = n_shot_eval(dataset=dataset, 
            intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=n_shots, 
            model=model, model_config=model_config, tokenizer=tokenizer, 
            filter_set = filter_set, generate_str=generate_str, metric=metric, 
            shuffle_labels=True, prefixes=prefixes, separators=separators,
            intervention_type='mlp_O',
        )
    else:
        fs_shuffled_results[dict_key] = n_shot_eval(dataset=dataset, 
            intervention_vector=mlp_O, edit_layer=edit_layer, n_shots=n_shots, 
            model=model, model_config=model_config, tokenizer=tokenizer, 
            filter_set = filter_set, 
            shuffle_labels=True, prefixes=prefixes, separators=separators,
            intervention_type='mlp_O',
            )

    # Save results
    zs_results_file_suffix = f'_{mlp_O_name}_Fv_{Fv_edit_layer}_layer_sweep.json'
    fs_shuffled_results_file_suffix = f'_{mlp_O_name}_Fv_{Fv_edit_layer}_layer_sweep.json'

    zs_results_file_name = make_valid_path_name(
        f'{results_dir}/zs_results{zs_results_file_suffix}'
    )
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)

    fs_shuffled_results_file_name = make_valid_path_name(
        f'{results_dir}/fs_shuffled_results{fs_shuffled_results_file_suffix}'
    )
    args.fs_shuffled_results_file_name = fs_shuffled_results_file_name
    with open(fs_shuffled_results_file_name, 'w') as results_file:
        json.dump(fs_shuffled_results, results_file, indent=2)

if __name__ == "__main__":
    """
    mlp_layer: the layer where the mlp outputs vector is extracted from 
    fv_layer: the layer where the function vector is integrated into the model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', 
        type=str, required=True)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--model_name', help='Name of model to be loaded',
        type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='/oscar/data/epavlick/zyang220/results/fv_comm')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',
        type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
    parser.add_argument("--universal_set", help="Flag for whether to use the univeral or task-specific set of heads for Fv", 
        action="store_true", required=False, default=False)
    parser.add_argument("--mlp_layer", help="target layer for patching or caching mlp_out",
        type=int, required=True)
    parser.add_argument("--mlp_O_FS", help="Whether to intervene with mlp_O_FS",
        type=bool, required=False, default=True)
    parser.add_argument("--mlp_O_Fv_patch_attn", help="Whether to intervene with mlp_O_Fv_patch_attn",
        type=bool, required=False, default=True)
    parser.add_argument("--Fv_edit_layer", help="Layer of Fv intervention that generates mlp output vector",
        type=int, required=True)
    
    args = parser.parse_args() 
    root_data_dir = args.root_data_dir
    save_path_root = args.save_path_root
    dataset_name = args.dataset_name
    model_name = args.model_name
    n_shots = args.n_shots
    device = args.device
    metric = args.metric
    seed = args.seed
    test_split = args.test_split
    prefixes = args.prefixes
    separators = args.separators
    generate_str = args.generate_str
    mlp_layer = args.mlp_layer
    mlp_O_FS = args.mlp_O_FS
    mlp_O_Fv_patch_attn = args.mlp_O_Fv_patch_attn
    universal_set = args.universal_set
    Fv_edit_layer = args.Fv_edit_layer
    if universal_set:
        head_name = 'universal_heads'
    else:
        head_name = 'task_specific_heads'

    # Create directories if they don't exist
    results_dir = make_valid_path_name(
        f'{save_path_root}/{dataset_name}/mlp_O_eval_{head_name}'
    )
    os.makedirs(results_dir, exist_ok=True)

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    # Load fs_resutls 
    print(f"Loading fs_results from {save_path_root}")
    fs_results_file_path = os.path.join(save_path_root, 
        f'{dataset_name}',
        f'fs_results_{n_shots}shots.json'
    )
    with open(fs_results_file_path, 'r') as indata:
        fs_results = json.load(indata)
        key = 'score' if generate_str else 'clean_rank_list'
        target_val = 1 if generate_str else 0
        filter_set = np.where(np.array(fs_results[key]) == target_val)[0]

    if mlp_O_FS: 
        print("mlp_O_FS eval")
        mlp_O_name = f'mlp_O_{mlp_layer}_FS'
        mlp_O_path = os.path.join(save_path_root, 
            f'{dataset_name}',
            f'{mlp_O_name}.pt',
        )
        mlp_O = torch.load(mlp_O_path)
        
        eval_patch_mlp(mlp_O, mlp_O_name, results_dir=results_dir, 
            dataset=dataset, model=model, model_config=model_config, 
            tokenizer=tokenizer, filter_set=filter_set, 
            generate_str=generate_str, metric=metric, prefixes=prefixes, 
            separators=separators, seed=seed, 
            mlp_layer=mlp_layer, n_shots=n_shots)

    if mlp_O_Fv_patch_attn:
        print("mlp_O_Fv_patch_attn eval")
        mlp_O_name = f'mlp_O_{mlp_layer}_Fv_patch_attn'
        mlp_O_path = os.path.join(save_path_root, 
            f'{dataset_name}',
            f'Fv_eval_{head_name}',
            f'{mlp_O_name}_layer_sweep.pt',
        )
        mlp_O = torch.load(mlp_O_path)
        mlp_O = mlp_O[Fv_edit_layer]
        
        eval_patch_mlp(mlp_O, mlp_O_name, results_dir=results_dir, 
            dataset=dataset, model=model, model_config=model_config, 
            tokenizer=tokenizer, filter_set=filter_set, 
            generate_str=generate_str, metric=metric, prefixes=prefixes, 
            separators=separators, seed=seed, 
            mlp_layer=mlp_layer, n_shots=n_shots)

    # Write args to file
    args_file_name = make_valid_path_name(f'{results_dir}/args.txt')
    with open(args_file_name, 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)