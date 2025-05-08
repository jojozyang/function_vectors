import os, json
import torch, numpy as np
import argparse


# Include prompt creation helper functions
from utils.prompt_utils import *
from utils.intervention_utils import *
from utils.model_utils import *
from utils.eval_utils import *
from utils.extract_utils import *
from compute_indirect_effect import compute_indirect_effect

import copy
IGNORE_INDEX = -100
from datasets import Dataset
import transformers

zs_prefixes = [{'input': 'Q:', 'output': 'A:', 'instructions': ''},
                {'input': 'A:', 'output': 'B:', 'instructions': ''},
                {'input': 'input:', 'output': 'output:', 'instructions': ''},
                {'input': 'Input:', 'output': 'Output:', 'instructions': ''},
                {'input': 'Input:', 'output': 'A:', 'instructions': ''},
                {'input': 'In:', 'output': 'Out:', 'instructions': ''},
                {'input': 'in:', 'output': 'out:', 'instructions': ''},
                {'input': 'question:', 'output': 'answer:', 'instructions': ''},
                {'input': 'Question:', 'output': 'Answer:', 'instructions': ''},
                {'input': 'Question:', 'output': 'A:', 'instructions': ''},
                {'input': 'word:', 'output': 'output:', 'instructions': ''},
                {'input': '', 'output': ' ->', 'instructions': ''},
                {'input': '', 'output': ' :', 'instructions': ''},
                {'input': '', 'output': '->', 'instructions': ''},
                {'input': '', 'output': '→', 'instructions': ''},
                {'input': '', 'output': ' →', 'instructions': ''},
                {'input': 'text:', 'output': 'label:', 'instructions': ''},
                {'input': 'x:', 'output': 'f(x):', 'instructions': ''},
                {'input': 'x:', 'output': 'y:', 'instructions': ''},
                {'input': 'X:', 'output': 'Y:', 'instructions': ''}]

zs_separators=[{'input': ' ', 'output': ' ', 'instructions': ''},
                {'input': ' ', 'output': '\n', 'instructions': ''},
                {'input': ' ', 'output': '\n\n', 'instructions': ''},
                {'input': '\n', 'output': '\n', 'instructions': ''},
                {'input': '\n', 'output': '\n\n', 'instructions': ''},
                {'input': '\n\n', 'output': '\n\n', 'instructions': ''},
                {'input': ' ', 'output': '|', 'instructions': ''},
                {'input': '\n', 'output': '|', 'instructions': ''},
                {'input': '|', 'output': '\n', 'instructions': ''},
                {'input': '|', 'output': '\n\n', 'instructions': ''}]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    
def load_task_specific_prefixes_or_separators(
    dataset_names, root_data_dir, is_prefixes=False):
    
    return_list = []
    
    for dataset_name in dataset_names:
        
        dataset_prefixes_or_separators = []
        
        files_path = os.path.join(root_data_dir, dataset_name)
        
        for template_idx in os.listdir(files_path):
            
            if is_prefixes:
                dataset_prefixes_or_separators.append(json.load(open(os.path.join(files_path, template_idx, "prefixes.json"), 'r')))
            else:
                dataset_prefixes_or_separators.append(json.load(open(os.path.join(files_path, template_idx, "separators.json"), 'r')))
        
        return_list.append(dataset_prefixes_or_separators)
    
    return return_list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument("--universal_set", help="Flag for whether to evaluate using the univeral set of heads", 
        type=str2bool, required=True, default=False)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=True, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=True, type=int, default=10)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, 
        default="/oscar/data/epavlick/zyang220/results/function_vector_instructions")
    parser.add_argument('--ie_path_root', help='File path to load indirect effects from', type=str, required=False, default=None)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--indirect_effect_path', help='Path to file containing indirect_effect scores for the specified task', required=False, type=str, default=None)    
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_mean_activations_trials', help="Number of in-context prompts to average over for mean_activations", 
        type=int, required=False, default=100)
    parser.add_argument('--n_indirect_effect_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=25)
    # parser.add_argument('--zs_prefixes', help='Prompt template prefixes to be used for ZS eval', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    # parser.add_argument('--zs_separators', help='Prompt template separators to be used for ZS eval', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--template_index', help='Index of template to be used, if None, templates will be randomly sampled', 
        type=int, required=False, default=None)
    # parser.add_argument('--compute_baseline', help='Whether to compute the model baseline 0-shot -> n-shot performance', 
    #     type=str2bool, required=False, default=False)
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
    parser.add_argument('--revision', help='Specify model checkpoints for pythia or olmo models', type=str, required=False, default=None)
        
    args = parser.parse_args()  

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{model_name.split('/')[-1]}/{dataset_name}"
    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer
    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_mean_activations_trials = args.n_mean_activations_trials
    n_indirect_effect_trials = args.n_indirect_effect_trials
    #compute_baseline = args.compute_baseline
    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set
    template_index = args.template_index
    # zs_prefixes = args.zs_prefixes
    # zs_separators = args.zs_separators

    # Load instruction template
    if template_index is None: # will sample from all templates later
        inst_prefixes = load_task_specific_prefixes_or_separators(
            [dataset_name], '../template_files/', is_prefixes=True)[0]
        inst_separators = load_task_specific_prefixes_or_separators(
            [dataset_name], '../template_files/', is_prefixes=False)[0]
    else: # only use a specifc template 
        inst_prefixes = load_task_specific_prefixes_or_separators(
            [dataset_name], '../template_files/', is_prefixes=True)[0][template_index]
        inst_separators = load_task_specific_prefixes_or_separators(
            [dataset_name], '../template_files/', is_prefixes=False)[0][template_index]

    print(f"inst_prefixes type: {type(inst_prefixes)}")
    print(f"universal_set: {universal_set}")
    print(f"save_path_root: {save_path_root}")
    print(f"ie_path_root: {ie_path_root}")

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, revision=args.revision)

    if args.edit_layer == -1: # sweep over all layers if edit_layer=-1
        eval_edit_layer = [0, model_config['n_layers']]

    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # 1. Compute or Load Model Instruction Baseline & 2. Filter test set to cases where model gets it correct
    print(f"Filtering Dataset")
    instruction_results_file_name = f'{save_path_root}/instruction_results_layer_sweep.json'
    print(instruction_results_file_name)
    if os.path.exists(instruction_results_file_name):
        print(f"Loading instruction results from {instruction_results_file_name}")
        with open(instruction_results_file_name, 'r') as indata:
            instruction_results = json.load(indata)
        key = 'score' if generate_str else 'clean_rank_list'
        target_val = 1 if generate_str else 0
        filter_set = np.where(np.array(instruction_results[key]) == target_val)[0]
        filter_set_validation = None
    elif generate_str:
        print("Computing instruction results")
        set_seed(seed+42)
        instruction_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=0,
            model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False,
            generate_str=True, metric=metric, test_split='valid', 
            prefixes=inst_prefixes, separators=inst_separators)
        filter_set_validation = np.where(np.array(instruction_results_validation['score']) == 1)[0]
        set_seed(seed)
        instruction_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, 
            model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False,
            generate_str=True, metric=metric, 
            prefixes=inst_prefixes, separators=inst_separators)
        filter_set = np.where(np.array(instruction_results['score']) == 1)[0]

        args.instruction_results_file_name = instruction_results_file_name
        with open(instruction_results_file_name, 'w') as results_file:
            json.dump(instruction_results, results_file, indent=2)
    else:
        set_seed(seed+42)
        instruction_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=0, 
            model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='valid', 
            prefixes=inst_prefixes, separators=inst_separators)
        filter_set_validation = np.where(np.array(instruction_results_validation['clean_rank_list']) == 0)[0]
        set_seed(seed)
        instruction_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=0,
            model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, 
            prefixes=inst_prefixes, separators=inst_separators)
        filter_set = np.where(np.array(instruction_results['clean_rank_list']) == 0)[0]
    
        args.instruction_results_file_name = instruction_results_file_name
        with open(instruction_results_file_name, 'w') as results_file:
            json.dump(instruction_results, results_file, indent=2)

    set_seed(seed)
    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        print(f"Loading Mean Activations from {mean_activations_path}")
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{ie_path_root}/{dataset_name}_mean_head_activations.pt'
        print(f"Loading Mean Activations from {mean_activations_path}")
        mean_activations = torch.load(mean_activations_path)        
    else:
        print("Computing Mean Activations")
        set_seed(seed)
        mean_activations = get_mean_head_activations(dataset, model=model, 
            model_config=model_config, tokenizer=tokenizer, n_icl_examples=0,
            N_TRIALS=n_mean_activations_trials, 
            prefixes=inst_prefixes, separators=inst_separators, 
            filter_set=filter_set_validation)  # (Layers, Heads, Tokens, head_dim)
 
        args.mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        torch.save(mean_activations, args.mean_activations_path)

    # Load or Re-Compute indirect_effect values
    if indirect_effect_path is not None and os.path.exists(indirect_effect_path):
        print(f"Loading Indirect Effect from {indirect_effect_path}")
        indirect_effect = torch.load(indirect_effect_path)
    elif indirect_effect_path is None and os.path.exists(f'{ie_path_root}/{dataset_name}_indirect_effect.pt'):
        indirect_effect_path = f'{ie_path_root}/{dataset_name}_indirect_effect.pt'
        print(f"Loading Indirect Effect from {indirect_effect_path}")
        indirect_effect = torch.load(indirect_effect_path) 
    elif not universal_set:     # Only compute indirect effects if we need to
        print("Computing Indirect Effects")
        set_seed(seed)
        indirect_effect = compute_indirect_effect(dataset, mean_activations, 
            model=model, model_config=model_config, tokenizer=tokenizer, n_shots=0,
            n_trials=n_indirect_effect_trials, last_token_only=True, 
            prefixes=zs_prefixes, separators=zs_separators, # prefixes and separators for base, which is zs prompt for instructions! 
            filter_set=filter_set_validation)
        args.indirect_effect_path = f'{save_path_root}/{dataset_name}_indirect_effect.pt'
        torch.save(indirect_effect, args.indirect_effect_path)
        
    # Compute Function Vector
    if universal_set:
        print("Computing Function Vector for Universal Heads") 
        fv, top_heads = compute_universal_function_vector_instructions(mean_activations,
                model, model_config, n_top_heads)
        file_name_end = '_uni_heads'

    else:
        print("Computing Function Vector for Task-Specific Heads")
        fv, top_heads = compute_function_vector(mean_activations, indirect_effect, 
            model, model_config=model_config, n_top_heads=n_top_heads) 
        file_name_end = ''
    
    # Run Evaluation
    if isinstance(eval_edit_layer, int):
        print(f"Running ZS Eval with edit_layer={eval_edit_layer}")
        set_seed(seed)
        if generate_str:
            pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_ZS_intervention_layer{eval_edit_layer}.txt"
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                generate_str=generate_str, metric=metric, pred_filepath=pred_filepath, 
                prefixes=zs_prefixes, separators=zs_separators)
        else:
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, 
                prefixes=zs_prefixes, separators=zs_separators)
        zs_results_file_suffix = f'_editlayer_{eval_edit_layer}{file_name_end}.json'   

        # print(f"Running {n_shots}-Shot Shuffled Eval")
        # set_seed(seed)
        # if generate_str:
        #     pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_{n_shots}shots_shuffled_intervention_layer{eval_edit_layer}.txt"
        #     fs_shuffled_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=n_shots, 
        #                                       model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, shuffle_labels=True,
        #                                       generate_str=generate_str, metric=metric, pred_filepath=pred_filepath, prefixes=prefixes, separators=separators)
        # else:
        #     fs_shuffled_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=n_shots, 
        #                                       model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set, shuffle_labels=True, prefixes=prefixes, separators=separators)
        # fs_shuffled_results_file_suffix = f'_editlayer_{eval_edit_layer}{file_name_end}.json'   
        
    else:
        print(f"Running sweep over layers {eval_edit_layer}")
        zs_results = {}
        # fs_shuffled_results = {}
        for edit_layer in range(eval_edit_layer[0], eval_edit_layer[1]):
            set_seed(seed)
            if generate_str:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, 
                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                    generate_str=generate_str, metric=metric, 
                    prefixes=zs_prefixes, separators=zs_separators)
            else:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, 
                    model=model, model_config=model_config, tokenizer=tokenizer, filter_set=filter_set,
                    prefixes=zs_prefixes, separators=zs_separators)
            # set_seed(seed)
            # if generate_str:
            #     fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=n_shots, 
            #                                         model=model, model_config=model_config, tokenizer=tokenizer, filter_set = filter_set,
            #                                         generate_str=generate_str, metric=metric, shuffle_labels=True, prefixes=prefixes, separators=separators)
            # else:
            #     fs_shuffled_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=n_shots, 
            #                                         model=model, model_config=model_config, tokenizer=tokenizer, filter_set = filter_set, shuffle_labels=True, prefixes=prefixes, separators=separators)
        zs_results_file_suffix = f'_layer_sweep{file_name_end}.json'
        #fs_shuffled_results_file_suffix = f'_layer_sweep{file_name_end}.json'


    # Save results to files
    zs_results_file_name = make_valid_path_name(f'{save_path_root}/zs_results' + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)
    
    # fs_shuffled_results_file_name = make_valid_path_name(f'{save_path_root}/fs_shuffled_results' + fs_shuffled_results_file_suffix)
    # args.fs_shuffled_results_file_name = fs_shuffled_results_file_name
    # with open(fs_shuffled_results_file_name, 'w') as results_file:
    #     json.dump(fs_shuffled_results, results_file, indent=2)

    # Write args to file
    args_file_name = make_valid_path_name(f'{save_path_root}/fv_eval_args{file_name_end}.txt')
    with open(args_file_name, 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    print("Done!")
