# def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             max_length=1024,
#             truncation=True,
#         )
#         for text in strings
#     ]
#     input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
#     input_ids_lens = labels_lens = [
#         tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
#     ]
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )

# def preprocess(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]    
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)
    
#     input_ids = [ids[:-1] for ids in input_ids] # remove the last token
#     labels = [label[1:] for label in labels]    # remove the first token
    
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len - 1] = IGNORE_INDEX
        
#     if len(sources) == len(targets) == 1: 
#         return dict(input_ids=input_ids[0], labels=labels[0])
#     else:
#         return dict(input_ids=input_ids, labels=labels)
    
# def prep_dataset(ds, num_data_points=50):
#     """
#     """
#     pred_inds = ds['source_predictive_token_idxs']
#     sentences = [ds['source_input_ids'][i][:pred_inds[i]+1].cuda() for i in range(num_data_points)]
#     target_ids = torch.stack([ds['source_labels'][i][pred_inds[i]].long() for i in range(num_data_points)]).cuda()
#     return sentences, target_ids

# def process_dataset(dataset, model_config, tokenizer, n_shots,
#     data_split, prefixes, separators, n_trials=None, 
#     ablation_method="zero_shot", draw_source_from_split=True,
#     Parrot_shot=False
# ):
#     """
#     Process the dataset
#     Arguments:
#         Parrot_shot: Whether to construct Parrot-shot prompt where input and output of the 
#             examplars are the same. 
#     Outputs: 
#         A Dataset that contains the following fields:
#         {
#             "source_input_ids": torch.Tensor,
#             "source_labels": torch.Tensor,
#             "source_predictive_token_idxs": torch.Tensor,
#         }
#         If ablation_method is not "none", then the Dataset will also contain the following fields:
#         {
#             "base_input_ids": torch.Tensor,
#             "base_labels": torch.Tensor,
#             "base_predictive_token_idxs": torch.Tensor
#         }
#     """
#     assert ablation_method in ["zero_shot", "noninformative", "none"]
#     torch_dataset = []
    
#      # If the model already prepends a bos token by default, we don't want to add one
#     prepend_bos =  False if model_config['prepend_bos'] else True
    
#     if n_trials is None:
#         sample_idxs = range(len(dataset[data_split]))
#     else:
#         sample_idxs = np.random.choice(len(dataset[data_split]), n_trials, replace=True).tolist()

#     first_idx = sample_idxs[0]

#     for i in sample_idxs:
        
#         data_pair = {}
        
#         word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]  
        
#         if not draw_source_from_split:
#             word_pairs_test = dataset[data_split][i]
#             source_test_word_pair = dataset[data_split][np.random.choice(len(dataset[data_split]), 1, replace=False)]
#         else:
#             source_test_word_pair = dataset[data_split][i]
#             word_pairs_test = dataset[data_split][np.random.choice(len(dataset[data_split]), 1, replace=False)]
            
#         if Parrot_shot:
#             word_pairs['input'] = word_pairs['output']
#             word_pairs_test['input'] = word_pairs_test['output']
#             source_test_word_pair['input'] = source_test_word_pair['output']

#         if type(prefixes) == dict and type(separators) == dict:
#             prefix = prefixes
#             separator = separators
#         elif type(prefixes) == list and type(separators) == list:
#             rand_idx = np.random.choice(len(prefixes))
#             prefix = prefixes[rand_idx]
#             sep_rand_idx = np.random.choice(len(separators))
#             separator = separators[sep_rand_idx]
#         else:
#             raise ValueError("prefixes and separators should be either both list or dict")
        
#         prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=source_test_word_pair, 
#             prepend_bos_token=prepend_bos, shuffle_labels=False, prefixes=prefix, separators=separator)
        
#         query = prompt_data['query_target']['input']
#         target = prompt_data['query_target']['output']
        
#         source_token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query, 
#             prepend_bos=model_config['prepend_bos'])
#         source_batch = preprocess([prompt_string], [target], tokenizer)
        
#         if i == first_idx:
#             print("source prompt_string:\n", prompt_string)
        
#         data_pair["source_input_ids"] = source_batch["input_ids"]
#         data_pair["source_labels"] = source_batch["labels"]
        
#         assert source_token_labels[-1][2] == "query_predictive_token"
#         source_predictive_token_idxs = source_token_labels[-1][0]
#         data_pair["source_predictive_token_idxs"] = source_predictive_token_idxs
        
#         if ablation_method == "none":
#             pass
#         else:
#             if ablation_method == "zero_shot": #TODO: not necessary to have this?
#                 base_word_pairs = {'input':[], 'output':[]}
#             elif ablation_method == "noninformative":
#                 base_word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_shots, replace=False)]
#             else:
#                 raise ValueError(f"ablation_method {ablation_method} is not supported.")
            
#             ablation_prefix = {"input": prefix["input"], "output": prefix["output"], "instructions": ""}
#             ablation_separator = {"input": separator["input"], "output": separator["output"], "instructions": ""}
            
#             base_prompt_data = word_pairs_to_prompt_data(
#                 base_word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
#                 shuffle_labels=True, prefixes=ablation_prefix, separators=ablation_separator)
            
#             base_query = base_prompt_data['query_target']['input']
#             base_target = base_prompt_data['query_target']['output']
            
#             token_labels, base_prompt_string = get_token_meta_labels(
#                 base_prompt_data, tokenizer, base_query, prepend_bos=model_config['prepend_bos'])
            
#             if i == first_idx:
#                 print("base prompt_string:\n", base_prompt_string)
            
#             base_batch = preprocess([base_prompt_string], [base_target], tokenizer)
#             data_pair["base_input_ids"] = base_batch["input_ids"]
#             data_pair["base_labels"] = base_batch["labels"]
            
#             assert token_labels[-1][2] == "query_predictive_token"
#             base_predictive_token_idxs = token_labels[-1][0]
#             data_pair["base_predictive_token_idxs"] = base_predictive_token_idxs
            
#         torch_dataset.append(data_pair)
            
#     torch_dataset = Dataset.from_list(torch_dataset)
#     torch_dataset.set_format(type='torch')
#     return torch_dataset