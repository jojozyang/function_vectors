import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def intervention_acc_layer_sweep(result_path, 
    intervention_type="fv", fv_intervention="add_resid", 
    mlp_O_name="", mlp_layer=0, Fv_edit_layer=0, 
    top_k=0, universal_set=False):
    """
    Plot before and after intervention accuracies across layers where intervention is applied

    Parameters:  
    result_path: The path where the result json files are stored 
    intervention_type: what intervention eval results to plot
        fv: function vector
        mlp_O: mlp output
    fv_intervention: FV intervention method
        add to resid (add_resid) 
        patch to attn outputs (patch_attn)
        path patch from attn to mlp (path_patch_attn)
    (related to "mlp_O" intervention_type)
        mlp_O_name: name of the mlp output for intervention 
            FS: cached from fewshot runs 
            Fv_patch_attn: cached from when function vector is patched to attn outputs
            Fv_path_patch_attn: cached from when function vector is path patched from attn to mlp outputs
        mlp_layer: layer where mlp output vector is extracted from 
        Fv_edit_layer: layer where FV intervention is applied to product of mlp_O of evaluation 
    top_k: which top k result to extract and plot
    """
    # Load JSON data for both sources
    file_name_1 = "fs_shuffled_results_"  # First JSON (Few Shots)
    file_name_2 = "zs_results_"  # Second JSON (Zero Shot)
    if universal_set:
        head_name = 'universal_heads'
    else:
        head_name = 'task_specific_heads'
    if intervention_type == "fv":
        folder_name = f"Fv_eval_{head_name}"
        file_suffix = f"{fv_intervention}_layer_sweep.json"
    elif intervention_type == "mlp_O":
        folder_name = f"mlp_O_eval_{head_name}"
        file_suffix = f"mlp_O_{mlp_layer}_{mlp_O_name}_Fv_{Fv_edit_layer}_layer_sweep.json"

    file_path_1 = os.path.join(result_path, folder_name, 
        file_name_1 + file_suffix)
    file_path_2 = os.path.join(result_path, folder_name, 
        file_name_2 + file_suffix)

    with open(file_path_1, "r") as f:
        data_1 = json.load(f)
    with open(file_path_2, "r") as f:
        data_2 = json.load(f)

    # Extract layers and accuracies for both JSON files
    layers = list(data_1.keys()) 

    intervention_accuracies_1 = [data_1[layer]["intervention_topk"][top_k][1] for layer in layers]
    intervention_accuracies_2 = [data_2[layer]["intervention_topk"][top_k][1] for layer in layers]

    # Extract baselines from a layer 
    layer = layers[0]
    baseline_accuracy_1 = data_1[layer]["clean_topk"][top_k][1]
    baseline_accuracy_2 = data_2[layer]["clean_topk"][top_k][1]

    baseline_accuracies_1 = [baseline_accuracy_1] * len(layers)
    baseline_accuracies_2 = [baseline_accuracy_2] * len(layers)

    # Create subplots with specific titles
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Few Shots", "Zero Shot")
    )

    # x axis 
    if intervention_type == "fv":
        x = [int(layer) for layer in layers]
    elif intervention_type == "mlp_O": # last item (key) is a layer range like "25-26"
        x = [int(layer) for layer in layers[:-1]]
        x = x + [int(x[-1]+1)]
            
    # Add solid line for intervention accuracies (Few Shots)
    fig.add_trace(go.Scatter(
        x=x,
        y=intervention_accuracies_1,
        mode="lines",
        line=dict(color="lightblue", width=3),
        name="Intervention Accuracy",
    ), row=1, col=1)

    # Add dotted line for baseline accuracy (Few Shots)
    fig.add_trace(go.Scatter(
        x=x,
        y=baseline_accuracies_1,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        name="Baseline Accuracy",
    ), row=1, col=1)

    # Add solid line for intervention accuracies (Zero Shot)
    fig.add_trace(go.Scatter(
        x=x,
        y=intervention_accuracies_2,
        mode="lines",
        line=dict(color="lightblue", width=3),
        showlegend=False,
    ), row=1, col=2)

    # Add dotted line for baseline accuracy (Zero Shot)
    fig.add_trace(go.Scatter(
        x=x,
        y=baseline_accuracies_2,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    # Update layout for better spacing
    fig.update_layout(
        title_text=f"Capitalize top_{top_k}",
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        template="plotly_dark",
        font=dict(size=10),
        width=1000,  # Increased width for side-by-side plots
        height=400,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)

    fig.update_xaxes(
        tickvals=x,
        ticktext=layers,
    )

    # Show the plot
    fig.show()

def cosine_similarity_heatmap(
    mlp_O_FS=None, mlp_O_Fv_patch_attn=None):
    """
    Plot cosine similarity heatmap of row vectors in a matrix
    """
    # Concatenate matrices 
    vector_matrix = np.concatenate([mlp_O_FS, mlp_O_Fv_patch_attn], axis=0)
    
    # Labels 
    labels = ['FS'] + [f'Fv_patch_{i}' for i in range(len(mlp_O_Fv_patch_attn))]

    # Convert to numpy array 
    data = np.array(vector_matrix)

    # Compute the cosine similarity matrix
    cos_sim_matrix = cosine_similarity(data)

    # Create a heatmap to visualize the similarity matrix
    sns.heatmap(cos_sim_matrix, annot=False, 
        cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title('Cosine Similarity Heatmap')
    plt.show()
        
