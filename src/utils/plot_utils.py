import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def intervention_acc_layer_sweep(save_path_root, 
    intervention_type="fv", fv_intervention="add_resid", 
    mlp_O_name="", mlp_layer=0, Fv_edit_layer=0, 
    top_k=-1, universal_set=False, n_top=10):
    """
    Plot before and after intervention accuracies across layers where intervention is applied

    Parameters:  
    save_path_root: The path where the results are stored (task folder)
        e.g. '/oscar/data/epavlick/zyang220/results/fv_comm/gpt-j-6b/capitalize'
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
           if top_k == -1, plot top_k=0, 1, and 2 in separate subplots
    n_top: number of top layers or top heads to extract fv or fev (mlp_O)
    """
    # Load JSON data for both sources
    if universal_set:
        head_name = 'universal_heads'
    else:
        head_name = 'task_specific_heads'
    results_dir = f'{save_path_root}/{intervention_type}_{n_top}/eval_{head_name}'
    file_name_1 = "fs_shuffled_results_layer_sweep.json"  # First JSON (Few Shots)
    file_name_2 = "zs_results_layer_sweep.json"  # Second JSON (Zero Shot)
    
    file_path_1 = os.path.join(results_dir, file_name_1)
    file_path_2 = os.path.join(results_dir, file_name_2)

    with open(file_path_1, "r") as f:
        data_1 = json.load(f)
    with open(file_path_2, "r") as f:
        data_2 = json.load(f)

    # Extract layers and accuracies for both JSON files
    layers = list(data_1.keys()) 

    # Determine if we're plotting multiple top_k values or just one
    if top_k == -1:
        # Plot top_k = 0, 1, and 2 in separate subplots
        top_k_values = [0, 1, 2]
        rows = 3
        cols = 2
        subplot_titles = []
        for k in top_k_values:
            subplot_titles.extend([f"Few Shots (top-{k+1})", f"Zero Shot (top-{k+1})"])
    else:
        # Plot just the specified top_k value
        top_k_values = [top_k]
        rows = 1
        cols = 2
        subplot_titles = ["Few Shots", "Zero Shot"]

    # Create subplots with specific titles
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=subplot_titles
    )

    # x axis 
    x = [int(layer) for layer in layers]
    
    # For each top_k value we want to plot
    for i, k in enumerate(top_k_values):
        row = i + 1 if top_k == -1 else 1
        
        # Extract accuracies for this specific top_k value
        intervention_accuracies_1 = [data_1[layer]["intervention_topk"][k][1] for layer in layers]
        intervention_accuracies_2 = [data_2[layer]["intervention_topk"][k][1] for layer in layers]

        # Extract baselines from a layer 
        layer = layers[0]
        baseline_accuracy_1 = data_1[layer]["clean_topk"][k][1]
        baseline_accuracy_2 = data_2[layer]["clean_topk"][k][1]

        baseline_accuracies_1 = [baseline_accuracy_1] * len(layers)
        baseline_accuracies_2 = [baseline_accuracy_2] * len(layers)

        # Add solid line for intervention accuracies (Few Shots)
        fig.add_trace(go.Scatter(
            x=x,
            y=intervention_accuracies_1,
            mode="lines",
            line=dict(color="lightblue", width=3),
            name=f"Intervention Accuracy (top-{k+1})" if top_k == -1 else "Intervention Accuracy",
            showlegend=(row == 1)  # Only show legend for the first row
        ), row=row, col=1)

        # Add dotted line for baseline accuracy (Few Shots)
        fig.add_trace(go.Scatter(
            x=x,
            y=baseline_accuracies_1,
            mode="lines",
            line=dict(color="blue", width=3, dash="dash"),
            name=f"Baseline Accuracy (top-{k+1})" if top_k == -1 else "Baseline Accuracy",
            showlegend=(row == 1)  # Only show legend for the first row
        ), row=row, col=1)

        # Add solid line for intervention accuracies (Zero Shot)
        fig.add_trace(go.Scatter(
            x=x,
            y=intervention_accuracies_2,
            mode="lines",
            line=dict(color="lightblue", width=3),
            showlegend=False,
        ), row=row, col=2)

        # Add dotted line for baseline accuracy (Zero Shot)
        fig.add_trace(go.Scatter(
            x=x,
            y=baseline_accuracies_2,
            mode="lines",
            line=dict(color="blue", width=3, dash="dash"),
            showlegend=False,
        ), row=row, col=2)

        # Set y-axis range for each subplot
        fig.update_yaxes(range=[0, 1], row=row, col=1)
        fig.update_yaxes(range=[0, 1], row=row, col=2)

    # Update layout for better spacing
    title_text = "Capitalize" if top_k == -1 else f"Capitalize top_{top_k}"
    fig.update_layout(
        title_text=title_text,
        template="plotly_dark",
        font=dict(size=10),
        width=1000,  # Increased width for side-by-side plots
        height=400 * rows,  # Adjust height based on number of rows
        margin=dict(l=60, r=40, t=60, b=60)
    )

    # Add axis titles
    for i in range(1, rows+1):
        fig.update_xaxes(title_text="Layer", row=i, col=1)
        fig.update_xaxes(title_text="Layer", row=i, col=2)
        fig.update_yaxes(title_text="Accuracy", row=i, col=1)

    # Set x-axis ticks
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            fig.update_xaxes(
                tickvals=x,
                ticktext=layers,
                row=i, col=j
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

def indirect_effect_heatmap(indirect_effect):
    """
    Plot indirect effect heatmap
    """
    sns.heatmap(indirect_effect, annot=False, cmap='coolwarm')
    plt.title('Indirect Effect Heatmap')
    plt.show()
        
