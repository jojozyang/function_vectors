import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#TODO: calculate mean acc across datasets
def intervention_mean_acc_layer_sweep_across_datasets(save_path_root=None, results_dir=None,
    intervention_type="fv", fv_intervention="add_resid", 
    mlp_O_name="", mlp_layer=0, Fv_edit_layer=0, datasets=None,
    top_k=-1, universal_set=False, n_top=1, exclude_last_layer=False,
    fs_file_name=None, zs_file_name=None, multi_last=False):
    """
    Plot before and after intervention accuracies across layers where intervention is applied,
    Average across datasets 

    Parameters:  
    save_path_root: The path where the task is stored (task folder)
        e.g. '/oscar/data/epavlick/zyang220/results/fv_comm/gpt-j-6b/capitalize'
    results_dir: The path to the results (json) directory 
    intervention_type: what intervention eval results to plot
        fv: function vector
        mlp_O: mlp output
    fv_intervention: FV intervention method
        add to resid (add_resid) 
        patch to attn outputs (patch_attn)
        path patch from attn to mlp (path_patch_attn)
    Related to "mlp_O" intervention_type:
        mlp_O_name: name of the mlp output for intervention 
            FS: cached from fewshot runs 
            Fv_patch_attn: cached from when function vector is patched to attn outputs
            Fv_path_patch_attn: cached from when function vector is path patched from attn to mlp outputs
        mlp_layer: layer where mlp output vector is extracted from 
        Fv_edit_layer: layer where FV intervention is applied to product of mlp_O of evaluation 
    top_k: which top k result to extract and plot
           if top_k == -1, plot top_k=0, 1, and 2 in separate subplots
    n_top: number of top layers or top heads to extract fv or fev (mlp_O)
           can be a single integer or a list of integers to compare multiple n_top values
    multi_last: if last eval is a multi-layer intervention 
    """
    if exclude_last_layer:
        folder_tail = '_no_last_layer'   
    else:
        folder_tail = ''

    # Convert n_top to a list if it's a single integer
    if isinstance(n_top, int):
        n_top_values = [n_top]
    else:
        n_top_values = n_top
    
    # Define colors for different n_top values
    colors = {
        1: "purple",
        3: "yellow",
        5: "lightblue",
        10: "red",
        15: "blue",
        20: "darkblue",
    }
    
    # Default colors if n_top is not in the predefined colors
    default_colors = ["purple", "yellow", "lightblue", "red", "blue", "darkblue"]
    
    # Define line styles for different n_top values
    line_styles = {
        1: dict(width=4, dash=None),  # Solid line
        3: dict(width=4, dash="dashdot"),  
        5: dict(width=4, dash="dot"),  
        10: dict(width=4, dash="longdash"),  
        15: dict(width=4, dash="dash")  
    }
    
    # Default line styles if n_top is not in the predefined styles
    default_line_styles = [
        dict(width=4, dash=None),  # Solid line
        dict(width=4, dash="dashdot"),  # Dashed-dot line
        dict(width=4, dash="dot"),  # Dotted line
        dict(width=4, dash="dash"),  # Dash line
        dict(width=4, dash="longdash")  # Long dash line
    ]
    
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
    
    # For each top_k value we want to plot
    for i, k in enumerate(top_k_values):
        row = i + 1 if top_k == -1 else 1
        
        # For each n_top value
        for j, n in enumerate(n_top_values):
            # Get color for this n_top value
            if n in colors:
                color = colors[n]
            else:
                color = default_colors[j % len(default_colors)]
            
            # Get line style for this n_top value
            if n in line_styles:
                line_style = line_styles[n]
            else:
                line_style = default_line_styles[j % len(default_line_styles)]
            
            # Combine color and line style
            line_config = {**line_style, "color": color}
            
            # Load JSON data for this n_top value
            #TODO: delete 
            # if universal_set:
            #     head_name = 'universal_heads'
            # else:
            #     head_name = 'task_specific_heads'
            # if results_dir is None:
            #     results_dir = f'{save_path_root}/{intervention_type}_{n}/eval_{head_name}{folder_tail}'
            intervention_accuracies_1_across_datasets = []
            intervention_accuracies_2_across_datasets = []
            baseline_accuracies_1_across_datasets = []
            baseline_accuracies_2_across_datasets = []
            for d_name in datasets:
                if fs_file_name is None and zs_file_name is None:
                    if universal_set:
                        fs_file_name = f"fs_shuffled_results_layer_sweep_uni_heads.json"  
                        zs_file_name = f"zs_results_layer_sweep_uni_heads.json"  # Second JSON (Zero Shot)
                    else: 
                        fs_file_name = "fs_shuffled_results_layer_sweep.json"  # First JSON (Few Shots)
                        zs_file_name = "zs_results_layer_sweep.json"  # Second JSON (Zero Shot)         
                file_path_1 = os.path.join(results_dir, d_name, fs_file_name)
                file_path_2 = os.path.join(results_dir, d_name, zs_file_name)

                with open(file_path_1, "r") as f:
                    data_1 = json.load(f)
                with open(file_path_2, "r") as f:
                    data_2 = json.load(f)

                # Extract layers and accuracies for both JSON files
                layers = list(data_1.keys())
                if multi_last: 
                    x = [int(layer) for layer in layers[:-1]]
                    x.append(x[-1]+1)
                    x_ticks = [str(layer) for layer in x]
                else: 
                    x = [int(layer) for layer in layers]
                    x_ticks = [str(layer) for layer in x]
                
                # Extract accuracies for this specific top_k value
                intervention_accuracies_1 = [data_1[layer]["intervention_topk"][k][1] for layer in layers]
                intervention_accuracies_2 = [data_2[layer]["intervention_topk"][k][1] for layer in layers]
                intervention_accuracies_1_across_datasets.append(intervention_accuracies_1)
                intervention_accuracies_2_across_datasets.append(intervention_accuracies_2)

                # Extract baselines from a layer 
                layer = layers[0]
                baseline_accuracy_1 = data_1[layer]["clean_topk"][k][1]
                baseline_accuracy_2 = data_2[layer]["clean_topk"][k][1]
                baseline_accuracies_1 = [baseline_accuracy_1] * len(layers)
                baseline_accuracies_2 = [baseline_accuracy_2] * len(layers)
                baseline_accuracies_1_across_datasets.append(baseline_accuracies_1)
                baseline_accuracies_2_across_datasets.append(baseline_accuracies_2)
            
            # Get mean across datasets 
            intervention_accuracies_1_mean = np.mean(intervention_accuracies_1_across_datasets, axis=0)
            intervention_accuracies_2_mean = np.mean(intervention_accuracies_2_across_datasets, axis=0)
            baseline_accuracies_1_mean = np.mean(baseline_accuracies_1_across_datasets, axis=0)
            baseline_accuracies_2_mean = np.mean(baseline_accuracies_2_across_datasets, axis=0)

            # Add solid line for intervention accuracies (Few Shots)
            fig.add_trace(go.Scatter(
                x=x,
                y=intervention_accuracies_1_mean,
                mode="lines",
                line=line_config,  # Use combined line style and color
                name=f"+FV Intervention",
                showlegend=(row == 1),  # Only show legend for the first row
                text=x_ticks,  # Add text labels for x-axis ticks
            ), row=row, col=1)

            # Add dotted line for baseline accuracy (Few Shots)
            # Only add baseline once per row since it's the same for all n_top values
            if j == 0:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=baseline_accuracies_1_mean,
                    mode="lines",
                    line=dict(color="white", width=2, dash="dot"),  # Use white dotted line for baseline
                    name=f"Baseline",
                    showlegend=(row == 1),  # Only show legend for the first row
                    text=x_ticks, 
                ), row=row, col=1)

            # Add solid line for intervention accuracies (Zero Shot)
            fig.add_trace(go.Scatter(
                x=x,
                y=intervention_accuracies_2_mean,
                mode="lines",
                line=line_config,  # Use combined line style and color
                showlegend=False,
                text=x_ticks, 
            ), row=row, col=2)

            # Add dotted line for baseline accuracy (Zero Shot)
            # Only add baseline once per row since it's the same for all n_top values
            if j == 0:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=baseline_accuracies_2_mean,
                    mode="lines",
                    line=dict(color="white", width=2, dash="dot"),  # Use white dotted line for baseline
                    showlegend=False,
                    text=x_ticks, 
                ), row=row, col=2)

        # Set y-axis range for each subplot
        fig.update_yaxes(range=[0, 1.02], row=row, col=1)
        fig.update_yaxes(range=[0, 1.02], row=row, col=2)

    # Update layout for better spacing
    n_top_str = "-".join([str(n) for n in n_top_values])

    fig.update_layout(
        #title_text=title_text,
        template="plotly_dark",
        font=dict(size=10),
        width=1000,  # Increased width for side-by-side plots
        height=420 * rows,  # Adjust height based on number of rows
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        )
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

def intervention_acc_layer_sweep(save_path_root=None, results_dir=None,
    intervention_type="fv", fv_intervention="add_resid", 
    mlp_O_name="", mlp_layer=0, Fv_edit_layer=0, 
    top_k=-1, universal_set=False, n_top=1, exclude_last_layer=False,
    fs_file_name=None, zs_file_name=None, multi_last=False):
    """
    Plot before and after intervention accuracies across layers where intervention is applied

    Parameters:  
    save_path_root: The path where the task is stored (task folder)
        e.g. '/oscar/data/epavlick/zyang220/results/fv_comm/gpt-j-6b/capitalize'
    results_dir: The path to the results (json) directory 
    intervention_type: what intervention eval results to plot
        fv: function vector
        mlp_O: mlp output
    fv_intervention: FV intervention method
        add to resid (add_resid) 
        patch to attn outputs (patch_attn)
        path patch from attn to mlp (path_patch_attn)
    Related to "mlp_O" intervention_type:
        mlp_O_name: name of the mlp output for intervention 
            FS: cached from fewshot runs 
            Fv_patch_attn: cached from when function vector is patched to attn outputs
            Fv_path_patch_attn: cached from when function vector is path patched from attn to mlp outputs
        mlp_layer: layer where mlp output vector is extracted from 
        Fv_edit_layer: layer where FV intervention is applied to product of mlp_O of evaluation 
    top_k: which top k result to extract and plot
           if top_k == -1, plot top_k=0, 1, and 2 in separate subplots
    n_top: number of top layers or top heads to extract fv or fev (mlp_O)
           can be a single integer or a list of integers to compare multiple n_top values
    multi_last: if last eval is a multi-layer intervention 
    """
    if exclude_last_layer:
        folder_tail = '_no_last_layer'   
    else:
        folder_tail = ''

    # Convert n_top to a list if it's a single integer
    if isinstance(n_top, int):
        n_top_values = [n_top]
    else:
        n_top_values = n_top
    
    # Define colors for different n_top values
    colors = {
        1: "purple",
        3: "yellow",
        5: "lightblue",
        10: "red",
        15: "blue",
        20: "darkblue",
    }
    
    # Default colors if n_top is not in the predefined colors
    default_colors = ["purple", "yellow", "lightblue", "red", "blue", "darkblue"]
    
    # Define line styles for different n_top values
    line_styles = {
        1: dict(width=4, dash=None),  # Solid line
        3: dict(width=4, dash="dashdot"),  
        5: dict(width=4, dash="dot"),  
        10: dict(width=4, dash="longdash"),  
        15: dict(width=4, dash="dash")  
    }
    
    # Default line styles if n_top is not in the predefined styles
    default_line_styles = [
        dict(width=4, dash=None),  # Solid line
        dict(width=4, dash="dashdot"),  # Dashed-dot line
        dict(width=4, dash="dot"),  # Dotted line
        dict(width=4, dash="dash"),  # Dash line
        dict(width=4, dash="longdash")  # Long dash line
    ]
    
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
    
    # For each top_k value we want to plot
    for i, k in enumerate(top_k_values):
        row = i + 1 if top_k == -1 else 1
        
        # For each n_top value
        for j, n in enumerate(n_top_values):
            # Get color for this n_top value
            if n in colors:
                color = colors[n]
            else:
                color = default_colors[j % len(default_colors)]
            
            # Get line style for this n_top value
            if n in line_styles:
                line_style = line_styles[n]
            else:
                line_style = default_line_styles[j % len(default_line_styles)]
            
            # Combine color and line style
            line_config = {**line_style, "color": color}
            
            # Load JSON data for this n_top value
            #TODO: delete 
            # if universal_set:
            #     head_name = 'universal_heads'
            # else:
            #     head_name = 'task_specific_heads'
            # if results_dir is None:
            #     results_dir = f'{save_path_root}/{intervention_type}_{n}/eval_{head_name}{folder_tail}'
            if fs_file_name is None and zs_file_name is None:
                if universal_set:
                    fs_file_name = f"fs_shuffled_results_layer_sweep_uni_heads.json"  
                    zs_file_name = f"zs_results_layer_sweep_uni_heads.json"  # Second JSON (Zero Shot)
                else: 
                    fs_file_name = "fs_shuffled_results_layer_sweep.json"  # First JSON (Few Shots)
                    zs_file_name = "zs_results_layer_sweep.json"  # Second JSON (Zero Shot)         
            file_path_1 = os.path.join(results_dir, fs_file_name)
            file_path_2 = os.path.join(results_dir, zs_file_name)

            with open(file_path_1, "r") as f:
                data_1 = json.load(f)
            with open(file_path_2, "r") as f:
                data_2 = json.load(f)

            # Extract layers and accuracies for both JSON files
            layers = list(data_1.keys())
            if multi_last: 
                 x = [int(layer) for layer in layers[:-1]]
                 x.append(x[-1]+1)
                 x_ticks = [str(layer) for layer in x]
            else: 
                x = [int(layer) for layer in layers]
                x_ticks = [str(layer) for layer in x]
            
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
                line=line_config,  # Use combined line style and color
                name=f"+FV Intervention",
                showlegend=(row == 1),  # Only show legend for the first row
                text=x_ticks,  # Add text labels for x-axis ticks
            ), row=row, col=1)

            # Add dotted line for baseline accuracy (Few Shots)
            # Only add baseline once per row since it's the same for all n_top values
            if j == 0:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=baseline_accuracies_1,
                    mode="lines",
                    line=dict(color="white", width=2, dash="dot"),  # Use white dotted line for baseline
                    name=f"Baseline",
                    showlegend=(row == 1),  # Only show legend for the first row
                    text=x_ticks, 
                ), row=row, col=1)

            # Add solid line for intervention accuracies (Zero Shot)
            fig.add_trace(go.Scatter(
                x=x,
                y=intervention_accuracies_2,
                mode="lines",
                line=line_config,  # Use combined line style and color
                showlegend=False,
                text=x_ticks, 
            ), row=row, col=2)

            # Add dotted line for baseline accuracy (Zero Shot)
            # Only add baseline once per row since it's the same for all n_top values
            if j == 0:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=baseline_accuracies_2,
                    mode="lines",
                    line=dict(color="white", width=2, dash="dot"),  # Use white dotted line for baseline
                    showlegend=False,
                    text=x_ticks, 
                ), row=row, col=2)

        # Set y-axis range for each subplot
        fig.update_yaxes(range=[0, 1.02], row=row, col=1)
        fig.update_yaxes(range=[0, 1.02], row=row, col=2)

    # Update layout for better spacing
    n_top_str = "-".join([str(n) for n in n_top_values])
    #title_text = 
    fig.update_layout(
        #title_text=title_text,
        template="plotly_dark",
        font=dict(size=10),
        width=1000,  # Increased width for side-by-side plots
        height=420 * rows,  # Adjust height based on number of rows
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1
        )
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

def cosine_similarity_heatmap_matrix(
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

def cosine_similarity_heatmap(
    vector_1_name, tensor_1,
    vector_2_name, tensor_2,
):
    """
    Plot cosine similarity heatmap of row vectors in a matrix
    """
    # Convert tensors to numpy if needed
    if hasattr(tensor_1, 'numpy'):
        tensor_1 = tensor_1.numpy()
    if hasattr(tensor_2, 'numpy'):
        tensor_2 = tensor_2.numpy()

    # Compute cosine similarity for each row vector
    cos_sim = cosine_similarity(tensor_1, tensor_2)  # Shape: (layer,)

    # Reshape to a 2D array for heatmap visualization
    cos_sim_matrix = cos_sim.reshape(-1, 1)  # (layer, 1)
    n_layers = cos_sim_matrix.shape[0]

    # Create a plotly figure for better control over the visualization
    fig = go.Figure(data=go.Heatmap(
        z=cos_sim_matrix,
        x=["Similarity"],
        y=list(range(n_layers)),
        colorscale='RdBu_r',  # Red-Blue color scale
        zmin=-1,              # Set minimum value to -1
        zmax=1,               # Set maximum value to 1
        colorbar=dict(title="Cosine Similarity"),
        hovertemplate='Layer: %{y}<br>Similarity: %{z:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f"Cosine Similarity of {vector_1_name} and {vector_2_name}",
        xaxis_title="",
        yaxis_title="Layer",
        yaxis=dict(
            autorange='reversed',  # This makes the first value at the top
            tickmode='array',
            tickvals=list(range(n_layers)),
            ticktext=list(range(n_layers))
        ),
        width=500,
        height=800,
    )

    fig.show()
    return cos_sim_matrix

def indirect_effect_heatmap(indirect_effect):
    """
    Plot indirect effect heatmap
    """
    sns.heatmap(indirect_effect, annot=False, cmap='coolwarm')
    plt.title('Indirect Effect Heatmap')
    plt.show()
        
