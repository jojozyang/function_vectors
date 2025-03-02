import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def intervention_acc_layer_sweep(result_path, fv_intervention="resid", top_k=0):
    """
    Plot before and after intervention accuracies across layers where intervention is applied

    Parameters:  
    result_path: The path where the result json files are stored 
    fv_intervention: FV intervention method
        add to resid (resid) 
        patch to attn outputs (attn_output)
        path patch from attn to mlp (path_patch_attn_mlp)
    top_k: which top k result to extract and plot

    """
    # Load JSON data for both sources
    file_name_1 = "fs_shuffled_results_layer_sweep_"  # First JSON (Few Shots)
    file_name_2 = "zs_results_layer_sweep_"  # Second JSON (Zero Shot)

    file_path_1 = result_path + file_name_1 + f"{fv_intervention}.json"
    file_path_2 = result_path + file_name_2 + f"{fv_intervention}.json"

    with open(file_path_1, "r") as f:
        data_1 = json.load(f)
    with open(file_path_2, "r") as f:
        data_2 = json.load(f)

    # Extract layers and accuracies for both JSON files
    layers = sorted(data_1.keys(), key=int)  # Ensure layers are sorted numerically

    intervention_accuracies_1 = [data_1[layer]["intervention_topk"][top_k][1] for layer in layers]
    intervention_accuracies_2 = [data_2[layer]["intervention_topk"][top_k][1] for layer in layers]

    # Extract baselines from layer 0
    baseline_accuracy_1 = data_1["0"]["clean_topk"][top_k][1]
    baseline_accuracy_2 = data_2["0"]["clean_topk"][top_k][1]

    baseline_accuracies_1 = [baseline_accuracy_1] * len(layers)
    baseline_accuracies_2 = [baseline_accuracy_2] * len(layers)

    # Create subplots with specific titles
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Few Shots", "Zero Shot")
    )

    # Add solid line for intervention accuracies (Few Shots)
    fig.add_trace(go.Scatter(
        x=[int(layer) for layer in layers],
        y=intervention_accuracies_1,
        mode="lines",
        line=dict(color="lightblue", width=3),
        name="Intervention Accuracy",
    ), row=1, col=1)

    # Add dotted line for baseline accuracy (Few Shots)
    fig.add_trace(go.Scatter(
        x=[int(layer) for layer in layers],
        y=baseline_accuracies_1,
        mode="lines",
        line=dict(color="blue", width=3, dash="dash"),
        name="Baseline Accuracy",
    ), row=1, col=1)

    # Add solid line for intervention accuracies (Zero Shot)
    fig.add_trace(go.Scatter(
        x=[int(layer) for layer in layers],
        y=intervention_accuracies_2,
        mode="lines",
        line=dict(color="lightblue", width=3),
        showlegend=False,
    ), row=1, col=2)

    # Add dotted line for baseline accuracy (Zero Shot)
    fig.add_trace(go.Scatter(
        x=[int(layer) for layer in layers],
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

    # Show the plot
    fig.show()