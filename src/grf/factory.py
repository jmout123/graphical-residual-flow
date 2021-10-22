import torch

from grf.residual_blocks import GraphicalLipschitzResidualBlock
from grf.graphical_residual_flow import GraphicalResidualFlow

from grf.normal_log_density import StandardNormalLogDensity

from graph.utils import adjacency_matrix, graph_masks, autoregressive_masks


def build_residual_flow(num_residual_blocks, graph, hidden_dims, coeff,
                         n_power_iterations, activation_function, device):
    residual_blocks = []
    latent_dim = graph.get_num_latent()
    cond_dim = graph.get_num_obs()

    masks = graph_masks(
        graph.inverse_graph, 
        input_vars=[*range(latent_dim+cond_dim)], 
        output_vars=[*range(latent_dim)], 
        hidden_dims=hidden_dims,
        self_dependent=True)
    masks = [torch.from_numpy(mask).to(device) for mask in masks]
    args = {
        'in_dim': latent_dim, 
        'hidden_dims': hidden_dims, 
        'cond_dim': cond_dim,
        'masks': masks,
        'activation_function': activation_function
    }
    
    residual_block = GraphicalLipschitzResidualBlock
    args['coeff'] = coeff
    args['n_power_iterations'] = n_power_iterations

    # Create blocks of flow
    for block in range(num_residual_blocks):
        block = residual_block(**args)
        residual_blocks.append(block)

    return GraphicalResidualFlow(residual_blocks, graph, StandardNormalLogDensity(), device)


def build_residual_flow_density_estimation(num_residual_blocks, graph, 
        hidden_dims, coeff, n_power_iterations, activation_function,
        device):
    residual_blocks = []
    D = graph.get_num_vertices()

    masks = graph_masks(
        graph.forward_graph, 
        input_vars=[*range(D)], 
        output_vars=[*range(D)], 
        hidden_dims=hidden_dims,
        self_dependent=True)
    masks = [torch.from_numpy(mask).to(device) for mask in masks]
    args = {
        'in_dim': D, 
        'hidden_dims': hidden_dims, 
        'cond_dim': 0,
        'masks': masks,
        'activation_function': activation_function
    }
    residual_block = GraphicalLipschitzResidualBlock
    args['coeff'] = coeff
    args['n_power_iterations'] = n_power_iterations

    # Create blocks of flow
    for block in range(num_residual_blocks):
        block = residual_block(**args)
        residual_blocks.append(block)

    return GraphicalResidualFlow(residual_blocks, graph, StandardNormalLogDensity(), device, generative=False)