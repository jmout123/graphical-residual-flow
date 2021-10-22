"""
Perform density estimation for given belief network. Learned likelihood will
 adhere to independencies specified by the BN structure. 
"""

import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import FileStorageObserver
from functools import partial
from torch.optim.lr_scheduler import StepLR

from grf.factory import build_residual_flow_density_estimation
from ex_utils import batch_iterator, count_parameters, sample_batch
from graph.belief_network import ArithmeticCircuit, GaussianBinaryTree, Tree, Protein


ex = Experiment('residual_nf_density_estimation', interactive=True)

@ex.config
def cfg():
    # Belief network [arithmetic, binary-tree, tree, protein]
    bn = 'arithmetic'
    # Params for flow with Lipschitz constraint
    coeff = 0.99
    n_power_iterations = 5

    num_blocks = 1
    hidden_dims = [100]
    activation_function = 'lipswish'

    batch_size = 100
    num_train_samples = 10000
    num_test_samples = 5000
    num_steps = 100
    num_train_batches = num_train_samples//batch_size
    num_test_batches = num_test_samples//batch_size

    lr = 1e-2
    lr_decay = 0.1
    lr_decay_step_size = 20
    seed = 6

    # Add observer
    ex_name = 'residual_nf_density_estimation'
    sub_folder = '{}_graphical-lipschitz_{}'.format(bn, str(coeff).split('.')[1])
    path = './experiment_logs/{}/{}'.format(ex_name, sub_folder)
    ex.observers.append(FileStorageObserver(path))


@ex.automain
def run(_config, _run, _rnd):
    # Training info
    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"
    batch_size = _config['batch_size']
    lr = _config['lr'] 
    lr_decay = _config['lr_decay']
    lr_decay_step_size = _config['lr_decay_step_size']
    num_train_batches = _config['num_train_batches']

    # BN initialization
    bn = _config['bn']
    if bn == 'arithmetic':
        graph = ArithmeticCircuit()
    elif bn == 'tree':
        graph = Tree()
    elif bn == 'protein':
        graph = Protein()
    elif bn == 'binary-tree':
        depth = 5
        coeffs = GaussianBinaryTree.rand_coeffs(_rnd, depth)
        graph = GaussianBinaryTree(depth, coeffs)
        ex.info['tree_coeffs'] = graph.coeffs.tolist()
    else:
        raise Exception("Unknown belief network: {}".format(bn))
    num_latent = graph.get_num_latent()
    n = graph.get_num_vertices()

    # Draw train and test sets and return minibatch iterators
    tree_train = np.zeros((_config['num_train_samples'], n))
    tree_test = np.zeros((_config['num_test_samples'], n))
    get_data = partial(graph.sample)
    iterators = partial(batch_iterator, get_data, tree_train, tree_test, batch_size)
   
    # Initialize model and optimizer
    coeff = _config['coeff']
    n_power_iterations = _config['n_power_iterations']
    hidden_dims = _config['hidden_dims']
    num_blocks = _config['num_blocks']
    activation_function = _config['activation_function']
    model = build_residual_flow_density_estimation(
        num_blocks, graph, hidden_dims, coeff,
        n_power_iterations, activation_function, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay)

    # Log model capacity
    ex.info['num model params'] = count_parameters(model)

    # Train
    for epoch in range(_config['num_steps']):
        train_batcher, test_batcher = iterators()
        train_nll = 0.0

        model.train()
        for _ in range(num_train_batches):
            train_batch = torch.tensor(train_batcher(), dtype=torch.float64).to(device)
            z0, j = model(train_batch)
            loss = -model.ll(z0, j)

            l = loss.detach().item()
            train_nll += l    

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log training metrics - training loss, kl-divergence
        _run.log_scalar("training.nll", value=train_nll/num_train_batches) 

        # Log test metrics - test loss, kl-divergence
        model.eval()
        with torch.no_grad():
            test_batch = torch.DoubleTensor(test_batcher()).to(device)
            z0, j = model(test_batch)
            test_nll = -model.ll(z0, j).item()
            _run.log_scalar("test.nll", test_nll)

            if bn == 'arithmetic':
                z = test_batch[:,:num_latent]
                x = test_batch[:,num_latent:]
                true_nll = (-graph.log_joint(x, z)).mean().item()

        # Decay learning rate every 100 epochs
        lr_scheduler.step()

        if epoch%1 == 0:
            if bn == 'arithmetic':
                print('['+'{}'.format(epoch+1).rjust(3)+']: train nll: {:.5f}; test nll: {:.5f} (model), {:.5f} (true)'.format( 
                train_nll/num_train_batches,
                test_nll,
                true_nll))
            else:
                print('['+'{}'.format(epoch+1).rjust(3)+']: train nll: {:.5f}; test nll: {:.5f}'.format( 
                train_nll/num_train_batches,
                test_nll))
    
    # Save the model
    path = _config['path']
    torch.save(model, path+'/{}/model.pt'.format(_run._id))

    # Inversion verification
    print('\n-- Invertibility Verification --')
    print('Largest singular values of the weight matrices of each layer of residual block i:')
    print('For invertibility: sigma_max < 1 and should approximately = {}'.format(coeff))
    for i, block in enumerate(model.blocks):
        sigmas = block.g.largest_singular_values()
        print('[{}] {}'.format(i, sigmas))

    model.eval()

    # Some plots
    with torch.no_grad():
        batch = sample_batch(graph.sample, 10000)
        true_sample = (batch).double().to(device)

        z0 = dist.Normal(
                loc=torch.zeros(graph.get_num_vertices(), dtype=torch.float64),
                scale=torch.ones(graph.get_num_vertices(), dtype=torch.float64)
            ).sample((10000,)).to(device)
        model_sample, _ = model.inverse(z0)

        model_z0,_ = model(true_sample)

        label1 = ['True','Flow']
        label2 = ['N(0,1)', 'Flow']
    
        for i in range(graph.get_num_vertices()):
            ztrue = true_sample[:,i].cpu().tolist()
            zflow = model_sample[:,i].cpu().tolist()
            plt.figure()
            plt.hist(x=[ztrue, zflow], bins=50, alpha=0.5, 
                        histtype='stepfilled', density=True,
                        color=['steelblue', 'red'], edgecolor='none',
                        label=label1)
            plt.xlabel("x "+str(i) +" (Data space)")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(path+'/{}/{}'.format(_run._id,i))

            ztrue = z0[:,i].cpu().tolist()
            zflow = model_z0[:,i].cpu().tolist()
            plt.figure()
            plt.hist(x=[ztrue, zflow], bins=50, alpha=0.5, 
                        histtype='stepfilled', density=True,
                        color=['steelblue', 'red'], edgecolor='none',
                        label=label2)
            plt.xlabel("x "+str(i)+" (Base distribution space)")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(path+'/{}/x0_{}'.format(_run._id,i))