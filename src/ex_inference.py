"""
Approximate the posterior of the latent variables in a belief network given the
observed using a Graphical Residual Flow. The learned posterior will adhere to 
the independencies specified through the belief network structure.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import FileStorageObserver
from functools import partial
from torch.optim.lr_scheduler import StepLR

from grf.factory import build_residual_flow
from ex_utils import batch_iterator, count_parameters, sample_batch
from graph.belief_network import ArithmeticCircuit, GaussianBinaryTree, Tree, Protein

ex = Experiment('residual_nf', interactive=True)


@ex.config
def cfg():
    # Belief network [arithmetic, binary-tree, tree, protein]
    bn = 'binary-tree'
    # Params for flow with Lipschitz constraint
    coeff = 0.99
    n_power_iterations = 5

    num_blocks = 5
    hidden_dims = [100]
    activation_function = 'lipswish'

    batch_size = 250
    num_train_samples = 25000
    num_test_samples = 250
    num_steps = 100
    num_train_batches = num_train_samples//batch_size
    num_test_batches = num_test_samples//batch_size

    lr = 1e-2
    lr_decay = 0.1
    lr_decay_step_size = 40
    seed = 6

    # Add observer
    ex_name = 'residual_nf_inference'
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

    # Get static batch for visualization over learning
    if bn == 'binary-tree':
        static_tree = graph.sample(batch_size=5)
        # Log static batches
        ex.info['static_batches'] = static_tree.tolist()
        static_samples = [np.tile(static_tree[i,:], (batch_size,1)) for i in range(5)]
        # log-likelihoods of samples from the inference network evaluated on the
        # analytical posterior, given the 5 static samples - saved as metric
        ll_p_metric_names = ["ll_p_1", "ll_p_2", "ll_p_3", "ll_p_4", "ll_p_5"]

    # Initialize model and optimizer
    coeff = _config['coeff']
    n_power_iterations = _config['n_power_iterations']
    hidden_dims = _config['hidden_dims']
    num_blocks = _config['num_blocks']
    activation_function = _config['activation_function']
    model = build_residual_flow(
        num_blocks, graph, hidden_dims, coeff,
        n_power_iterations, activation_function, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay)

    # Log model capacity
    ex.info['num model params'] = count_parameters(model)

    # Train
    print('[ ]: Shifted Reverse KL-divergence')
    for epoch in range(_config['num_steps']):
        train_batcher, test_batcher = iterators()
        train_shifted_kl = 0.0
        if bn == 'binary-tree':
            train_true_kl = 0.0

        model.train()
        for idx in range(num_train_batches):
            train_batch = torch.tensor(train_batcher(), dtype=torch.float64).to(device)
            x = train_batch[:,num_latent:]
            eps0, z, j = model(cond=x)
            loss = model.shifted_reverse_kl(x, z, eps0, j)

            l = loss.detach().item()
            train_shifted_kl += l    

            if bn == 'binary-tree':
                with torch.no_grad(): 
                    p = graph.log_posterior(torch.cat((z,x), dim=1)) 
                    q = model.ll(eps0, j)
                    train_true_kl += (q - p).item() 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # log training metrics - training loss, kl-divergence
        _run.log_scalar("training.shifted_kl", value=train_shifted_kl/num_train_batches) 
        if bn == 'binary-tree':
            _run.log_scalar("training.true_kl", value=train_true_kl/num_train_batches) 

        # Log test metrics - test loss, kl-divergence
        model.eval()
        with torch.no_grad():
            test_batch = torch.DoubleTensor(test_batcher()).to(device)
            x = test_batch[:,num_latent:]
            eps0, z, j = model(cond=x)
            test_shifted_kl = model.shifted_reverse_kl(x, z, eps0, j).item()
            _run.log_scalar("test.shifted_kl", test_shifted_kl)

            if bn == 'binary-tree':
                p = graph.log_posterior(torch.cat((z,x), dim=1))
                q = model.ll(eps0, j)
                _run.log_scalar("test.true_kl", (q - p).item())

        # Decay learning rate every 100 epochs
        lr_scheduler.step()

        # Calculate log-likelihood of samples from the inference
        # network evaluated on the analytical posterior
        if epoch%1 == 0:
            
            if bn == 'binary-tree':
                ll_p = [[],[],[],[],[]]
                for i in range(5):
                    with torch.no_grad():
                        static_sample = torch.DoubleTensor(static_samples[i]).to(device)
                        x = static_sample[:,num_latent:]
                        _, z, _ = model(cond=x)
                        ll = graph.log_posterior(torch.cat((z, x), dim=1))
                        # log training metrics - log-likelihood
                        ll_p[i] = -ll.item()/num_latent
                        _run.log_scalar(ll_p_metric_names[i], ll_p[i])

                print('[{}]: test: {}, train: shifted {}, true {}, post [{}, {}, {}, {}, {}]'.format(
                    epoch+1, 
                    test_shifted_kl, 
                    train_shifted_kl/num_train_batches, 
                    train_true_kl/num_train_batches, 
                    ll_p[0], ll_p[1], ll_p[2], ll_p[3], ll_p[4]))
            else:
                print('[{}]: test {}, train {}'.format(
                    epoch+1, 
                    test_shifted_kl, 
                    train_shifted_kl/num_train_batches))

    
    # Save the model
    path = _config['path']
    torch.save(model, path+'/{}/model.pt'.format(_run._id))
                    

    # Inversion verification
    print('\n-- Invertibility Verification --')
    print('Largest singular values of the weight matrices of each layer of residual block i:')
    print('For invertibility: sigma_max < 1 and will approximately = {}'.format(coeff))
    for i, block in enumerate(model.blocks):
        sigmas = block.g.largest_singular_values()
        print('[{}] {}'.format(i, sigmas))
    
    # Log-likelihood of samples drawn from the true posterior
    if bn == 'binary-tree':
        true_ll = 0.0
        for i in range(5):
            static_sample = torch.DoubleTensor(static_samples[i]).to(device)
            x = static_sample[:,num_latent:]
            z = graph.sample_posterior(x)
            ll = graph.log_posterior(torch.cat((z, x), dim=1))
            true_ll += ll/5
        print('Neg Log-likelihood of samples drawn from the true posterior: -log(p(z|x)) = ', -(true_ll).item()/num_latent)

    # Some plots of the posterior distributions
    # with torch.no_grad():
    #     batch = sample_batch(graph.sample, 100000)
    #     sample = (batch).double().to(device)
    #     x = sample[:,num_latent:]
    #     eps0, z, j = model(cond=x)
    #     inference_network_samples = torch.cat((z, x), dim=1)
    #     if bn == 'binary-tree':
    #         bn_samples = graph.sample_posterior(x)
    #         label = ['True','Inference network']
    #     else:
    #         bn_samples = sample[:,:num_latent]
    #         label = ['Joint','Inference network']

    #     for i in range(num_latent):
    #         ztrue = bn_samples[:,i].cpu().tolist()
    #         zinf = inference_network_samples[:,i].cpu().tolist()
    #         plt.figure()
    #         plt.hist(x=[ztrue, zinf], bins=50, alpha=0.5, 
    #                     histtype='stepfilled', density=True,
    #                     color=['steelblue', 'red'], edgecolor='none',
    #                     label=label)
    #         plt.xlabel("z_"+str(i))
    #         plt.ylabel("Density")
    #         plt.legend()
    #         plt.savefig(path+'/{}/z_{}'.format(_run._id,i))