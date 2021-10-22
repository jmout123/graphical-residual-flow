import json
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import torch

from sacred import Experiment
from ex_utils import sample_batch, kolmogorov_smirnov

ex = Experiment('results_visualization')


@ex.command(unobserved=True)
def shifted_reverse_kl_loss_curves(experiments, legend, _config):
    num_runs = _config['num_runs']
    iter = _config['iter']
    ylims = _config['ylims']
    log_scale = _config['log_scale']
    path = './experiment_logs/'
    exs_data = {'names':[]}
    num_experiments = len(experiments)
    palette = _config['palette'][:num_experiments]

    for p in range(num_experiments):
        ex_name = experiments[p]
        exs_data['names'].append(ex_name)
        
        for i in range(1,num_runs+1):
            with open(os.path.join(path, experiments[p], str(i), 'metrics.json')) as json_file:
                data = json.load(json_file)
                
                if p == 0 and i == 1:
                    exs_data['iter'] = data['training.shifted_kl']['steps'][:iter]
                    num_iters = len(data['training.shifted_kl']['steps'][:iter])
                    exs_data['train_kl'] = np.zeros((num_iters, num_runs*num_experiments))
                    exs_data['test_kl'] = np.zeros((num_iters, num_runs*num_experiments))

                exs_data['train_kl'][:,p*num_runs+i-1] = data['training.shifted_kl']['values'][:iter]
                exs_data['test_kl'][:,p*num_runs+i-1] = data['test.shifted_kl']['values'][:iter]

    columns = [[], []]
    for ex_name in exs_data['names']:
        columns[0] += [ex_name]*num_runs
        for i in range(1, num_runs+1):
            columns[1] += ['run_'+str(i)]
    columns = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(columns, names=["experiment", "runs"])

    # Training shifted reverse kl-divergence
    df = pd.DataFrame(exs_data['train_kl'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    s = sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    if log_scale:
        s.set(yscale='log')
    plt.ylabel('Shifted Reverse KL-divergence')
    plt.xlabel('epoch')
    if ylims is not None:
        plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()

    # Test shifted reverse kl-divergence
    df = pd.DataFrame(exs_data['test_kl'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    s = sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    if log_scale:
        s.set(yscale='log')
    plt.ylabel('Test Shifted Reverse KL-divergence')
    plt.xlabel('epoch')
    if ylims is not None:
        if len(ylims) > 1:
            plt.ylim(ylims[1])
        else:
            plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()


@ex.command(unobserved=True)
def binary_tree_plots(experiments, legend, _config):
    num_runs = _config['num_runs']
    iter = _config['iter']
    ylims = _config['ylims']
    log_scale = _config['log_scale']
    path = './experiment_logs/'
    exs_data = {'names':[]}
    num_experiments = len(experiments)
    palette = _config['palette'][:num_experiments]

    for p in range(num_experiments):
        ex_name = experiments[p]
        exs_data['names'].append(ex_name)
        
        for i in range(1,num_runs+1):
            with open(os.path.join(path, experiments[p], str(i), 'metrics.json')) as json_file:
                data = json.load(json_file)
                ll_values = []
                for j in range(1,6):
                    ll_values.append([data['ll_p_'+str(j)]['values'][:iter]])
                ll_values = np.concatenate(ll_values, axis=0).T
                ll_values = ll_values.mean(axis=1)
                
                if p == 0 and i == 1:
                    exs_data['iter'] = data['training.true_kl']['steps'][:iter]
                    num_iters = len(data['training.true_kl']['steps'][:iter])
                    exs_data['train_kl'] = np.zeros((num_iters, num_runs*num_experiments))
                    exs_data['test_kl'] = np.zeros((num_iters, num_runs*num_experiments))
                    exs_data['ll'] = np.zeros((num_iters, num_runs*num_experiments))

                exs_data['train_kl'][:,p*num_runs+i-1] = data['training.true_kl']['values'][:iter]
                exs_data['test_kl'][:,p*num_runs+i-1] = data['test.true_kl']['values'][:iter]
                exs_data['ll'][:,p*num_runs+i-1] = ll_values
                
    columns = [[], []]
    for ex_name in exs_data['names']:
        columns[0] += [ex_name]*num_runs
        for i in range(1, num_runs+1):
            columns[1] += ['run_'+str(i)]
    columns = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(columns, names=["experiment", "runs"])

    # Training kl-divergence
    df = pd.DataFrame(exs_data['train_kl'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    s = sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    if log_scale:
        s.set(yscale='log')
    plt.ylabel('Reverse KL-divergence')
    plt.xlabel('epoch')
    if ylims is not None:
        plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()

    # Test kl-divergence
    df = pd.DataFrame(exs_data['test_kl'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    plt.ylabel('Test Reverse KL-divergence')
    plt.xlabel('epoch')
    if ylims is not None:
        if len(ylims) > 1:
            plt.ylim(ylims[1])
        else:
            plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()

    # Negative log-likelihood
    df = pd.DataFrame(exs_data['ll'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    ax = sns.lineplot(data=df, x='level_2', y=0, hue='experiment', 
        palette=palette)
    plt.axhline(0.7535, color='indianred', linestyle='--', label='True')
    plt.ylabel('NLL')
    plt.xlabel('epoch')
    if ylims is not None:
        if len(ylims) > 2:
            plt.ylim(ylims[2])
        else:
            plt.ylim(ylims[0])
    labels = legend + ["True"]
    handles, _ = ax.get_legend_handles_labels()
    plt.legend(handles = handles, labels = labels)
    plt.show()


@ex.command(unobserved=True)
def infer(model, bn, sample):
    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"
    num_latent = bn.get_num_latent()

    x = sample[:,num_latent:]
    z_joint = sample[:,:num_latent]

    _, z_approx, _ = model(x)

    for i in range(num_latent):
        zj = z_joint[:,i].cpu().tolist()
        za = z_approx[:,i].cpu().tolist()
        # ks-test:
        ks, _ = kolmogorov_smirnov(za, zj)
        fig =  plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x=[zj, za], bins=50, alpha=0.5, histtype='stepfilled',
                density=True, color=['steelblue', 'red'], 
                edgecolor='none', label=['Joint','Inference network'])
        ax.set_xlabel("z_"+str(i))
        ax.set_ylabel("Density")
        ax.legend()
        ax.text(0.02,0.035,"ks={:.3f}".format(ks), transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))
        plt.show()


@ex.command(unobserved=True)
def nll_loss_curves(experiments, legend, _config):
    num_runs = _config['num_runs']
    iter = _config['iter']
    ylims = _config['ylims']
    log_scale = _config['log_scale']
    path = './experiment_logs/'
    exs_data = {'names':[]}
    num_experiments = len(experiments)
    palette = _config['palette'][:num_experiments]

    for p in range(num_experiments):
        ex_name = experiments[p]
        exs_data['names'].append(ex_name)
        
        for i in range(1,num_runs+1):
            with open(os.path.join(path, experiments[p], str(i), 'metrics.json')) as json_file:
                data = json.load(json_file)
                
                if p == 0 and i == 1:
                    exs_data['iter'] = data['training.nll']['steps'][:iter]
                    num_iters = len(data['training.nll']['steps'][:iter])
                    exs_data['train_nll'] = np.zeros((num_iters, num_runs*num_experiments))
                    exs_data['test_nll'] = np.zeros((num_iters, num_runs*num_experiments))

                exs_data['train_nll'][:,p*num_runs+i-1] = data['training.nll']['values'][:iter]
                exs_data['test_nll'][:,p*num_runs+i-1] = data['test.nll']['values'][:iter]

    columns = [[], []]
    for ex_name in exs_data['names']:
        columns[0] += [ex_name]*num_runs
        for i in range(1, num_runs+1):
            columns[1] += ['run_'+str(i)]
    columns = list(zip(*columns))
    columns = pd.MultiIndex.from_tuples(columns, names=["experiment", "runs"])

    # Training shifted reverse kl-divergence
    df = pd.DataFrame(exs_data['train_nll'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    s = sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    if log_scale:
        s.set(yscale='log')
    plt.ylabel('NLL')
    plt.xlabel('epoch')
    if ylims is not None:
        plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()

    # Test shifted reverse kl-divergence
    df = pd.DataFrame(exs_data['test_nll'], index=exs_data['iter'], columns=columns)
    df = df.unstack(level=1).reset_index()
    s = sns.lineplot(data=df, x='level_2', y=0, hue='experiment',
        palette=palette)
    if log_scale:
        s.set(yscale='log')
    plt.ylabel('Test NLL')
    plt.xlabel('epoch')
    if ylims is not None:
        if len(ylims) > 1:
            plt.ylim(ylims[1])
        else:
            plt.ylim(ylims[0])
    plt.legend(legend)
    plt.show()



@ex.config
def cfg():
    num_runs = 1
    iter = 100
    palette = ('teal', 'crimson', 'orange','yellowgreen', 'darkslateblue')
    ylims = None
    log_scale=False


@ex.automain
def main():
    pass