#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
import seaborn as sns

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Variables of interest
variables_dim_1 = ['p_0_x_position', 'p_1_x_position', 'p_2_x_position', 'p_3_x_position']
variables_dim_2 = ['p_0_y_position', 'p_1_y_position', 'p_2_x_position', 'p_3_x_position']


def load_observations(path, _variables):
    data = pd.read_csv(path)
    # reversing the data so that the least valued
    # 0 index represents most reset observation
    # collect variables of interest
    if _variables:
        data = data[_variables]
    print(data.head())
    var_names = data.columns.values
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=var_names)
    return dataframe


# Positions of particles
observations_dim_1 = load_observations('data/observations.csv', variables_dim_1)
observations_dim_2 = load_observations('data/observations.csv', variables_dim_2)

# Spring constants between particles
springs = load_observations('data/springs.csv', [])


def setup_pcmci(data_frame):
    pcmci = PCMCI(dataframe=data_frame,
                  cond_ind_test=ParCorr(),
                  verbosity=0)
    # Auto correlations are helpful to determine tau
    # correlations = pcmci.get_lagged_dependencies(tau_max=100, val_only=True)['val_matrix']
    return pcmci


def construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2):
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            avg_p_val = (p_values_dim_1[p_a][p_b][time_step] + p_values_dim_2[p_a][p_b][time_step])/2.0
            if graph.has_edge(f'particle_{p_a}', f'particle_{p_b}') and (np.abs(avg_p_val) > 0.25):
                graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')

    # variables_dim_1 is ok
    save_graph(time_step, graph, variables_dim_1)


def save_graph(time_step, causal_graph, _variables):
    # observations -> positions
    # springs -> spring constants
    # causal graph from predictions

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))


    # ----- Plotting Particle positions
    axes[0][0].set_title('Particle position')
    entries = []
    _observations = pd.read_csv('data/observations.csv')
    for particle_id in range(0, len(variables_dim_1)):
        data = {'particle': particle_id,
                'x_cordinate': _observations.iloc[time_step][f'p_{particle_id}_x_position'],
                'y_cordinate': _observations.iloc[time_step][f'p_{particle_id}_y_position']}
        entries.append(data)
    pdframe = pd.DataFrame(entries)
    pl = sns.scatterplot(data=pdframe,
                         x='x_cordinate',
                         y='y_cordinate',
                         hue='particle',
                         ax=axes[0][0])
    pl.set_ylim(-5.0, 5.0)
    pl.set_xlim(-5.0, 5.0)

    # ----- Plotting spring constants
    _springs = pd.read_csv('data/springs.csv')
    axes[0][1].set_title(f'Spring connections')
    columns = [f'particle_{i}' for i in range(len(_variables))]
    s_mat = []
    for p_a in range(len(_variables)):
        for p_b in range(len(_variables)):
            s_mat.append(_springs.iloc[time_step][f's_{p_a}_{p_b}'])
    s_mat = np.reshape(s_mat, (len(_variables), len(_variables)))
    sns.heatmap(pd.DataFrame(s_mat, columns=columns, index=columns),
                vmin=0.0, vmax=2.0, ax=axes[0][1])

    # ----- Plotting Ground Truth Causal graph
    axes[1][0].set_title(f'Ground truth causal graph (Springs)')
    _vars = [f'particle_{i}' for i in range(len(_variables))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if np.abs(_springs.iloc[time_step][f's_{p_a}_{p_b}']) == 0.0 and graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')
    nx.draw(graph,
            pos=nx.circular_layout(graph),
            with_labels=True,
            ax=axes[1][0],
            node_size=500)

    # ----- Plotting Predicted Causal graph
    axes[1][1].set_title(f'Predicted causal graph (Springs)')
    nx.draw(causal_graph,
            pos=nx.circular_layout(causal_graph),
            with_labels=True,
            ax=axes[1][1],
            node_size=500)

    confusion_matrix = np.zeros(shape=(2, 2))
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if np.abs(_springs.iloc[time_step][f's_{p_a}_{p_b}']) == 0.0:
                if causal_graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                    # false positive
                    confusion_matrix[0][1] += 1
                else:
                    # True negative
                    confusion_matrix[1][1] += 1
            else:
                if causal_graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                    # True positive
                    confusion_matrix[0][0] += 1
                else:
                    # False negative
                    confusion_matrix[1][0] += 1

    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

    fig.suptitle(f'Time step {time_step} - precision {precision}')

    #plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{time_step}.png'))
    plt.clf()
    plt.close(fig)


def get_parents(tau_max, tau_min):
    _vars = list(range(len(variables_dim_1)))
    _lags = list(range(-(tau_max), -tau_min + 1, 1))
    # Set the default as all combinations of the selected variables
    _int_sel_links = {}
    for j in _vars:
        _int_sel_links[j] = [(var, -lag) for var in _vars
                             for lag in range(tau_min, tau_max + 1)
                             if not (var == j and lag == 0)]
    # Remove contemporary links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[1] != 0]
    # Remove self links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[0] != j]

    return _int_sel_links


def main():
    # First they estimate all parents for last layer.
    # Using the same kin relationship as parent sets
    # The same set of parents are used for momemtary ci test backwards in time.
    # Running pcmci on dim 1

    # *** Control Variables ***
    tau_max = 100

    _springs = pd.read_csv('data/springs.csv')

    print('Running pcmci on dim 1')
    parents = get_parents(tau_min=1, tau_max=tau_max)
    pcmci = setup_pcmci(observations_dim_1)
    pcmci.verbosity = 1

    results = pcmci.run_pcmci(tau_max=tau_max,
                              selected_links=parents)
    p_values_dim_1 = results['p_matrix'].round(3)

    # Running pcmci on dim 2
    print('Running pcmci on dim 2')
    pcmci = setup_pcmci(observations_dim_2)
    pcmci.verbosity = 1

    results = pcmci.run_pcmci(tau_max=tau_max,
                              selected_links=parents)
    p_values_dim_2 = results['p_matrix'].round(3)

    time_step = tau_max-1
    while time_step != 0:
        construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2)
        time_step -= 1

    print('Done.')

# delete all png files.
fp_in = f"{os.getcwd()}/tmp/timestep_*.png"
for f in glob.glob(fp_in):
    os.remove(f)
logging.info('trajectory gif stores in media')


if __name__ == "__main__":
    main()