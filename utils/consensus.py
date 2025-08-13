from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def uniqueSensorCombos(listOfSensors):
    "returns a list of unique sensor combinations from a list of sensors. excludes empty set, includes full set"
    combos = []
    for i in range(1, len(listOfSensors) + 1):
        for combo in combinations(listOfSensors, i):
            combos.append(list(combo))
    return combos

def plotStateEstimates(indexes_to_test, EKFs, x_est_all, x_cov_all, t_sim_ekf_all, sim_data):
    x_est = x_est_all[indexes_to_test[0]]
    n_col = 3
    n_row = int(np.ceil(len(x_est.keys()) / n_col))
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(6.0 * n_col, 3.0 * n_row))
    fig.suptitle('State Estimates from EKFs', fontsize=16)

    colors = plt.cm.get_cmap('tab20b', len(indexes_to_test))

    handles = []
    labels = []

    for i, x in enumerate(x_est.keys()):
        ax.flat[i].set_title(x)
        # Plot all EKF estimates first (so "true" is on top)
        for j, idx in enumerate(indexes_to_test):
            label = f'EKF {idx}: {EKFs[idx].measurement_names}'
            l2, = ax.flat[i].plot(
                t_sim_ekf_all[idx],
                x_est_all[idx][x].values,
                label=label,
                color=colors(j),
                linestyle='-'
            )
            if i == 0:
                handles.append(l2)
                labels.append(label)
        # Plot the true line last so it is on top
        l1, = ax.flat[i].plot(
            t_sim_ekf_all[indexes_to_test[0]],
            sim_data[x].values,
            label='true',
            color='black',
            linestyle=':',
            linewidth=2
        )
        if i == 0:
            handles.insert(0, l1)
            labels.insert(0, 'true')
        lim = np.hstack([sim_data[x].values] + [x_est_all[idx][x].values for idx in indexes_to_test])
        if (lim.max() - lim.min()) < 0.1:
            ax.flat[i].set_ylim(lim.min() - 0.1, lim.max() + 0.1)

    # Place legend vertically on the left
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0, 0.5), fontsize=12)
    fig.subplots_adjust(hspace=0.5, wspace=0.5, left=0.22)  # Increase left margin for legend

def ignoreUnobservableCases(EKFs, x_est_all, x_cov_all):
    estimated_states = x_est_all[0].columns.tolist()
    print("estiamted states: ",estimated_states)

    trustworthy_combos = {state: [] for state in estimated_states}
    for i in range(len(EKFs)):
        EKF = EKFs[i]
        for state in estimated_states:
            trustworthy_combos[state].append(i)

    print(trustworthy_combos)

    for state in trustworthy_combos:
        print(f"Checking state: {state}")
        for j in range(len(trustworthy_combos[state])):
            print('checking ', state, 'with EKF ', trustworthy_combos[state][j])
            this_cov = x_cov_all[j][state]
            print(this_cov)
            print()