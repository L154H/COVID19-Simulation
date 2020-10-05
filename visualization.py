import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import networkx as nx
import seaborn as sns
import numpy as np
from util import prepare_data, merge_curves, merge_curves_std


def show_graph(LG, states):
    print("Spring_layout starting now")
    pos = nx.spring_layout(LG)
    print("end")
    create_fig(LG, states, pos)
    plt.show()
    plt.close()


def create_fig(LG, states, pos):
    fig = plt.figure()
    color_palette = sns.color_palette("muted", len(states))
    index_of_state = {state: i for i, state in enumerate(states)}

    # nodes
    for node in LG.nodes():
        state = LG.nodes[node]["state"]
        c_rgb = color_palette[index_of_state[state]]
        s = 1.0 / len(pos) * 150 * 70
        plt.scatter(
            pos[node][0], pos[node][1], s=s, alpha=0.8,
            zorder=15, c=[c_rgb], edgecolors='none'
        )
    # edges
    lw = min(3, 1.0 / len(LG.edges()) * 600)
    for e in LG.edges:
        pos_v1 = pos[e[0]]
        pos_v2 = pos[e[1]]
        plt.plot([pos_v1[0], pos_v2[0]], [pos_v1[1], pos_v2[1]], c='black',
                 alpha=0.5, zorder=10, linewidth=lw)

    # make axis invisible:
    plt.axis("off")

    # count number of nodes in each state:
    num_nodes_in_state = {state: 0 for state in states}
    for node in LG.nodes():
        state = LG.nodes[node]["state"]
        num_nodes_in_state[state] += 1
    # legend:
    patches = []
    #####
    max_len = None
    for state in states:
        if max_len is None or len(state) > max_len:
            max_len = len(state)
    #####
    for i, state in enumerate(states):
        label = "{}{}: {}".format(
            state, " "*(max_len - len(state)), num_nodes_in_state[state]
        )
        patches.append(mpatches.Patch(color=color_palette[i], label=label))
    tupel = (0.93, 1.17)
    plt.legend(
        handles=patches, bbox_to_anchor=tupel, loc="upper left", frameon=False
    )
    fig.canvas.draw()       # draw the canvas, cache the renderer
    return fig


def plot_curves(data, color_palette, grouping=None):
    if grouping is None:
        grouping = []
    states = color_palette.keys()
    plt.figure(figsize=(7, 4), dpi=200)
    timepoints = [element[0] for element in data]
    plotted_states = set()

    for group in grouping:
        y = [0] * len(data)
        for state in group:
            plotted_states.add(state)
            for i, element in enumerate(data):
                y[i] += element[1][state]
        plt.plot(
            timepoints, y, color=color_palette[group[0]],
            label=", ".join(group)
        )

    for state in states:
        if state not in plotted_states:
            y = [element[1][state] for element in data]
            plt.plot(timepoints, y, color=color_palette[state], label=state)

    plt.legend(loc="upper right")

    plt.xlabel("Time")
    plt.ylabel("Number of nodes in state")
    plt.show()
    plt.close()


def save_curves(data, color_palette, filename):
    states = color_palette.keys()
    plt.figure(figsize=(7, 4), dpi=200)
    timepoints = [element[0] for element in data]

    for state in states:
        y = [element[1][state] for element in data]
        plt.plot(timepoints, y, color=color_palette[state], label=state)

    plt.legend(loc="upper right")

    plt.xlabel("Time")
    plt.ylabel("Number of nodes in state")
    filename = "AutomatedSavings/{}.png".format(filename)
    plt.draw()
    plt.savefig(fname=filename)
    plt.pause(10)


def plot_merged_curves(color_palette, dataset, stepsize, states=None):
    plt.figure(figsize=(7, 4), dpi=200)

    if states is None:
        states = [[key] for key in color_palette.keys()]
    for group in states:
        data = prepare_data(dataset, group)
        timepoints, min, max, avg, t, m = merge_curves(
            data, len(dataset), stepsize
        )
        plt.plot(
            timepoints, avg, color=color_palette[group[0]],
            label=", ".join(group), zorder=2
        )
        plt.fill_between(
            timepoints, min, max,
            color=color_palette[group[0]], alpha=0.3, zorder=1
        )
        plt.scatter(t, m, color='red', zorder=3)

    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Number of nodes in state")
    plt.show()
    plt.close()


def fill_around_std(color_palette, dataset, stepsize, states=None, labels={}):
    fig = plt.figure(figsize=(7, 4), dpi=200)
    results = ({}, {}, {})
    if states is None:
        states = [[key] for key in color_palette.keys()]
    for group in states:
        data = prepare_data(dataset, group)
        timepoints, avg, std = merge_curves_std(data, len(dataset), stepsize)
        # save data:
        results[0][group[0]] = timepoints
        results[1][group[0]] = avg
        results[2][group[0]] = std
        std_low = [np.max([avg[i] - std[i], 0]) for i in range(len(avg))]
        std_high = [avg[i] + std[i] for i in range(len(avg))]
        label = "$" + labels.get(group[0], group[0]) + "$"
        plt.plot(
            timepoints, avg, color=color_palette[group[0]],
            label=label, zorder=2
        )
        plt.fill_between(
            timepoints, std_low, std_high,
            color=color_palette[group[0]],
            alpha=0.25, zorder=1)

    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Number of nodes in state")
    fig.canvas.set_window_title(sys.argv[1])
    plt.show()
    plt.close()
    return results
