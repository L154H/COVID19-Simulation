import sys
import time
import os
import math
import numpy as np
import json
import zlib
from multiprocessing import Pool


def timing(f):
    def g(*args, **kwargs):
        # print("start", f)
        t = time.time()
        res = (f(*args, **kwargs))
        print(time.time() - t, f)
        return res
    return g


def prepare_data(dataset, states):
    result = []
    for index, simulation in enumerate(dataset):
        for times, counts in simulation:
            count = sum([counts[state] for state in states])
            result.append((times, count, index))
    result.sort()
    return result


def merge_curves(dataset, num_sim, stepsize):
    status = [0]*num_sim
    begin = 0
    end = 0
    min = []
    max = []
    avg = []
    timepoints = []

    for index, (time, count, sim) in enumerate(dataset):
        status[sim] = count
        if timepoints == [] or timepoints[-1] != time:
            timepoints.append(time)
        else:
            continue
        while end < len(dataset)-1 and dataset[end][0] < time + stepsize/2.0:
            end += 1
            _, c, s = dataset[end]
            status[s] = c
        while begin < len(dataset)-1 and dataset[begin][0] < time - stepsize/2:
            begin += 1

        current_min = [None] * num_sim
        current_max = [None] * num_sim
        current_sum = [None] * num_sim
        current_num = [None] * num_sim
        for i in range(end - begin + 1):
            time, count, sim = dataset[i + begin]
            if current_min[sim] is None or count < current_min[sim]:
                current_min[sim] = count
            if current_max[sim] is None or count > current_max[sim]:
                current_max[sim] = count
            # For the average calculation:
            if current_sum[sim] is not None:
                current_sum[sim] += count
                current_num[sim] += 1
            else:
                current_sum[sim] = count
                current_num[sim] = 1
        # no datapoints in interval:
        for i in range(num_sim):
            if current_sum[i] is None:
                current_sum[i] = status[i]
                current_num[i] = 1
            if current_min[i] is None:
                current_min[i] = status[i]
            if current_max[i] is None:
                current_max[i] = status[i]
        # print(current_num)
        min.append(np.min(current_min))
        max.append(np.max(current_max))
        avg.append(
            sum([current_sum[i]/current_num[i]
                 for i in range(num_sim)])
            / num_sim
        )
    m = np.max(avg)
    t = timepoints[avg.index(m)]
    return (timepoints, min, max, avg, t, m)


def merge_curves_std(dataset, num_sim, stepsize):
    status = [0]*num_sim
    begin = 0
    end = 0
    std = []
    avg = []
    timepoints = []

    for index, (time, count, sim) in enumerate(dataset):
        status[sim] = count
        if timepoints == [] or timepoints[-1] != time:
            timepoints.append(time)
        else:
            continue
        while end < len(dataset)-1 and dataset[end][0] < time + stepsize/2:
            end += 1
            _, c, s = dataset[end]
            status[s] = c
        while begin < len(dataset)-1 and dataset[begin][0] < time - stepsize/2:
            begin += 1

        current_square = [None] * num_sim
        current_sum = [None] * num_sim
        current_num = [None] * num_sim
        for i in range(end - begin + 1):
            time, count, sim = dataset[i + begin]
            # For the average calculation:
            if current_sum[sim] is not None and current_square is not None:
                current_sum[sim] += count
                current_num[sim] += 1
                current_square[sim] += count * count
            else:
                current_sum[sim] = count
                current_num[sim] = 1
                current_square[sim] = count * count
        # no datapoints in interval:
        for i in range(num_sim):
            if current_sum[i] is None and current_square[i] is None:
                current_sum[i] = status[i]
                current_num[i] = 1
                current_square[i] = status[i]*status[i]
        avg_i = [current_sum[i]/current_num[i] for i in range(num_sim)]
        square_i = [a**2 for a in avg_i]
        avg.append(sum(avg_i)/num_sim)
        std.append(np.sqrt((sum(square_i)/num_sim) - (avg[-1]**2)))
    return (timepoints, avg, std)


def clear(height):
    print("\n"*height)


def plot_progress(p):
    width, height = os.get_terminal_size()
    clear(height)
    len_bar = width - 14
    hashes = int(math.floor(len_bar*p))
    dots = len_bar - hashes
    print("+-" + "-" * len_bar + "-+-" + "-"*7 + "-+")
    print(
        "| "+"#" * hashes + "·" * dots + " | " + "{:7.2f}".format(p * 100) +
        " |"
    )
    print("+-" + "-" * len_bar + "-+-" + "-"*7 + "-+")


def get_max(filename, groups):
    with open(filename, "rb") as f:
        data = f.read()
        data = zlib.decompress(data).decode()
        dataset = json.loads(data)
    indices = {group: 0 for group in groups}
    max_indices = {group: len(dataset[0][group]) for group in groups}
    value = {group: 0 for group in groups}
    max_value = 0
    max_time = 0
    while any([indices[group] < max_indices[group] for group in groups]):
        min_time = None
        next_group = None
        for group in groups:
            if indices[group] < max_indices[group]:
                times = dataset[0][group]
                if min_time is None or times[indices[group]] < min_time:
                    min_time = times[indices[group]]
                    next_group = group
        value[next_group] = dataset[1][next_group][indices[next_group]]
        x = sum(value.values())
        if x > max_value:
            max_value = x
            max_time = dataset[0][next_group][indices[next_group]]
            copy = value.copy()
        indices[next_group] += 1
    print(copy)
    return (round(max_time, ndigits=2), round(max_value, ndigits=2))


def get_value(filename, group, time):
    with open(filename, "rb") as f:
        data = f.read()
        data = zlib.decompress(data).decode()
        dataset = json.loads(data)
    times, avg = dataset[0][group], dataset[1][group]
    for index, t in enumerate(times):
        if t > time:
            break
    i = np.max([index-1, 0])
    m = avg[i]
    return (times[i], round(m, ndigits=3))


def getfilename(tupel):
    return f"Results/{sys.argv[1]} {tupel}"


def save_data(data, arguments):
    with open(getfilename(arguments), "wb") as f:
        resultsasbit = json.dumps(data).encode()
        print("not compressed: ", len(resultsasbit))
        restultscompressed = zlib.compress(resultsasbit, level=9)
        print("compressed: ", len(restultscompressed))
        f.write(restultscompressed)


def get_results(simulation, arguments, num_sim):
    with Pool() as p:
        data = p.starmap(simulation, [arguments] * num_sim)
    return data


def complete_group(groups, states):
    for i, state in enumerate(states):
        if not any([state in group for group in groups]):
            groups.insert(i, [state])


if __name__ == "__main__":
    group = ["Q1", "Q2", "Q3", "I1", "I2", "I3"]
    t, m = get_max(sys.argv[1], sys.argv[2:])
    print(f"timepoint: {t}, maxvalue: {m}")
