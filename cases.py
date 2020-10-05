import matplotlib.pyplot as plt
import numpy as np
import sys
from multiprocessing import Pool
from visualization import save_curves, fill_around_std
from simulation import simulate
from observer import CountObserver
from setup import setup_CH, setup_LG
from setup import color_palette, qolor_palette, quolor_palette, kolor_palette
from util import get_results, prepare_data, merge_curves, save_data
from util import complete_group
from model import CoronaQuarantineModel, QuorontracingModel, AgeModel
from interventions import UpdateMultipleLayerIntervention
from interventions import ContactTracingIntervention
from interventions import InfectiousTracingIntervention


N = 100
NUM_SIM = 10

# Kids
B_KIDS = 0.6
G_KIDS = 0.135
P_KIDS = 0.01
# Normal
B_NORMAL = 0.05
G_NORMAL = 0.1
P_NORMAL = 0.033
# Risk
B_NORMAL = 0.007
G_NORMAL = 0.05
P_NORMAL = 0.06


###############################################################################
# no_intervention                                                             #
###############################################################################

def simulation_no_intervention(n):
    LG, CH = setup_CH(n)
    c_observer = CountObserver(color_palette.keys())
    CH.init_states(LG, 5)
    simulate(LG, CH, set(), 50, [c_observer])
    return c_observer.data


def case_no_intervention():
    label = {
        "I1": "I_1",
        "I2": "I_2",
        "I3": "I_3"
    }
    data = get_results(simulation_no_intervention, (N,), NUM_SIM)
    times, avg, std = fill_around_std(color_palette, data, 1, labels=label)
    save_data([times, avg, std], (N, NUM_SIM))


###############################################################################
# contact_tracing                                                             #
###############################################################################


def simulation_contact_tracing(n, p, times):
    interventions = []
    mapping = {
        'S':  'QS',
        'E':  'QE',
        'I1': 'Q1',
        'I2': 'Q2',
        'I3': 'Q3',
        'R':  'R',
        'D':  'D',
        'Q1': 'Q1',
        'Q2': 'Q2',
        'Q3': 'Q3',
        'QS': 'QS',
        'QE': 'QE'
    }
    for time in times:
        interventions.append(
            ContactTracingIntervention(["I1", "I2", "I3"], mapping, p, time,)
        )
    LG = setup_LG(n)
    M = QuorontracingModel(0, 0, 0)
    c_observer = CountObserver(quolor_palette.keys())
    observers = [c_observer]
    M.init_states(LG, 5)
    simulate(LG, M, interventions, 50, observers)  # scale x -axis
    return c_observer.data


def contact_tracing(p):
    time = [5, 7.5, 10]
    data = get_results(simulation_contact_tracing, (N, p, time), NUM_SIM)
    states = [
        ["S"], ["E"], ["I1"], ["I2"], ["I3"], ["R"], ["D"], ["Q1", "Q2", "Q3"],
        ["QS", "QE"]
    ]
    labels = {
        "Q1": "Q_I", "QS": "Q_{S, E}",
        "I1": "I_1", "I2": "I_2", "I3": "I_3"
    }
    times, avg, std = fill_around_std(
        quolor_palette, data, 1, states=states, labels=labels
    )
    save_data([times, avg, std], (N, NUM_SIM, time, p))


def case_contact_tracing0():
    contact_tracing([1])


def case_contact_tracing1():
    contact_tracing([0.7, 0.5])


def case_contact_tracing2():
    contact_tracing([0.7, 0.5, 0.5])


def simulation_infectious_tracing(n, p_start, p_contact, times):
    interventions = []
    mapping = {
        'S':  'QS',
        'E':  'QE',
        'I1': 'Q1',
        'I2': 'Q2',
        'I3': 'Q3',
        'R':  'R',
        'D':  'D',
        'Q1': 'Q1',
        'Q2': 'Q2',
        'Q3': 'Q3',
        'QS': 'QS',
        'QE': 'QE'
    }
    for time in times:
        interventions.append(
            InfectiousTracingIntervention(
                ["I1", "I2", "I3"], mapping, p_start, p_contact, time
            )
        )
    LG = setup_LG(n)
    M = QuorontracingModel(0, 0, 0)
    c_observer = CountObserver(quolor_palette.keys())
    observers = [c_observer]
    M.init_states(LG, 5)
    simulate(LG, M, interventions, 50, observers)  # scale x -axis
    return c_observer.data


def case_infectious_tracing():
    time = [5, 7.5, 10]
    p = [0.7, 1]
    data = get_results(
        simulation_infectious_tracing, (N, p[0], p[1], time),
        NUM_SIM,

    )
    states = [
        ["S"], ["E"], ["I1"], ["I2"], ["I3"], ["R"], ["D"], ["Q1", "Q2", "Q3"],
        ["QS", "QE"]
    ]
    labels = {
        "Q1": "Q_{I}", "QS": "Q_{S, E}",
        "I1": "I_{1}", "I2": "I_{2}", "I3": "I_{3}"
    }
    times, avg, std = fill_around_std(
        quolor_palette, data, 1, states=states, labels=labels
    )
    save_data([times, avg, std], (N, NUM_SIM, time, p))


###############################################################################
# quarantine_rates                                                            #
###############################################################################


def simulation_max_quarantine_rates(n, q1, q2, q3):
    LG = setup_LG(n)
    CH = CoronaQuarantineModel(q1, q2, q3)
    # Observer
    c_observer = CountObserver(qolor_palette.keys())
    observers = [c_observer]
    # Simulation
    CH.init_states(LG, 5)
    simulate(LG, CH, set(), 50, observers)
    return c_observer.data


def case_average_quarantine_rates():
    q1 = 0.5
    output = get_results(
        simulation_max_quarantine_rates, (N, q1, 0, 0), NUM_SIM
    )
    labels = {
        "I1": "I_1", "I2": "I_2", "I3": "I_3",
        "Q1": "Q", "Q2": "Q_2", "Q3": "Q_3",
    }
    groups = [
        ["Q1", "Q2", "Q3"]
    ]
    complete_group(groups, qolor_palette)
    timepoints, avg, std = fill_around_std(
        qolor_palette, output, 1, labels=labels, states=groups
    )
    save_data([timepoints, avg, std], (N, NUM_SIM, q1))


###############################################################################
# case_max_time                                                               #
###############################################################################


def simulation_max_time(n, t):
    LG, CH = setup_CH(n)
    # Interventions
    layernames_to_p = {
        "Households": 1, "Schools": 0, "Workplaces": 0.01,
        "R_Workplaces": 0, "Social": 0, "parties": 0.00, "basic": 0.1}
    um_intervention1 = UpdateMultipleLayerIntervention(layernames_to_p, t)
    # Observer
    c_observer = CountObserver(color_palette.keys())
    observers = [c_observer]
    # Simulation
    CH.init_states(LG, 5)
    simulate(LG, CH, {um_intervention1}, 50, observers)
    return c_observer.data


def case_average_time():
    t = 5
    output = get_results(
        simulation_max_time, (N, t), NUM_SIM
    )
    timepoints, avg, std = fill_around_std(
        color_palette, output, 1
    )
    save_data([timepoints, avg, std], (N, NUM_SIM, t))


def case_max_time():
    time = np.linspace(0, 15, 30)
    group = ["I3", "I1", "I2"]
    arguments = [(simulation_max_time, (N, t), group) for t in time]
    with Pool() as p:
        y = p.starmap(max_sim, arguments)
    save_data([time.tolist(), y], (N, NUM_SIM))
    plt.xlabel("point in time social distancing takes place")
    plt.ylabel("maximum count of infected people")
    plt.scatter(time, y, c="red")
    plt.show()


def max_sim(simulation, arguments, group):
    dataset = [simulation(*arguments) for i in range(NUM_SIM)]
    data = prepare_data(dataset, group)
    _, _, _, _, _, m = merge_curves(data, len(dataset), 1)
    return m


###############################################################################
# half_edges                                                                  #
###############################################################################


def simulation_half_edges():
    LG, CH = setup_CH(N, p=0.5)
    c_observer = CountObserver(color_palette.keys())
    CH.init_states(LG, 5)
    simulate(LG, CH, set(), 50, [c_observer])
    return c_observer.data


def case_half_edges():
    data = get_results(simulation_half_edges, tuple(), NUM_SIM)
    labels = {
        "I1": "I_{1}", "I2": "I_{2}", "I3": "I_{3}"
    }
    times, avg, std = fill_around_std(
        color_palette, data, 1, labels=labels
    )
    save_data([times, avg, std], (N, NUM_SIM))


def case_max_half_edges():
    percentage = np.linspace(0, 1, 21)
    y = []
    group = ["I3", "I1", "I2"]
    arguments = [(simulation_max_time, (N, p), group) for p in percentage]
    with Pool() as p:
        y = p.starmap(max_sim, arguments)
    save_data([percentage.tolist(), y], (N, NUM_SIM))
    plt.xlabel("percentage of active edges")
    plt.ylabel("maximum count of infected people")
    plt.scatter(percentage, y, c="red")
    plt.gcf().canvas.set_window_title(sys.argv[1])
    plt.show()


###############################################################################
# second_wave                                                                 #
###############################################################################

def simulation_second_wave():
    LG, CH = setup_CH(N)
    # Interventions
    layernames_to_p1 = {
        "Households": 0.5, "Schools": 0, "Workplaces": 0.01,
        "R_Workplaces": 0, "Social": 0.01, "parties": 0.00,
        "basic": 0.01
    }
    layernames_to_p2 = {
        "Households": 1, "Schools": 1, "Workplaces": 1,
        "R_Workplaces": 1, "Social": 1, "parties": 1,
        "basic": 1
    }
    um_intervention1 = UpdateMultipleLayerIntervention(layernames_to_p1, 6)
    # 7.8
    um_intervention2 = UpdateMultipleLayerIntervention(layernames_to_p2, 28.7)
    # 29.7
    # Observer
    c_observer = CountObserver(color_palette.keys())
    # Simulation
    CH.init_states(LG, 5)
    interventions = {um_intervention1, um_intervention2}
    simulate(LG, CH, interventions, 50, [c_observer])
    return c_observer.data


def case_average_second_wave():
    data = get_results(simulation_second_wave, tuple(), NUM_SIM)
    labels = {
        "I1": "I_{1}", "I2": "I_{2}", "I3": "I_{3}"
    }
    times, avg, std = fill_around_std(
        color_palette, data, 1, labels=labels
    )
    save_data([times, avg, std], (N, NUM_SIM))


###############################################################################
# class_size                                                                  #
###############################################################################


def simulation_max_class(n, class_size, degree, p):
    if degree is None and p is None:
        LG, CH = setup_CH(n, classsize=class_size)
    else:
        LG, CH = setup_CH(n, classsize=class_size, degree=degree, p=p)
    CH.init_states(LG, 5)
    c_observer = CountObserver(color_palette.keys())
    simulate(LG, CH, set(), 50, [c_observer])
    return c_observer.data


def case_max_class():
    class_size = [5, 10, 15, 20, 25, 30]
    group = ["I3", "I1", "I2"]
    arguments = [
        (simulation_max_class, (N, cs, None, None), group) for cs in class_size
    ]
    with Pool() as p:
        y = p.starmap(max_sim, arguments)
    save_data([class_size, y], (N, NUM_SIM))
    plt.xlabel("average class size")
    plt.ylabel("maximum count of infected people")
    plt.scatter(class_size, y, c="red")
    plt.show()


def case_max_class_mod():
    class_sizes = [5, 10, 15, 20, 25, 30]
    degree = [1.25, 2.5, 3.75, 5, 6.25, 7.5]
    p = 0.3
    group = ["I3", "I1", "I2"]
    arguments = [
        (simulation_max_class, (N, cs, d, p), group)
        for cs, d in zip(class_sizes, degree)
    ]
    with Pool() as p:
        y = p.starmap(max_sim, arguments)
    save_data([class_sizes, y], (N, NUM_SIM))
    plt.xlabel("average class size")
    plt.ylabel("maximum count of infected people")
    plt.scatter(class_sizes, y, c="red")
    plt.show()


###############################################################################
# age_model                                                                   #
###############################################################################


def simulation_age_model(n, b_kids, g_kids, p_kids, b_normal, g_normal, p_normal, b_risk, g_risk, p_risk):
    LG = setup_LG(n)
    AM = AgeModel(b_kids, g_kids, p_kids, b_normal, g_normal, p_normal,
                  b_risk, g_risk, p_risk)
    c_observer = CountObserver(kolor_palette.keys())
    AM.init_states(LG, 5)
    simulate(LG, AM, set(), 50, [c_observer])
    return c_observer.data


def case_age_model():
    group = [
        ['S_kids', 'S_normal', 'S_risk'],
        ['E_kids', 'E_normal', 'E_risk'],
        ['I_kids', 'I_normal', 'I_risk'],
        ['I2'],
        ['I3'],
        ['R'],
        ['D']
    ]
    label = {
        "S_kids": "S",
        "E_kids": "E",
        "I_kids": "I_1",
        "I2": "I_2",
        "I3": "I_3"
    }
    data = get_results(
        simulation_age_model, (N, B_KIDS, G_KIDS, P_KIDS, B_NORMAL, G_NORMAL,
                               P_NORMAL, B_NORMAL, G_NORMAL, P_NORMAL), NUM_SIM
    )
    times, avg, std = fill_around_std(
        kolor_palette, data, 1, states=group, labels=label
    )
    save_data([times, avg, std], (N, NUM_SIM,))


def simulation_age_max_class(n, class_size, degree, p):
    if degree is None and p is None:
        LG = setup_LG(n, classsize=class_size)
    else:
        LG = setup_LG(n, classsize=class_size, degree=degree, p=p)
    AM = AgeModel(B_KIDS, G_KIDS, P_KIDS, B_NORMAL, G_NORMAL, P_NORMAL,
                  B_NORMAL, G_NORMAL, P_NORMAL)
    AM.init_states(LG, 5)
    c_observer = CountObserver(kolor_palette.keys())
    simulate(LG, AM, set(), 50, [c_observer])
    return c_observer.data


def case_age_max_class_mod():
    class_sizes = [5,    10,  15,   20, 25,   30]
    degree = [1.25, 2.5, 3.75, 5,  6.25, 7.5]
    p = 0.3
    group = ['I_kids', 'I_normal', 'I_risk', "I2", "I3"]
    arguments = [
        (simulation_age_max_class, (N, cs, d, p), group)
        for cs, d in zip(class_sizes, degree)
    ]
    with Pool() as p:
        y = p.starmap(max_sim, arguments)
    save_data([class_sizes, y],
              (N, NUM_SIM, B_KIDS, G_KIDS, P_KIDS, B_NORMAL, G_NORMAL,
               P_NORMAL, B_NORMAL, G_NORMAL, P_NORMAL))
    plt.xlabel("average class size")
    plt.ylabel("maximum count of infected people")
    plt.scatter(class_sizes, y, c="red")
    plt.show()
