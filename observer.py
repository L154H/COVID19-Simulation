import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


########################################################
# Observer                                             #
########################################################
class Observer():
    """
    superclass for any observer
    """
    def start(self, LG):
        pass

    def end(self, LG):
        pass

    def before_model_event(self, LG, event):
        pass

    def after_model_event(self, LG, event):
        pass

    def before_intervention_event(self, LG, event):
        pass

    def after_intervention_event(self, LG, event):
        pass


########################################################
# CountObserver                                        #
########################################################
class CountObserver(Observer):
    """
    observers the number of nodes in each state
    """
    def __init__(self, states):
        super().__init__()
        self.data = []
        self.states = states

    def start(self, LG):
        self.initialize_counts(LG, 0)

    def before_model_event(self, LG, event):
        _, count = self.data[-1]
        new_count = count.copy()
        new_count[LG.nodes[event.node]["state"]] -= 1
        new_count[event.state] += 1
        self.data.append((event.time, new_count))

    def after_intervention_event(self, LG, event):
        self.initialize_counts(LG, event.time)

    def initialize_counts(self, LG, time):
        counts = {state: 0 for state in self.states}
        for node in LG.nodes():
            counts[LG.nodes[node]["state"]] += 1
        self.data.append((time, counts))


########################################################
# InfectiousObserver                                   #
########################################################
class InfectiousObserver(Observer):
    """
    observers the duration and the number of secondary infections
    during infectious period.
    """
    def __init__(self, infect_states):
        super().__init__()
        self.infect_states = infect_states
        self.data = {}

    def start(self, LG):
        for node in LG.nodes():
            self.check_state_and_update(LG, node, 0)

    def before_model_event(self, LG, event):
        if LG.nodes[event.node]["state"] not in self.infect_states and \
           event.state in self.infect_states:
            if event.causer is not None:
                causer = np.random.choice(event.causer)
            else:
                print(LG.nodes[event.node]["state"])
                print(event)
            self.data[causer]["number_infected"] += 1

    def after_model_event(self, LG, event):
        self.check_state_and_update(LG, event.node, event.time)

    def after_intervention_event(self, LG, event):
        for node in LG.nodes():
            self.check_state_and_update(LG, node, event.time)

    def check_state_and_update(self, LG, node, time):
        if LG.nodes[node]["state"] in self.infect_states:
            if node not in self.data:
                self.data[node] = {
                    "start": time,
                    "end": float("inf"),
                    "number_infected": 0
                }
        else:
            if node in self.data:
                current = self.data[node]["end"]
                self.data[node]["end"] = min(current, time)

    def plot_one_interval(self, x, radius, size, res_t, res_R, res_v, R,
                          variance):
        group = []
        for node in self.data:
            start = self.data[node]["start"]
            end = self.data[node]["end"]
            if x-radius <= (start + end) / 2 <= x+radius:
                group.append(self.data[node]["number_infected"])
        print(len(group))
        if len(group) < len(self.data) * size:
            return

        res_t.append(x)
        if R:
            res_R.append(np.mean(group))
        if variance:
            res_v.append(np.var(group))

    def plot_results(self, timepoints, radius, size=0.1, R=True,
                     variance=True):
        plt.figure(figsize=(7, 4), dpi=200)
        plt.xlabel("Time")
        plt.ylabel("$R_t$")

        res_t = []
        res_R = []
        res_v = []
        for t in timepoints:
            self.plot_one_interval(
                t, radius, size, res_t, res_R, res_v, R, variance
            )
        if R:
            res_R = savgol_filter(res_R, 7, 3)
            plt.plot(res_t, res_R, "r-")
        if variance:
            plt.plot(res_t, res_v, "b-")
        plt.show()
