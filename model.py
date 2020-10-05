import numpy as np
import random
from events import ModelEvent
from typing import Dict, List, Tuple


########################################################
# Superclass for any spreading model                   #
########################################################

class SpreadingModel:
    """
    Superclass for any spreading model
    """
    def generate_event(self, LG, node, global_clock):
        raise NotImplementedError("Implement generate_event!")

    def perform_statuschanges(self, LG, node, new_state):
        LG.nodes[node]["state"] = new_state

    def check_condition(self, LG):
        return True


########################################################
# Class for Fitting a Model with parameters            #
########################################################
class FittingModel(SpreadingModel):
    """
    Creates a model by getting the underlying epidemiological model
    as a list of edges
    """

    def __init__(self, states, edges,
                 infectious_states: Dict[str, Tuple[List[str], List[float]]],
                 susceptible_states: Dict[str, Tuple[List[str], List[float]]],
                 causer_states=None, scale_by_mean_degree=True):
        SpreadingModel.__init__(self)
        self.states = states
        self.edges = edges
        if causer_states is None:
            self.causer_states = {}
        else:
            self.causer_states = causer_states
        self.scale_by_mean_degree = scale_by_mean_degree
        self.connected_states = {state: [] for state in states}
        for (state1, rate, state2) in edges:
            self.connected_states[state1].append((rate, state2))
        # print("Connected States", self.connected_states)
        self.susceptible_states = susceptible_states
        self.infectious_states = infectious_states

    def init_states(self, LG, number_of_seeds):
        # give all nodes a susceptible state
        for node in LG.nodes():
            cat = LG.category_of[node]
            states, p = self.susceptible_states[cat]
            state = np.random.choice(states, p=p)
            LG.nodes[node]["state"] = state
        #
        nodes = np.random.choice(LG.nodes(), number_of_seeds, replace=False)
        for node in nodes:
            cat = LG.category_of[node]
            states, p = self.infectious_states[cat]
            state = np.random.choice(states, p=p)
            LG.nodes[node]["state"] = state

    def generate_event(self, LG, node, global_clock):
        """
        Optional: None zurückgeben wenn es kein neues Event gibt
        """
        current_state = LG.nodes[node]["state"]
        min_firetime = None
        new_state = None
        causer = None
        count_states = {state: 0 for state in self.states}
        neighbors_of_state = {state: [] for state in self.states}

        for neighbor in LG.neighbors(node):
            count_states[LG.nodes[neighbor]["state"]] += 1
            neighbors_of_state[LG.nodes[neighbor]["state"]].append(neighbor)

        if self.scale_by_mean_degree:
            mean_degree = (2 * len(LG.edges())) / LG.number_of_nodes()
            count_states = {
                state: val/mean_degree for state, val in count_states.items()
            }

        for (rate, state2) in self.connected_states[current_state]:
            denominator = rate(count_states)
            if denominator != 0:
                curr_firetime = -np.log(random.random()) / denominator
                if min_firetime is None or curr_firetime < min_firetime:
                    min_firetime = curr_firetime
                    new_state = state2
        if new_state is None:
            return None

        if new_state in self.causer_states:
            causer = []
            for c_state in self.causer_states[new_state]:
                causer.extend(neighbors_of_state[c_state])

        new_time = global_clock + min_firetime
        return ModelEvent(new_time, node, new_state, causer)


########################################################
# CoronaHillModel                                      #
########################################################
class CoronaHillModel(FittingModel):
    """
    Model from
    https://www.medrxiv.org/content/early/2020/06/05/2020.06.04.20121673
    fitted by Hill et al.
    For an overview we refer to: https://alhill.shinyapps.io/COVID19seir/
    """
    def __init__(self):
        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050
        states = ['S', 'E', 'I1', 'I2', 'I3', 'R', 'D', 'Q']
        f = linearcombination([(b1, "I1"), (b2, "I2"), (b3, "I3")])
        edges = [("S", f, "E"),
                 ("E", constant(a), "I1"),
                 ("I1", constant(g1), "R"),
                 ("I2", constant(g2), "R"),
                 ("I3", constant(g3), "R"),
                 ("I1", constant(p1), "I2"),
                 ("I2", constant(p2), "I3"),
                 ("I3", constant(u), "D")]
        causer_states = {"E": ["I1", "I2", "I3"]}
        s_states = {
            "Kids": (["S"], [1]),
            "Normal": (["S"], [1]),
            "Risk": (["S"], [1])
        }
        i_states = {
            "Kids": (["I1"], [1]),
            "Normal": (["I2"], [1]),
            "Risk": (["I3", "I2"], [0.5, 0.5])
        }
        super().__init__(
            states, edges, i_states, s_states, causer_states, False
        )


#########################################################
class AgeModel(FittingModel):
    """
    Model including age structure
    """
    def __init__(self, b_kids, g_kids, p_kids, b_normal, g_normal, p_normal,
                 b_risk, g_risk, p_risk):
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p2 = 0.042  # i2 to i3
        u = 0.050
        states = ['S_kids', 'S_normal', 'S_risk',
                  'E_kids', 'E_normal', 'E_risk',
                  'I_kids', 'I_normal', 'I_risk', 'I2', 'I3',
                  'R', 'D', 'Q']
        f = linearcombination([
            (b_kids, "I_kids"), (b_normal, "I_normal"), (b2, "I2"),
            (b3, "I3"), (b_risk, "I_risk")
        ])
        edges = [("S_kids", f, "E_kids"),
                 ("S_normal", f, "E_normal"),
                 ("S_risk", f, "E_risk"),
                 ("E_kids", constant(a), "I_kids"),
                 ("E_normal", constant(a), "I_normal"),
                 ("E_risk", constant(a), "I_risk"),
                 ("I_risk", constant(p_risk), "I2"),
                 ("I_risk", constant(g_risk), "R"),
                 ("I_kids", constant(g_kids), "R"),
                 ("I_kids", constant(p_kids), "I2"),
                 ("I_normal", constant(g_normal), "R"),
                 ("I2", constant(g2), "R"),
                 ("I3", constant(g3), "R"),
                 ("I_normal", constant(p_normal), "I2"),
                 ("I2", constant(p2), "I3"),
                 ("I3", constant(u), "D")]
        causer_states = {"E": ["I_normal", "I2", "I3"]}
        s_states = {
            "Kids": (["S_kids"], [1]),
            "Normal": (["S_normal"], [1]),
            "Risk": (["S_risk"], [1])
        }
        i_states = {
            "Kids": (["I_kids"], [1]),
            "Normal": (["I_normal"], [1]),
            "Risk": (["I2"], [1])
        }
        super().__init__(
            states, edges, i_states, s_states, causer_states, False
        )


##########################################################
class CoronaQuarantineModel(FittingModel):
    """
    Model including quarantine.
    Infectious nodes go to isolation with the rates q1, q2, q3
    """
    def __init__(self, q1, q2, q3):
        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050
        states = ["S", "E", "I1", "I2", "I3", "R", "D", "Q1", "Q2", "Q3"]
        edges = [("S", linearcombination([(b1, "I1"), (b2, "I2"), (b3, "I3")]),
                  "E"),
                 ("E", constant(a), "I1"),
                 ("I1", constant(g1), "R"),
                 ("I2", constant(g2), "R"),
                 ("I3", constant(g3), "R"),
                 ("I1", constant(p1), "I2"),
                 ("I2", constant(p2), "I3"),
                 ("I3", constant(u), "D"),
                 ("I1", constant(q1), "Q1"),
                 ("Q1", constant(g1), "R"),
                 ("Q1", constant(p1), "Q2"),
                 ("I2", constant(q2), "Q2"),
                 ("Q2", constant(g2), "R"),
                 ("Q2", constant(p2), "Q3"),
                 ("I3", constant(q3), "Q3"),
                 ("Q3", constant(g3), "R"),
                 ("Q3", constant(u), "D")]
        causer_states = {"E": ["I1", "I2", "I3"]}
        s_states = {
            "Kids": (["S"], [1]),
            "Normal": (["S"], [1]),
            "Risk": (["S"], [1])
        }
        i_states = {
            "Kids": (["I1"], [1]),
            "Normal": (["I2"], [1]),
            "Risk": (["I3", "I2"], [0.5, 0.5])
        }
        super().__init__(states, edges, i_states, s_states, causer_states,
                         False)


class QuorontracingModel(FittingModel):
    """
    Model allows an intervention to put nodes into quarantine.
    Note that there a no transitions from the infectious states to
    the quarantine states. This is the difference to the previous model.
    """
    def __init__(self, q1=0, q2=0, q3=0):
        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050
        states = [
            "S", "E",
            "I1", "I2", "I3",
            "Q1", "Q2", "Q3", "QS", "QE",
            "R", "D"
        ]
        edges = [("S", linearcombination([(b1, "I1"), (b2, "I2"), (b3, "I3")]),
                  "E"),
                 ("E", constant(a), "I1"),
                 ("QE", constant(a), "Q1"),
                 ("I1", constant(g1), "R"),
                 ("I2", constant(g2), "R"),
                 ("I3", constant(g3), "R"),
                 ("I1", constant(p1), "I2"),
                 ("I2", constant(p2), "I3"),
                 ("I3", constant(u), "D"),
                 ("I1", constant(q1), "Q1"),
                 ("Q1", constant(g1), "R"),
                 ("Q1", constant(p1), "Q2"),
                 ("I2", constant(q2), "Q2"),
                 ("Q2", constant(g2), "R"),
                 ("Q2", constant(p2), "Q3"),
                 ("I3", constant(q3), "Q3"),
                 ("Q3", constant(g3), "R"),
                 ("Q3", constant(u), "D")]
        causer_states = {"E": ["I1", "I2", "I3"]}
        s_states = {
            "Kids": (["S"], [1]),
            "Normal": (["S"], [1]),
            "Risk": (["S"], [1])
        }
        i_states = {
            "Kids": (["I1"], [1]),
            "Normal": (["I2"], [1]),
            "Risk": (["I3", "I2"], [0.5, 0.5])
        }
        super().__init__(states, edges, i_states, s_states, causer_states,
                         False)


#########################################################
# usefull functions                                     #
#########################################################
def constant(c):
    def f(_, c=c):
        return c
    return f


def linearcombination(factors):
    def f(population, factors=factors):
        return sum([factor * population[state] for factor, state in factors])
    return f
