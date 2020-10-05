from events import InterventionEvent
import random


########################################################
# Superclass for any intervention
########################################################
class Intervention:
    """
    Superclass for any intervention
    """
    def check_condition(self, LG):
        return True

    def check_precond_and_generate_event(self, LG, global_clock):
        raise NotImplementedError("Implement check_precond_and_generate_event")

    def perform_intervention(self, LG):
        raise NotImplementedError("Implement perform_intervention")


class TimedIntervention(Intervention):
    """
    Superclass for any timed intervention
    """
    def __init__(self, time):
        super().__init__()
        self.time = time

    def check_precond_and_generate_event(self, LG, global_clock):
        event = InterventionEvent(self.time, self)
        return event


class UpdateLayerIntervention(TimedIntervention):
    """
    Intervention that operates only on layers
    """
    def __init__(self, p, layer_name, time):
        super().__init__(time)
        self.p = p
        self.layer_name = layer_name

    def perform_intervention(self, LG):
        LG.update_layer(self.layer_name, self.p)
        return LG.nodes()


class AddLayerIntervention(TimedIntervention):
    """
    Intervention that adds layer
    """
    def __init__(self, layer, time):
        super().__init__(time)
        self.layer = layer

    def perform_intervention(self, LG):
        LG.add_layer(self.layer)
        return LG.nodes()


class UpdateMultipleLayerIntervention(TimedIntervention):
    """
    Update multiple interventions on p
    layer_dict layer_name : p
    """
    def __init__(self, layer_dict, time):
        super().__init__(time)
        self.layer_dict = layer_dict

    def perform_intervention(self, LG):
        for layer_name, p in self.layer_dict.items():
            # print("LG.update_layer()", layer_name)
            LG.update_layer(layer_name, p)
        return LG.nodes()


class ChangeStateIntervention(TimedIntervention):
    """
    Changes state of nodes
    inf_states_p state : (p, new_state)
    """
    def __init__(self, state_p, time):
        super().__init__(time)
        self.state_p = state_p

    def perform_intervention(self, LG):
        changed_nodes = []
        for node in LG.nodes():
            state = LG.nodes[node]["state"]
            if state in self.state_p:
                p, new_state = self.state_p[state]
                r = random.random()
                if r < p:
                    LG.nodes[node]["state"] = new_state
                    changed_nodes.append(node)
        return changed_nodes


class ContactTracingIntervention(TimedIntervention):
    """
    Intervention that performs contact tracing.
    """
    def __init__(self, group, mapping, p, time):
        super().__init__(time)
        hops = len(p) - 1  # depth of contact tracing 0 = just isolation
        self.group = group
        self.mapping = mapping
        self.hops = hops
        self.p = p

    def perform_intervention(self, LG):
        changed_nodes = set()
        nodes_in_state = [
            node for node in LG.nodes()
            if LG.nodes[node]["state"] in self.group
        ]
        current_nodes = self.random_choice(nodes_in_state, self.p[0])
        self.update_state(LG, current_nodes)
        changed_nodes.update(current_nodes)

        for i in range(self.hops):
            new_nodes = set()
            for current_node in current_nodes:
                neighbors = LG.neighbors(current_node)
                new_neighbors = self.random_choice(neighbors, self.p[i+1])
                for n in new_neighbors:
                    new_nodes.add(n)
            self.update_state(LG, new_nodes)
            changed_nodes.update(new_nodes)
            current_nodes = new_nodes

        return list(changed_nodes)

    def update_state(self, LG, nodes):
        for node in nodes:
            state = LG.nodes[node]["state"]
            LG.nodes[node]["state"] = self.mapping[state]

    def random_choice(self, items, p):
        return [item for item in items if random.random() < p]


class InfectiousTracingIntervention(TimedIntervention):
    """
    Intervention that performs contact tracing recursively and isolates only
    infectious nodes.
    """
    def __init__(self, group, mapping, p_start, p_contact, time):
        super().__init__(time)
        self.group = group
        self.mapping = mapping
        self.p_start = p_start
        self.p_contact = p_contact

    def perform_intervention(self, LG):
        changed_nodes = set()
        nodes_in_state = [
            node for node in LG.nodes()
            if LG.nodes[node]["state"] in self.group
        ]
        current_nodes = self.random_choice(nodes_in_state, self.p_start)
        self.update_state(LG, current_nodes)
        changed_nodes.update(current_nodes)
        while current_nodes:
            new_nodes = set()
            for current_node in current_nodes:
                neighbors = LG.neighbors(current_node)
                new_neighbors = self.random_choice(neighbors, self.p_contact)
                for n in new_neighbors:
                    state = LG.nodes[n]["state"]
                    if state in self.group:
                        new_nodes.add(n)
            self.update_state(LG, new_nodes)
            changed_nodes.update(new_nodes)
            current_nodes = new_nodes
        return list(changed_nodes)

    def update_state(self, LG, nodes):
        for node in nodes:
            state = LG.nodes[node]["state"]
            LG.nodes[node]["state"] = self.mapping[state]

    def random_choice(self, items, p):
        return [item for item in items if random.random() < p]
