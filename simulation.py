import heapq
from output import Output
from events import InterventionEvent, ModelEvent


def simulate(LG, model, interventions, max_time, observers=None):
    """
    One simulation run with the model model on the multilayered graph LG
    max_time denotes the end of the simulation
    observers get notified when events happen
    """
    output = Output()
    global_clock = 0.0
    PQ = PriorityQueue()
    if observers is None:
        observers = []

    for observer in observers:
        observer.start(LG)

    # Initialize Eventqueue:
    for node in LG.nodes():
        # Initialize model events:
        push_model_event(PQ, LG, node, global_clock, model)

    # Initialize intervention events:
    check_and_push_interventions(PQ, LG, global_clock, interventions)

    # Handle events in queue:
    while global_clock <= max_time and not PQ.is_empty():
        event, valid = PQ.pop_event()
        global_clock = event.time
        # Model event:
        if valid and isinstance(event, ModelEvent):
            if model.check_condition(LG):
                for observer in observers:
                    observer.before_model_event(LG, event)
                model.perform_statuschanges(LG, event.node, event.state)
                for observer in observers:
                    observer.after_model_event(LG, event)
                update_nodes(PQ, LG, [event.node], global_clock, model)
                output.add_event(event)
        # Intervention event:
        elif valid and isinstance(event, InterventionEvent):
            if event.intervention.check_condition(LG):
                for observer in observers:
                    observer.before_intervention_event(LG, event)
                changed_nodes = event.intervention.perform_intervention(LG)
                for observer in observers:
                    observer.after_intervention_event(LG, event)
                update_nodes(PQ, LG, changed_nodes, global_clock, model)
                output.add_event(event)
        elif not valid:
            push_model_event(PQ, LG, event.node, global_clock, model)

        # check for new possible interventions:
        check_and_push_interventions(PQ, LG, global_clock, interventions)

    for observer in observers:
        observer.end(LG)

    return output

########################################################
# Help Functions                                       #
########################################################


def update_nodes(PQ, LG, changed_nodes, global_clock, model):
    for node in changed_nodes:
        push_model_event(PQ, LG, node, global_clock, model)
        for neighbor in LG.neighbors(node):
            push_model_event(PQ, LG, neighbor, global_clock, model)


def transform(event, id):
    return(event.time, id, event)


def push_model_event(PQ, LG, node, global_clock, model):
    event = model.generate_event(LG, node, global_clock)
    PQ.push_model_event(event, node)


def check_and_push_interventions(PQ, LG, global_clock, interventions):
    used_interventions = []
    for intervention in interventions:
        event = intervention.check_precond_and_generate_event(LG, global_clock)
        if event is not None:
            PQ.push_intervention_event(event)
            used_interventions.append(intervention)
    # remove interventions who have been added to the queue:
    for intervention in used_interventions:
        interventions.remove(intervention)

########################################################
# PriorityQueue                                        #
########################################################


class PriorityQueue():
    """
    implements a priority queue as binary heap
    used to store the events
    """
    def __init__(self):
        self.event_queue = []
        self.current_event_id = {}

    def push_model_event(self, event, node):
        event_id = self.get_event_id(node)
        self.current_event_id[node] += 1
        if event is not None:
            heapq.heappush(self.event_queue, transform(event, event_id + 1))

    def push_intervention_event(self, event):
        if event is not None:
            heapq.heappush(self.event_queue, transform(event, None))

    def get_event_id(self, node):
        if node in self.current_event_id:
            event_id = self.current_event_id[node]
        else:
            self.current_event_id[node] = 0
            event_id = 0
        return event_id

    def pop_event(self):
        """
        returns event, valid
        """
        time, event_id, event = heapq.heappop(self.event_queue)
        if event_id is None or event_id == self.current_event_id[event.node]:
            return event, True
        else:
            return event, False

    def is_empty(self):
        return self.event_queue == []
