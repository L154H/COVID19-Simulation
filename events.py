from collections import namedtuple

"""
Intervention: (Zeitpunkt, Intervention)
Model: (Zeitpunkt, event_id, src_node, new_state , notanonymanymore)

"""
InterventionEvent = namedtuple("InterventionEvent", ["time", "intervention"])
ModelEvent = namedtuple("ModelEvent", ["time", "node", "state", "causer"])
