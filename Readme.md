# Code for a network-based COVID-19 Simulation


## Execution
- Execute the command `python run.py` to see the names of all cases
- Choose a case and execute `python run.py <casename>`, e.g. `python run.py no_intervention`
- The number of nodes and number of simulations can be changed in `cases.py` by specifying `N` and `NUM_SIM` repectively


![Results of `python run.py no_intervention`](no_interventions.png)

Results of `python run.py no_intervention` with `N = 1000` and `NUM_SIM = 100`

## Overview of all cases

| Case   |      Description     
|----------|-------------|
| `no_intervention` |  simulation with no interventions |
| `average_quarantine_rates`|  simulation with quarantine rate $q_1=0.5$ |
| `contact_tracing0`|  performing contact tracing with 0 hops |
| `contact_tracing1`|  performing contact tracing with 1 hop |
| `contact_tracing2`|  performing contact tracing with 2 hops |
| `infectious_tracing`|  starting with infectious nodes we trace their contacts until we find an infectious neighbour put them in isolation and continue recursively |
| `average_time`|  early and strict social distancing |
| `max_time`|  maximum counts depending on time when social distancing takes place |
| `half_edges`|  simulation with half of the edges |
| `max_half_edges`|  maximum counts depending on a percentage of active edges |
| `average_second_wave`|  second wave scenario |
| `max_class`|  maximum counts depending on the class size in school |
| `max_class_mod`|  maximum counts depending on the class size and the degree in school|
| `age_model`|  simulation with no interventions and the age model |
| `age_max_class_mod`|  maximum counts depending on the class size and the degree in school with the model including age structure |

## Reference
- For more details we refer to the bachelor thesis:
*Effects of Interventions on the COVID-19 Outbreak: A Network-based Approach*
