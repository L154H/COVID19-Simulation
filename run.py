import cases
import sys
from util import get_max

# Latex font
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

if len(sys.argv) > 1:
    name = sys.argv[1]
    if name == "load":
        filename = sys.argv[2]
        tuple = get_max(filename, sys.argv[3:])
        print(sys.argv[3:], tuple)
    else:
        case = getattr(cases, "case_" + name)
        case()
else:
    for name in dir(cases):
        if name.startswith("case_"):
            print(name[5:])
