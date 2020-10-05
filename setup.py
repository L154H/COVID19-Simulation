from layer import LayerGraph, LayerFactory
from model import CoronaHillModel
import seaborn as sns


color_palette = {
    'S': sns.xkcd_rgb['blue'],
    'E': sns.xkcd_rgb['dandelion'],
    'I1': sns.xkcd_rgb['orange'],
    'I2': sns.xkcd_rgb['red'],
    'I3': sns.xkcd_rgb['dark red'],
    'R': sns.xkcd_rgb['green'],
    'D': sns.xkcd_rgb['black']
}

qolor_palette = {
    'S': sns.xkcd_rgb['blue'],
    'E': sns.xkcd_rgb['dandelion'],
    'I1': sns.xkcd_rgb['orange'],
    'I2': sns.xkcd_rgb['red'],
    'I3': sns.xkcd_rgb['dark red'],
    'R': sns.xkcd_rgb['green'],
    'D': sns.xkcd_rgb['black'],
    'Q1': sns.xkcd_rgb['bright purple'],
    'Q2': sns.xkcd_rgb['purple'],
    'Q3': sns.xkcd_rgb['dark purple']
}

quolor_palette = {
    'S':  sns.xkcd_rgb['blue'],
    'E':  sns.xkcd_rgb['dandelion'],
    'I1': sns.xkcd_rgb['orange'],
    'I2': sns.xkcd_rgb['red'],
    'I3': sns.xkcd_rgb['dark red'],
    'R':  sns.xkcd_rgb['green'],
    'D':  sns.xkcd_rgb['black'],
    'Q1': sns.xkcd_rgb['bright purple'],
    'Q2': sns.xkcd_rgb['purple'],
    'Q3': sns.xkcd_rgb['dark purple'],
    'QS': sns.xkcd_rgb['pink'],
    'QE': sns.xkcd_rgb['dark pink']
}

kolor_palette = {
    'S_kids': sns.xkcd_rgb['blue'],
    'S_normal': sns.xkcd_rgb['blue'],
    'S_risk': sns.xkcd_rgb['blue'],
    'E_kids': sns.xkcd_rgb['dandelion'],
    'E_normal': sns.xkcd_rgb['dandelion'],
    'E_risk': sns.xkcd_rgb['dandelion'],
    'I_kids': sns.xkcd_rgb['orange'],
    'I_normal': sns.xkcd_rgb['orange'],
    'I_risk': sns.xkcd_rgb['orange'],
    'I2': sns.xkcd_rgb['red'],
    'I3': sns.xkcd_rgb['dark red'],
    'R': sns.xkcd_rgb['green'],
    'D': sns.xkcd_rgb['black']
}


def setup_CH(n, classsize=20, degree=5, p=1):
    LG = setup_LG(n, classsize, degree, p)
    CH = CoronaHillModel()
    return LG, CH


def setup_LG(n, classsize=20, degree=5, p=1):
    """
    construction of our basic network
    """
    categories = ["Kids", "Normal", "Risk"]
    percentage = [0.15, 0.48, 0.37]  # stastica
    LG = LayerGraph(n, categories, percentage)
    LF = LayerFactory(LG)

    # create layers:
    household_layer = LF.layer_dividing_Graph("Households", 2, None,
                                              categories, fully_connected=True)
    school_layer = LF.layer_dividing_Graph("Schools", classsize, degree,
                                           ["Kids"])
    working_layer = LF.layer_dividing_Graph("Workplaces", 6, 3, ["Normal"])
    risk_layer = LF.create_layer("R_Workplaces", 6, int(percentage[1] * n / 3),
                                 [0, 0.7, 0.3], 3)
    social_layer = LF.layer_dividing_Graph("Social", 10, 3, categories)
    party_layer = LF.create_layer("parties", 20, int(n * percentage[0]/6),
                                  [0.6, 0.4, 0], 6)
    basic_connect = LF.layer_dividing_Graph("basic", n, 1, categories)
    # Add layers:
    LG.add_layer(household_layer,   p)
    LG.add_layer(school_layer,      p)
    LG.add_layer(working_layer,     p)
    LG.add_layer(risk_layer,        p)
    LG.add_layer(social_layer,      p)
    LG.add_layer(party_layer,       p)
    LG.add_layer(basic_connect,     p)

    return LG
