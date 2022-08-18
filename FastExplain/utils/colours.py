from itertools import cycle

COLOURS = {
    "blue": "#0052CC",
    "dark_blue": "#172B4D",
    "light_blue": "#00B8D9",
    "white": "#FFFFFF",
    "grey": "#C1C7D0",
    "red": "#FF5630",
    "yellow": "#FFAB00",
    "green": "#36B37E",
    "purple": "#6554C0",
    "black": "#091E42",
    "light_grey": "#EBECF0",
}

CORE_COLOURS = ["blue", "red", "yellow", "green", "purple", "black"]
CORE_COLOURS = [COLOURS[i] for i in CORE_COLOURS]


def cycle_colours(colours=CORE_COLOURS):
    return cycle(CORE_COLOURS)
