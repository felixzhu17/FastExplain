from itertools import cycle

COLOURS = {
    "blue": "#0052CC",
    "dark_blue": "#172B4D",
    "light_blue": "#00B8D9",
    "white": "#FFFFFF",
    "grey": "#C1C7D0",
    "red": "#FF5630",
    "yellow": "FFAB00",
    "green": "36B37E",
    "purple": "6554C0",
    "black": "091E42",
    "light_grey": "EBECF0",
}


def cycle_colours(colours=["blue", "red", "yellow", "green", "purple", "black"]):
    return cycle([COLOURS[i] for i in colours])
