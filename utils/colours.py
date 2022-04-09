from itertools import cycle


class ColourUtils:
    def __init__(self):
        self._init_colours()

    def _init_colours(self):
        self.blue = "#0052CC"
        self.dark_blue = "#172B4D"
        self.light_blue = "#00B8D9"
        self.white = "#FFFFFF"
        self.grey = "#C1C7D0"
        self.red = "#FF5630"
        self.yellow = "#FFAB00"
        self.green = "#36B37E"
        self.purple = "#6554C0"
        self.black = "#091E42"
        self.light_grey = "#EBECF0"

    def cycle_colours(self):
        return cycle(
            [self.blue, self.red, self.yellow, self.green, self.purple, self.black]
        )
