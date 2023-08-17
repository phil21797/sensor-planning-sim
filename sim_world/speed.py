"""
A class to track and allow simple update to parameter values.

History:
    2021-01-07 - P. David - Created.
"""


class Speed:
    """
    A class to track and update values associated with camera parameters
    (translation, rotation, zoom, and framerate).
    """

    def __init__(self):
        """
        Initialize ranges and values of parameters.
        """

        # Names and units of parameters that this object controls.
        self.pnames = ['Trans', 'Rot', 'FPS', 'Zoom']
        self.punits = ['meters/sec', 'º/sec', 'fps', 'º/sec']
        self.numparams = len(self.pnames)
        self.selected = 0               # which paramater is currently selected?

        # The following define all allowed values of the tracked parameters. The
        # order of the lists within "self.values" must correspond to the order
        # of the names in "self.names".
        self.values = [[0.1, 0.5, 1, 2, 3, 5, 7, 10],            # trans, meters/sec
                       [0.1, 0.5, 1, 5, 10, 25, 45, 90, 180],    # rot, degrees/sec
                       [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 60],  # fps, frames/sec
                       [0.1, 0.5, 1, 2, 5, 10, 20, 30, 45]]      # zoom, degrees/sec


        # Number of allowed values of each parameter.
        self.numvals = [len(self.values[k]) for k in range(self.numparams)]

        # Index of currently selected value of each parameter.
        self.idxcur = [7, 5, 2, 5]
        # self.idxcur = [int(self.numvals[k]/2 + 0.5) for  k in range(self.numparams)]

        # Dict to map parameter names to index in "self.values".
        self.name2idx = {self.pnames[k].lower():k for k in range(len(self.pnames))}

        # Set the quick-access parameter attributes.
        self.trans = self.values[self.name2idx['trans']][self.idxcur[self.name2idx['trans']]]
        self.rot = self.values[self.name2idx['rot']][self.idxcur[self.name2idx['rot']]]
        self.fps = self.values[self.name2idx['fps']][self.idxcur[self.name2idx['fps']]]
        self.zoom = self.values[self.name2idx['zoom']][self.idxcur[self.name2idx['zoom']]]

        # Set the generic parameter attributes.
        self.name = self.pnames[self.selected]
        self.value = self.values[self.selected][self.idxcur[self.selected]]
        self.units = self.punits[self.selected]


    def next(self):
        """
        Select the next parameter to adjust.
        """
        self.selected = (self.selected + 1) % self.numparams
        self.name = self.pnames[self.selected]
        self.value = self.values[self.selected][self.idxcur[self.selected]]
        self.units = self.punits[self.selected]


    def inc(self):
        """
        Increment the currently selected parameter.
        """
        idx = self.idxcur[self.selected]
        if idx < self.numvals[self.selected] - 1:
            idx += 1
            self.idxcur[self.selected] = idx
        self.value = self.values[self.selected][idx]
        if self.selected == self.name2idx['trans']:
            self.trans = self.value
        elif self.selected == self.name2idx['rot']:
            self.rot = self.value
        elif self.selected == self.name2idx['fps']:
            self.fps = self.value
        elif self.selected == self.name2idx['zoom']:
            self.zoom = self.value
        else:
            raise Exception('Unrecognized parameter selection:{}'.format(self.selected))


    def dec(self):
        """
        Decrement the currently selected parameter.
        """
        idx = self.idxcur[self.selected]
        if idx > 0:
            idx -= 1
            self.idxcur[self.selected] = idx
        self.idxcur[self.selected] = idx
        self.value = self.values[self.selected][idx]
        if self.selected == self.name2idx['trans']:
            self.trans = self.value
        elif self.selected == self.name2idx['rot']:
            self.rot = self.value
        elif self.selected == self.name2idx['fps']:
            self.fps = self.value
        elif self.selected == self.name2idx['zoom']:
            self.zoom = self.value
        else:
            raise Exception('Unrecognized parameter selection:{}'.format(self.selected))