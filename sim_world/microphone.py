"""
Microphone model.

AUTHOR: Phil David, Parsons

HISTORY:
    2023-06-14: P. David, Created.

"""


from skimage import draw
from PIL import Image
from fig import *
from phutils import *
from numbers import Number
from typing import Union
import math
import time
import numpy as np
import numpy.matlib

sound_tags = {'person', 'car', 'motorcycle', 'quadrotor', 'bird', 'dog', 'cat',
              'cow', 'sheep', 'robot', 'background'}


class Microphone:
    """
    Microphone model.
    """

    def __init__(self, pos=(0,0,0), mountedon=None, inctime=None):
        """
        Initialize a microphone.

        Usage:
            mic = Microphone(pos=(0,0,2), mountedon=None, inctime=None)

        Arguments:
            mountedon: The object that microphone is mounted on. This object, if
            it has a 'pos' or 'position' attribute, determines the position of
            the microphone (and the microphone's internal position is ignored).
            If 'mountedon' is None, the microphone position is taken from the
            microphone's own 'pos' attribute. Default is None.

            pos: The default position of the microphone if it is *not* mounted
            on any object. This is a 3-element array-like of floats. Default is
            (0,0,2). If the microphone is mounted on an object, then this is the
            relative position of the microphone on that object.

            inctime: (method) "inctime" is the method of the simulation
            environemnt that is used by the microphone to automatically
            increment the simulation time to account for the time it takes the
            microphone to acquire and process one frame of data. "inctime"
            should take a single argument, deltatime, the time in seconds to
            increment the simulation's clock. If "inctime" is None, then the
            simulation time should be incremented by some external method.
            Default is None.

        Returns:
            mic: The new Microphone object.

        Description:
            Create a microphone object.
        """

        # Check inputs.
        assert len(pos) == 3, 'POS must be 3D array-like'

        # Method to increment the simulation environment's clock.
        self.inctime = inctime

        # Setup the microphone position and orientaion.
        self.pos = pos                      # use this only if camera is not mounted
        self.mount(mountedon, relpos=pos)   # object the camera is mounted on may be None

        self.power_on = True


    def power(self, mode:Union[bool,int,None]=None):
        """
        Turn the microphone power on or off.

        Arguments:
            mode:bool, int, None -- The new microphone power mode, 1/0 or
            True/False. If None, then toggle the microphone power.
        """
        if mode == None:
            self.power_on = False if self.power_on else True
        elif type(mode) == int:
            self.power_on = True if mode != 0 else False
        elif type(mode) == bool:
            self.power_on = mode
        else:
            raise ValueError('Microphone mode must be True/False or 1/0 or None')


    def identical(self, mic):
        """
        Set all attributes of the current microphone to be identical those of
        another microphone.

        Usage:
            Microphone().identical(mic)

        Arguments:
            mic: The attributes of 'mic' are copied into the current microphone
            (self).
        """
        for atrb in mic.__dict__.keys():
            setattr(self, atrb, getattr(mic, atrb))


    def copy(self):
        """
        Create a copy of the micriphone.

        Usage:
            newmic = Micriphone().copy()

        Returns:
            newmic: A copy of the Micriphone() object.
        """
        newmic = Microphone()
        newmic.identical(self)
        return newmic


    def mount(self, obj, relpos=(0,0,0), relorient=(0,0,0)):
        """
        Mount the micriphone on an object.

        Usage:
            Micriphone().mount(obj)

        Arguments:
            obj: The object to mount the micriphone on.

            relpos: (float 3D array-like) The relative position of the micriphone
            with respect to the position of the object that the micriphone is
            mounted on.

            relorient: (float 3D array-like) The relative rotation angles (in
            radians) about the object's X, Y, and Z axes that define the zero
            pan/tilt angle of the micriphone.

        Description:
            The position and orientation of the object that the microphone is
            mounted on affects the absolute position and orientation of the
            microphone. The object should have an attribute 'pos' or 'position,' a
            3-element float array-like, that defines the X, Y, and Z position of
            the object. The position of the object needn't be defined when the
            microphone is mounted; the position will be accessed when needed.
        """
        self.mountedon = obj
        self.relpos = relpos                # microphone position offset relative to object position
        self.relorient = relorient          # microphone angle offset relative to object angle
        self.rot = Rot3d(angles=relorient)  # pan/tilt device's 3D orientation
        self.orient = relorient             # save for making microphone copies


    def unmount(self, obj):
        """
        Unmount the microphone from an object.
        """
        if hasattr(self, 'mountedon'):
            self.mountedon = None


    def get_pos(self):
        """
        Get the position of the microphone.
        """
        if hasattr(self.mountedon, 'pos'):
            pos = np.array(self.mountedon.pos) + np.array(self.relpos)
        elif hasattr(self.mountedon, 'position'):
            pos = np.array(self.mountedon.position) + np.array(self.relpos)
        else:
            pos = np.array(self.pos)
        return pos


    def get_audio(self, env, duration:float=3.0, maxdist:float=300,
                  verbose=False, map2d=None) -> numpy.ndarray:
        """
        Get an audio recording from the microphone.

        Usage:
            audio = Microphone.get_audio(env, duration=3.0, maxdist=500,
                                         map2d=None, verbose=False)

        Arguments:
            env: The SimWorld environment.

            duration: (float) Duration (seconds) of the recording to fetch from
            the microphone. The time period of the recording will begin at
            `duration` seconds prior to the current time and end at the current
            time.

            maxdist: (float) The maximum distance (meters) of any object from
            the microphone for its audio signal to be heard by the microphone.
            This is used to filter out objects that are too far away to
            be heard, to speed up the computation.

            map2d: (Map2D) The 2D map of the environment. This is used only
            for displaying the ground truth audio tracks on the 2D map display.
            If `verbose` is False or `map2d` is None, then the ground truth
            audio tracks are not displayed.

            verbose: (bool) If True, print information about ground truth audio
            tracks and, if `map2d` is not None, display these tracks on the 2D
            map.

        Returns:
            audio: (dict) Dictionary with the following keys:
                'signal': The audio signal recorded by the microphone during the
                previous `duration` seconds. The sample rate of this signal is
                the default sample rate for the simulation.

                'samplerate': The sample frequency (Hz) of the signal.

                'psd': The power spectal density of the recorded audio signal.

                'freq': The frequency (Hz) at each sample in `psd`.
        """
        if self.power == "off":
            # No audio signal when power is off.
            audio = None
        else:
            pos = self.get_pos()
            audio = env.get_audio(pos, duration=duration, maxdist=maxdist,
                                 map2d=map2d, verbose=verbose)

        return audio


