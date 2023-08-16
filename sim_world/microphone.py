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
        """
        pos = self.get_pos()
        return env.get_audio(pos, duration=duration, maxdist=maxdist,
                             map2d=map2d, verbose=verbose)


