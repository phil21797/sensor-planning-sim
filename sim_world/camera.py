"""
PTZ camera.

AUTHOR: Phil David, US Army Research Laboratory

HISTORY:
    2017-11-21: P. David, Created.

"""


from skimage import draw
from PIL import Image
from fig import *
from phutils import *
from numbers import Number
from panoimage import inv_cylindrical_proj
import math
import time
import numpy as np
import numpy.matlib
import vtkutils as vtu


class PTZCamera:
    """
    Pan-Tilt-Zoom camera.
    """

    def __init__(self, imsize=(1920,1080), rnghfov=(2.3,63.7), rngpan=(-180,180),
                 rngtilt=(-10,60), pan=0, tilt=0, zoom=0, pos=(0,0,2),
                 orient=(0,0,0), mountedon=None, speeds=(60,60,2.5),
                 inctime=None):
        """
        Initialize a PTZ camera.

        Usage:
            cam = PTZCamera(imsize=(1920,1080), rnghfov=(2, 60),
                            rngpan=(-180, 180), rngtilt=(-10, 60),
                            pan=0, tilt=0, zoom=0, speeds=(60,60,2.5),
                            pos=(0,0,2), orient=(0,0,0), mountedon=None,
                            inctime=None)

        Arguments:
            imsize: (WIDTH, HEIGHT). Size of camera image in pixels. Default is
            (1920,1080).

            rnghfov: (MINHFOV, MAXHFOV). Range of camera horizontal field of
            view (degrees).  MINHFOV <= MAXHFOV. Default is (3,54).

            rngpan: (MINPAN, MAXPAN). Range of camera pan angles (degrees).
            Default is (-180,180).

            rngtilt: (MINTILT, MAXTILT). Range of camera tilt angles
            (degrees). Default is (-10,60).

            mountedon: The object that camera is mounted on. This object, if it
            has a 'pos' or 'position' attribute, determines the position of the
            camera (and the camera's internal position is ignored). If
            'mountedon' is None, the camera position is taken from the camera's
            own 'pos' attribute. Default is None.

            pos: The default position of the camera if it is *not* mounted on
            any object. This is a 3-element array-like of floats. Default is
            (0,0,2).

            orient: (tuple) Orient is a tuple (RX, RY, RZ). This is 3-element
            array-like vector gives the rotation angles (in radians) of the
            pan/tilt unit in the base coordinate system. The base coordinate
            system is either the world coordinate sytem, if the camera is not
            mounted, or an object's local corrdinate system, if the camera is
            mouned on an object. Default is (0,0,0).

            speeds: (tuple) Speeds is the tuple (panspeed, tiltspeed, zoomtime),
            where "panspeed", "tiltspeed", and "zoomtime" are the pan, tilt, and
            zoom speeds, respectively, of the camera. The pan and tilt speeds
            are given in degrees/second. The zoom time is the time, in seconds,
            that it takes the camera to go from minimum to mximum zoom.

            pan: Initial camera pan angle (degrees). Default is 0.

            tilt: Initial camera tilt angle (degrees). Default is 0.

            zoom: Initial camera zoom ([0,1]). Default is 0.

            inctime: (method) "inctime" is the method of the simulation
            environemnt that is used by the camera to automatically increment
            the simulation time to account for the time it takes the camera to
            change its pan, tilt, and zoom. "inctime" should take a single
            argument, deltatime, the time in seconds to increment the
            simulation's clock. If "inctime" is None, then the simulation time
            should be incremented by some external method. Default is None.

        Returns:
            cam: The new PTZCamera object.

        Description:
            Camera Zoom == 0 represents the minimum zoom, which is the widest
            FOV. Camera Zoom == 1 represents the maximum zoom, which is the
            narrowest FOV.

            The default camera properties are based on this camera:
                Model: Sony FCB-EV7520A Block Camera
                URL: https://www.sony.co.jp/Products/ISP/english/download/pdf/
                    catalog/21_FCB_EV7520A_S.pdf
                Resolution: 1920x1080, 1280x720
                Horizontal Viewing Angle: 63.7° (wide) to 2.3° (tele)
                Zoom: 30x Optical Zoom, f=4.3 mm (wide) to 129.0 mm (tele),
                    F1.6 to F4.7
                Zoom speed: 2.5 sec. to go from full wide to full tele with
                    focus tracking off. 5.0 sec. with focus tracking on.
                Shutter speed: 1 sec. to 1/10,000 sec.

            The default pan/tilt properties are based on this unit:
                Model: FLIR Motion Control Systems, Inc., PTU-E46-70W
                Pan/tilt speed: 60°/sec
                Pan/tilt axis resolution: 0.003°
                Payload: 9 lbs.

            The pan/tilt device's 3D orientation is determined by the camera's
            3x3 rotation matrix. This gives the direction of the front of the
            pan/tilt unit relative to the Y-axis of the base coordinate system.
            This enables the pan/tilt unit can be oriented in any direction
            relative to the object that it's mounted on, or in the world if the
            camera is not mounted on anything. If the camera is mounted on an
            object, then the camera's rotation matrix gives the orientation of
            the pan/tilt unit relative to the object's local coordinate system.
        """

        # Check inputs.
        assert len(imsize) == 2, 'IMSIZE must be 2D array-like'
        assert len(rnghfov) == 2 and rnghfov[0] <= rnghfov[1], \
            'RNGHFOV must be 2D array-like with RNGHFOV[0] <= RNGHFOV[1]'
        assert len(rngpan) == 2 and rngpan[0] <= rnghfov[1], \
            'RNGPAN must be 2D array-like with RNGPAN[0] <= RNGPAN[1]'
        assert len(rngtilt) == 2 and rngtilt[0] <= rngtilt[1], \
            'RNGTILT must be 2D array-like with RNGTILT[0] <= RNGTILT[1]'
        assert type(speeds) is tuple and len(speeds) == 3, \
            'SPEEDS must be a tuple (panspeeed, tiltspeed, zoomtime)'

        # Resolution of camera image.
        self.imsize = imsize                         # (width, height) in pixels
        self.ncols = imsize[0]
        self.nrows = imsize[1]
        self.aratio = self.ncols/self.nrows          # image aspect ratio

        # Range of pan and tilt angles (degrees).
        self.minpan = rngpan[0]
        self.maxpan = rngpan[1]
        self.mintilt = rngtilt[0]
        self.maxtilt = rngtilt[1]

        # Pan/tilt/zoom speeds.
        self.pan_speed = speeds[0]      # (degrees/second)
        self.tilt_speed = speeds[1]     # (degrees/second)
        self.zoom_time = speeds[2]      # time (sec.) to go from min to max zoom

        # Method to increment the simulation environment's clock.
        self.inctime = inctime

        # Range of camera fields-of-view (FOV) (degrees) and focal lengths
        # (pixels).
        self.minhfov = rnghfov[0]
        self.maxhfov = rnghfov[1]
        self.minfoclen = self.ncols/(2*np.tan(np.deg2rad(self.maxhfov/2)))
        self.maxfoclen = self.ncols/(2*np.tan(np.deg2rad(self.minhfov/2)))
        if True:
            self.minvfov = self.minhfov/self.aratio
            self.maxvfov = self.maxhfov/self.aratio
        else:
            self.minvfov = 2*np.rad2deg(np.arctan(self.nrows/(2*self.maxfoclen)))
            self.maxvfov = 2*np.rad2deg(np.arctan(self.nrows/(2*self.minfoclen)))
        self.nonzooming = True if abs(rnghfov[1]-rnghfov[0]) < 1e-3 else False

        # Set initial camera pan, tilt, and zoom.
        self.pan = pan
        self.tilt = tilt
        self.zoom = 0
        self.set(zoom=zoom)

        # Setup the camera position and orientaion.
        self.pos = pos                  # use this only if camera is not mounted
        self.mount(mountedon)      # object the camera is mounted on may be None
        self.rot = Rot3d(angles=orient)       # pan/tilt device's 3D orientation
        self.orient = orient                     # save for making camera copies


    def identical(self, cam):
        """
        Set all attributes of the current camera to be identical those of
        another camera.

        Usage:
            PTZCamera().identical(cam)

        Arguments:
            cam: The attributes of 'cam' are copied into the current camera (self).
        """
        for atrb in cam.__dict__.keys():
            setattr(self, atrb, getattr(cam, atrb))


    def copy(self):
        """
        Create a copy of the camera.

        Usage:
            newcam = PTZCamera().copy()

        Returns:
            newcam: A copy of the PTZCamera() object.
        """
        newcam = PTZCamera()
        newcam.identical(self)

        # newcam = PTZCamera(imsize=(self.ncols, self.nrows),
                           # rnghfov=(self.minhfov, self.maxhfov),
                           # rngpan=(self.minpan, self.maxpan),
                           # rngtilt=(self.mintilt, self.maxtilt),
                           # pan=self.pan, tilt=self.tilt, zoom=self.zoom,
                           # pos=self.pos, orient=self.orient,
                           # mountedon=self.mountedon)

        return newcam


    def mount(self, obj, relpos=(0,0,0), relorient=(0,0,0)):
        """
        Mount the camera on an object.

        Usage:
            PTZCamera().mount(obj)

        Arguments:
            obj: The object to mount the camera on.

            relpos: (float 3D array-like) The relative position of the camera
            with respect to the position of the object that the camera is
            mounted on.

            relorient: (float 3D array-like) The relative rotation angles (in
            radians) about the object's X, Y, and Z axes that define the zero
            pan/tilt angle of the camera.

        Description:
            The position and orientation of the object that the camera is
            mounted on affects the absolute position and orientation of the
            camera. The object should have an attribute 'pos' or 'position,' a
            3-element float array-like, that defines the X, Y, and Z position of
            the object. The position of the object needn't be defined when the
            camera is mounted; the position will be accessed when needed.
        """
        self.mountedon = obj
        self.relpos = relpos     # camera position offset relative to object position
        self.relorient = relorient # camera angle offset relative to object angle
        self.rot = Rot3d(angles=relorient)    # pan/tilt device's 3D orientation
        self.orient = relorient               # save for making camera copies


    def unmount(self, obj):
        """
        Unmount the camera from an object.
        """
        if hasattr(self, 'mountedon'):
            self.mountedon = None


    def set(self, pan=None, tilt=None, zoom=None, hfov=None, vfov=None,
            copycam=None):
        """
        Set the camera pan, tilt, and/or zoom to specific values.

        Usage:
            proctime = PTZCamera().set(pan, tilt, zoom, hfov, vfov, copycam)

        Arguments:
            pan: The camera pan angle (degrees).

            tilt: The camera tilt angle (degrees).

            zoom: The camera zoom, a number in [0,1]. 0 sets the camera to
            its widest FOV. 1 sets the camera to its narrowest FOV. The field
            of views (horizontal and vertical) of the camera are linear
            functions of the zoom.

            hfov: Set the camera's zoom such that its horizontal field-of-view
            is 'hfov' (degrees).

            vfov: Set the camera's zoom such that its vertical field-of-view
            is 'vfov' (degrees).

            copycam: Make the current camera a copy of 'copycam'. All other
            arguments are ignored if 'copycam' is not None. Default is None.

        Returns:
            proctime: The time (seconds) required to process the request.

        Description:
            This function implements absolute adjustments to the camera's pan,
            tilt, and zoom. Use PTZCamera().inc() to make incremental changes to
            pan, tilt, or zoom.

            Any combination of the arguments may be defined. Other camera
            parameters are changed to be consistent with the provided arguments.
        """

        panorig = self.pan
        tiltorig = self.tilt
        zoomorig = self.zoom

        if copycam is not None:
            # Set all camera parameters to the same as copycam.
            self.identical(copycam)
        else:
            if pan is not None and (pan < self.minpan or pan > self.maxpan):
                raise ValueError('PAN out of range: {}'.format(pan))
            if tilt is not None and (tilt < self.mintilt or tilt > self.maxtilt):
                raise ValueError('TILT out of range: {}'.format(tilt))
            if zoom is not None and (zoom < 0 or zoom > 1):
                raise ValueError('ZOOM must be in [0, 1]: {}'.format(zoom))
            if hfov is not None:
                if hfov < self.minhfov - 1e-7 or hfov > self.maxhfov + 1e-7:
                    raise ValueError('HFOV is out of range for this camera: {}'.format(hfov))
                else:
                    hfov = max(self.minhfov, min(self.maxhfov, hfov))
            if vfov is not None:
                if vfov < self.minvfov - 1e-7 or vfov > self.maxvfov + 1e-7:
                    raise ValueError('VFOV is out of range for this camera: {}'.format(vfov))
                else:
                    vfov = max(self.minvfov, min(self.maxvfov, vfov))

            if pan is not None:
                self.pan = pan

            if tilt is not None:
                self.tilt = tilt

            if zoom is not None:
                self.zoom = zoom
                self.hfov = self.maxhfov + zoom*(self.minhfov - self.maxhfov)
                self.vfov = self.maxvfov + zoom*(self.minvfov - self.maxvfov)
                self.foclen = self.ncols/(2*np.tan(np.deg2rad(self.hfov/2)))
                self.vfoclen = self.nrows/(2*np.tan(np.deg2rad(self.vfov/2)))
                self.horizres = self.ncols/self.hfov   # horiz. res. (pixels/degree)
                self.vertres = self.nrows/self.vfov    # vert. res. (pixels/degree)

            if hfov is not None:
                if self.nonzooming:
                    self.foclen = self.minfoclen
                    self.zoom = 0
                else:
                    self.foclen = self.ncols/(2*np.tan(np.deg2rad(hfov/2)))
                    self.zoom = (self.maxhfov-hfov)/(self.maxhfov-self.minhfov)
                self.hfov = self.maxhfov + self.zoom*(self.minhfov - self.maxhfov)
                self.vfov = self.maxvfov + self.zoom*(self.minvfov - self.maxvfov)
                self.horizres = self.ncols/self.hfov   # horiz. res. (pixels/degree)
                self.vertres = self.nrows/self.vfov    # vert. res. (pixels/degree)
                self.vfoclen = self.nrows/(2*np.tan(np.deg2rad(self.vfov/2)))

            if vfov is not None:
                if self.nonzooming:
                    self.foclen = self.minfoclen
                    self.zoom = 0
                else:
                    self.foclen = self.nrows/(2*np.tan(np.deg2rad(vfov/2)))
                    self.zoom = (self.maxvfov-vfov)/(self.maxvfov-self.minvfov)
                self.hfov = self.maxhfov + self.zoom*(self.minhfov - self.maxhfov)
                self.vfov = self.maxvfov + self.zoom*(self.minvfov - self.maxvfov)
                self.horizres = self.ncols/self.hfov   # horiz. res. (pixels/degree)
                self.vertres = self.nrows/self.vfov    # vert. res. (pixels/degree)
                self.vfoclen = self.nrows/(2*np.tan(np.deg2rad(self.vfov/2)))

        # Get the time (in seconds) to process the current PTZ request. We
        # assume pan and tilt occur sequentially, and that zoom occurs
        # simultaneously with pan and tilt.
        #   pantime = abs(self.pan - panorig)/self.pan_speed
        #   tilttime = abs(self.tilt - tiltorig)/self.tilt_speed
        #   zoomtime = self.zoom_time*abs(self.zoom - zoomorig)
        #   proctime = max(pantime+tilttime, zoomtime)
        proctime = self.get_ptz_time(newpan=panorig, newtilt=tiltorig,
                                     newzoom=zoomorig)

        # If defined, call the simulation time updater. This will update the
        # positions of dynamic objects.
        if self.inctime: self.inctime(proctime)

        return proctime


    def get_ptz_time(self, newpan:float=None, newtilt:float=None,
                     newzoom:float=None) -> float:
        """
        Get the time required to change the PTZ, but don't actually make any
        changes.

        Usage:
            time = PTZCamera.get_ptz_time(newpan=None, newtilt=None,
                                          newzoom=None)

        Arguments:
            newpan, newtilt, newzoom: (floats) The pan, tilt, and zoom position
            that the camera is supposed to be moved to. "newpan" and "newtilt"
            are in degrees and "newzoom" is in [0,1]. Any of these arguments may
            be None, in which case that aspect of the camera is assumed to
            remain unchanged.

        Returns:
            time: (float) Time in seconds for the camera to pan, tilt, and zoom
            from its current position to the new position.

        Description:
            We assume that pan and tilt occur sequentially, and that zoom occurs
            simultaneously with pan and tilt.
        """
        pantime = 0 if newpan is None else abs(self.pan - newpan)/self.pan_speed
        tilttime = 0 if newtilt is None else abs(self.tilt - newtilt)/self.tilt_speed
        zoomtime = 0 if newzoom is None else self.zoom_time*abs(self.zoom - newzoom)
        total = max(pantime+tilttime, zoomtime)
        return total


    def inc(self, dpan=None, dtilt=None, dzoom=None):
        """
        Increment the pan, tilt, and/or zoom of a camera.

        Usage:
            proctime = PTZCamera().inc(dpan=None, dtilt=None, dzoom=None)

        Arguments:
            dpan: Increment for the camera pan (degrees).
            dtilt: Increment for the camera tilt (degrees).
            dzoom: Increment for the camera zoom, a number in [-1,1]. Camera
                zoom is always maintained in the range [0,1].

        Returns:
            proctime: The time (seconds) required to process the request.

        Description:
            This function implements relative adjustments to the camera's pan,
            tilt, and zoom. Use PTZCamera().set() to change pan, tilt, or zoom
            to an absolute position. Decrement the pan, tilt, or zoom by passing
            a negative value for an argument.
        """

        panorig = self.pan
        tiltorig = self.tilt
        zoomorig = self.zoom

        if dpan is not None:
            new = self.pan + dpan
            if new < self.minpan:
                new = self.minpan
            elif new > self.maxpan:
                new = self.maxpan
            self.pan = new

        if dtilt is not None:
            new = self.tilt + dtilt
            if new < self.mintilt:
                new = self.mintilt
            elif new > self.maxtilt:
                new = self.maxtilt
            self.tilt = new

        if dzoom is not None:
            new = self.zoom + dzoom
            if new < 0:
                new = 0
            elif new > 1:
                new = 1
            self.zoom = new
            self.hfov = self.maxhfov + self.zoom*(self.minhfov - self.maxhfov)
            self.vfov = self.maxvfov + self.zoom*(self.minvfov - self.maxvfov)
            self.foclen = self.ncols/(2*np.tan(np.deg2rad(self.hfov/2)))
            self.horizres = self.ncols/self.hfov   # horiz. res. (pixels/degree)
            self.vertres = self.nrows/self.vfov    # vert. res. (pixels/degree)

        # Get the time (in seconds) to process the current PTZ request. We
        # assume pan and tilt occur sequentially, and that zoom occurs
        # simultaneously with pan and tilt.
        pantime = abs(self.pan - panorig)/self.pan_speed
        tilttime = abs(self.tilt - tiltorig)/self.tilt_speed
        zoomtime = self.zoom_time*abs(self.zoom - zoomorig)
        proctime = max(pantime+tilttime, zoomtime)

        # If possible, increment the simulation time.
        if self.inctime: self.inctime(proctime)

        return proctime


    def whatzoom(self, hfov=None, vfov=None):
        """
        Get the camera zoom for a specified camera horizontal or vertical field
        of view (without changing the camera zoom).

        Usage:
            zoom = Camera.whatzoom(hfov=None, vfov=None)

        Arguments:
            hfov, vfov: (float) The horizonatal and vertical camera FOVs,
            respectively, in degrees. Only one of "hfov" or "vfov" should be
            provided.

        Returns:
            zoom: The cmera zoom value, in [0, 1].
        """
        if hfov is not None:
            return (self.maxhfov-hfov)/(self.maxhfov-self.minhfov)
        elif vfov is not None:
            return (self.maxvfov-vfov)/(self.maxvfov-self.minvfov)
        else:
            raise ValueError('One of arguments "hfov" or "vfov" must be provided')


    def whathvf(self, zoom=None, hfov=None, vfov=None, foclen=None):
        """
        Get the camera horizonatal FOV, vertical FOV and focal length for a
        specified camera zoom, hfov, vfov or foclen (without changing the camera
        zoom).

        Usage:
            hfov, vfov, foclen = Camera.whathvf(zoom=None, hfov=None,
                                                vfov=None, foclen=None)

        Arguments:
            zoom: The cmera zoom value, in [0, 1].  Only one of 'zoom', 'hfov',
            'vfov' or 'foclen' should be provided.

            hfov: Horizontal FOV (degrees).

            vfov: Vertical FOV (degrees).

            foclen: The camera focal length (in pixels).

        Returns:
            hfov, vfov: The horizonatal and vertical camera FOVs, respectively,
            in degrees.

            foclen: The camera focal length (in pixels).
        """

        if zoom is not None:
            assert isinstance(zoom,(int,float)), 'ZOOM must be a number:'
            assert zoom >= 0 and zoom <= 1, 'ZOOM is out of range'
            hfov = self.maxhfov + zoom*(self.minhfov - self.maxhfov)
            vfov = self.maxvfov + zoom*(self.minvfov - self.maxvfov)
            foclen = self.ncols/(2*np.tan(np.deg2rad(hfov/2)))
        elif foclen is not None:
            assert foclen >= self.minfoclen and foclen <= self.maxfoclen
            hfov = 2*np.rad2deg(np.arctan(self.ncols/(2*foclen)))
            vfov = 2*np.rad2deg(np.arctan(self.nrows/(2*foclen)))
            # zoom = (self.maxvfov-vfov)/(self.maxvfov-self.minvfov)
        elif hfov is not None:
            if hfov < self.minhfov or hfov > self.maxhfov:
                raise ValueError('HFOV is out of range for this camera: {:.2f}'.format(hfov))
            if self.nonzooming:
                foclen = self.minfoclen
                zoom = 0
            else:
                foclen = self.ncols/(2*np.tan(np.deg2rad(hfov/2)))
                zoom = (self.maxhfov-hfov)/(self.maxhfov-self.minhfov)
            vfov = self.maxvfov + zoom*(self.minvfov - self.maxvfov)
        elif vfov is not None:
            if vfov < self.minvfov or vfov > self.maxvfov:
                raise ValueError('VFOV is out of range for this camera: ', vfov)
            if self.nonzooming:
                foclen = self.minfoclen
                zoom = 0
            else:
                foclen = self.nrows/(2*np.tan(np.deg2rad(vfov/2)))
                zoom = (self.maxvfov-vfov)/(self.maxvfov-self.minvfov)
            hfov = self.maxhfov + zoom*(self.minhfov - self.maxhfov)
        else:
            raise ValueError('One of arguments ZOOM, HFOV, VFOV or FOCLEN must be provided')

        hfov = min(self.maxhfov, max(self.minhfov, hfov))
        vfov = min(self.maxvfov, max(self.minvfov, vfov))
        return hfov, vfov, foclen


    def get_pt(self, pntdir):
        """
        Get the pan and tilt angles corrsponding to a 3D pointing direction.

        Usage:
            pan, tilt = Camera().get_pt(pntdir)

        Arguments:
            pntdir: (3D array-like) The 3D vector (X,Y,Z) from the camera's
            focal position and aligned with the central axis of the camera
            direction. This vector is absolute, not relative to the orientation
            of any object that the camera may be mounted on.

        Returns:
            pan: (float) Pan angle in degrees. If the camera is mounted on an
            object, then this pan angle is relative to the orientation of that
            object.

            tilt: (float) Tilt angle in degrees. If the camera is mounted on an
            object, then this tilt angle is relative to the orientation of that
            object.

        Description:
            The pan and tilt angles returned are relative to a right-handed
            coordinate system. If the camera is mounted on an object, then these
            angles are relative to the local coordinate system of the object;
            otherwise, these angles are relative to the world coordinate system.
            In either case, the positive Y-axis corresponds to a pan angle of 0
            degrees. The negative X-axis is a pan angle of 90 degrees, and the
            positive X-axis is a pan angle of -90 degrees. The positive Z-axis
            corresponds to a tilt angle of 90 degrees, and the negative Z-axis
            to a tilt angle of -90 degrees.
        """

        pan = np.rad2deg(np.arctan2(-pntdir[0], pntdir[1]))
        d = np.linalg.norm(pntdir[0:2])
        tilt = np.rad2deg(np.arctan2(pntdir[2], d))

        return pan, tilt


    def get_pos_fp(self):
        """
        Get the 3D position and focus point of the camera in world coordinates.

        Usage:
            pos, fp = PTZCamera().get_pos_fp()

        Returns:
            pos: (Numpy array) The camera position, a tuple (PX,PY,PZ).

            fp: (Numpy array) The camera focus point, a tuple (FPX,FPY,FPZ).

        Description:
            The focus point of the camera is a finite 3D world point in front of
            the camera that the camera's optical axis passes through. This is
            the point that the camera is looking directly at. In the VTK API,
            even when the camera changes position, it will still be looking at
            the same focus point unless the focus point is explicitly changed.
            The focus point is obtained by rotating the point (0,1,0) to account
            for the base object's orientation, the pan/tilt unit's orientation
            relative to the base object, and the actual pan and tilt of the
            camera. This orientation is defined by a 3x3 rotation matrix that is
            the product of the base object's rotation matrix, the pan/tilt
            unit's rotation matrix, and the rotation matrix for the final camera
            pan and tilt.
        """
        # Get position.
        if hasattr(self.mountedon, 'pos'):
            pos = np.array(self.mountedon.pos) + np.array(self.relpos)
            delta_elev = self.mountedon.delta_elev
            delta_tilt = self.mountedon.delta_tilt
        elif hasattr(self.mountedon, 'position'):
            pos = np.array(self.mountedon.position) + np.array(self.relpos)
            delta_elev = self.mountedon.delta_elev
            delta_tilt = self.mountedon.delta_tilt
        else:
            pos = np.array(self.pos)
            delta_elev = 0
            delta_tilt = 0

        # Get translation-invariant focus point.
        tilt = np.deg2rad(self.tilt + delta_tilt)
        fp = np.array([[0,1,0]]).T              # initial VTK camera focal point
        fp = Rot3d(rx=tilt, ry=0, rz=0)*fp      # tilt about x axis
        fp = Rot3d(rx=0, ry=0, rz=np.deg2rad(self.pan))*fp    # pan about z axis
        fp = self.rot*fp                             # rotation of pan/tilt unit
        if hasattr(self.mountedon, 'rot'):
            fp = self.mountedon.rot*fp                 # rotation of base object

        # Translate camera FP to give an absolute world point.
        fp[0] += pos[0]
        fp[1] += pos[1]
        fp[2] += pos[2] + delta_elev
        fp = np.squeeze(np.asarray(fp))

        return pos, fp


    def get_images(self, myworld, imlist=[], filtersize=1):
        """
        Get this camera's current images.

        Usage:
            imgs = PTZCamera.get_images(myworld, imlist=[], filtersize=1)

        Arguments:
            myworld: The SimWorld object to take an image of.

            imlist: (list) List of names of images to retrieve. Each image name
            is a string from the allowed images described below. If this list
            is empty, then all valid images are returned. Default is empty (all
            images returned).

            filtersize: (int) Radius (in pixels) of filter used to remove small
            regions from the non-ground-truth semantic label image. We attempt
            to simulate real image perception algorithms where very small
            objects are not recognized. Default is 1.

        Returns:
            imgs: A dictionary of images containing zero or more of the
            following:

                'color': RGB color image.

                'label': Single-channel semantic label image. The value of each
                pixel is one of the IDs from the `label2id` dictionary. To
                simulate reality, the labels of very small objects (as
                determined by the `filtersize` argument) can be removed from
                this image.

                'label_gt': Single-channel ground-truth semantic label image.
                The value of each pixel is one of the IDs from the `label2id`
                dictionary.

                'label_rgb': RGB semantic label image. Objects in this image are
                colored using the RGB values from the `label_colors` list. This
                image is probably useful only for display purposes.

                'objectid': Image of object IDs, which index into the list of
                objects, self.objs.

                'depth': Single-channel, float-valued, depth image. Depth values
                are in meters. Currently, the depth and ground-truth depth
                images are identical.

                'depth_gt': Single-channel, float-valued, ground-truth depth
                image. Depth values are in meters. Currently, the depth and
                ground-truth depth images are identical.

        Description:
            The camera's current pan, tilt, and zoom, and the position and
            zerodir of the object that the camera is mounted on, all determine
            the images generated.
        """
        # Get the position and focus point of the camera in 3D world coordinates.
        pos, fp = self.get_pos_fp()

        # Render the scene.
        myworld.render_scene(pos, fp, self.vfov)

        # Get the rendered images.
        imgs = myworld.get_images(imlist=imlist, filtersize=filtersize)

        return imgs


    def xy_to_pt(self, ix, iy):
        """
        Get the absolute pan/tilt position of the points (ix,iy) in the image
        acquired by the camera in its current pan/tilt position. (ix,iy) are the
        pixel locations relative to the image origin at the upper left corner of
        the image.

        Usage:
            pan, tilt = PTZCamera().xy_to_pt(ix, iy)

        Arguments:
            ix: Horizontal positions (columns) of the image points, a float or
            an Nx1 Numpy array.

            iy: Vertical positions (rows) of the image points, a float or an
            Nx1 Numpy array.

        Returns:
            pan: Pan angles in degrees, a single float or Nx1 Numpy array of
            floats.

            tilt: Tilt angles in degrees, a single float or Nx1 Numpy array of
            floats.

        Implementation Notes:
            The meshgrid 'g' gives a set of 2D column and row indicies in
            a pan-tilt space that is to be mapped onto a planar image: g[0]
            are column indicies, g[1] are the row indicies. 'gridsize' is the
            number of rows and number of columns in the pan-tilt space. The
            range of pan and tilt angles of the space is defined by ('pmin',
            'pmax') and ('tmin', 'tmax'), respectively.

            'ptcr' is a Nx2 array of column and row indices in the pan-tilt
            space that are to be mapped onto the planar image. The size of the
            planar image is defined by 'mapargs["foclen"]', 'mapargs["nrows"]',
            and 'mapargs["ncols"]'.

            The orientation of the planar image with respect to the cylindrical
            pan-tilt image is defined by 'mapargs["pan"]' and 'mapargs["tilt"]'.

            'imcr' is a Nx2 array where imcr[K,:] gives the position in the
            planar image of the projection of the pan-tilt point corresponding
            to ptcr[K,:].
        """

        # Project all pan/tilt angles spanning the current image to (x,y)
        # positions in the current image.
        pmin, pmax = self.pan-1.1*self.hfov/2, self.pan+1.1*self.hfov/2
        tmin, tmax = self.tilt-1.1*self.vfov/2, self.tilt+1.1*self.vfov/2
        mapargs = {'prange':(pmin, pmax), 'trange':(tmin, tmax),
                   'nrows':self.nrows, 'ncols':self.ncols,
                   'hfoclen':self.foclen, 'vfoclen':self.vfoclen,
                   'pan':self.pan, 'tilt':self.tilt}
        gridsize = 1000
        g = np.meshgrid(np.arange(gridsize),np.arange(gridsize))
        ptcr = np.hstack((g[0].reshape(g[0].size,1),g[1].reshape(g[1].size,1)))
        imcr = inv_cylindrical_proj(ptcr, **mapargs)

        # Find index of the pan/tilt coordinate that projects closest to image
        # location (ix,iy).
        d = imcr - np.array([ix,iy])
        idx = np.argmin((d*d).sum(axis=1))

        # Get pan/tilt angle corresponding to pan/tilt index.
        c, r = ptcr[idx,:]
        pan = pmax + (pmin-pmax)*c/gridsize     # pan angle (degrees)
        tilt = tmax + (tmin-tmax)*r/gridsize    # tilt angle (degrees)

        return pan,tilt


    def xy_to_pt_old(self, ix, iy):
        """
        Get the absolute pan/tilt position of the points (ix,iy) in the image
        acquired by the camera in its current pan/tilt position. (ix,iy) are the
        pixel locations relative to the image origin at the upper left corner of
        the image.

        Usage:
            pan, tilt = PTZCamera().xy_to_pt(ix, iy)

        Arguments:
            ix: Horizontal positions (columns) of the image points, a float or
            an Nx1 Numpy array.

            iy: Vertical positions (rows) of the image points, a float or an
            Nx1 Numpy array.

        Returns:
            pan: Pan angles in degrees, a single float or Nx1 Numpy array of
            floats.

            tilt: Tilt angles in degrees, a single float or Nx1 Numpy array of
            floats.
        """

        if True:
            # For debugging... delta pan = 10 deg., delta tilt = 0
            if False:
                ix = self.ncols-1
                iy = (self.nrows-1)/2
            elif False:
                ix = (self.ncols-1)/2 - self.foclen*np.tan(np.deg2rad(0))
                iy = (self.nrows-1)/2
            elif False:
                ix = 148 #(self.ncols-1)/2 - self.foclen*np.tan(np.deg2rad(1))
                iy = 532 # (self.nrows-1)/2

            pos, fp = self.get_pos_fp()      # camera absolute pos. & focus point
            n = fp - pos
            n = n/np.linalg.norm(n)          # 3D normal to camera image plane
            fp = pos + self.foclen*n         # focal point at camera focal length


            # In mapping world points to the cylindrical pan-tilt image, the
            # camera is first tilted (rotate about X axis) and then panned
            # (rotate about original Z axis). This is actually
            # accomplished by rotating the world points in the opposite
            # direction. The opposite rotation of the world points must
            # therefore first inverse pan, and then inverse tilt.
            pan = -np.arctan2(n[0], n[1])                      # rot. about Z axis
            tilt = np.arctan(n[2]/np.sqrt(n[0]**2 + n[1]**2))  # rot. about X axis
            R = Rot3d(angles=[-tilt,0,0])*Rot3d(angles=[0,0,-pan])

            xbasis = np.asarray(R*np.array([1,0,0]).reshape((3,1))).reshape((1,3)).flatten()
            ybasis = np.asarray(R*np.array([0,0,1]).reshape((3,1))).reshape((1,3)).flatten()
            # ybasis = np.cross(xbasis.reshape((1,3)),n)
            # ybasis = ybasis/np.linalg.norm(ybasis)

            # Shift image origin to center of image.
            x = ix - (self.ncols-1)/2        # world X axis parallel to image X axis
            # y = self.foclen*np.ones_like(x)  # units of foclen is pixels
            # z = (self.nrows-1)/2 - iy        # world Z axis parallel to image Y axis
            y = (self.nrows-1)/2 - iy        # invert Y axis


            # Get world coordinates.
            p = fp + x*xbasis + y*ybasis

            # v = np.asarray(R*np.vstack((x,y,z))).flatten()
            # v = v/np.linalg.norm(v)
            # print('Delta direction =', d/np.linalg.norm(d)-v)
            # p = pos + v

            # fp = np.asarray(fp).reshape((1,3)).flatten()
            # xbasis = np.asarray(xbasis).reshape((1,3)).flatten()
            # ybasis = np.asarray(ybasis).reshape((1,3)).flatten()
            # p = fp + x*xbasis + y*ybasis

        pan, tilt = self.mountedon.xyz_to_pt(p[0],p[1],p[2])

        return pan, tilt


    def ptzstr(self) ->  str:
        """
        Get a string displaying the current camera pan, tilt, and zoom setting.

        Usage:
            str = Camera.ptzstr()
        """
        return '{:^7s} ║ {:^7s} ║ {:^7s}'.format("{:.1f}˚".format(self.pan),
                                                 "{:.1f}˚".format(self.tilt),
                                                 "{:.2f}".format(self.zoom))


    def deg_to_zerone(self, pt:(float,float)) -> (float,float):
        """
        Map a degree-based pan/tilt pair to a [0,1]-based pan/tilt pair.

        Usage:
            val = Camera.deg_to_zerone(pt:(float,float))

        Arguments:
            pt: (float,float) A tuple of (pan, tilt) angles (in degrees) to
            convert to [0,1]-based angles.

        Returns:
            val: (float, float) A tuple of (pan, tilt) angles, each of which is
            in [0,1].

        Description:
            A degree-based value D is mapped to 0 when D is the minimum pan or
            tilt, and mapped to 1 when D is the maximum pan or tilt.
        """
        assert pt is not None, 'Missing "pt" argument'
        p = (pt[0] - self.minpan)/(self.maxpan - self.minpan)
        t = (pt[1] - self.mintilt)/(self.maxtilt - self.mintilt)
        return (p,t)






if __name__ == '__main__':
    import simworld as sim

    camera = PTZCamera(imsize=(1280,720), rnghfov=(2,60), rngpan=(-180,180),
                       rngtilt=(-10,20), pos=(0,0,2), pan=0, tilt=0, zoom=0)

    # Create a world.
    w = sim.SimWorld(imsize=camera.imsize, textures='textures')

    with Fig(figsize=(8,4), axpos=[121,122]) as f:
        for t in [0, 10, 20]:
            camera.set(tilt=t)
            for p in np.arange(-40,50,10):
                camera.set(pan=p)
                cur = camera.get_images(w)
                f.set(axisnum=0, image=cur['color'],
                      axistitle='P: {}, T: {}, Z: {:.2f}'.format(camera.pan,
                                                                 camera.tilt,
                                                                 camera.zoom))
                f.set(axisnum=1, image=cur['depth'], axistitle='Depth')
                f.wait(event='key_press_event')

    print('Done')