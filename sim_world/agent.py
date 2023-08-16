"""
The agent class implements agents that interact with the world.

Notes:
    The color of each target box in the displayed images indicates the target
    confidence. Blue is low, green is medium, and red is high confidence,
    respectively.

Author:
    Phil David, 2020-04-01, US Army Research Laboratory
"""

import os, datetime
import numpy as np
import simworld as sim
import vtkutils as vtu
import panoimage as pano
import ptzexplorer as ptzx
import phutils as phu
import munkres as munk
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from fig import *
from map2d import *

figsuper = None
axsuper = [None]*9
ghsuper = [None]*9
framesuper = 0
from matplotlib.colors import ListedColormap
semlabcm = ListedColormap(phu.cmdistinct(numcolors=150, amp=1.0),
                         name='SemLabels')

class Agent:
    """
    The agent class implements agents that interact with the world.
    """

    agent_count = 0

    def __init__(self, rtype='ground', name=None, env=None, pos=[0,0,0],
                 orient=[0,0,0], cam=None, mic=None, objdet=None, semseg=None,
                 map2d=None, markpath=False, panoramics=False, panores=10,
                 showgui=False, showinfo={}, verbosity='low'):
        """
        Create a new agent.

        Usage:
            agent = Agent(rtype='ground', name=None, env=None, pos=[0,0,0],
                          orient=[0,0,0], cam=None, mic=None, objdet=None,
                          semseg=None, map2d=None, markpath=False, panoramics=False,
                          panores=10, showgui=False, showinfo={}, verbosity='low')

        Arguments:
            name: (str) The name of the agent, a string. Default is 'Agent_N'
            where N is the agent number (1,2,3,...).

            rtype: (str) Agent type. Currently supported types include 'ground'
            and 'air'. Default is 'ground'.

            pos: (tuple of floats) The initial position of the agent in world
            coordinates, a 3-element array-like (X,Y,Z) where X, Y, and Z are
            floats. If a camera is mounted on the agent, then this should be the
            position of the camera mount. Default is (0,0,0).

            orient: (tuple of floats) The initial orientation of the agent in
            world coordinates. This is a 3-element array-like vector (rx, ry,
            rz) giving the rotation angles (in radians) of the agent about the
            world's X, Y, and Z axes. The front of the agent is assumed to be
            initially aligned with the world's Y axis. Default is (0,0,0).

            cam: (Camera) The agent's camera. The camera is "mounted" on the
            agent, so it moves with the agent.

            mic: (Microphone) The agent's microphone. The microphone is "mounted"
            on the agent, so it moves with the agent.

            env: (SimWorld) The agent's operating environment.

            objdet: (funtion). The object detection function. This function
            takes an RGB image (a numpy array with shape [nrows, ncols, 3]
            with values in the range [0,255]) as input and produces a
            list of dictionaries, one dictionary for each detected object.

            semseg: (function) The semantic image segmentation function. This
            function takes an RGB image (a numpy array with shape [nrows, ncols,
            3] with values in the range [0,255]) as input and produces an image
            where every pixel is assigned a semantic label.

            map2d: (Map2D) If not None, this is the 2D map of the environment
            that the agent is operating in. The agent's camera positoin and FOV
            will be displayed on this map. Default is None.

            markpath: (bool) Should the agent's path be marked in the 2D map?
            Default is False.

            panoramics: (bool) Should the agent's panoramic images be
            initialized? Default is False.

            panores: (float) Angular resolution (in pixels per degree) of all
            panoramic images. Default is 10 pixels/degree.

            showgui: (bool) Should the agent's GUI be displayed?  Default
            is False.

            showinfo: (set) Set of strings indicating what information should be
            displayed, either as text output or graphics overlays. Default is
            empty. Possible entries include: 'costmatrix', 'matches', trgbox',
            'trgid'.

            verbosity: (str) Level of information provided during normal
            operations, one of {'off', 'low', 'medium', 'high'}. Default is
            'low'
        """

        Agent.agent_count += 1
        self.rtype = rtype
        self.name = 'Agent '+str(Agent.agent_count) if name is None else name
        self.id = Agent.agent_count
        self.panoramics = panoramics
        self.panores = panores
        self.showgui = showgui
        self.showinfo = showinfo
        self.search_init_done = False
        self.busy = False
        self.show_static_plan = True
        self.markpath = markpath
        self.fig = None                             # GUI figure

        # Convert the verbosity string to a verbosity level in [0, 1, 2, 3].
        if verbosity.lower() not in {'off','low','medium','high'}:
            raise ValueError('**Verbosity**' + \
                             ' must be one of {"off", "low", "medium", "high"}')
        self.verbosity = ['off','low','medium','high'].index(verbosity.lower())

        if cam is None and mic is None:
            raise Exception(self.name,' needs a camera or a microphone')

        self.pos = np.array(pos, dtype=float)
        self.trgcm = cmjet1(101)          # colormap for displaying target boxes
        self.map2d = map2d
        if self.map2d:
            if cam is not None:
                self.map2d.AddCamera(cam, camid=self.id)
            if mic is not None:
                self.map2d.AddMicrophone(mic, micid=self.id)
        else:
            self.markpath = False       # cannot mark agent's path without a map

        # The agent's 3D orientation in the world is defined by its 3x3 rotation
        # matrix. The front of the agent is always aligned with the rotated Y
        # axis: forward = self.rot*np.array([[0,1,0]]).T.
        self.orient = np.array(orient)
        self.rot = Rot3d(angles=self.orient)

        # Small perturbation in height and angle due to terrain randomness.
        self.delta_elev = 0
        self.delta_tilt = 0

        if cam is not None:
            cam.mount(self, relpos=cam.pos, relorient=cam.orient)  # mount camera on agent
        if mic is not None:
            mic.mount(self, relpos=mic.pos, relorient=mic.orient)  # mount microphone on agent

        self.cam = cam               # the agent's camera
        self.mic = mic               # the agent's microphone
        self.env = env               # the environment the agent is operating in
        self.objdet = objdet         # the object detector
        self.semseg = semseg         # the semantic image segmenter
        self.tracked_objs = []       # list of targets -- objects of interest
        self.targetgt = []           # list of ground truth targets
        self.targets = {'person'}    # set of object types to search for

        # The following allows the user to dynamically change the displayed images.
        self.imname = ['Color', 'Depth', 'Labels',
                       'Color', 'Depth', 'Label', 'LabelGT',
                       'TrgConf', 'TCVel', 'MinZoom', 'MaxZoom',
                       'GndTruth', 'Mask']
        self.im = [None]*13          # list of all standard and panoramic images
        self.ax0 = 0                 # image # to display in axis 0 (standard size)
        self.ax1 = 1                 # image # to display in axis 1 (standard size)
        self.ax2 = 3                 # image # to display in axis 2 (panoramic)

        self.panocolor = None
        self.panodepth = None
        self.panolabel = None
        self.panotrgconf = None
        self.panotrgconfvel = None
        self.panomaxzoom = None      # maximum zoom so far
        self.panominzoom = None      # minimum zoom to satisfy PPM requirements
        self.panolabelgt = None

        if self.panoramics:
            self.InitPanoImages()
            if self.showgui:
                self.MakeGUI()       # create the agent GUI

        self.search_init_done = False

        # "mybody" is the agent's WorldObj in the simulated environment.
        self.mybody = env.NewAgent(pos[0], pos[1])

        # Initialize for agent commands.
        self.tspeed = 1.0           # translation speed, meters/sec.
        self.rspeed = 10.0          # rotation speed, degrees/sec.
        self.zspeed = 1.0           # zoom speed, zoom-units/sec.
        self.cmd_goal = 0           # goal of command
        self.cmd_progress = 0       # progress toward command completion


    def cmd_new(self, cmd=None):
        """
        Issue a new agent command.

        Usage:
            Agent.cmd_new(cmd)

        Arguments:
            cmd: (tuple) A tuple (CMD, ARG1, ARG2, ..., ARGN) where CMD is a
            string giving the command name and ARGK, K=1,...,N are the command
            arguments.

            The following commands are recognized:

                ('speed', TRANS, ROT, ZOOM) sets the default translation speed
                (meters/sec.), rotation speed (degrees/sec.), and zoom speed
                (zoom-units/sec.)

                ('forward', DISTANCE [, SPEED]) move forward DISTANCE meters at
                the given speed (meters/sec), or if no speed is provided, at the
                default translation speed.

                ('reverse', DISTANCE [, SPEED]) move backward DISTANCE meters at
                the given speed (meters/sec), or if no speed is provided, at the
                default translation speed.

                ('turn', ANGLE, RADIUS [, SPEED]) make a turn of angle (degrees)
                following an arc of radius RADIUS (meters). A positive ANGLE
                correspond to counter-clockwise rotation (i.e., a left turn),
                and a negative ANGLE corresponds to a clockwise rotation (i.e.,
                a right turn). If RADIUS <= 0, then the turn is a skid-steer
                (i.e., turn in place). SPEED, if provided, gives the linear
                speed (meters/sec) for arc turns (RADIUS > 0) or rotational
                speed (degrees/sec) for skid-steer turns (RADIUS == 0).

                ('ptz', DPAN, DTILT, DZOOM, TIME) moves the camera, over a
                period of TIME seconds, by the amount (DPAN, DTILT, DZOOM). If
                DPAN, DTILT, or DZOOM is zero or None, then that parameter of
                the camera will not change.

                ('pause', TIME) pause for TIME seconds.

        Description:
            Call Agent.cmd_step(dtime) to incrementally perform the agent
            command. Any previous incomplete command is aborted when a new
            command is issued.

            For turn in-place (skid-steer), rotational speed determines the
            speed of the manuver. However, for normal turns (that follow the arc
            of a circle), the translational speed determines the speed of the
            manuver.
        """

        if type(cmd) != tuple or type(cmd[0]) != str:
            raise ValueError('CMD must be a tuple (CMD_NAME, ARG1, ARG2, ...)')
        cmdname = cmd[0].lower()
        self.command = cmdname          # name of currently executing command
        self.cmd_progress = 0           # progress towards achieving command

        if cmdname == 'speed':

            # Set the default translation, rotation, and zoom speeds.
            if len(cmd) != 4:
                raise ValueError('Expected command ("speed", TRANS, ROT, ZOOM)')
            self.command = ''
            self.tspeed = cmd[0]
            self.rspeed = cmd[1]
            self.zspeed = cmd[2]

        elif cmdname == 'forward':

            if len(cmd) not in {2,3}:
                raise ValueError('Expected command ("forward", DIST [, SPEED])')
            if cmd[1] < 0:
                raise ValueError('Distance must be >= 0')
            self.cmd_goal = cmd[1]
            self.cmd_speed = self.tspeed if len(cmd) == 2 else cmd[2]
            self.cmd_dir = np.array(self.rot*np.array([[0,1,0]]).T).squeeze()

        elif cmdname == 'reverse':

            if len(cmd) not in {2,3}:
                raise ValueError('Expected command ("reverse", DIST [, SPEED])')
            if cmd[1] < 0:
                raise ValueError('Distance must be >= 0')
            self.cmd_goal = cmd[1]
            self.cmd_speed = self.tspeed if len(cmd) == 2 else cmd[2]
            self.cmd_dir = np.array(-self.rot*np.array([[0,1,0]]).T).squeeze()

        elif cmdname == 'turn':

            if len(cmd) not in {3,4}:
                raise ValueError('Expected command ("turn", ANGLE, RADIUS [, SPEED])')
            if cmd[2] < 0:
                raise ValueError('RADIUS must be >= 0')
            if len(cmd) == 4:
                self.cmd_speed = cmd[3]
            else:
                # Use rotational speed for skid-turn, translational speed otherwise.
                self.cmd_speed = self.tspeed if cmd[2] > 0 else self.rspeed
            self.cmd_goal = abs(cmd[1])
            self.cmd_dir = np.sign(cmd[1])  # direction of turn: + is counter-clockwise
            self.cmd_radius = cmd[2]
            if cmd[2] <= 0:
                # Skid-streer - turn in place.
                self.cmd_skid_turn = True
            else:
                # Follow the arc of a circle.
                self.cmd_skid_turn = False
                curdir = np.array(self.rot*np.array([[0,1,0]]).T).squeeze() # cur. dir. of motion
                self.cmd_turn_ctr = self.pos[:2] + self.cmd_dir * cmd[2] * \
                                        np.array([-curdir[1],curdir[0]])    # 2D center of arc

        elif cmdname == 'ptz':

            if len(cmd) != 5:
                raise ValueError('Expected command ("ptz", PAN, TILT, ZOOM, TIME)')
            self.cmd_goal = cmd[4]       # progress measured againt elapsed time
            self.cmd_pan_goal = cmd[1]
            self.cmd_tilt_goal = cmd[2]
            self.cmd_zoom_goal = cmd[3]

        elif cmdname == 'pause':

            if len(cmd) != 2:
                raise ValueError('Expected command ("pause", TIME)')
            self.cmd_goal = cmd[1]


    def cmd_step(self, dtime):
        """
        Perform one step of a previously issued command.

        Usage:
            done = Agent.cmdstep(dtime)

        Arguments:
            dtime: (float) Time (in seconds) to work on the current command.

        Returns:
            done: (bool) True if the command has been completed; otherwise,
            False.
        """

        # Increment the enviroment's clock and update dynamic objects.
        self.env.inc_time(dtime)

        if self.command == '':
            return True

        if self.command in {'forward', 'reverse'}:

            # Move forward or backward.
            trans = self.cmd_speed*dtime*self.cmd_dir
            length = np.linalg.norm(trans)              # length of current step
            if self.cmd_progress + length > self.cmd_goal:
                # Time step is large enough that total distance traveled on this
                # step would exceed the goal distance. Stop instead exactly at
                # the goal distance.
                d = self.cmd_goal - self.cmd_progress
                trans = d*trans/length
            self.inc(pos=trans)
            self.cmd_progress += np.linalg.norm(trans)
            dist = self.cmd_progress                  # total distance traveled
            S = 1.0
            if self.markpath:
                self.map2d.mark(self.pos[0], self.pos[1])

        elif self.command == 'turn':

            # Turn in place or follow the arc of a circle. Below, "dist" is the
            # total distance traveled around the arc of the turn. This distance
            # is the main variable for computing vehicle vibration. For turn-in-
            # place, the arc radius is fixed at 0.5 m. Note: the distance of an
            # arc of radius R and angle A (radians) is R*A.
            S = 1.2
            if self.cmd_skid_turn:
                # Turn in place.
                angle = self.cmd_dir*self.cmd_speed*dtime  # turn angle, degrees
                if self.cmd_progress + abs(angle) > self.cmd_goal:
                    # Time step is large enough that total angle traveled on
                    # this step would exceed the goal angle. Stop instead
                    # exactly at the goal angle.
                    angle = self.cmd_dir*(self.cmd_goal - self.cmd_progress)
                self.cmd_progress += abs(angle)
                self.inc(orient=(0,0,np.deg2rad(angle)))
                dist = 0.5*np.deg2rad(self.cmd_progress)
            else:
                # Follow the arc of a circle.
                dist = self.cmd_speed*dtime # meters along arc of circle to move
                angle = self.cmd_dir*dist/self.cmd_radius  # turn angle, radians
                if self.cmd_progress + np.rad2deg(abs(angle)) > self.cmd_goal:
                    # Time step is large enough that total angle traveled on
                    # this step would exceed the goal angle. Stop instead
                    # exactly at the goal angle.
                    angle = np.deg2rad(self.cmd_dir*(self.cmd_goal - self.cmd_progress))
                R = phu.Rot3d(angles=(0,0,angle))           # 3D rotation matrix
                v = self.pos[:2] - self.cmd_turn_ctr  # vect from ctr to cur pos
                v = np.hstack((v,0))                           # add Z dimension
                vrot = R*v.reshape((3,1))       # new pos relative to circle ctr
                vrot = np.array(vrot).squeeze()         # remove empty dimension
                trans = vrot - v                       # translation of position
                self.inc(pos=trans, orient=(0,0,angle))
                self.cmd_progress += abs(np.rad2deg(angle))
                dist = self.cmd_radius*np.deg2rad(self.cmd_progress)
                if self.markpath:
                    self.map2d.mark(self.pos[0], self.pos[1])

        elif self.command == 'ptz':

            # Move the agent's PTZ camera.
            if dtime + self.cmd_progress > self.cmd_goal:
                # Don't move PTZ past the goal position.
                dtime = self.cmd_goal - self.cmd_progress
            r = dtime/self.cmd_goal
            dpan = dtilt = dzoom = None
            self.cmd_progress += dtime
            if self.cmd_pan_goal != None:
                dpan = r*self.cmd_pan_goal
            if self.cmd_tilt_goal != None:
                dtilt = r*self.cmd_tilt_goal
            if self.cmd_zoom_goal != None:
                dzoom = r*self.cmd_zoom_goal
            self.cam.inc(dpan=dpan, dtilt=dtilt, dzoom=dzoom)

        elif self.command == 'pause':

            # Pause a fixed amount of time.
            self.cmd_progress += dtime

        else:

            raise Exception('Unrecognized  command:{}'.format(self.command))

        if self.command in {'forward', 'reverse', 'turn'}:
            # Perturb the elevation and angle of the agent to simulate random
            # variations in pose due to terrain, wind, etc. "W" is the
            # wavelength (meters) of the main perturbations. "E" and "A" are the
            # amplitudes of the elevation (meters) and angle (degrees)
            # perturbations, respectively. "V" is the amplitude of additional
            # higher-frequency vibration.
            if self.rtype == 'air':
                # Flying through air.
                W, E, A, V = 15.0, 0, 0.5, 0
            elif 'road' in self.env.map3d.get(rect=[*self.pos[:2],0.1,0.1],out='L'):
                # Traveling on a road.
                W, E, A, V = 7.0, 0.01, 0.005, 0.001
            else:
                # Traveling off-road.
                W, E, A, V = 2.0, 0.01, 0.01, 0.005
            p = 2*np.pi*S*dist/W
            self.delta_elev = E*np.sin(p) + V*np.cos(10*p)
            self.delta_tilt = A*np.cos(p) + V*np.cos(10*p)/5

        if np.around(self.cmd_progress, decimals=6) >= self.cmd_goal:
            # Current command is complete.
            done = True
            self.command = ''
            self.delta_elev = 0
            self.delta_tilt = 0
        else:
            done = False

        if self.map2d: self.map2d.Update()
        return done


    def you_drive(self, map2d=None, holdpos=False, verbose=False):
        """
        The user interactively moves the agent around the world while rendering
        the scene. When done, the agent remembers its final position and
        orientation, and the pan, tilt, and zoom of its camera.

        Usage:
            Agent.you_drive(map2d=None, holdpos=False)

        Arguments:
            map2d: (Map2D) If not None, this is the 2D map on which to display
            the agent's camera position. Default is None. If this is None, then
            the agent's internal map, if it exists, will be displayed during
            the driving process.

            holdpos: (bool) If True, do not allow the agent to move (translate);
            allow only camera pan, tilt, and zoom. Default is False.

        Description:
            The agent's orientation accounts for the horizontal viewing angle of
            the camera (what could be the camera's pan), so camera pan is always
            zero. I.e., the agent pans, not the camera. Since the agent body is
            assumed to sit flat on a plane, any vertical viewing angle is
            accounted for by the camera tilt.
        """
        if map2d is None and self.map2d:
            map2d = self.map2d

        # Start the driving simulation. Self.cam is updated by the driving
        # routine and returns the camera settings when driving ends.
        self.env.you_drive(self.cam, holdpos=holdpos, map2d=map2d)


    def get_audio(self, time=3.0, maxdist=300, verbose=False):
        """
        Get audio signal from the agent's microphone.

        Usage:
            audio =  Agent.get_audio(dtime=1.0)

        Arguments:
            time: (float) The time period (in seconds) over which to sample the
            scene.

            maxdist: (float) The maximum distance (meters) of any object from
            the microphone for its audio signal to be heard by the microphone.

        Description:
            A dictionary of the audio signal properties is returned.

        """
        if self.mic is None:
            return None

        mymap = self.map2d if verbose else None

        audio = self.mic.get_audio(self.env, duration=3.0, maxdist=maxdist,
                                   verbose=verbose, map2d=mymap)

        return audio


    def get_images(self, imlist=[], filtersize=1):
        """
        Get images from the agent's cameras.

        Usage:
            imgs = Agent.get_images(imlist=[], filtersize=1)

        Arguments:
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

                'labelgt': Single-channel ground-truth semantic label image.
                The value of each pixel is one of the IDs from the `label2id`
                dictionary.

                'labelrgb': RGB semantic label image. Objects in this image are
                colored using the RGB values from the `label_colors` list. This
                image is probably useful only for display purposes.

                'objectid': Image of object IDs, which index into the list of
                objects, self.objs.

                'depth': Single-channel, float-valued, depth image. Depth values
                are in meters. Currently, the depth and ground-truth depth
                images are identical.

                'depthgt': Single-channel, float-valued, ground-truth depth
                image. Depth values are in meters. Currently, the depth and
                ground-truth depth images are identical.

        Description:
            The camera's pan, tilt, and zoom, and the position and orientation
            of the agent (on which the camera is mounted), all determine the
            images generated.
        """

        if self.cam is None:
            return {}

        imgs = self.cam.get_images(self.env, imlist=imlist, filtersize=filtersize)

        if self.semseg is not None:
            self.semseg.process(imgs['color'])
            if 'label' in imgs.keys():
                imgs['label'] = self.semseg.imlabel.copy()
                imgs['labelrgb'] = self.semseg.imlabelrgb
            if 'labelgt' in imgs.keys():
                # Map SimWorld labels to SemanticSegmenter labels.
                intlabel = list(self.semseg.names.keys())
                strlabel = list(self.semseg.names.values())
                labelgt = imgs['labelgt']
                newlabelgt = np.zeros_like(labelgt)
                for sid, slabel in sim.id2label.items():
                    intnew = intlabel[strlabel.index(sim.simworld2mit150[slabel])]-1
                    newlabelgt[labelgt == sid] = intnew
                imgs['labelgt'] = newlabelgt

        self.stdimgs = imgs                          # images for display in GUI
        return imgs


    def show_2d_map(self, size=7, pos=(20,1200)):
        """
        Display a 2D map of the world and the agent's position in it.

        Arguments:
            size: (float) Width and height of map window on screen (inches).

            pos: (float array-like) The (row, col) position on the screen to
            place the upper left corner of the map's window.
        """
        if self.map2d == None:
            self.map2d = Map2D(maps=self.env.map3d, label_colors=sim.label_colors,
                               ptzcam=self.cam, size=size, pos=pos)


    def update_2d_map(self):
        """
        Update the agent's position on the 2D map.
        """
        if self.map2d != None:
            self.map2d.Update()


    def update_target_conf(self, zoom:float, newobjs:list) -> None:
        """
        Update the panoramic image storing the target detection confidences.

        Arguments:
            zoom (float): Zoom of camera (in [0,1]) that was used to acquire
            current list of object detections.

            newobjs (list): List of object detections.

        Description:
            Update the panoramic image storing the confidences that detected
            objects are in fact targets. All detected objects in "newobjs" are
            used to update this history. All detected objects, whether or not
            they are classified as a target, must have a target confidence
            value, object.targetconf.
        """

        if newobjs == []:
            return

        imcur = np.zeros(self.imgs['color'].shape[:2])
        prevtrgconf = self.panotrgconf.image.copy()
        panodmaxzoom = self.panodeltazoom.image.copy()

        # Sort the target confidence values from lowest to highest in case
        # objects overlap. Lower confidence values will be updated prior to
        # higher confidence values so that the higher confidence overwrites the
        # lower confidence.
        conf = [o.targetconf for o in newobjs]
        idx = np.argsort(conf)

        # Create a standard image with the confidences of all new objects.
        for k in idx:
            o = newobjs[k]
            cmin = o.xmin_orig
            cmax = o.xmin_orig + o.width_orig
            rmin = o.ymin_orig
            rmax = o.ymin_orig + o.height_orig
            imcur[rmin:rmax, cmin:cmax] = conf[k]

        # Update the panoramic image of target confidences.
        self.panotrgconf.update(imcur, self.cam.pan, self.cam.tilt,
                                self.cam.foclen, op='mask',
                                panomaskimage=(panodmaxzoom >= 0), order=0)

        # Wherever the maximum zoom has increased, compute the velocity of the
        # target confidence: (delta target conf)/(delta zoom).
        pdmz = panodmaxzoom.copy()
        panodmaxzoom[panodmaxzoom <= 0] = -1
        panotrgconfvel = (self.panotrgconf.image - prevtrgconf)/panodmaxzoom
        self.panotrgconfvel.image[panodmaxzoom > 0] = panotrgconfvel[panodmaxzoom > 0]

        if False:
            f, ax = plt.subplots(5,1,sharex=True,sharey=True)
            ax[0].imshow(self.panocolor.image)
            ax[0].set_title('Color')
            ax[1].imshow(self.panomaxzoom.image)
            ax[1].set_title('Max Zoom')
            ax[2].imshow(pdmz)
            ax[2].set_title('Delta Max Zoom')
            ax[3].imshow(self.panotrgconf.image)
            ax[3].set_title('Trg Conf')
            ax[4].imshow(self.panotrgconfvel.image)
            ax[4].set_title('Vel Trg Conf')
            plt.pause(1)
            plt.close(f)


    def camera_pos(self):
        """
        Get the absolute position and orientation of the agent's camera.

        Usage:
            pos, orient = Agent.camera_pos()

        Returns:
            pos: (Numpy.ndarray) 3D position of the camera focal point in world
            coordinates. This position is determined by the agent's position and
            the relative position of the camera on the agent.

            orient: (Numpy.ndarray) Rotation angles, in degrees, about the
            world coordinate X, Y, and Z axes. Camera pan is rotation about
            the Z axis. Camera tilt is rotation about the Y axis. Normally,
            there should never be any rotation about the X axis. These rotation
            angles account for rotation of the camera with respect to the agent
            as well as rotation of the agent itself.  So, rotation angles
        """
        pos = np.array(self.pos) + np.array(self.cam.pos)
        if np.any(np.array(self.cam.relorient) != 0):
            raise Exception('cam_pos does not yet handle nonstandard camera mounts')
        orient = np.rad2deg(self.orient) + np.array([0,self.cam.tilt,self.cam.pan])
        return pos, orient


    def MoveActors(self, pos=None, dpos=None):
        """
        Move the agent's WorldObj actors in the simulated environment.

        Arguments:
            pos: (3D array-like) The 3D position to move to. Default is None.
            Only one of "pos" or "dpos" should be not None. If any element of
            "pos" is None, then that corresponding element of the actors'
            position will not be changed.

            dpos: (3D array-like) The 3D change in position. Default is None.
            Only one of "pos" or "dpos" should be not None. If any element of
            "dpos" is None, then that corresponding element of the actors'
            position will not be changed.
        """
        if self.mybody:
            # Change the centroid of the agent's WorldObj.
            if pos is not None:
                dims = len(pos)
                if dims > 0 and pos[0] is not None: self.mybody.xctr = pos[0]
                if dims > 1 and pos[1] is not None: self.mybody.yctr = pos[1]
                if dims > 2 and pos[2] is not None: self.mybody.zctr = pos[2]
            else:
                dims = len(dpos)
                if dims > 0 and dpos[0] is not None: self.mybody.xctr += dpos[0]
                if dims > 1 and dpos[1] is not None: self.mybody.yctr += dpos[1]
                if dims > 2 and dpos[2] is not None: self.mybody.zctr += dpos[2]
            pos = (self.mybody.xctr, self.mybody.yctr, self.mybody.zctr)
            for actor in self.mybody.actors:
                actor.SetPosition(pos)


    def inc(self, pos=None, orient=None):
        """
        Increment the position and/or orientation of a agent.

        Usage:
            Agent.inc(pos=None, orient=None)

        Arguments:
            pos: (tuple of floats) The change in the agent's position in world
            coordinates, a 3-element array-like (dX,dY,dZ) where dX, dY, and dZ
            are floats. Default is None.  Any of dX, dY, or dZ may be None, in
            which case the corresponding element of the agent's position will
            not be changed.

            orient: (tuple of floats) The change in orientation of the agent in
            world coordinates. This is a 3-element array-like vector (rx, ry,
            rz) giving the change in rotation angles (in radians) of the agent
            about the world's X, Y, and Z axes. Default is None.
        """
        if pos is not None:
            dims = len(pos)
            if dims > 0 and pos[0] is not None:
                self.pos[0] += pos[0]
            if dims > 1 and pos[1] is not None:
                self.pos[1] += pos[1]
            if dims > 2 and pos[2] is not None:
                self.pos[2] += pos[2]
            self.MoveActors(dpos=pos)

        if orient is not None:
            orient = np.array(orient)
            self.orient = self.orient + orient
            self.rot = Rot3d(angles=self.orient)


    def set(self, pos=None, orient=None):
        """
        Set the position and/or orientation of a agent.

        Usage:
            Agent.set(pos=None, orient=None)

        Arguments:
            pos: (tuple of floats) The agent's position in world coordinates, a
            3-element array-like (X,Y,Z) where X, d, and Z are floats. Default
            is None. Any of X, Y, or Z may be None, in which case the
            corresponding element of the agent's position will not be changed.

            orient: (tuple of floats) The orientation of the agent in world
            coordinates. This is a 3-element array-like vector (rx, ry, rz)
            giving the rotation angles (in radians) of the agent about the
            world's X, Y, and Z axes. Default is None.
        """
        if pos is not None:
            dims = len(pos)
            if dims > 0 and pos[0] is not None: self.pos[0] = pos[0]
            if dims > 1 and pos[1] is not None: self.pos[1] = pos[1]
            if dims > 2 and pos[2] is not None: self.pos[2] = pos[2]
            self.MoveActors(pos=pos)

        if orient is not None:
            self.orient = np.array(orient) # rotaion angles (radians) about X,Y,Z axes
            self.rot = Rot3d(angles=self.orient)


    def move(self, pos=None, fdir=None):
        """
        Move the agent to a 2D position and orientation.

        Usage:
            Agent().move(pos=None, fdir=None)

        Arguments:
            pos: (float array-like) The 2D position (X,Y) of the agent. The
            elevation of the agent is not changed.  Default is None, in which
            case the agent's position does not change.

            fdir: (float array-like) The 2D direction vector (X,Y) of the front
            of the agent. The agent is always on a horizontal ground plane, so
            the z component of the front direction is always 0. Default is None,
            in which case the direction of the front of the agent does not
            change.

        Description:
            This sets the pose of the agent, which may change the absolute pan
            angle of the agent's camera. The agent's camera pose is defined with
            respect to the position and orientation of the agent. So, changing
            the agent position and/or orientation will change the camera
            position and/or orientation by the same amount.
        """
        if type(pos) == type(None):         # this needed to handle numpy arrays
            pos = self.pos
        elif type(pos) in [np.ndarray, list, tuple]:
            assert len(pos) == 2
            self.pos = np.hstack((pos, self.pos[2]))    # 3D position

        if fdir != None:
            assert len(fdir) == 2
            zrot = vang((0,1), fdir)              # rotation (rad) about Z axis
            self.orient = np.array((0, 0, zrot))  # rot angles around X,Y,Z axes
            self.rot = Rot3d(angles=self.orient)  # 3x3 rotation matrix

        if self.cam is not None:
        # Update the VTK camera position.
            vtkcam = self.env.renderers[0].GetActiveCamera()
            cpos, cfp = self.cam.get_pos_fp()
            vtkcam.SetPosition(cpos)
            vtkcam.SetFocalPoint(cfp)

        self.targetgt = []                        # list of ground truth targets
        self.search_init_done = False    # need to rescan environment after moving

        # Update all renderer windows.
        vtu.update_renderers(self.env, self.env.vtkcamera, self.env.renderers)

        fdir = [0,0,0] if self.cam is None else cfp - cpos
        if self.verbosity > 0:
            print('{:s} moved to pos = ({:.1f},{:.1f},{:.1f}), dir = ({:.2f},{:.2f})'
               .format(self.name, self.pos[0], self.pos[1], self.pos[2],
                       fdir[0], fdir[1]))

        self.update_2d_map()

        self.MoveActors(pos=(self.pos[0],self.pos[1],None))


    def move_random(self, to='ground', align='random', outskirts=0.6):
        """
        Move to a random location in the environment.

        Usage:

            Agent().move_random(to='random', align='random', outskirts=0.6)

        Arguments:

            to: (str) This is a constraint on where to move. Default is
            'ground'. It may be any of the following:

                'road_intersection' -- Move to the center of a road intersection.

                'outside_building' -- Move just outside a building, as if walking
                out a door.

                'ground' -- Move to any unoccupied ground location. Ground
                locations include roads.

            align: (str) This is a constraint on the viewing direction.  Default
            is 'random'.  It may be any of the following:

                'random': Choose a random viewing direction.

                'to_road': If located on a road, view down the road.

            outskirts: (float) The fraction of the city radius (a float in
            [0,1]) considered to be the start of the outskirts of the
            environment. "outskirts" == 0 corresponds to the center of the
            environment and "outskirts" == 1 to the outside radius. When the
            agent is in the outskirts, its viewing direction is constrained
            (proportional to its distance from the start of the outskirts) so
            that it looks more inward, providing more interesting views. Default
            is 0.6.
        """
        to = to.lower()
        align = align.lower()

        if to == 'random':
            raise NotImplementedError('Not implemented yet: to="random"')
            # pos2d, viewdir = sef.env.rand_pos()
        elif to == 'road_intersection':
            pos2d, viewdir = self.env.rand_road_intersection(align=align)
        elif to == 'outside_building':
            pos2d, viewdir = self.env.rand_building_egress()
        elif to == 'ground':
            pos2d = self.env.FindMapPts(dist=[('plant', 1, None),
                                              ('barrier', 0.5, None),
                                              ('clutter', 1, None),
                                              ('animal', 1, None),
                                              ('water', 2, None),
                                              ('person', 3, None),
                                              ('building', 5, None)],
                                        numpts=1, sample=2)

            pos2d = pos2d[0,:]
            outskirts = min(0.99, outskirts)
            d = (np.linalg.norm(pos2d) - outskirts*self.env.env_radius)/ \
                (self.env.env_radius*(1 - outskirts))
            if d > 0:
                # The agent is in the outskirts of the environment. Constrain
                # the agent's viewing direction so that it looks more inward the
                # further out it gets. In this case, 0 < D <= 1. At the start of
                # the outskirts, any view direction (in a +/-180 degree region)
                # is allowed. At the end of the outskirts (at the city radius),
                # the view direction is constrained to be in a 22.5 degree (+/-
                # 11.25 degree) region centered on the city center.
                dang = np.deg2rad(180/(2**(4*d)))    # +/- view angle from city center
                dtheta = np.random.uniform(-dang, dang)
                theta = np.arctan2(pos2d[0],-pos2d[1]) + dtheta
            else:
                theta = np.random.uniform(0, 2*np.pi)
            viewdir = (-np.sin(theta), np.cos(theta), 0)    # 2D direction of view
        else:
            raise ValueError("Don't understand where to move: {:s}".format(to))

        self.move(pos2d, viewdir[0:2])


    def xyz_to_pt(self, x, y, z):
        """
        Get the absolute pan/tilt positions of the world points (x,y,z)
        as seen by the agent in its current position.

        Usage:
            pan, tilt = Agent.xyz_to_pt(x, y, z)

        Arguments:
            x, y, z: Each is a float or Nx1 Numpy array giving the X, Y, and Z
                coordinates of a set of world points.

        Returns:
            pan: Pan angles in degrees, a single float or Nx1 Numpy array of
            floats.

            tilt: Tilt angles in degrees, a single float or Nx1 Numpy array of
            floats.

        Description:
            The world coordinate system is right-handed with Z pointing toward
            the sky.
        """

        # The absolute position of the camera in world coordinates.
        campos = np.array(self.pos).reshape((3,1)) + np.array(self.cam.relpos).reshape((3,1))

        # Translate and then rotate the world points into the agent's coordinate
        # system.
        w = np.vstack((x,y,z))    # 3xN vector needed for matrix multiplications
        w = w - campos
        w = np.linalg.inv(self.rot)*w
        wx = np.squeeze(np.asarray(w[0,:]))        # final agent coordinates
        wy = np.squeeze(np.asarray(w[1,:]))
        wz = np.squeeze(np.asarray(w[2,:]))

        # Get the pan/tilt angle from the world coordinates. Note: np.arctan2()
        # returns positive values for clockwise angles. Camera pan angles are
        # positive for counter-clockwise angles, so we negate the pan angle
        # below.
        d = np.sqrt(wx**2 + wy**2)
        tilt = np.rad2deg(np.arctan(wz/d))        # tilt in [-90,90] degrees
        pan = -np.rad2deg(np.arctan2(wx,wy))      # pan range [-180,180] degrees

        return pan, tilt


    def InitPanoImages(self):
        """
        Initialize the panoramic images and display.
        """
        # Record detection statistics to a file: [target (T/F), resolution
        # (ppm), confidence].
        self.numstats = 0
        self.maxstats = 10000
        self.stats = np.zeros((self.maxstats, 3), dtype=np.float)
        folder = os.getcwd() + '/stats'
        if not os.path.isdir(folder):
            os.makedirs(folder)
            print('Created folder "{}"'.format(folder))
        self.statsfile = phu.goodpath(folder+'/stats_{}.npy')

        # Create panoramic images. The ARES argument is the angular resolution
        # (in pixels per degree) of the panoramic images. It may affect the
        # quality of the displayed images and processing time, but shouldn't
        # significantly affect the search accuracy.
        if True:
            # Create panoramas only over pan-tilt range that the camera can
            # fully zoom in on.
            panrange = (self.cam.minpan, self.cam.maxpan)
            tiltrange = (self.cam.mintilt, self.cam.maxtilt)
        else:
            # Create panoramas over pan-tilt range that the camera can view with
            # some level of zoom, not necessarily full zoom.
            panrange = (self.cam.minpan-self.cam.maxhfov/2,
                        self.cam.maxpan+self.cam.maxhfov/2)
            tiltrange = (self.cam.mintilt-self.cam.maxvfov/2,
                         self.cam.maxtilt+self.cam.maxvfov/2)

        self.panocolor = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                        numchan=3, dtype=np.uint8)
        self.panodepth = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                        numchan=1, dtype=np.float, init=0.1)
        self.panolabel = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                        numchan=1, dtype=np.uint8)
        self.panosearchmask = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                             numchan=1, dtype=np.uint8, init=1)
        self.panolabelgt = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                          numchan=1, dtype=np.uint8)
        self.panotrgconf = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                          numchan=1, dtype=np.float)
        self.panotrgconfvel = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                             numchan=1, dtype=np.float)
        self.panominzoom = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                          numchan=1, dtype=np.float)
        self.panomaxzoom = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                          numchan=1, dtype=np.float, init=-1.0)
        self.panodeltazoom = pano.PanoImage(panrange, tiltrange, ares=self.panores,
                                            numchan=1, dtype=np.float)

        self.pause_op = False
        self.step_op = False
        self.panoramics = True


    def ClearPanoImages(self):
        """
        Clear all panoramic images.
        """
        self.panocolor.image[:] = 0
        self.panodepth.image[:] = 0.1
        self.panolabel.image[:] = 0
        self.panosearchmask.image[:] = 1
        self.panotrgconf.image[:] = 0
        self.panotrgconfvel.image[:] = 0
        self.panomaxzoom.image[:] = -1
        self.panominzoom.image[:] = 0
        self.panodeltazoom.image[:] = 0
        self.panolabelgt.image[:] = 0

        if self.showgui:
            # Update the image in the GUI.
            self.fig.set(axisnum=2, image=self.im[self.ax2], clearoverlays=True)
            self.fig.set(shownow=True)


    def UpdateGUI(self, imgs, imnames):
        """
        Update the agent's GUI.

        Arguments:
            imgs: (dict) Dictionary of images.

            imnames: (list) List of keys into dictionary "imgs" of images to
            display.  At most three image keys should be given corresponding
            to GUI image axis 0, 1, and 2. Axes 0 and 1 are for display of
            original resolution camera images. Axis 2 is for display of a
            panoramic image.
        """
        if self.showgui:
            self.gui_time.set_text(self.env.timestr())
            self.gui_frame.set_text('{:06d}'.format(self.frame))
            self.gui_ptz.set_text(self.cam.ptzstr())

            for k, imname in enumerate(imnames):
                self.fig.set(axisnum=k, image=imgs[imname],
                             axistitle=imname.title(),
                             xticks=([],[]), yticks=([],[]))

            plt.pause(0.01)


    def MakeGUI(self, winwh=(12,8), winpos=None):
        """
        Make the agent GUI.

        Arguments:
            winwh: (tuple) Tuple giving (width, height) of the GUI window, in
            inches. Default is (12,8).

            winpos: (tuple) The (row, col) position (in pixels) on the screen to
            place the upper left corner of the GUI window. The coordinates of
            the upper left corner of the screen are (0,0). If None, then the
            system choses the window position. Default is None.

        """
        self.showgui = True

        # Dictionaries holding standard and panoramic images to display.
        self.stdimgs = {}                 # this is filled by Agent.get_images()
        self.panoimgs = {'color': self.panocolor.image,
                         'depth': self.panodepth.image,
                         'label': self.panolabel.image,
                         'labelgt': self.panolabelgt.image,
                         'trgconf': self.panotrgconf.image,
                         'trgconfvel': self.panotrgconfvel.image,
                         'minzoom': self.panominzoom.image,
                         'maxzoom': self.panomaxzoom.image,
                         'deltazoom': self.panodeltazoom.image}

        # Names of selected images to display in each GUI axis (0, 1, 2).
        self.imselect = ['color', 'depth', 'color']

        # Min and max values of pixels in each single-channel image. Set to None
        # for RGB images.
        if self.semseg is None:
            numclass = max(sim.label2id.values())
        else:
            numclass = self.semseg.num_classes
        self.vminmax = {'color':None, 'depth':(0,500), 'depthgt':(0,500),
                        'label':(0,numclass), 'labelgt':(0,numclass),
                        'labelrgb':None, 'trgconf':(0,1), 'trgconfvel':(-10,10),
                        'minzoom':(0,1), 'maxzoom':(0,1), 'deltazoom':(-10,10),
                        'mask':(0,1)}

        # Create figure to display this agent's images and user interface.
        # Axis positions are given by (rstart, colstart, rowspan, colspan, label)
        # where "label" turns on or off all axis labels.
        layout = [(130,100),           # Grid layout: (numrows,numcols)
                  (0,0,38,48,0),       # 0: image in upper left   (0,0,38,48)
                  (0,50,38,48,0),      # 1: image in upper right  (0,50,38,48,0)
                  (42,0,54,98,1),      # 2: panoramic image       (42,0,50,98,0)
                  (126,0,4,8,0),       # 3: btn - init
                  (126,9,4,8,0),       # 4: btn - show path
                  (126,18,4,8,0),      # 5: btn - search
                  (126,27,4,8,0),      # 6: btn - manual
                  (126,36,4,8,0),      # 7: btn - pause
                  (126,45,4,8,0),      # 8: btn - step
                  (126,54,4,8,0),      # 9: btn - unused
                  (126,63,4,8,0),      # 10: btn - unused
                  (126,72,4,7,0),      # 11: btn - display axis 0
                  (126,80,4,7,0),      # 12: btn - display axis 1
                  (126,88,4,7,0),      # 13: btn - display axis 2
                  (126,96,4,4,0),      # 14: btn - quit
                  (119,0,4,9,0),       # 15: Text box - Time
                  (119,10,4,6,0),      # 16: Text box - Frame number
                  (119,17,4,16,0),     # 17: Text box - PTZ
                  (119,34,4,10,0),     # 18: Text box - Mode
                  (119,45,4,60,0)]     # 19: Text box - Messages

        self.fig = Fig(figsize=winwh, winpos=winpos, figtitle=self.name,
                       link=[0,1], grid=layout)

        # Define the button colors.
        c1n = (0.94,0.93,0.57)         # button color 1 - normal
        c2n = (0.94,0.84,0.55)         # button color 2 - normal
        c3n = (0.94,0.74,0.55)         # button color 3 - normal
        c4n = (0.94,0.65,0.57)         # button color 4 - normal
        c5  = (0.95,0.89,0.74)         # text box color
        c1h = (1.0,0.98,0.008)         # button color 1 - hover
        c2h = (1.0,0.74,0.008)         # button color 2 - hover
        c3h = (1.0,0.49,0.008)         # button color 3 - hover
        c4h = (1.0,0.24,0.008)         # button color 4 - hover

        # Set figure button font sizes based on figure width.
        if winwh[0] > 10:
            fs1 = 7
            fs2 = 8
        else:
            fs1 = 5
            fs2 = 6

        # Setup each axis of the user interface.
        self.fig.set(axisnum=0, image=np.zeros((*self.cam.imsize[::-1],3),dtype=int),
                     axisoff=True)
        self.fig.set(axisnum=1, image=np.zeros(self.cam.imsize[::-1],dtype=int),
                     axisoff=True)
        self.fig.set(axisnum=2, image=np.zeros(self.panocolor.image.shape,dtype=int))
        self.fig.set(axisnum=3, button=('Init',self.btn_init_search,c1n,c1h,fs2))
        self.fig.set(axisnum=4, button=('Show Plan',self.btn_show_plan,c1n,c1h,fs2))
        self.fig.set(axisnum=5, button=('Search',self.btn_deep_search,c1n,c1h,fs2))
        self.fig.set(axisnum=6, button=('Manual',self.btn_manual,c2n,c2h,fs2))
        self.fig.set(axisnum=7, button=('Pause',self.btn_pause,c2n,c2h,fs2))
        self.fig.set(axisnum=8, button=('Step',self.btn_onestep,c2n,c2h,fs2))
        self.fig.set(axisnum=9, button=(' ',lambda *args: None,c2n,c2n,fs2))
        self.fig.set(axisnum=10, button=(' ',lambda *args: None,c2n,c2n,fs2))
        self.fig.set(axisnum=11, button=(self.imselect[0], self.ax0_display,
                                         c3n, c3h, fs1))
        self.fig.set(axisnum=12, button=(self.imselect[1], self.ax1_display,
                                         c3n, c3h, fs1))
        self.fig.set(axisnum=13, button=(self.imselect[2], self.ax2_display,
                                         c3n ,c3h, fs1))
        self.fig.set(axisnum=14, button=('Quit',self.btn_quit,c4n,c4h,fs2))

        self.fig.set(axisnum=15, axistitle='Time', axisfontsize=fs2,
                     xticks=([],[]), yticks=([],[]), axiscolor=c5)
        self.gui_time = self.fig.text(0.1, 0.26, '00:00:00.00', axisnum=15,
                                      fontsize=fs2)
        self.fig.set(axisnum=16, axistitle='Frame', axisfontsize=fs2,
                     xticks=([],[]), yticks=([],[]), axiscolor=c5)
        self.gui_frame = self.fig.text(0.12,0.26, '000000', axisnum=16,
                                       fontsize=fs2)
        self.fig.set(axisnum=17, axistitle='PTZ', axisfontsize=fs2,
                     xticks=([],[]), yticks=([],[]), axiscolor=c5)
        self.gui_ptz = self.fig.text(0.04,0.26,'           ║            ║     ',
                                     axisnum=17, fontsize=fs2)
        self.fig.set(axisnum=18, axistitle='Mode',axisfontsize=fs2,
                     xticks=([],[]), yticks=([],[]), axiscolor=c5)
        self.gui_mode = self.fig.text(0.07, 0.26, ' ', axisnum=18, fontsize=fs2)
        self.fig.set(axisnum=19, axistitle='Messages', axisfontsize=fs2,
                     xticks=([],[]), yticks=([],[]), axiscolor=c5)
        self.gui_msg = self.fig.text(0.014, 0.26, 'Initializing...',
                                     axisnum=19, fontsize=fs2)

        self.axis_pause = 7           # axis of pause button
        self.axis_ax0_ctrl = 11       # axis of button to control axis 0 display

        # Display a panoramic image.
        self.fig.set(axisnum=2, image=self.panoimgs[self.imselect[2]],
                     xlabel='Pan', ylabel='Tilt', labelpos='bl', aspect='equal',
                     vminmax=self.vminmax[self.imselect[2]],
                     imextent=(self.panocolor.pmax, self.panocolor.pmin,
                               self.panocolor.tmin, self.panocolor.tmax)) # (L,R,B,T)


    def do_scan(self, pth, imagefile=None):
        """
        Do the specified scan of the environment.
        """

        maxdist = 5.0

        for k in range(pth.shape[0]):
            if k == 0:
                pthfull = [list(pth[k,:])+[True]]
            else:
                d = pth[k,:] - pth[k-1,:]
                nsteps = np.ceil(np.linalg.norm(d[0:2])/maxdist).astype(int)
                curpth = pth[k-1,:]
                delta = d/nsteps
                for j in range(nsteps):
                    curpth = curpth + delta
                    curpth[2] = max(self.cam.minhfov, curpth[2])
                    pthfull.append(list(curpth)+[j == nsteps-1])

        cnt = 0

        for curpth in pthfull:
            self.cam.set(pan=curpth[0])
            self.cam.set(tilt=curpth[1])
            self.cam.set(hfov=curpth[2])
            print('Pan = {}, Tilt = {}, Hfov = {}'.format(curpth[0],curpth[1],curpth[2]))
            if curpth[3]:
                self.process_image(updatepanos=True, detobjects=True)
            else:
                self.process_image(updatepanos=True, detobjects=False)
            if imagefile != None:
                self.fig.savefig(imagefile.format(cnt))
                cnt += 1

            if self.pause_op:
                while self.step_op is False:
                    plt.pause(0.1)
                self.step_op = False


    def add_views(self, newobjects):
        """
        Given newly detected objects, decide which ones need additional views
        and add PTZ views to acquire more information on those objects.

        Description:
            Objects are selected for follow-up views when the object's current
            detection resolution (in pixels per meter) is less than the
            maximum-pixels-per-meter parameter set during the call to
            Agent.init_search(). An additional view is also allowed for objects
            that may be cut off by an edge of the image.

            The main job that this function performs is to find the minimum
            number of new views so that all objects needing follow-up views are
            covered at the maximum-pixels-per-meter level.

            This function assumes that the camera's current FOV is the FOV which
            all objects in the `newobjects` list were detected. So, this
            function must be called before changing the camera FOV.
        """
        fovscale = 2.0                # how much to expand view around an object
        ppmscale = 1.1                # get a little more PPM than required

        # if self.cam.zoom > 0.95:
            # # Camera can't be zoomed in enough.
            # return

        # Collect data (odata) on objects that need additional views. Each row
        # of odata is [pancenter,tiltcenter,panmin,panmax,tiltmin,tiltmax,dist].
        odata = np.empty((0,7), dtype=np.float)
        for obj in newobjects:
            if not obj.closelook:
                if obj.nearedge.any():
                    # This object was near one or more edges of the image. Expand
                    # the pan/tilt size of this object. We expand by 33% the object
                    # dimension for each edge that is is near. (Note: positive pan
                    # is to the left and positive tilt is up.)
                    xmin, xmax = obj.xmin, obj.xmax
                    ymin, ymax = obj.ymin, obj.ymax
                    dx = (xmax - xmin)/3
                    dy = (ymax - ymin)/3
                    if obj.nearedge[0]: xmax += dx      # near left edge
                    if obj.nearedge[1]: xmin -= dx      # near right edge
                    if obj.nearedge[2]: ymax += dy      # near top edge
                    if obj.nearedge[3]: ymin -= dy      # near bottom edge
                    xctr = (xmin + xmax)/2
                    yctr = (ymin + ymax)/2
                    odata = np.vstack((odata, [xctr, yctr, xmin, xmax,
                                               ymin, ymax, obj.dist]))
                    obj.closelook = True          # don't repeat for this object
                elif obj.zoom < 1.0 and obj.confidence < self.highconf \
                        and obj.dist < np.Inf and obj.res < self.maxpixpermeter:
                    # This object needs a closer look.
                    odata = np.vstack((odata, [obj.xctr, obj.yctr, obj.xmin,
                                               obj.xmax, obj.ymin, obj.ymax,
                                               obj.dist]))
                    obj.closelook = True          # don't repeat for this object

                if obj.closelook:
                    # Update image of min zoom values.
                    print("Closer look at Obj {:d} from Det {:d}".format(
                        obj.id, obj.detid))
                    # cstart, rstart = self.panominzoom.pt2xy(obj.xmax, obj.ymax)
                    # cend, rend = self.panominzoom.pt2xy(obj.xmin, obj.ymin)
                    hfov = np.rad2deg(2*np.arctan(self.cam.ncols/
                                                  (2*self.minpixpermeter*obj.dist)))
                    hfov = np.maximum(np.minimum(hfov, self.cam.maxhfov), self.cam.minhfov)
                    zoom = self.cam.whatzoom(hfov=hfov)
                    self.panominzoom.image[obj.tilt_rstart:obj.tilt_rend+1,
                                           obj.pan_cstart:obj.pan_cend+1] = \
                        np.maximum(self.panominzoom.image[obj.tilt_rstart:obj.tilt_rend+1,
                                                          obj.pan_cstart:obj.pan_cend+1],
                                   zoom)

        if odata.shape[0] > 0:
            # One or more objects need closer views.

            if odata.shape[0] == 1:
                # Only one object needs a closer view.
                if True:
                    # Get the minimum camera FOV that fully covers the object.
                    hfov = odata[0,3] - odata[0,2]
                    vfov = odata[0,5] - odata[0,4]
                    minhfov = max(hfov, self.cam.minhfov)
                    minvfov = max(vfov, self.cam.minvfov)
                    hfov, _, _ = self.cam.whathvf(vfov=minvfov)
                    hfov = max(hfov, minhfov)
                    _, vfov, _ = self.cam.whathvf(hfov=hfov)
                else:
                    # OLD CODE...
                    # What HFOV will provide the required pixels/meter
                    # resolution on this object? The selected HFOV should be no
                    # more than this. Using the ppmscale parameter, we aim to
                    # get a little better resolution than what is required.
                    maxhfov = np.rad2deg(2*np.arctan(self.cam.ncols/
                                                     (2*ppmscale*self.maxpixpermeter*odata[0,6])))
                    hfov = max(maxhfov, self.cam.minhfov)
                    hfov = min(hfov, self.cam.maxhfov)

                    ehfov = fovscale*(odata[0,3] - odata[0,2])  # expanded HFOV around object
                    hfov = min(ehfov, hfov)
                    hfov = max(hfov, self.cam.minhfov)
                    _, vfov, _ = self.cam.whathvf(hfov=hfov)

                ptctr = [[np.asscalar(odata[0,0]), np.asscalar(odata[0,1]),
                          hfov, vfov]]
            else:
                # Multiple objects need closer views. Check if one zoomed-in
                # image can cover all objects and provide the required
                # resolution.

                # Get the camera horiz. & vert. FOV to cover all objects.
                hfov = odata[:,3].max() - odata[:,2].min()
                if hfov > self.cam.maxhfov:
                    # Case when FOV crosses discontinuity at +/- 180 degrees.
                    hfov = 360 + odata[:,2].min() - odata[:,3].max()
                vfov = odata[:,5].max() - odata[:,4].min()
                minhfov = max(hfov, self.cam.minhfov)
                minvfov = max(vfov, self.cam.minvfov)
                hfov, _, _ = self.cam.whathvf(vfov=minvfov)
                hfov = max(hfov, minhfov)
                _, vfov, _ = self.cam.whathvf(hfov=hfov)

                # What's the maximum HFOV that will provide the required
                # minimum pixels/meter resolution?
                dmax = odata[:,6].max()     # max distance of any object in group
                maxhfov = np.rad2deg(2*np.arctan(self.cam.ncols/(2*ppmscale*self.maxpixpermeter*dmax)))
                maxhfov = max(maxhfov, self.cam.minhfov)

                # hfov = max(maxhfov, self.cam.minhfov)  # HFOV must be in range
                # hfov = min(hfov, self.cam.maxhfov)     #    of camera

                # Expanded HFOV to view all objects in one image. This may
                # be smaller than MAXHFOV.
                # ehfov = fovscale*(odata[:,3].max() - odata[:,2].min())

                # Tentative FOVs.
                # hfov = min(ehfov, hfov)
                # hfov = max(hfov, self.cam.minhfov)
                # _, vfov, _ = self.cam.whathvf(hfov=hfov)

                if hfov <= maxhfov:
                    # Objects are close enough to view with one image.
                    # pt = odata[:,0:2].mean(axis=0)
                    width = odata[:,3].max() - odata[:,2].min()
                    height = odata[:,5].max() - odata[:,4].min()
                    pan = odata[:,2].min() + width/2
                    tilt = odata[:,4].min() + height/2
                    ptctr = [pan, tilt, hfov, vfov]
                else:
                    # Objects are spread out. Multiple views are needed.
                    nc = 2              # initial number of k-means clusters
                    done = False

                    while not done:
                        done = True
                        ptctr = []

                        # Cluster the pan/tilt centers.
                        kmeans = KMeans(n_clusters=nc).fit(odata[:,0:2])

                        # Check if each cluster can be covered by one image
                        for c in range(nc):
                            idx = np.argwhere(kmeans.labels_ == c).flatten()
                            width = odata[idx,3].max() - odata[idx,2].min()
                            height = odata[idx,5].max() - odata[idx,4].min()
                            if idx.shape[0] == 1 or width <= self.cam.minhfov \
                                      or height <= self.cam.minvfov:
                                # There is one object in this cluster or the
                                # cluster is too small to zoom in on further.

                                # xctr = odata[idx,0].mean()
                                # yctr = odata[idx,1].mean()
                                # depth = odata[idx,6].max()

                                # Get the camera horiz. & vert. FOV to cover
                                # all objects.
                                hfov = odata[idx,3].max() - odata[idx,2].min()
                                vfov = odata[idx,5].max() - odata[idx,4].min()
                                minhfov = max(hfov, self.cam.minhfov)
                                minvfov = max(vfov, self.cam.minvfov)
                                hfov, _, _ = self.cam.whathvf(vfov=minvfov)
                                hfov = max(hfov, minhfov)
                                _, vfov, _ = self.cam.whathvf(hfov=hfov)

                                # What HFOV will provide the required
                                # pixels/meter resolution on this object? We
                                # aim to get a little better resolution (via
                                # the ppmscale parameter) than what is required.
                                # maxhfov = np.rad2deg(2*np.arctan(self.cam.ncols/
                                              # (2*ppmscale*self.maxpixpermeter*depth)))
                                # hfov = max(maxhfov, self.cam.minhfov)
                                # hfov = min(hfov, self.cam.maxhfov)

                                # ehfov = fovscale*width      # expanded HFOV around object
                                # hfov = min(ehfov, hfov)
                                # hfov = max(hfov, self.cam.minhfov)
                                # _, vfov, _ = self.cam.whathvf(hfov=hfov)

                                pan = odata[idx,2].min() + width/2
                                tilt = odata[idx,4].min() + height/2
                                ptctr += [[pan, tilt, hfov, vfov]]
                            else:
                                # Multiple objects in this cluster.  Check if all
                                # can be adequately viewed with a single image.

                                # Get the camera horiz. & vert. FOV to cover
                                # all objects.
                                hfov = odata[idx,3].max() - odata[idx,2].min()
                                if hfov > self.cam.maxhfov:
                                    # Case when FOV crosses discontinuity at +/- 180 degrees.
                                    hfov = 360 + odata[idx,2].min() - odata[idx,3].max()
                                vfov = odata[idx,5].max() - odata[idx,4].min()
                                minhfov = max(hfov, self.cam.minhfov)
                                minvfov = max(vfov, self.cam.minvfov)
                                hfov, _, _ = self.cam.whathvf(vfov=minvfov)
                                hfov = max(hfov, minhfov)
                                _, vfov, _ = self.cam.whathvf(hfov=hfov)

                                # What's the maximum HFOV that will provide
                                # the required minimum pixels/meter resolution?
                                dmax = odata[idx,6].max()     # max distance of any object in group
                                maxhfov = np.rad2deg(2*np.arctan(self.cam.ncols/(2*ppmscale*self.maxpixpermeter*dmax)))
                                maxhfov = max(maxhfov, self.cam.minhfov)

                                # hfov = max(maxhfov, self.cam.minhfov) # HFOV must be in range
                                # hfov = min(hfov, self.cam.maxhfov)    #    of camera

                                # Expanded HFOV to view all objects in one image. This may
                                # be smaller than MAXHFOV.
                                # ehfov = fovscale*(odata[idx,3].max() - odata[idx,2].min())

                                # Tentative FOVs.
                                # hfov = min(ehfov, hfov)
                                # hfov = max(hfov, self.cam.minhfov)
                                # _, vfov, _ = self.cam.whathvf(hfov=hfov)

                                if hfov <= maxhfov:
                                    # Objects are close enough to view with one image.
                                    # pt = odata[idx,0:2].mean(axis=0)
                                    pan = odata[idx,2].min() + width/2
                                    tilt = odata[idx,4].min() + height/2
                                    ptctr += [[pan, tilt, hfov, vfov]]
                                else:
                                    # This cluster too large. Add a cluster and retry.
                                    nc += 1
                                    done = False
                                    break

            self.ptzexp.addviews(ptctr, timestamp=self.frame)



    def track_objects(self, newobjs, zoom=None):
        """
        Update tracked objects using new object detections.

        Usage:
            Agent.track_objects(newobjs)

        Arguments:
            newobjs: (Blob list) A list of newly detected objects. It is
            currently required that these objects are tracked (by this routine)
            before changing the position of the camera (self.cam). This is
            because the current camera position is used to filter which
            previously tracked objects the set of new objects may match to.
            It would not be difficult to eliminate this requiremnt.

            zoom: (float) Zoom of camera (in [0,1]) used to detect current
            set of new objects.

        Description:
            Associate newly detected objects with previously seen objects.

            This method maintains two lists of objects. Agent.tracked_objs is a
            list of all tracked objects, detected at any time and anywhere in
            the scene. Agent.cur_objs is a subset of Agent.tracked_objs that
            includes all objects detected during the current time step.
        """
        old_objs_inside = []
        old_objs_outside = []
        old_objs_part_outside = []
        for n in newobjs: n.matched = False

        halfhfov = self.cam.hfov/2
        halfvfov = self.cam.vfov/2
        panctr = self.cam.pan
        tiltctr = self.cam.tilt
        panmin = panctr - halfhfov
        panmax = panctr + halfhfov
        tiltmin = tiltctr - halfvfov
        tiltmax = tiltctr + halfvfov

        # Separate old objects into those predicted to be inside the current FOV
        # and those predicted to be outside of it. Only old objects inside the
        # current FOV need to be tracked. An object is "inside" the current FOV
        # if any part of the object's bounding box is inside the current FOV.
        for o in self.tracked_objs:
            if abs(o.yctr - tiltctr) < halfvfov + o.height/2 \
                     and abs(o.xctr - panctr) < halfhfov + o.width/2:
                # This object is inside the current FOV.
                old_objs_inside += [o]
                o.matched = False

                # Check if any part of the object lies outside the current FOV.
                # If so, temporarily replace the object's bounding box with a
                # box that is fully inside the FOV. This results in better match
                # (IOU) scores.
                if o.xmax > panmax or o.xmin < panmin \
                               or o.ymax > tiltmax or o.ymin < tiltmin:
                    # Object is partially outside FOV.
                    old_objs_part_outside += [o]
                    o.partoutside = True
                    o.xmin_save = o.xmin
                    o.ymin_save = o.ymin
                    o.width_save = o.width
                    o.height_save = o.height
                    xmax = min(o.xmin + o.width, panmax)
                    ymax = min(o.ymin + o.height, tiltmax)
                    o.xmin = max(o.xmin, panmin)
                    o.ymin = max(o.ymin, tiltmin)
                    o.width = xmax - o.xmin
                    o.height = ymax - o.ymin
                else:
                    # Old object is completely inside FOV.
                    o.partoutside = False
            else:
                old_objs_outside += [o]

        if newobjs and old_objs_inside:
            # Create a new-to-old object association cost matrix. The cost
            # depends only on the position of the objects' bounding boxes.
            maxcost = 0.9
            costs = 1 - iou_mat(newobjs, old_objs_inside)
            costs = costs.tolist()

        # Restore the original bounding boxes of old objects partially outside
        # the current FOV.
        for o in old_objs_part_outside:
            o.xmin = o.xmin_save
            o.ymin = o.ymin_save
            o.width = o.width_save
            o.height = o.height_save

        if newobjs and old_objs_inside:
            if 'costmatrix' in self.showinfo:
                # Print the cost matrix.
                print('\nNew-to-old cost matrix:')
                print('        Old')
                print('New', end='')
                for o in old_objs_inside:
                    print('{:7d}'.format(o.id), end='')
                print('')
                for idxn in range(len(newobjs)):
                    print('{:3d}: '.format(newobjs[idxn].id), end='')
                    for idxo in range(len(old_objs_inside)):
                        if costs[idxn][idxo] == munk.DISALLOWED:
                            print('    X  ', end='')
                        else:
                            print('  {:5.2f}'.format(costs[idxn][idxo]), end='')
                    print('')

            # Find the least-cost assignment of new to old objects.
            m = munk.Munkres()
            assign = m.compute(costs)

            # Update matched old objects.
            if 'matches' in self.showinfo: print('\nMatches:')
            for idxn, idxo in assign:
                if costs[idxn][idxo] < maxcost:
                    n = newobjs[idxn]
                    o = old_objs_inside[idxo]
                    n.matched = True
                    o.matched = True
                    o.detid = n.id        # ID of most recent matching detection
                    if 'matches' in self.showinfo: print('  new {} --> old {}'.format(n.id, o.id))
                    if n.zoom > o.zoom or n.confidence > o.confidence:
                        # New object is greater resolution or higher confidence
                        # than old object.
                        o.label = n.label
                        if o.partoutside:
                            # The old object is partially outside the current
                            # FOV. Update old object based on new object.
                            xmax = max(o.xmin+o.width, n.xmin+n.width)
                            ymax = max(o.ymin+o.height, n.ymin+n.height)
                            o.xmin = min(o.xmin, n.xmin)
                            o.ymin = min(o.ymin, n.ymin)
                            o.width = xmax - o.xmin
                            o.height = ymax - o.ymin
                            o.area = o.width*o.height
                            o.xctr = o.xmin + o.width/2
                            o.yctr = o.ymin + o.height/2
                            o.dist = min(o.dist, n.dist)
                            o.confidence = max(n.confidence, o.confidence)
                            o.res = max(n.res, o.res)
                            o.zoom = n.zoom
                            o.gt = n.gt
                            o.target |= n.target
                        else:
                            # The old object is fully inside the current FOV.
                            # The new object's position should be more accurate
                            # than old object's. Replace old object with new
                            # object.
                            o.xmin = n.xmin
                            o.ymin = n.ymin
                            o.width = n.width
                            o.height = n.height
                            o.area = n.area
                            o.xcenter = n.xctr
                            o.ycenter = n.yctr
                            o.dist = n.dist
                            o.confidence = n.confidence
                            o.zoom = n.zoom
                            o.res = n.res
                            o.gt = n.gt
                            o.partoutside = False
                            o.target = n.target
                    else:
                        # Resolution of new object detection is lower or equal
                        # to that of the old object. Update old object with the
                        # new object's position only if the old object is
                        # partially outside the current FOV.
                        if o.partoutside:
                            # Update old object based on new object.
                            xmax = max(o.xmin+o.width, n.xmin+n.width)
                            ymax = max(o.ymin+o.height, n.ymin+n.height)
                            o.xmin = min(o.xmin, n.xmin)
                            o.ymin = min(o.ymin, n.ymin)
                            o.width = xmax - o.xmin
                            o.height = ymax - o.ymin
                            o.area = o.width*o.height
                            o.xctr = o.xmin + o.width/2
                            o.yctr = o.ymin + o.height/2
                            o.dist = min(o.dist, n.dist)
                            o.confidence = max(n.confidence, o.confidence)
                            o.res = max(n.res, o.res)
                            o.gt = n.gt
            if 'matches' in self.showinfo: print('\n')

        # Save final object lists. Old objects in the current FOV whose zoom
        # value is less or equal to the current zoom and which don't match any
        # new object are eliminated from the set of tracked objects.
        unmatched_new = [n for n in newobjs if not n.matched]
        matched_old = []
        keep_old_inside = []
        for o in old_objs_inside:
            o.keep = False
            if o.matched:
                o.keep = True
                keep_old_inside += [o]
                matched_old += [o]
            elif o.partoutside or o.zoom > zoom:
                o.keep = True
                keep_old_inside += [o]
        self.tracked_objs = old_objs_outside + keep_old_inside + unmatched_new
        self.cur_objs = matched_old + unmatched_new

        if self.showgui:
            # Update the panoramic image display of detected objects.

            # Remove displayed boxes of old objects that are no longer tracked.
            for o in old_objs_inside:
                if not o.keep:
                    if o.patch: o.patch.remove()
                    if o.text: o.text.remove()

            # Display/update new objects. The color of each target box indicates
            # the detection confidence. Blue is low, green is medium, and red is
            # high confidence, respectively.
            for o in self.cur_objs:
                if o.patch:
                    o.patch.remove()
                    o.patch = None
                if o.target and o.confidence >= self.minconftarget:
                    o.patch = self.fig.draw(axisnum=2, rect=o, shownow=False,
                                edgecolor=self.trgcm[int(100*o.confidence+0.5),:])
                    if 'id' in self.showinfo:
                        # Write ID next next to each object.
                        if o.text: o.text.remove()
                        o.text = self.fig.text(o.xctr, o.yctr, '{}'.format(o.id),
                                               axisnum=2, color='w', fontsize=6,
                                               fontweight='bold', shownow=False)


    def update_searchmask(self):
        """
        Update the panoramic search mask image.

        Description:
            Set pixels in the panoramic search mask to 0 wherever none of the
            search location labels (self.contexts) are true. This eliminates any
            requirement to zoom in on these pixels.
        """
        minholesize = 0.25           # hole size (degrees) for closing operation
        mask = self.panosearchmask.image

        # Pixels in the desired search locations are initially set to 0, then
        # the image is complemented.
        mask[:] = 1
        for k in self.contexts:
            mask[self.panolabel.image == k] = 0
        mask = 1 - self.panosearchmask.image

        # Remove small 0-valued holes in the mask.
        se = ndi.generate_binary_structure(2, 2)
        numiters = int(round(self.panocolor.ares*minholesize/2))
        mask = ndi.binary_closing(mask, structure=se, iterations=numiters,
                                  border_value=1, output=mask)

        self.panosearchmask.image[:] = mask[:]


    def get_min_zoom(self, imgs:dict, where={'ground','road','building'}) -> np.ndarray:
        """
        Get an image showing the minimum zoom needed at each pixel.

        Arguments:
            where: (set) Set of semantic labels (each a string) where objects
            being searched for may be found. Once the initial search is
            complete, only areas of the scene containing at least one of these
            labels will be further examined.

            imgs: Dictionary of images, including 'label' and 'depth'.

        Description:
            Given the depth (in meters) of a scene point (corresponding to a
            pixel in the image), compute the minimum camera zoom needed to view
            that scene point such that the image of that scene point has a given
            minimum spatial resolution (in pixels per horizontal meter). This is
            done for all pixels in the image.
        """
        imdepth = imgs['depth']
        imlabel = imgs['label']

        assert self.minpixpermeter > 0, 'Parameter "minpixpermeter" must be > 0'
        assert np.all(imdepth > 0), 'Depth image must be > 0'

        # Get the minimum zoom based on the depth of each scene point.
        hfov = np.rad2deg(2*np.arctan(self.cam.ncols/
                                      (2*self.minpixpermeter*imdepth)))
        hfov = np.maximum(np.minimum(hfov, self.cam.maxhfov), self.cam.minhfov)
        imzoom = (self.cam.maxhfov-hfov)/(self.cam.maxhfov-self.cam.minhfov)

        # Map semantic labels from strings to integers.
        contexts = self.semlab_str2int(where)

        # Mask out scene points that don't need to be viewed again due to their
        # semantic labels not being of the type where objects of interest might
        # appear. Pixels in the desired search locations are initially set to 0,
        # then the image is complemented.
        immask = np.ones_like(imdepth, dtype=bool)
        for k in contexts:
            immask[imlabel == k] = 0
        immask = np.invert(immask)

        minholesize = 3           # hole size for closing operation
        if minholesize > 0:
            # Remove small 0-valued holes in the mask image.
            se = ndi.generate_binary_structure(2, 2)
            numiters = int(np.ceil(minholesize/2))
            immask = ndi.binary_closing(immask, structure=se, iterations=numiters,
                                      border_value=1, output=immask)

        # Apply the semantic label mask to the zoom image.
        imzoom[immask == False] = 0

        return imzoom


    def update_minzoom(self):
        """
        Update the panoramic image of required minimum zoom values.

        Description:
            Given the depth (in meters) of a scene point (corresponding to a
            pixel in the image), compute the minimum camera zoom needed to view
            that scene point such that the image of that scene point has a given
            minimum spatial resolution (in pixels per horizontal meter). This is
            done for all pixels in the panoramic image.
        """
        assert self.minpixpermeter > 0, 'Parameter "minpixpermeter" must be > 0'
        assert np.all(self.panodepth.image > 0), 'Depth image must be > 0'

        hfov = np.rad2deg(2*np.arctan(self.cam.ncols/
                                      (2*self.minpixpermeter*self.panodepth.image)))
        hfov = np.maximum(np.minimum(hfov, self.cam.maxhfov), self.cam.minhfov)
        self.panominzoom.image[:] = (self.cam.maxhfov-hfov)/(self.cam.maxhfov-self.cam.minhfov)


    def update_maxzoom(self):
        """
        Update the panoramic images recording the maximum zoom (panomaxzoom)
        used to view each pan/tilt position, and the pan/tilt positions where
        the max zoom has increased (panozoominc).
        """
        panmin = self.cam.pan - self.cam.hfov/2
        panmax = panmin + self.cam.hfov
        tiltmin = self.cam.tilt - self.cam.vfov/2
        tiltmax = tiltmin + self.cam.vfov
        rstart = np.argmin(abs(self.panomaxzoom.row2tilt - tiltmax))
        rend = np.argmin(abs(self.panomaxzoom.row2tilt - tiltmin))
        cstart = np.argmin(abs(self.panomaxzoom.col2pan - panmax))
        cend = np.argmin(abs(self.panomaxzoom.col2pan - panmin))
        prevmaxzoom = self.panomaxzoom.image[rstart:rend+1, cstart:cend+1].copy()
        self.panomaxzoom.image[rstart:rend+1, cstart:cend+1] = \
            np.maximum(self.panomaxzoom.image[rstart:rend+1, cstart:cend+1],
                       self.cam.zoom)
        self.panodeltazoom.image[rstart:rend+1, cstart:cend+1] = \
            self.panomaxzoom.image[rstart:rend+1, cstart:cend+1] > prevmaxzoom
        print('')


    def save_stats(self):
        datetimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file = self.statsfile.format(datetimestr)
        np.save(file, self.stats[0:self.numstats,:])
        print('Stats saved to "{}"'.format(file))


    def load_stats(self):
        pass


    def btn_init_search(self, event):
        """
        Code to run when GUI "Init" button is pressed.
        """
        print('Doing initial scan...')
        self.InitSearch(where={'ground','road','building'},
                        targets={'person'},
                        objofinterest={'person','dog','cow'},
                        minpixpermeter=100, maxpixpermeter=200, highconf=0.9,
                        saveguifile=None, saveimgfile=None)
        print('Done')


    def btn_deep_search(self, event):
        """
        Code to run when GUI "Search" button is pressed.
        """
        print('Doing full scan...')
        if not self.search_init_done:
            self.InitSearch(targets={'person'}, where={'ground','road','building'},
                            minpixpermeter=25, maxpixpermeter=100, highconf=0.9)
        self.ptzexp.reset()
        self.DeepSearch(method='baseline')
        print('Done')


    def stop(self, event):
        """
        Code to run when GUI "Stop" button is pressed.
        """
        print('Stop')


    def detect_objs(self, imcolor, imdepth, zoom=None, bottomdepth=False):
        """
        Detect objects of interest in an image.

        Usage:
            newobjs = detectobjs(imcolor, imdepth, zoom, bottomdepth=False)

        Arguments:
            imcolor: (Numpy.ndarry) The color image, which is input to the
            object detector.  This image has shape [rows, cols, channels] where
            color channels are ordered RGB and must be [0,255]-valued.
            [0,1]-valued colors will *not* work.

            imdepth: (Numpy.ndarry) The corresponding depth image.

            zoom: (float) Zoom of camera (in [0,1]) that was used to acquire
            current images.

            bottomdepth: (bool) Should the depth of a detected object be set to
            the depth map value at the center of the bottom edge of the
            detection's bounding box (bottomdepth == True), or from the depth
            map value at the middle of its bounding box (bottomdepth == False)?
            Default is False.

        Returns:
            newobjs: A list of object detections (Blob objects).

        Description:
            The types of objects to be detected by this method are specified in
            the call to Agent.InitSearch() via the "objofinterest" argument.
            Any detection with one of these labels, regardless of the confidence
            score, is returned in "newobjs". If "objofinterest" contains the
            string "all", then all detected objects are considered interesting
            and are returned by this function.

            "newobjs" returns a list of Blob objects that describe all newly
            detected objects. Each blob will have the object's bounding box in
            the original image (in pixel coordinates) as well as in the
            panoramic image (in pan/tilt angles). The bounding box of blob B in
            the original image is given in pixel coordinates:

                [B.xmin_orig, B.ymin_orig, B.width_orig, B.height_orig]

            and in the panoramic image, the bounding box is given in pan/tilt
            angles (in degrees):

                [B.xmin, B.ymin, B.width, B.height].
        """
        assert zoom != None, 'Argument camera "zoom" must be assigned.'
        nrows = imcolor.shape[0]
        ncols = imcolor.shape[1]

        # Define a rectangle to identify when blobs are close to the edge of
        # an image.
        r = 0.05
        leftside = r*ncols
        rightside = (1-r)*ncols
        topside = r*nrows
        bottomside = (1-r)*nrows

        # Run the object detector. Detections are saved in self.objdet.dets.
        self.objdet.process(imcolor, targets=self.targets)

        getalldetects = 'all' in self.objofinterest

        # Get object bounding boxes.
        newobjs = []

        if 'tracks' in self.showinfo: print('\nDetections:')

        for det in self.objdet.dets:
            if getalldetects or det['label'] in self.objofinterest:
                b = Blob(props=det, pixels=True)
                b.objclass = sim.objclass[b.label] if b.label in sim.objclass.keys() else b.label
                b.incurframe = True
                b.zoom = zoom
                b.target = b.label in self.targets
                b.targetconf = det['targetconf']
                b.detid = b.id
                b.closelook = False

                if self.showgui:
                    # Overlay some information on the displayed original image.
                    if 'box' in self.showinfo:
                        self.fig.draw(axisnum=0, rect=b, shownow=False,
                              edgecolor=self.trgcm[int(100*b.targetconf+0.5),:])
                    if 'id' in self.showinfo:
                        self.fig.text(b.xctr, b.yctr, '{}'.format(b.id),
                                      axisnum=0, color='w', shownow=False,
                                      fontsize=6, fontweight='bold')

                # Get the position of the blob in the current image.
                x0 = int(b.xmin)              # left
                x1 = int(b.xmin+b.width)      # right+1
                x2 = int(b.xmin+b.width/2)    # horizontal center
                y0 = int(b.ymin)              # top
                y1 = int(b.ymin+b.height)     # bottom+1
                y2 = int(b.ymin+b.height/2)   # vertical center
                x0 = max(0, x0)
                x1 = min(ncols, x1)
                y0 = max(0, y0)
                y1 = min(nrows, y1)

                # Save position of blob in original image.
                b.xmin_orig = x0          # b.xmin
                b.ymin_orig = y0          # b.ymin
                b.width_orig = x1 - x0    # b.width
                b.height_orig = y1 - y0   # b.height
                b.area_orig = b.width*b.height

                # Check if the blob is near any edge of the image.
                b.nearedge = np.zeros(4, dtype=int)
                b.nearedge[0] = int(x0 <= leftside)
                b.nearedge[1] = int(x1 >= rightside)
                b.nearedge[2] = int(y0 <= topside)
                b.nearedge[3] = int(y1 >= bottomside)

                # Get the depth (meters) and resolution (pixels per horizontal
                # meter) of the detection.
                if bottomdepth:
                    b.dist = imdepth[y1-1, x2]
                else:
                    b.dist = imdepth[y2, x2]
                b.res = self.cam.ncols/(2*b.dist*np.tan(np.deg2rad(self.cam.hfov/2)))
                b.zoom = self.cam.zoom

                # Record groundtruth: does this blob overlaps with a true object
                # having the same label?
                # if np.any(labelgt[y0:y1,x0:x1] == sim.label2id[b.objclass]):
                    # b.gt = b.objclass
                # else:
                    # b.gt = 'unknown'
                b.gt = 'unknown'

                # Convert image coordinates to pan/tilt angles.
                pmax, tmax = self.cam.xy_to_pt(b.xmin, b.ymin)
                pmin, tmin = self.cam.xy_to_pt(b.xmin+b.width, b.ymin+b.height)
                b.xmin, b.ymin = pmin, tmin
                b.xmax, b.ymax = pmax, tmax
                b.width = pmax - pmin
                b.height = tmax - tmin
                b.xctr = (pmax + pmin)/2
                b.yctr = (tmax + tmin)/2
                b.area = b.width*b.height

                # Get row and column numbers of blob in pan/tilt image.
                b.pan_cstart, b.tilt_rstart = self.panominzoom.pt2xy(b.xmax, b.ymax)
                b.pan_cend, b.tilt_rend = self.panominzoom.pt2xy(b.xmin, b.ymin)

                # Is any part of this detection in a search region?
                in_search_region = np.max(self.panosearchmask.image[
                                                    b.tilt_rstart:b.tilt_rend+1,
                                                    b.pan_cstart:b.pan_cend+1])
                if not in_search_region:
                    continue

                # Initialize display of this blob.
                b.patch = None
                b.text = None

                # Keep detections only if in the camera's pan/tilt range.
                if b.xctr >= self.cam.minpan and b.xctr <= self.cam.maxpan and \
                   b.yctr >= self.cam.mintilt and b.yctr <= self.cam.maxtilt:
                    newobjs += [b]

                    if 'tracks' in self.showinfo:
                        print(('  {} {}, conf={:.2f}, tconf={:.1g}, pt={:.1f}º,{:.1f}º,'+
                               ' wh={:.1f}º,{:.1f}º, dist={:.1f}, res={:.1f}, Z={:.2f}'). \
                                    format(b.label, b.id, b.confidence, b.targetconf,
                                           b.xctr, b.yctr, b.width, b.height,
                                           b.dist, b.res, b.zoom))

                if self.numstats < self.maxstats:
                    self.stats[self.numstats,:] = [b.gt == b.label, b.res, b.confidence]
                    self.numstats += 1
                else:
                    print('Detection statistics array is full!')

        if self.showgui: self.fig.set(shownow=True)

        return newobjs


    def cur_image(self, image='color'):
        """
        Get the last acquired camera image.

        Usage:
            Agent().cur_image()
        """
        image = image.lower()
        if image == 'color':
            return self.im[0]
        elif image == 'depth':
            return self.im[1]
        elif image == 'label':
            return self.im[2]


    def mycamfov(self, pano, cam, f, axisnum=0):
        """
        Display the current camera FOV in a figure of the panoraminc image.
        """
        try:
            # Remove the previous FOV graphic.
            for p in self.mypatchcurfov:
                p.remove()
        except:
            pass
        self.mypatchcurfov = []

        # Get pan/tilt angles at evenly spaced points around edge of image.
        cols = np.linspace(0, cam.ncols-1, num=10, endpoint=True)
        rows = np.linspace(0, cam.nrows-1, num=10, endpoint=True)
        x = np.hstack((cols, np.full_like(rows, cam.ncols-1), cols[-1::-1],
                       np.zeros_like(rows)))
        y = np.hstack((np.zeros_like(cols), rows, np.full_like(cols, cam.nrows-1),
                       rows[-1::-1]))
        pan, tilt = cam.xy_to_pt(x, y)
        pt = np.column_stack((pan, tilt))

        # Check if the FOV overlaps two sides of the FOV discontinuity.
        discont = (np.abs(pan[:-1] - pan[1:]) > 180).nonzero()
        if discont[0] != []:
            # The FOV overlaps the discontinuity at +/- 180 degrees. Draw the
            # FOV in two pieces, one on each side of the discontinuity.
            print("Discontinuous FOV")
            p1, p2 = discont[0]
            top = (tilt[p1]+tilt[p1+1])/2  # tilt at FOV top on discontinuity
            bot = (tilt[p2]+tilt[p2+1])/2  # tilt at FOV bottom on discontinuity
            pt1 = np.vstack(([pano.pmax,top],pt[p1+1:p2+1,:],[pano.pmax,bot]))
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt1, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt1, linewidth=1,
                                           edgecolor='b', linestyle=':'))
            pt2 = np.vstack((pt[:p1+1,:], [pano.pmin,top],
                             [pano.pmin,bot], pt[p2+1:,:]))
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt2, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt2, linewidth=1,
                                           edgecolor='b', linestyle=':'))
        else:
            # Draw a dashed white rectangle on top of a solid blue rectangle.
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.mypatchcurfov.append(f.draw(axisnum=axisnum, poly=pt, linewidth=1,
                                           edgecolor='b', linestyle=':'))

    def process_image(self, imgs=None, updatepanos=True, detobjects=True):
        """
        Process the camera's current image.

        Arguments:
            imgs: (dict) Dictionary of images to process. If None, then new
            images are acquired from the camera. Otherwise, the `imgs`
            dictionary must include the following images (keys in the dict):
            `color`, `depth`, `label`, `labelgt`.  Default in None.

            updatepanos: (bool) Should the panoramic images be updated using the
            current camera images? When viewing part of the scene that has
            already been viewed from the same location, the displayed panoramic
            images should not change significantly, so probably don't need to be
            updated, thus saving some processing time. Dafault is True.

            detobjects: (bool) Should the object detector be applied to the
            images? In some situations, it may be desirable to collect a set
            of images before analyzing any of them. Default is True.

        Description:
            Upon return, Agent.imgs is a dictionary of the images processed by
            the current call. See the`imgs` argument above.

            If "detobjects" is True, then upon return, Agent.cur_objs will
            contain a list of Blob objects that describe all newly detected
            objects. Each blob will have the object's bounding box in the
            original image (in pixel coordinates) as well as in the panoramic
            image (in pan/tilt angles). The bounding box of blob B in the
            original image is
                [B.xmin_orig, B.ymin_orig, B.width_orig, B.height_orig]
            and in the panoramic image, the bounding box is given by
                [B.xmin, B.ymin, B.width, B.height].
        """

        # Get camera images to process.
        curzoom = self.cam.zoom
        if imgs == None:
            imgs = self.get_images()
        self.imgs = imgs

        if self.saveimgfile:
            fname = self.saveimgfile.format(self.frame)
            imageio.imwrite(fname, imgs['color'])

        if self.showgui:
            # Update the standard images in the user interface.
            self.fig.set(axisnum=0, image=self.stdimgs[self.imselect[0]],
                         axistitle=self.imselect[0].title(),
                         xticks=([],[]), yticks=([],[]))
            self.fig.set(axisnum=1, image=self.stdimgs[self.imselect[1]],
                         axistitle=self.imselect[1].title(),
                         xticks=([],[]), yticks=([],[]))

        self.cur_objs = []

        # Update panoramic image of the maximum zoom used to view each
        # discretized pan/tilt position. Panodeltazoom is the amount of positive
        # increase in the maximum zoom ever used at each pan/tilt location.
        # Panodeltazoom == -1 prior to viewing a pan/tilt location and
        # Panodeltazoom == 0 on the 1st view of a pan/tilt location (or if
        # viewed with zoom == 0).
        prevmaxzoom = self.panomaxzoom.image.copy()
        self.panomaxzoom.update(self.cam.zoom*np.ones(imgs['color'].shape[:2],
                                                      dtype=np.float),
                                self.cam.pan, self.cam.tilt, self.cam.foclen,
                                op='max')
        self.panodeltazoom.image[:,:] = self.panomaxzoom.image - prevmaxzoom
        self.panodeltazoom.image[prevmaxzoom < 0] = -1
        self.panodeltazoom.image[np.logical_and(prevmaxzoom < 0,
                                                self.panomaxzoom.image >= 0)] = 0

        if updatepanos:
            # Assuming the agent is stationary, these panoramic images should
            # need to be updated only during the agent's initial scan. Once the
            # initial scan is done, these should only change when the agent
            # moves to a new location and a needs to do a new initial scan. This
            # assumes that the scene is static and that the spatial resolution
            # of the initial images is sufficient to capture all pertinent
            # contextual information about the environment.
            self.panocolor.update(imgs['color'], self.cam.pan, self.cam.tilt,
                                  self.cam.foclen)
            self.panolabelgt.update(imgs['labelgt'], self.cam.pan, self.cam.tilt,
                                  self.cam.foclen, order=0)
            self.panolabel.update(imgs['label'], self.cam.pan, self.cam.tilt,
                                  self.cam.foclen, order=0)
            self.panodepth.update(imgs['depth'], self.cam.pan, self.cam.tilt,
                                  self.cam.foclen, order=0)

        if detobjects:
            # Detect objects in the new images.
            newobjs = self.detect_objs(imgs['color'], imgs['depth'], zoom=curzoom)

            # Update target confidence and confidence velocity panoramic images.
            self.update_target_conf(curzoom, newobjs)

            # Update tracked objects.
            self.track_objects(newobjs, zoom=curzoom)

            if self.search_phase == 'deep':
                # Check if any additional views are needed.
                self.add_views(self.cur_objs)

            if self.showgui:
                if len(self.cur_objs) > 0:
                    self.gui_msg.set_text('Detected '+ self.objstr(self.cur_objs))
                else:
                    self.gui_msg.set_text('No objects of interest')
                plt.pause(0.01)

        if self.showgui:
            self.fig.set(axisnum=2, image=self.panoimgs[self.imselect[2]],
                         clearoverlays=False)
            self.fig.set(shownow=True)


        if False:
            # Show a figure with a lot of different images.
            global figsuper, axsuper, ghsuper, framesuper, semlabcm
            framesuper += 1
            if framesuper == 1:
                plt.subplots_adjust(wspace=0.25, hspace=0.25)
                figsuper = Fig(figsize=(12,6.75),   # (width, height)
                               grid=[(3,9),
                                     (0,0,1,2), (0,2,1,2), (0,4,1,2),
                                     (1,0,1,3), (1,3,1,3), (1,6,1,3),
                                     (2,0,1,3), (2,3,1,3), (2,6,1,3)],
                               figtitle="Where's Waldo?", link=[3,4,5,6,7,8])

            figsuper.set(axisnum=0, image=self.imgs['color'],
                         axistitle='Color', axisfontsize=9, xticks=([],[]), yticks=([],[]))
            figsuper.set(axisnum=1, image=self.imgs['depth'],
                         vminmax=(0,100), cmapname='jet',
                         axistitle='Depth', axisfontsize=9, xticks=([],[]), yticks=([],[]))
            figsuper.set(axisnum=2, image=self.imgs['label'],
                         vminmax=(0,150), cmap=semlabcm,
                         axistitle='Semantic Label', axisfontsize=9, xticks=([],[]), yticks=([],[]))

            # figsuper.set(axisnum=3, image=self.panocolor.image, aspect='equal',
                         # imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   # self.panocolor.tmin, self.panocolor.tmax),
                         # axistitle='Color', axisfontsize=9, xticks=([],[]), yticks=([],[]))
            figsuper.set(axisnum=3, image=self.panocolor.image,
                         axistitle='Color', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax) )
            self.mycamfov(self.panocolor, self.cam, figsuper, axisnum=3)
            plt.pause(1)
            d = self.panodepth.image.copy()
            d[d == np.inf] = 500
            figsuper.set(axisnum=4, image=d,
                         vminmax=(0,100), cmapname='jet',
                         axistitle='Depth', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))
            figsuper.set(axisnum=5, image=self.panolabel.image,
                         vminmax=(0,150), cmap=semlabcm,
                         axistitle='Semantic Label', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))
            figsuper.set(axisnum=6, image=self.panomaxzoom.image,
                         vminmax=(-1,1), cmapname='jet',
                         axistitle='Max Zoom', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))
            figsuper.set(axisnum=7, image=self.panotrgconf.image,
                         vminmax=(-1,1), cmapname='jet',
                         axistitle='Target Confidence', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))
            figsuper.set(axisnum=8, image=self.panotrgconfvel.image,
                         vminmax=(-5,5), cmapname='jet',
                         axistitle='Target Conf. Vel.', axisfontsize=9, xticks=([],[]), yticks=([],[]),
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))
            # self.mycamfov(self.panocolor, self.cam, figsuper, axisnum=3)
            plt.pause(1)
            figsuper.savefig(filename='waldo/frame_{:06d}.png'.format(framesuper))
            print('')


    def semlab_str2int(self, slabels:list) -> list:
        """
        Map SimWorld semantic string labels to integer semantic labels. The
        integer labels will be from the SemanticSegmenter class, if it exists,
        otherwise from the SimWorld class.
        """
        intlabels = []
        if self.semseg is not None:
            # Map SimWorld string labels to SemanticSegmenter integer labels.
            intlabel = list(self.semseg.names.keys())
            strlabel = list(self.semseg.names.values())
            for label in slabels:
                try:
                    c = intlabel[strlabel.index(sim.simworld2mit150[label.lower()])]-1
                    intlabels.append(c)
                except:
                    raise Exception('"{:s}" is not a recognized semantic label'\
                                    .format(label.lower()))
        else:
            # Map SimWorld string labels to SimWorld integer labels.
            for label in slabels:
                try:
                    intlabels.append(sim.label2id[label.lower()])
                except:
                    raise Exception('"{:s}" is not a recognized semantic label'\
                                    .format(label.lower()))

        return intlabels


    def prep_deep_search(self):
        """
        Prepare for the deep search.
        """

        # Map string labels to integer labels.
        self.contexts = self.semlab_str2int(self.search_where)

        # Update regions that should be searched.
        self.update_searchmask()

        # Update the required minimum zoom at all scene points.
        self.update_minzoom()

        # Plan the deep PTZ exploration of the scene.
        self.ptzexp = ptzx.PTZExplorer(self.panodepth, self.panolabel,
                                       self.panominzoom, self.panosearchmask,
                                       self.cam)
        self.ptzexp.analyze()


    def InitSearch(self, targets={'person'}, objofinterest={'person'},
                   where={'ground','road','building'},
                   minpixpermeter=25, maxpixpermeter=75, highconf=0.9,
                   minconftarget=0.5, saveimgfile=None, saveguifile=None,
                   ptzstep=None):
        """
        Perform a quick scan of the environment for objects of interest.

        Usage:
            Agent.InitSearch(targets={'person'}, objofinterest={'person'},
                             where={'ground','road','building'},
                             minpixpermeter=25, maxpixpermeter=75,
                             highconf=0.9, minconftarget=0.5,
                             saveimgfile=None, saveguifile=None)

        Arguments:
            targets: (set) The set of labels (as assigned by YOLO) of the objects
            to search for. Each label is a string identifying the object class.

            objofinterest: (set) The set of labels (as assigned by YOLO) of the
            objects of interest which, if detected, should be further
            investigated. Each label is a string. If 'objofinterest' contains
            the string "all", then all detected objects are considered
            interesting and will be investigated.

            where: (set) Set of semantic labels (each a string) where objects
            being searched for may be found. Once the initial search is
            complete, only areas of the scene containing at least one of these
            labels will be further examined.

            minpixpermeter: (float) The desired minimum number of pixels per
            meter horizontally across objects to expect reliable detections. We
            will try to obtain this resolution on all objects of interest in
            order to provide reliable detections.

            maxpixpermeter: (float) The number of pixels per meter horizontally
            across objects at which we should have full confidence in
            detections.

            highconf: (float) Any object with detection confidence that is at
            least "highconf" and whose resolution is at least "minpixpermeter"
            will not require further views. Default is 0.9.

            minconftarget: (float) Minimum detection confedence for an object
            to be displayed as a target in the panoramic image. Default is 0.5.

            saveimgfile: (str) If not None, then save each original color image
            to the folder/file specified by "saveimgfile". This should include a
            format string such as "{:07d}" to accept frame numbers. Default is
            None.

            saveguifile: (str) If not None, then save each frame of the GUI
            display to the folder/file specified by "saveguifile". This should
            include a format string such as "{:07d}" to accept frame numbers.
            Default is None.
        """

        assert type(targets) == set, 'Argument "targets" must be a set of object labels'
        assert type(where) == set, 'Argument "where" must be a set of strings'

        self.minpixpermeter = minpixpermeter
        self.maxpixpermeter = maxpixpermeter
        self.highconf = highconf
        self.minconftarget = minconftarget
        self.imagecnt = 0
        self.frame = 0
        self.saveimgfile = saveimgfile
        self.saveguifile = saveguifile

        if not self.panoramics:
            self.InitPanoImages()
        if self.showgui and self.fig is None:
            self.MakeGUI()                      # create the agent GUI

        # "Targets" is a list of the types of objects that the agent should look
        # for. "objofinterest" is a list of the types of objects of interest
        # that should be examined closely if detected. These objects of interest
        # may be confused with targets when image spatial resolution is low.
        self.targets = {s.lower() for s in targets}
        self.objofinterest = {s.lower() for s in objofinterest}
        self.search_where = where
        if self.verbosity >= 1:
            print('Doing quick scan:')
            print('   In regions', where)
            print('   For targets', targets)
            print('   With objects of interest', objofinterest)

        # The initial search is done with the camera set to its widest FOV with
        # at least 10% overlap at each edge.
        hoverlap = 2 # 0.2*self.cam.maxhfov       # HFOV overlap (degrees)
        voverlap = 2 # 0.2*self.cam.maxvfov       # VFOV overlap (degrees)
        hfov = self.cam.maxhfov - hoverlap
        vfov = self.cam.maxvfov - voverlap

        # Range of pan and tilt angles needed for initial search.
        pan0 = self.cam.minpan + self.cam.maxhfov/2
        pan1 = self.cam.maxpan - self.cam.maxhfov/2
        tilt0 = self.cam.mintilt + self.cam.maxvfov/2
        tilt1 = self.cam.maxtilt - self.cam.maxvfov/2
        if pan0 > pan1:
            pan0 = pan1 = (self.cam.minpan + self.cam.maxpan)/2
            numhfov = 1
        else:
            numhfov = int(np.ceil((self.cam.maxpan - self.cam.minpan)/hfov))
        if tilt0 > tilt1:
            tilt0 = tilt1 = (self.cam.mintilt + self.cam.maxtilt)/2
            numvfov = 1
        else:
            numvfov = int(np.ceil((self.cam.maxtilt - self.cam.mintilt)/vfov))

        self.cam.set(zoom=0)
        self.search_phase = 'init'              # initialization phase of search
        imgdict = None

        if self.showgui:
            self.gui_mode.set_text('Quick scan')
            plt.pause(0.01)
            if self.saveguifile != None:
                # Save the empty GUI display to a file.
                self.fig.savefig(self.saveguifile.format(self.imagecnt))

        # Do the quick scan of the pan/tilt range.
        for t in np.linspace(tilt0, tilt1, num=numvfov):
            self.cam.set(tilt=t)
            for p in np.linspace(pan0, pan1, num=numhfov):
                self.cam.set(pan=p)
                self.frame += 1

                self.update_2d_map()              # show camera pan angle on map
                if self.showgui:
                    self.panocolor.draw_fov(self.cam, self.fig, axisnum=2)
                    self.gui_time.set_text(self.env.timestr())
                    self.gui_frame.set_text('{:06d}'.format(self.frame))
                    self.gui_ptz.set_text(self.cam.ptzstr())
                    plt.pause(0.01)
                print('Frame {:d}: Time={:.2f}, Pan={:.1f}º, Tilt={:.1f}º, Zoom={:.1f}'.format(
                      self.frame, self.env.time, p, t, self.cam.zoom))

                self.process_image(updatepanos=True, detobjects=True, imgs=imgdict)

                if 'tracks' in self.showinfo:
                    print('\nTracked objects:')
                    for trg in self.tracked_objs:
                        print('  {} {}, C={:.2f}, PT={:.1f}º,{:.1f}º, WH={:.1f}º,{:.1f}º, D={:.1f}, R={:.1f}, GT={}, ID={:d}'. \
                                    format(trg.label, trg.id, trg.confidence, trg.xctr,
                                           trg.yctr, trg.width, trg.height, trg.dist,
                                           trg.res, trg.gt, trg.detid))
                    print('')

                if self.showgui and self.saveguifile != None:
                    # Save the current GUI display to a file.
                    self.imagecnt += 1
                    self.fig.savefig(self.saveguifile.format(self.imagecnt))

                if self.pause_op:
                    # User pressed the pause button on the GUI.
                    while self.step_op is False:
                        plt.pause(0.1)
                    self.step_op = False

            # Reverse the pan direction on the next pass.
            pan0, pan1 = pan1, pan0

        self.prep_deep_search()
        self.search_init_done = True

        if self.showgui:
            self.gui_msg.set_text('Quick scan done')
            # self.fig.set(shownow=True)
            plt.pause(0.01)


    def DeepSearch(self, method='baseline'):
        """
        Do a full search of previously viewed regions of the scene.

        Arguments:
            method: (str) A string specifying what PTZ search algorithm to use.
            Default is 'baseline'. Options include:
                'none': Do not perform any search.
                'baseline': Use the baseline search algorithm.

        Description:
            Agent.InitSearch() must be called prior to this funtion to
            initialize the search process.
        """

        assert type(method) == str, 'Argument "method" must be a string'
        if not self.search_init_done:
            raise Exception('Agent.InitSearch() must be called before Agent.DeepSearch()')

        self.search_method = method.lower()
        self.search_phase = 'deep'
        print('Performing {} {} search...'.format(self.search_phase,
                                                  self.search_method))

        if self.showgui:
            self.gui_mode.set_text('Deep search')
            plt.pause(0.01)

        if self.search_method == 'baseline':
            for pthv in self.ptzexp.scanner(self.cam.pan, self.cam.tilt):
                self.frame += 1
                self.cam.set(pan=pthv[0], tilt=pthv[1], hfov=pthv[2])
                if self.showgui:
                    self.gui_time.set_text(self.env.timestr())
                    self.gui_frame.set_text('{:06d}'.format(self.frame))
                    self.gui_ptz.set_text(self.cam.ptzstr())
                    self.fig.set(shownow=True)
                    self.panocolor.draw_fov(self.cam, self.fig, axisnum=2)
                    plt.pause(0.01)

                print('Frame {}: Time={:.2f}, P={:.1f}º, T={:.1f}º, H={:.1f}º, V={:.1f}º, Z={:.2f}'.\
                      format(self.frame, self.env.time, *pthv[0:4], self.cam.zoom))
                self.update_2d_map()
                self.process_image(updatepanos=self.env.dynamic_env,
                                   detobjects=True)

                if 'tracks' in self.showinfo:
                    print('\nTracked objects:')
                    for t in self.tracked_objs:
                        print('  {} {}, C={:.2f}, PT={:.1f}º,{:.1f}º, WH={:.1f}º,{:.1f}º, D={:.1f}, R={:.1f}, Z={:.2f}, ID={:d}'. \
                                    format(t.label, t.id, t.confidence, t.xctr,
                                           t.yctr, t.width, t.height, t.dist,
                                           t.res, t.zoom, t.detid))
                    print('')

                if self.showgui and self.saveguifile != None:
                    # Save the current GUI display to a file.
                    self.imagecnt += 1
                    self.fig.savefig(self.saveguifile.format(self.imagecnt))

                if self.pause_op:
                    while self.step_op is False:
                        plt.pause(0.1)
                    self.step_op = False
        else:
            raise Exception('Unimplemented search method: {}'.format(self.search_method))

        if False:
            # Display the detection statistics.
            f = Fig(figsize=(4,4), figtitle='Detection Statistics')
            trg = self.stats[self.stats[:,0]==True, 1:3]
            nontrg = self.stats[self.stats[:,0]==False, 1:3]
            f.draw(point=[trg[:,0], trg[:,1]], markersize=5, markercolor='r')
            f.draw(point=[nontrg[:,0], nontrg[:,1]], markersize=5, markercolor='b',
                   xlabel='Res.', ylabel='Conf.')
            plt.show()

        if self.showgui:
            self.gui_msg.set_text('Deep search done')
            # self.fig.set(shownow=True)
            plt.pause(0.01)

        self.search_phase = 'done'
        print('Done')


    def btn_manual(self, event=None):
        print('You drive')
        self.you_drive(holdpos=True)
        print('\nPan = {:.1f}, Tilt = {:.1f}, Zoom = {:.1f}'.
                      format(self.cam.pan, self.cam.tilt, self.cam.zoom))
        self.process_image(updatepanos=True, detobjects=True)


    def ax0_display(self, event=None):
        # Change the 1st displayed standard image.
        if self.showgui and len(self.stdimgs) > 0:
            keys = list(self.stdimgs.keys())
            if self.imselect[0] == '' or self.imselect[0] not in keys:
                self.imselect[0] = keys[0]
            else:
                idx = keys.index(self.imselect[0])
                idx = (idx + 1) % len(keys)
                self.imselect[0] = keys[idx]

            self.fig.btn[self.axis_ax0_ctrl].label.set_text(self.imselect[0])
            self.fig.set(axisnum=0, image=self.stdimgs[self.imselect[0]],
                         vminmax=self.vminmax[self.imselect[0]],
                         axistitle=self.imselect[0].title())


    def ax1_display(self, event=None):
        # Change the 2nd displayed standard image.
        if self.showgui and len(self.stdimgs) > 0:
            keys = list(self.stdimgs.keys())
            if self.imselect[1] == '' or self.imselect[1] not in keys:
                self.imselect[1] = keys[0]
            else:
                idx = keys.index(self.imselect[1])
                idx = (idx + 1) % len(keys)
                self.imselect[1] = keys[idx]

            self.fig.btn[self.axis_ax0_ctrl+1].label.set_text(self.imselect[1])
            self.fig.set(axisnum=1, image=self.stdimgs[self.imselect[1]],
                         vminmax=self.vminmax[self.imselect[1]],
                         axistitle=self.imselect[1].title())


    def ax2_display(self, event=None):
        # Change the displayed panoramic image.
        if self.showgui and len(self.panoimgs) > 0:
            keys = list(self.panoimgs.keys())
            if self.imselect[2] == '' or self.imselect[2] not in keys:
                self.imselect[2] = keys[0]
            else:
                idx = keys.index(self.imselect[2])
                idx = (idx + 1) % len(keys)
                self.imselect[2] = keys[idx]

            self.fig.btn[self.axis_ax0_ctrl+2].label.set_text(self.imselect[2])
            self.fig.set(axisnum=2, image=self.panoimgs[self.imselect[2]],
                         vminmax=self.vminmax[self.imselect[2]],
                         axistitle=self.imselect[2].title(),
                         clearoverlays=False, xlabel='Pan', ylabel='Tilt',
                         labelpos='bl', aspect='equal',
                         imextent=(self.panocolor.pmax, self.panocolor.pmin,
                                   self.panocolor.tmin, self.panocolor.tmax))


    def btn_pause(self, event=None):
        """
        Pause or continue processing.
        """
        if self.pause_op:
            # Continue processing.
            self.pause_op = False
            self.step_op = True
            self.fig.btn[self.axis_pause].label.set_text('Pause')
        else:
            # Pause processing.
            self.pause_op = True
            self.step_op = False
            self.fig.btn[self.axis_pause].label.set_text('Continue')


    def btn_onestep(self, event=None):
        """
        Perform one step of the operation.
        """
        self.step_op = True


    def btn_quit(self, event=None):
        print('Quitting.')
        if self.showgui:
            self.fig.close()
        self.env.close()
        exit()


    def btn_show_plan(self, event=None):
        """
        Show the PTZ search plan.  Alternate between a static view and dynamic
        view on each call of this function.
        """
        if self.busy: return
        if self.show_static_plan:
            self.show_views()
        else:
            self.show_route()
        self.show_static_plan = False if self.show_static_plan else True


    def show_views(self, showtime=None):
        """
        Display all views in the agent's ptz search plan.

        Notes:
            The rectangles drawn in the PT image are only approximate.
        """
        if not self.showgui:
            return

        self.ptzexp.reset()
        if self.ptzexp.views == []:
            print('There is no search plan.')
            self.gui_msg.set_text('There is no search plan')
            plt.pause(0.01)
            return

        self.busy = True
        patches = []
        colors = 'w'     # 'rgbycm'  # multi-colored boxes.
        self.gui_msg.set_text('Search plan...')

        # Draw the views a little smaller than they are to make each stand out.
        d = self.ptzexp.pt_degperpix

        # for k in range(self.ptzexp.pthvz.shape[0]):
        for pthv in self.ptzexp.scanner(self.cam.pan, self.cam.tilt):
            pan, tilt, hfov, vfov, timestamp = pthv
            r = [pan-hfov/2+d, tilt-vfov/2+d, hfov-2*d, vfov-2*d]  # [xmin, ymin, width, height]
            color = colors[np.random.randint(0,len(colors))]
            p = self.fig.draw(axisnum=2, rect=r, edgecolor=color, shownow=False)
            patches += [p]

        self.fig.update()

        if showtime is None:
            self.gui_msg.set_text('Search plan... press any key to continue')
            plt.pause(0.01)
            self.fig.wait('key_press_event')
            self.gui_msg.set_text('')
            plt.pause(0.01)
        else:
            # Pause for a bit, then remove all patches.
            plt.pause(showtime)
        try:
            for p in patches:
                p.remove()
        except:
            pass
        self.fig.update()
        self.busy = False



    def show_route(self, showtime=5):
        """
        Show the PTZ route that will be used to search the scene.
        """
        if self.busy: return

        self.ptzexp.reset()
        if self.ptzexp.views == []:
            print('There is no search plan.')
            self.gui_msg.set_text('There is no search plan')
            plt.pause(0.01)
            return

        if not self.showgui:
            return

        self.busy = True
        patches = []
        self.gui_msg.set_text('Search plan...')

        # Draw the views a little smaller than they are to make each stand out.
        d = self.ptzexp.pt_degperpix

        for pthv in self.ptzexp.scanner(self.cam.pan, self.cam.tilt):
            pan, tilt, hfov, vfov, timestamp = pthv
            # print('P={0:.1f},T={1:.1f}, H={0:.1f}, V={1:.1f}'.format(*pthv))
            r = [pan-hfov/2+d, tilt-vfov/2+d, hfov-2*d, vfov-2*d]  # [xmin, ymin, w, h]
            patch = self.fig.draw(axisnum=2, rect=r, edgecolor='w', shownow=True)
            patches += [patch]
            plt.pause(0.01)

        if True:
            self.gui_msg.set_text('Search plan... press any key to continue')
            plt.pause(0.01)
            self.fig.wait('key_press_event')
            self.gui_msg.set_text('')
            plt.pause(0.01)
        else:
            # Pause for a bit, then remove all patches.
            plt.pause(showtime)

        try:
            for p in patches:
                p.remove()
        except:
            pass
        self.fig.update()
        self.busy = False


    def objstr(self, objlist):
        """
        Create a short string description of the given list of objects.
        """
        odict = dict()
        for o in objlist:
            if o.label in odict.keys():
                odict[o.label].append(o.detid)
            else:
                odict[o.label] = [o.detid]
        ostr = ''
        for label in odict.keys():
            if ostr != '':
                ostr += ', '
            odict[label].sort()
            ostr += label + ' ' + str(odict[label]).replace(' ','')
        return ostr


