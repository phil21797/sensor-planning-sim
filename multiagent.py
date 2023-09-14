"""
Create and run multiple agents operating in a random environment.

Author:
    Phil David, Parsons, May 1, 2023.

"""


from simworld import *
from camera import *
from microphone import *
from agent import *
from datetime import datetime
from fig import *
import os
import audio_io

audiorows = 0.5          # fraction of image rows (in [0,1]) over which to
                         # overlay audio signals


class MultiAgentEnv:
    """
    Class for multiagent environments.
    """

    def __init__(self, imdet=False):
        """
        Initialize a multiagent environment.

        Arguments:
            imdet:bool -- Should the image-based object detector be loaded?
            The default value is False. The current detector is YOLO v4 from
            https://github.com/AlexeyAB/darknet. For more details, see the YOLO
            v4 paper (https://arxiv.org/abs/2004.10934) or the website
            http://pjreddie.com/darknet/yolo.
        """
        if imdet:
            from detector import Detector
            self.imdet_weights_file = "weights/yolov4.pth"
            self.image_thresh = 0.5
            self.imdet_device = "cpu"
            print(f'Loading YOLO detector weights from {self.imdet_weights_file}')
            self.imdet = Detector(weights_file=self.imdet_weights_file,
                                  detect_thresh=self.image_thresh,
                                  device=self.imdet_device)
        else:
            self.imdet = None


    def create(self, envradius=500, randseed=None, numagents=0,
               imsize=(1280,720), dtime=0.1, playaudio=False, prob_has_cam=1.0,
               prob_has_mic=1.0, envdefsfile=None, outfolder=None, showmap=True,
               verbose=True):
        """
        Create a multiagent environment.

        Arguments:
            randseed:int -- random seed or None. If None, then a new random
            seed is chosen.

            numagents:int -- This is the number of agents to create, each of
            which is moved to a random location in the environment.

            imsize:tuple -- size (cols,rows) of all rendered images.

            dtime:float -- time (seconds) between agent updates.

            playaudio:bool --  play recorded audio from agents?

            prob_has_cam:float -- probability that an agent has a camera.

            prob_has_mic:float -- probability that an agent has a microphone.

            envdefsfile:str or None -- file containing external environment
            definitions. Currently, moving objects and agents may be defined in
            this file. If this is not None, then the numagents argument is
            ignored.

            outfolder:str or None -- name of output folder, or None.

            showmap:bool -- show the 2D groundtruth map?

            verbose:bool -- display a lot of output?
        """

        self.envradius = envradius
        self.randseed = randseed
        self.imsize = imsize
        self.dtime = dtime
        self.playaudio = playaudio
        self.prob_has_cam = prob_has_cam
        self.prob_has_mic = prob_has_mic
        self.outfolder = outfolder
        self.showmap = showmap
        self.verbose = verbose
        self.agentdefs = []

        if self.outfolder != None:
            # Create a folder to save output in.
            dt = datetime.now().strftime("%Y%m%d%H%M%S")
            self.outfolder = self.outfolder + '_' + dt
            if os.path.exists(self.outfolder):
                print(f'Output folder "{self.outfolder}" already exists')
                exit(1)
            else:
                try:
                    os.mkdir(self.outfolder)
                    print(f'Created output folder "{self.outfolder}"')
                except:
                    raise Exception(f'Unable to create output folder "{self.outfolder}"')

        if type(envdefsfile) is str:
            # Read external environment definitions.
            self.read_env_defs(envdefsfile)
            self.numagents = len(self.agent_defs)
            self.defined_agents = True
        else:
            self.numagents = numagents
            self.defined_agents = False

        # Create a random outdoor environmeent.
        self.sim = SimWorld(imsize=self.imsize, timeofday=[700,1800],
                            env_radius=self.envradius,
                            bldg_density=1, road_density=1.0, clutter_density=0.5,
                            plant_density=0.5, people_density=0, animal_density=0,
                            vehicle_density=0, airborne_density=0,
                            bldg_plant_density=0.5, barrier_density=1, gndfeat_density=0.2,
                            lookouts=True, probwindowoccupied=0.25,
                            p_over_building={'person':0.5, 'clutter':0.1, 'animal':1.0},
                            p_over_road={'person':0.1, 'clutter':0.05, 'animal':0.2},
                            textures='textures', rand_seed=randseed,
                            dynamic_env=True)

        if showmap:
            # Create and display a 2D map of the environment groundtruth.
            self.map2d = Map2D(maps=self.sim.map3d, size=8,
                               label_colors=self.sim.label_colors)
        else:
            self.map2d = None

        self.sim.insert_fixed_path_objs(self.obj_defs)
        self.create_agents()


    def create_agents(self, camstep=0.25):
        """
        Create agents, each with their own camera and/or microphone. Each
        agent should have a least one of these sensors.

        If the agents are defined by the user (i.e., self.defined_agents is
        True), then self.agentdef will be a 2D array-like where each row defines
        one agent:
            [xpos, ypos, xfwd, yfwd, hascam, hasmic, initzoom, panrange].

                (xpos, ypos) is the agent's 2D position (z is 0) in the
                environment.

                (xfwd, yfwd) is the 2D orientation of the agent, its forward
                direction. This is the direction that an agent's camera points
                when its pan angle is zero.

                hascam is true (1) if the agent has a camera

                hasmic is True (1) if the agent has a microphone.

                initzoom is the initial zoom of the agent's camera (in [0,1]).
                This is ignored if the agent does not have a camera.

                panrange is the range of angles (degrees) that an agent's camera
                can pan over. E.g., if panrange is 180, then the agent's camera
                can pan back and forth from -90 to 90 degrees. This is ignored
                if the agent does not have a camera.

        Arguments:
            camstep:float -- When an agent's camera is turned on (assuming the
            agent has a camera), the change in camera pan angle from one frame
            to the next (as implemented in Agent.config_sensors()) will be
            camstep*HFOV degrees. The default value is 0.25.
        """

        print(f'Creating {self.numagents} agents...')
        self.agent = [[]]*self.numagents

        for k in range(self.numagents):

            # What sensors does this agent have?
            if self.defined_agents:
                # Sensors are defined by user.
                camera = True if self.agent_defs[k][4] else None
                microphone = True if self.agent_defs[k][5] else None
                iz = self.agent_defs[k][6]                 # initial camera zoom
                pr = self.agent_defs[k][7]/2              # 1/2 camera pan range
            else:
                # Randomly assign sensors.
                camera = True if np.random.rand() <= self.prob_has_cam else None
                if not camera:
                    microphone = True
                else:
                    microphone = True if np.random.rand() <= self.prob_has_mic else None
                    iz = 0.5                               # initial camera zoom
                    pr = 90                               # 1/2 camera pan range

            if camera:
                # Setup the agent's camera.
                camera = PTZCamera(imsize=self.imsize, rnghfov=(3,54),
                                   rngpan=(-pr,pr), rngtilt=(-45,60),
                                   pos=(0,0,1), pan=0, tilt=0, zoom=iz)

            if microphone:
                # Setup the agent's microphone.
                microphone = Microphone(pos=(0,0,1))

            # Create an agent with the given sensors.
            self.agent[k] = Agent(env=self.sim, cam=camera, mic=microphone,
                                  map2d=self.map2d, objdet=None)

            # Detections are used to decide when to turn on/off sensors.
            self.agent[k].mic_detect = False
            self.agent[k].cam_detect = False
            self.agent[k].mic_time_onoff = self.sim.time  # time of last on or off

            if camera:
                # Choose a random initial pan direction.
                self.agent[k].panstep = np.random.choice([-1,1])*camstep*camera.hfov

            if self.defined_agents:
                # Move agent to user-defined position
                self.agent[k].move(pos=self.agent_defs[k][0:2],
                                   fdir=self.agent_defs[k][2:4])
            elif True:
                # Move agent to a random ground location.
                self.agent[k].move_random(to='ground')
            else:
                # User manually drives the agent into position.
                self.agent[k].you_drive()



    def read_env_defs(self, envdefsfile:str=''):
        """
        Read additional environment definitions from a file.

        Arguments:
            envdefsfile:str -- Text file containing additional environment
            definitions. The general format of this file is:

                OBJECT <Object_1_tags>      # this is a comment
                START <x> <y> <z>
                <move_command_1>
                <move_command_2>
                ...
                <move_command_N>
                END

                AGENT <x> <y> <xfwd> <yfwd> <hascam> <hasmic> <initzoom> <panrange>
                ...

            Any number of objects and agents my be defined. "Objects" are the
            things that move around the environment that the agents are expected
            to detect and track. The trajectory of each object is defined by a
            sequence of object movement commands:

                START <x> <y> <z> -- The starting position of the object (at
                                     time 0).

                TIME <time> <x> <y> <z>     -- Move to position (x,y,z) at
                                               absolute time <time> sec.

                DTIME <dtime> <x> <y> <z>   -- Move for <dtime> sec. to position
                                               (x,y,z).

                SPEED <speed> <x> <y> <z>   -- Move at speed <speed> to position
                                               (x,y,z) where <speed> > 0.

                ARC <dir> <rad> <deg> <speed>
                                            -- Move through an arc in direction
                                               <dir>, either LEFT or RIGHT,
                                               radius <rad>, for <deg> degrees,
                                               and at speed <speed>.

                STOP <dtime>                -- Stop for <dtime> sec.

            Times for an object must be listed sequentially and be increasing.
            After an object reaches its final defined position, the object jumps
            back to its starting position on the next frame of the simulation.

            Any number of agents may be defined. Agents, which are stationary
            except when moved by control algorithms external to the environment
            simulation, possess sensors for detecting objects. Each agent
            definition has the following parameters:

                 <x> <y>       -- The 2D position (z is 0) of the agent in the
                                  environment.

                 <xfwd> <yfwd> -- The 2D forward-facing direction of the agent.

                 <hascam>      -- 1 if the agent has a camera; otherwise, 0.

                 <hasmic>      -- 1 if the agent has a microphone; otherwise, 0

                 <initzoom>    -- Initial zoom (in [0,1]) of the agent's camera.

                 <panrange>    -- Range of pan angles (in degrees) of the
                                  agent's camera (e.g., 180 => pan from -90° to
                                  90°). Note: A pan angle of 0° will point the
                                  agent's camera in the direction (<xfwd>,
                                  <yfwd>).

            For reference, vehicles typically move at speeds between 22 and 36 m/s
            on highways, and move at speeds up to 22 m/s on non-highways; and humans
            typically walk at around 3 m/s, and jog at around 5-6 m/s.

        Returns:
            self.obj_defs: A list of object definitions. Each item in this list
            is a list [tags, txyz] where tags is a string giving the texture tags
            of the object and txyz is a list of lists [t, x, y, z] giving the
            position of the object (x,y,z) at each time t.

            self.agent_defs: A list of agent definitions. Each item in this list
            is a list [x, y, xfwd, yfwd, hascam, hasmic, initzoom, panrange]
            defining the properties of one agent.
        """

        self.obj_defs = []
        self.agent_defs = []

        if envdefsfile == '':
            return

        linenum = 0
        objectcnt = 0
        agentcnt = 0
        objstate = 0        # object state: 0=none, 1=need "start", 2=need "end"

        with open(envdefsfile) as f:
            for line in f:
                linenum += 1
                line = line.split('#', 1)[0]
                line = line.rstrip()
                if line == "":
                    continue
                token = [s.lower() for s in line.split(' ') if s != '']
                # print(f'Line {linenum}: {line}')
                if token[0] == 'object':
                    if objstate != 0:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}":'\
                                         f' new object before end of previous object: "{line}"')
                    objstate = 1
                    tags = token[1]
                    txyz = []
                elif token[0] == "start":
                    # Start at time 0.
                    if objstate != 1:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}":'\
                                         f' "start" before "object" def: "{line}"')
                    txyz.append([0]+[float(v) for v in token[1:]])
                    objstate = 2
                elif token[0] == "end":
                    if objstate != 2:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}":'\
                                         f' "end" without "object" def: "{line}"')
                    if txyz == []:
                        raise ValueError(f'Line {linenum} of file "{pathsfile}": '\
                                         'Missing object time and position data')
                    print('\nNew object:\n', np.array(txyz))
                    self.obj_defs.append([tags, txyz])
                    objstate = 0
                    objectcnt += 1
                elif token[0] == "agent":
                    # Process an agent definition.
                    try:
                        x, y, xfwd, yfwd, hascam, hasmic, initzoom, panrange = \
                                           [float(v) for v in token[1:]]
                    except:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                         f'Expected: AGENT x y xfwd yfwd hascam hasmic initzoom panrange.\n' \
                                         f'Got: "{line}"')
                    self.agent_defs.append([x,y,xfwd,yfwd,hascam,hasmic,initzoom,panrange])
                    agentcnt += 1
                else:
                    # Process an object movement command.
                    if objstate != 2:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}": '\
                                         f'Expected object movement command. Got: "{line}"')
                    if token[0] == "time":
                        try:
                            t, x, y, z = [float(v) for v in token[1:]]
                        except:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Expected: TIME time x y x\n' \
                                             f'Got: "{line}"')
                        if t <= txyz[-1][0]:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Time must be > {txyz[-1][0]} (previous time): "{line}"')
                        newtxyz = [[t, x, y, z]]
                    elif token[0] == "dtime":
                        try:
                            dt, x, y, z = [float(v) for v in token[1:]]
                        except:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Expected: DTIME dtime x y z\n' \
                                             f'Got: "{line}"')
                        t = txyz[-1][0] + dt
                        newtxyz = [[t, x, y, z]]
                    elif token[0] == "speed":
                        try:
                            spd, x, y, z = [float(v) for v in token[1:]]
                        except:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Expected: SPEED speed x y z\n' \
                                             f'Got: "{line}"')
                        if spd <= 0:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Speed must be > 0: "{line}"')
                        dist = np.linalg.norm(txyz[-1][1:4]-np.array([x,y,z]))
                        t = txyz[-1][0] + dist/spd
                        newtxyz = [[t, x, y, z]]
                    elif token[0] == "stop":
                        try:
                            dt = float(token[1])
                        except:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Expected: STOP dtime\n' \
                                             f'Got: "{line}"')
                        t = txyz[-1][0] + dt
                        x, y, z = txyz[-1][1:]
                        newtxyz = [[t, x, y, z]]
                    elif token[0] == "arc":
                        if token[1] not in {"left", "right"}:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             'Expected "left/right" for ARC direction.'\
                                             f' Got: "{line}"')
                        if len(txyz) < 2:
                            # Need to know direction of motion to known what is left or right.
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             'Can use ARC only after object has moved.'\
                                             f': "{line}"')
                        turnleft = True if token[1] == "left" else False
                        try:
                            rad, deg, spd = [float(v) for v in token[2:5]]
                        except:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'Expected: ARC dir rad deg speed\n' \
                                             f'Got: "{line}"')
                        if rad <= 0 or deg <= 0 or spd <= 0:
                            raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                             f'ARC parameters must be > 0: "{line}"')
                        newtxyz = arc_points(txyz, turnleft, rad, deg, spd)
                    else:
                        raise ValueError(f'Line {linenum} of file "{envdefsfile}":\n'\
                                         'Expected object movement command.'\
                                         f' Got: "{line}"')

                    txyz.extend(newtxyz)

        if objstate == 2:
            raise ValueError(f'Line {linenum} of file "{envdefsfile}":'\
                             f' last object def is missing "end"')

        print(f'Read {objectcnt} object and {agentcnt} agent definitions from '\
              f'"{envdefsfile}"')


    def config_sensors(self):
        """
        Update the configuartion of all agents' sensors.
        """

        for k in range(self.numagents):
            print(f'Agent {k+1}: camera', end='')

            if self.agent[k].cam:
                # Turn the camera on or off?
                if self.agent[k].mic_detect:
                    if not self.agent[k].cam.power_on:
                        self.agent[k].cam.power(1)
                        print(f' turn', end='')
                elif self.agent[k].cam.power_on:
                    self.agent[k].cam.power(0)
                    print(f' turn', end='')

                if self.agent[k].cam.power_on:
                    # Pan the camera.
                    nextpan = self.agent[k].cam.pan + self.agent[k].panstep
                    if nextpan < self.agent[k].cam.minpan or \
                       nextpan > self.agent[k].cam.maxpan:
                        # Change pan direction.
                        self.agent[k].panstep *= -1
                    self.agent[k].cam.inc(self.agent[k].panstep)

                print(' on' if self.agent[k].cam.power_on else ' off', end='')
            else:
                print(' not present', end='')

            print(', microphone', end='')

            if self.agent[k].mic:
                dt = self.sim.time - self.agent[k].mic_time_onoff
                if self.agent[k].mic_detect:
                    # The microphone must be on since there is a current
                    # detection. Keep the microphone on for at least
                    # cyle_times[0] seconds longer to try to get addional
                    # detections.
                    self.agent[k].mic_time_onoff = self.sim.time
                elif self.agent[k].mic.power_on:
                    # The microphone is currently on. Turn the microphone off if
                    # there are no current detections and it's been at least
                    # cyle_times[0] seconds since the last detection.
                    if dt >= self.cycle_times[0]:
                        self.agent[k].mic.power(0)
                        self.agent[k].mic_time_onoff = self.sim.time
                        print(f' turn', end='')
                elif dt >= self.cycle_times[1]:
                    # The microphone is currently off. Turn is on if it's been
                    # off for past cycle_times[1] seconds.
                    self.agent[k].mic.power(1)
                    self.agent[k].mic_time_onoff = self.sim.time
                    print(f' turn', end='')
                print(' on' if self.agent[k].mic.power_on else ' off')
            else:
                print(' not present')

        print()


    def collect_sensor_data(self, f:Fig):
        """
        Collect and display sensor data from all agents.
        """

        blackimage = np.zeros(list(self.imsize[::-1])+[3], dtype=np.uint8)

        for k in range(self.numagents):
            if self.verbose:
                print(f'Agent {k+1}')

            # Clear agent display.
            f.clearaxis(axisnum=k, keepimage=True)
            f.set(axisnum=k, image=blackimage,
                  axistitle=self.agent[k].name, axisoff=True, shownow=True)

            if self.agent[k].cam:
                if self.agent[k].cam.power_on:
                    # Get an image from the agent.
                    imgs = self.agent[k].get_images(imlist=['color'])
                    imcolor = imgs['color']
                    f.set(axisnum=k, image=imcolor,
                          axistitle=self.agent[k].name, axisoff=True)

                    if self.imdet:
                        # Run the object detector on this image.
                        self.imdet.process(imcolor, detect_thresh=self.image_thresh)
                        self.imdet.show(fig=f.fig, info=self.verbose)
                        self.cam_detect = True if len(self.imdet.dets) > 0 else False

            if self.agent[k].mic:
                if self.agent[k].mic.power_on:
                    # Get audio recording.
                    audio = self.agent[k].get_audio(duration=3.0, maxdist=250,
                                                    verbose=self.verbose)
                    signal = audio['signal']
                    sigmax = signal.max()
                    print(f'  Max sum of signals = {sigmax:.1f}')
                    self.overlay_audio(signal, f, k)
                    self.agent[k].mic_detect = True if sigmax >= self.audio_thresh else False
                    if self.playaudio:
                        audio_io.play_audio(signal, audio['samplerate'])
                else:
                    self.agent[k].mic_detect = False


    def overlay_audio(self, sig, f, axnum):
        """
        Overlay an audio signal on top of an image.
        """
        ncols = self.imsize[0]
        nrows = self.imsize[1]
        s = np.ceil(len(sig)/ncols).astype(int)   # resample to fit across image
        sig2 = sig[0:-1:s]
        x = np.arange(len(sig2))
        y = nrows/2 + audiorows*nrows*sig2/WorldObj.audio_max_measured
        f.set(axisnum=axnum)
        plt.plot(x, y, 'r', linewidth=0.5)
        plt.pause(0.1)


    def run(self, sim_run_time=1e10, cycle_times=(5,10), audio_thresh=100,
            image_thresh=0.7):
        """
        Run the multiagent simulation.

        Arguments:
            sim_run_time:float -- time (sec.) to run the simulation to. Default
            is 1e10 seconds (almost infinity).

            cycle_times:(float,flot) -- The microphone cycle times (on_time,
            off_time), the durations (in sec.) that the microphone is on and
            then off.

            audio_thresh:float -- threshold on audio signal amplitude to detect
            something.

            image_thresh:float -- Image-based object detection threshold, in
            [0:1]. The default value is 0.7.
        """

        self.cycle_times = cycle_times
        self.audio_thresh = audio_thresh
        self.image_thresh = image_thresh

        fnum = 0                      # frame number of saved images
        sim_run_time += 1e-4          # account for round-off errors

        if False:
            # Display color, semantic label, and depth images from one agent.
            imgs = self.agent[0].get_images()
            with Fig(axpos=[131,132,133], figtitle='My World', figsize=(10,4),
                     link=[0,1,2]) as f:
                f.set(axisnum=0, image=imgs['color'], axistitle='Color')
                f.set(axisnum=1, image=imgs['label'], axistitle='Semantic labels')
                f.set(axisnum=2, image=imgs['depth'], axistitle='Depth')
                print('Press any key to continue...')
                f.wait(event='key_press_event')

        # Determine layout of figure to display sensor data of all agents.
        nr = int(np.ceil(np.sqrt(self.numagents))) # num rows of axis in display
        nc = int(np.ceil(self.numagents/nr))       # num cols of axis in display
        axpos = [(nr,nc,k+1) for k in range(self.numagents)]   # 2D grid of axes

        #----------------------------------------------------------------------
        # Collect and display agent sensor data and occasionally change sensor
        # configurations.
        #----------------------------------------------------------------------

        print('Starting agents... Close the figure to quit')

        with Fig(axpos=axpos, figtitle='Agent Cameras', figsize=(10,8)) as f:
            while self.sim.time <= sim_run_time:
                fnum += 1
                print(f'\n[[ Frame {fnum}, Time {self.sim.time:.2f} sec. ]]\n')
                f.fig.suptitle(f'⟦ Time: {self.sim.time:.2f} sec. ⟧', fontsize=10)

                # Update the configuartion of all agent sensors.
                self.config_sensors()

                if self.showmap:
                    # Update the 2D groundtruth map.
                    self.map2d.Update()

                # Collect and display sensor data from all agents.
                self.collect_sensor_data(f)

                if self.outfolder is not None:
                    # Save figures to files.
                    f.savefig(f'{self.outfolder}/agents_{fnum:07d}.png')
                    if self.showmap:
                        self.map2d.mfig.savefig(f'{self.outfolder}/map_{fnum:07d}.png')

                self.sim.inc_time(self.dtime)


def arc_points(txyz:list, turnleft:bool, rad:float, deg:float, spd:float):
    """
    Get points around an arc.
    """
    # Get two most recent distinct points on trajectory.
    cur = np.array(txyz[-1][1:3])
    idx = -1
    for k in range(len(txyz)-1,-1,-1):
        if np.any(txyz[k][1:3] != cur):
            idx = k
            break
    if idx < 0:
        raise ValueError('arc_points: no object motion found')
    prev = np.array(txyz[idx][1:3])

    v = cur - prev                  # 2D vector pointing in direction of motion
    dv = np.array([v[1], -v[0]])    # perpindicular vector
    dv = rad*dv/np.linalg.norm(dv)  # normalize to length `rad`
    if turnleft:
        origin = cur - dv           # center of rotation
        rdir = 1                    # rotation direction: counter-clockwise
    else:
        origin = cur + dv           # center of rotation
        rdir = -1                   # rotation direction: clockwise

    dist = 2*np.pi*rad*deg/360      # distance object will travel along arc
    npts = np.ceil(deg/10).astype(int) # one control point every 5° along arc
    npts = max(1, npts)
    dt = dist/(npts*spd)            # time between control points

    # Rotate the current point arond the origin `npts` times.
    newtxyz = []
    dtheta = rdir*np.deg2rad(deg/npts)
    angle = 0
    t = txyz[-1][0]
    for k in range(npts):
        t += dt
        angle += dtheta
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        p = np.round(R @ (cur.T - origin.T) + origin.T, 2)
        newtxyz.append([t, p[0], p[1], 0])

    return newtxyz


if __name__ == '__main__':

    mae = MultiAgentEnv(imdet=True)

    # Create the multiagent environment.
    mae.create(randseed=3, envradius=400, showmap=True,
               envdefsfile='envdefs.txt', outfolder='./outputs')

    # Run the simulation.
    mae.run(sim_run_time=60, cycle_times=(1,5), audio_thresh=1500,
            image_thresh=0.8)
