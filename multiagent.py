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


    def create(self, envradius=500, randseed=None, agentdef=None,
               imsize=(1280,720), dtime=0.1, playaudio=False, prob_has_cam=1.0,
               prob_has_mic=1.0, pathsfile=None, outfolder=None, showmap=True,
               verbose=True):
        """
        Create a multiagent environment.

        Arguments:
            randseed:int -- random seed or None.

            agentdef:int or 2D array-like -- If an int, this is the number of
            agents to create, and each is moved to a random location in the
            environment. Otherwise, this must be a 2D array-like where each row
            of the array is
              [xpos, ypos, xfwd, yfwd, hascam, hasmic, initzoom, panrange]
            defining one agent:

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

            imsize:tuple -- size (cols,rows) of all rendered images.

            dtime:float -- time (seconds) between agent updates.

            playaudio:bool --  play recorded audio from agents?

            prob_has_cam:float -- probability that an agent has a camera.

            prob_has_mic:float -- probability that an agent has a microphone.

            pathsfile:str -- file defining object trajectories, or None.

            outfolder:str -- name of output folder, or None.

            showmap:bool -- show the 2D groundtruth map?

            verbose:bool -- display a lot of output?
        """

        self.envradius = envradius
        self.randseed = randseed
        self.agentdef = agentdef
        self.imsize = imsize
        self.dtime = dtime
        self.playaudio = playaudio
        self.prob_has_cam = prob_has_cam
        self.prob_has_mic = prob_has_mic
        self.pathsfile = pathsfile
        self.outfolder = outfolder
        self.showmap = showmap
        self.verbose = verbose

        if type(agentdef) is int:
            self.numagents = agentdef
            self.defined_agents = False
        elif type(agentdef) in [list or np.ndarray]:
            self.numagents = len(agentdef)
            self.defined_agents = True
        else:
            raise ValueError('"agentdef" argument must be an int or 2D array-like')

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

        # Create a random outdoor environmeent.
        self.sim = SimWorld(imsize=self.imsize, timeofday=[700,1800],
                            env_radius=self.envradius,
                            bldg_density=1, road_density=2.0, clutter_density=0.5,
                            plant_density=1, people_density=0, animal_density=0,
                            vehicle_density=0, airborne_density=0,
                            bldg_plant_density=0.5, barrier_density=1, gndfeat_density=0.2,
                            lookouts=True, probwindowoccupied=0.25,
                            p_over_building={'person':0.5, 'clutter':0.1, 'animal':1.0},
                            p_over_road={'person':0.1, 'clutter':0.05, 'animal':0.2},
                            textures='textures', rand_seed=randseed,
                            dynamic_env=True, pathsfile=self.pathsfile)

        if showmap:
            # Create and display a 2D map of the environment groundtruth.
            self.map2d = Map2D(maps=self.sim.map3d, size=8,
                               label_colors=self.sim.label_colors)
        else:
            self.map2d = None

        self.create_agents()


    def create_agents(self, camstep=0.25):
        """
        Create agents, each with their own camera and/or microphone. Each
        agent should have a least one of these sensors.

        If the agents are defined by the user (i.e., self.defined_agents is
        True), then self.agentdef will be a 2D array-like where each row defines
        one agent:
            [xpos, ypos, xfwd, yfwd, hascam, hasmic, initzoom, panrange].
        See MultiAgentEnv.create() for more details.

        Arguments:
            camstep:float -- If an agent has a camera, and if it's turned on,
            then change the camera pan angle by camstep*HFOV degrees on each
            camera position update. The default value is 0.25.
        """

        print(f'Creating {self.numagents} agents...')
        self.agent = [[]]*self.numagents

        for k in range(self.numagents):

            # What sensors does this agent have?
            if self.defined_agents:
                # Sensors are defined by user.
                camera = True if self.agentdef[k][4] else None
                microphone = True if self.agentdef[k][5] else None
                iz = self.agentdef[k][6]                   # initial camera zoom
                pr = self.agentdef[k][7]/2                # 1/2 camera pan range
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
                self.agent[k].move(pos=self.agentdef[k][0:2],
                                   fdir=self.agentdef[k][2:4])
            elif True:
                # Move agent to a random ground location.
                self.agent[k].move_random(to='ground')
            else:
                # User manually drives the agent into position.
                self.agent[k].you_drive()


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


if __name__ == '__main__':

    mae = MultiAgentEnv(imdet=True)

    # Define the position, orientation, and sensors of each agent. Each row is
    #     [xpos, ypos, xfwd, yfwd, hascam, hasmic, initzoom, panrange].
    agentdef = [[-24, -12, -1, 0, 1, 1, 0.5, 180],
                [-15, 10, 1, 0, 1, 1, 0.5, 180],
                [63, -10, -1, 0, 1, 1, 0.5, 180],
                [-35, -61, 0, 1, 1, 1, 0.5, 180]]

    # Create the multiagent environment.
    mae.create(randseed=1234, envradius=100, agentdef=agentdef, showmap=True,
               pathsfile='obj_paths.txt', outfolder='./outputs')

    # Run the simulation.
    mae.run(sim_run_time=60, cycle_times=(1,5), audio_thresh=1500,
            image_thresh=0.8)
