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

    def __init__(self):
        """
        Initialize a multiagent environment.

        Arguments:
        """
        pass


    def create(self,
               envradius = 500,
               simtime = 120.0,
               randseed = None,
               agentdef = None,
               imsize = (1280,720),
               dtime = 0.1,
               playaudio = False,
               prob_have_cam = 1.0,
               prob_have_mic = 1.0,
               prob_cam_onoff = 0.2,
               prob_mic_onoff = 0.2,
               prob_cam_reset = 0.1,
               pathsfile = None,
               outfolder = None,
               showmap = True,
               verbose = True
               ):
        """
        Create a multiagent environment.

        Arguments:
            simtime:float -- time (sec.) to run the simulation to.

            randseed:int -- random seed or None.

            agentdef:int or 2D array-like -- If an int, this is the number of
            agents to create, and each is moved to a random location in the
            environment. Otherwise, this must be a 2D array-like where each row
            of the array is [x, y, dx, dy, cam, mic] defining one agent: (x,y)
            is the agent's 2D position (z is 0) in the environment, (dx, dy) is
            the 2D orientation of the agent, cam is true (1) if the agent has a
            camera, and mic is True (1) if the agent has a microphone.

            imsize:tuple -- size (cols,rows) of all rendered images.

            dtime:float -- time (seconds) between agent updates.

            playaudio:bool --  play recorded audio from agents?

            prob_have_cam:float -- probability that an agent has a camera.

            prob_have_mic:float -- probability that an agent has a microphone.

            prob_cam_onoff:float -- probability of switching a camera on/off on any step.

            prob_mic_onoff:float -- probability of switching a microphone on/off on any step.

            prob_cam_reset:float -- probability of resetting a camera pan & zoom on any step.

            pathsfile:str -- file defining object trajectories, or None.

            outfolder:str -- name of output folder, or None.

            showmap:bool -- show the 2D groundtruth map?

            verbose:bool -- display a lot of output?
        """

        self.envradius = envradius
        self.sim_end_time = simtime
        self.randseed = randseed
        self.agentdef = agentdef
        self.imsize = imsize
        self.dtime = dtime
        self.playaudio = playaudio
        self.prob_have_cam = prob_have_cam
        self.prob_have_mic = prob_have_mic
        self.prob_cam_onoff = prob_cam_onoff
        self.prob_mic_onoff = prob_mic_onoff
        self.prob_cam_reset = prob_cam_reset
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


    def create_agents(self):
        """
        Create agents, each with their own camera and/or microphone. Each
        agent should have a least one of these sensors.
        """

        print(f'Creating {self.numagents} agents...')
        self.agent = [[]]*self.numagents

        for k in range(self.numagents):

            # Decide what sensors this agent will have.
            if self.defined_agents:
                # Sensors are defined by user.
                camera = True if self.agentdef[k][4] else None
                microphone = True if self.agentdef[k][5] else None
            else:
                # Randomly assign sensors.
                camera = True if np.random.rand() <= self.prob_have_cam else None
                if not camera:
                    microphone = True
                else:
                    microphone = True if np.random.rand() <= self.prob_have_mic else None

            if camera:
                z = np.random.rand()                      # initial camera zoom
                camera = PTZCamera(imsize=self.imsize, rnghfov=(3,54),
                                   rngpan=(-np.Inf,np.Inf), rngtilt=(-45,60),
                                   pos=(0,0,1), pan=0, tilt=0, zoom=z)

            if microphone:
                microphone = Microphone(pos=(0,0,1))

            self.agent[k] = Agent(env=self.sim, cam=camera, mic=microphone,
                                  map2d=self.map2d, objdet=None)

            if camera:
                self.agent[k].panstep = np.deg2rad((1 if np.random.rand() > 0.5 else -1)
                                                   *(5-3*camera.zoom))

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
                if np.random.rand() < self.prob_cam_onoff:
                    self.agent[k].cam.toggle_power()
                    print(f' toggle_power', end='')

                if self.agent[k].cam.power == "on":
                    if np.random.rand() < self.prob_cam_reset:
                        # Reset agent camera pan speed and zoom.
                        self.agent[k].cam.set(zoom=np.random.rand())
                        self.agent[k].panstep = np.deg2rad((1 if np.random.rand() > 0.5 else -1)
                                                *(5-3*self.agent[k].cam.zoom))
                        print(f' reset_p/t', end='')
                    self.agent[k].inc(orient=[0,0,self.agent[k].panstep])

                print(f' {self.agent[k].cam.power}', end='')
            else:
                print(' not present', end='')

            print(', microphone', end='')

            if self.agent[k].mic:
                if np.random.rand() < self.prob_mic_onoff:
                    self.agent[k].mic.toggle_power()
                    print(f' toggle_power', end='')
                print(f' {self.agent[k].mic.power}')
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
                if self.agent[k].cam.power == "on":
                    # Get agent image.
                    imgs = self.agent[k].get_images(imlist=['color'])
                    f.set(axisnum=k, image=imgs['color'],
                          axistitle=self.agent[k].name, axisoff=True)

            if self.agent[k].mic:
                if self.agent[k].mic.power == "on":
                    # Get agent audio.
                    audio = self.agent[k].get_audio(maxdist=250, verbose=self.verbose)
                    self.overlay_audio(audio['signal'], f, k)
                    if self.playaudio:
                        audio_io.play_audio(audio['signal'], audio['samplerate'])


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


    def run(self):
        """
        Run the multiagent simulation.
        """

        fnum = 0                    # frame number of saved images

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

        # Determine layout of figure that will display agent sensor data.
        nr = int(np.ceil(np.sqrt(self.numagents))) # num rows of axis in display
        nc = int(np.ceil(self.numagents/nr))       # num cols of axis in display
        axpos = [(nr,nc,k+1) for k in range(self.numagents)]   # 2D grid of axes

        #----------------------------------------------------------------------
        # Collect and display agent sensor data and occasionally change sensor
        # configurations.
        #----------------------------------------------------------------------

        print('Started agents... Close the figure to quit')

        with Fig(axpos=axpos, figtitle='Agent Cameras', figsize=(10,8)) as f:
            while self.sim.time <= self.sim_end_time:
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

    mae = MultiAgentEnv()

    # Define the position, orientation, and available sensors of each agent.
    # Each row is [x, y, dx, dy, cam, mic].
    agentdef = [[-24, -18, -1, 0, 1, 1],
                [7, 14, 0, -1, 1, 1],
                [45, -18, -1, 0, 1, 1],
                [-44, -58, 0, 1, 1, 1] ]

    # Create the multiagent environment.
    mae.create(randseed=1234, envradius=100, agentdef=agentdef, showmap=True,
               pathsfile='obj_paths.txt', outfolder='./outputs')

    # Run the simulation.
    mae.run()
