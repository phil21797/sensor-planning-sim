"""
Run the outdoor simulation with multiple agents operating in the environment.

Author:
    Phil David, Parsons, May 1, 2023.

"""


from simworld import *
from camera import *
from microphone import *
from agent import *
from datetime import datetime
from audio_io import *
from fig import *
import os

audiorows = 0.5          # fraction of image rows (in [0,1]) over which to
                         # overlay audio signals


def create_agents(numagents):
    """
    Create agents, each with their own camera and/or microphone. Each
    agent will have a least one of these sensors.
    """

    print(f'Creating {numagents} agents...')
    agent = [[]]*numagents

    for k in range(numagents):
        camera = microphone = None

        if np.random.rand() <= prob_have_cam:
            z = np.random.rand()
            camera = PTZCamera(imsize=imsize, rnghfov=(3,54),
                               rngpan=(-np.Inf,np.Inf), rngtilt=(-45,60),
                               pos=(0,0,1), pan=0, tilt=0, zoom=z)

        if camera is None or np.random.rand() <= prob_have_mic:
            microphone = Microphone(pos=(0,0,1))

        agent[k] = Agent(env=sim, cam=camera, mic=microphone,
                            map2d=mymap, objdet=None)

        if camera is not None:
            agent[k].panstep = np.deg2rad((1 if np.random.rand() > 0.5 else -1)
                                       *(5-3*camera.zoom))

        if True:
            # Move agent to a random ground location.
            agent[k].move_random(to='ground')
        else:
            # User manually drives the agent into position.
            agent[k].you_drive()

    return agent


def config_agent_sensors(agent):
    """
    Update the configuartion of all agents' sensors.
    """

    numagents = len(agent)

    for k in range(numagents):
        print(f'Agent {k+1}: camera', end='')
        if agent[k].cam:
            if np.random.rand() < prob_cam_onoff:
                agent[k].cam.toggle_power()
                print(f' toggle_power', end='')
            if agent[k].cam.power == "on":
                if np.random.rand() < prob_cam_reset:
                    # Reset agent camera pan speed and zoom.
                    agent[k].cam.set(zoom=np.random.rand())
                    agent[k].panstep = np.deg2rad((1 if np.random.rand() > 0.5 else -1)
                                            *(5-3*agent[k].cam.zoom))
                    print(f' reset_p/t', end='')
                agent[k].inc(orient=[0,0,agent[k].panstep])
            print(f' {agent[k].cam.power}', end='')
        else:
            print(' not present', end='')

        print(', microphone', end='')
        if agent[k].mic:
            if np.random.rand() < prob_mic_onoff:
                agent[k].mic.toggle_power()
                print(f' toggle_power', end='')
            print(f' {agent[k].mic.power}')
        else:
            print(' not present')
    print()


def collect_agent_sensor_data(agent, f:Fig, playaudio:bool=False,
                              verbose:bool=False):
    """
    Collect and display sensor data from all agents.
    """

    numagents = len(agent)

    for k in range(numagents):
        if verbose:
            print(f'Agent {k+1}')

        # Clear agent display.
        f.clearaxis(axisnum=k, keepimage=True)
        f.set(axisnum=k, image=blackimage,
              axistitle=agent[k].name, axisoff=True, shownow=True)

        if agent[k].cam:
            if agent[k].cam.power == "on":
                # Get agent image.
                imgs = agent[k].get_images(imlist=['color'])
                f.set(axisnum=k, image=imgs['color'],
                      axistitle=agent[k].name, axisoff=True)

        if agent[k].mic:
            if agent[k].mic.power == "on":
                # Get agent audio.
                audio = agent[k].get_audio(maxdist=250, verbose=verbose)
                overlay_audio(audio['signal'], f, k, imsize)
                if playaudio:
                    play_audio(audio['signal'], audio['samplerate'])


def overlay_audio(sig, f, axnum, imsize):
    """
    Overlay an audio signal on top of an image.
    """
    ncols = imsize[0]
    nrows = imsize[1]
    s = np.ceil(len(sig)/ncols).astype(int)   # resample to fit across image
    sig2 = sig[0:-1:s]
    x = np.arange(len(sig2))
    y = nrows/2 + audiorows*nrows*sig2/WorldObj.audio_max_measured
    f.set(axisnum=axnum)
    plt.plot(x, y, 'r', linewidth=0.5)
    plt.pause(0.1)


if __name__ == '__main__':

    # Program parameters.
    sim_end_time = 45.0       # time (sec.) to run the simulation to
    randseed = 8370646        # random seed or None
    numagents = 6             # number of agents to place in the environment
    imsize = (1280,720)       # size (cols,rows) of all rendered images
    dtime = 0.1               # time (seconds) between agent updates
    playaudio = False         # play recorded audio from agents?
    prob_have_cam = 0.6       # probability that an agent has a camera
    prob_have_mic = 0.8       # probability that an agent has a microphone
    prob_cam_onoff = 0.3      # probability of switching a camera on/off on any step
    prob_mic_onoff = 0.3      # probability of switching a microphone on/off on any step
    prob_cam_reset = 0.1      # probability of resetting a camera pan & zoom on any step
    outfolder = './outputs'   # name of output folder, or None
    verbose = True            # display a lot of output?

    fnum = 0                  # frame number of saved images
    blackimage = np.zeros(list(imsize[::-1])+[3], dtype=np.uint8)

    if outfolder != None:
        # Create a folder to save output in.
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        outfolder = outfolder + '_' + dt
        if os.path.exists(outfolder):
            print(f'Output folder "{outfolder}" already exists')
            exit(1)
        else:
            try:
                os.mkdir(outfolder)
                print(f'Created output folder "{outfolder}"')
            except:
                raise Exception(f'Unable to create output folder "{outfolder}"')

    # Create a random outdoor environmeent.
    sim = SimWorld(imsize=imsize, timeofday=[700,1800], env_radius=200,
                   bldg_density=1, road_density=2.0, clutter_density=0.5,
                   plant_density=1, people_density=0.1, animal_density=0.1,
                   vehicle_density=0.1, airborne_density=0.05,
                   bldg_plant_density=0.5, barrier_density=1, gndfeat_density=0.2,
                   lookouts=True, probwindowoccupied=0.25,
                   p_over_building={'person':0.5, 'clutter':0.1, 'animal':1.0},
                   p_over_road={'person':0.1, 'clutter':0.05, 'animal':0.2},
                   textures='textures', rand_seed=randseed)

    # Create and display a 2D map of the environment.
    mymap = Map2D(maps=sim.map3d, size=8, label_colors=sim.label_colors)

    # Create the agents, each with their own camera and/or microphone. Each
    # agent will have a least one of these sensors.
    agent = create_agents(numagents)

    if False:
        # Display color, semantic label, and depth images from one agent.
        imgs = agent[0].get_images()
        with Fig(axpos=[131,132,133], figtitle='My World', figsize=(10,4), link=[0,1,2]) as f:
            f.set(axisnum=0, image=imgs['color'], axistitle='Color')
            f.set(axisnum=1, image=imgs['label'], axistitle='Semantic labels')
            f.set(axisnum=2, image=imgs['depth'], axistitle='Depth')
            print('Press any key to continue...')
            f.wait(event='key_press_event')

    # Determine layout of figure that will display agent sensor data.
    nr = int(np.ceil(np.sqrt(numagents)))      # num rows of axis in display
    nc = int(np.ceil(numagents/nr))            # num cols of axis in display
    axpos = [(nr,nc,k+1) for k in range(numagents)]        # 2D grid of axes

    #----------------------------------------------------------------------
    # Collect and display agent sensor data and occasionally change sensor
    # configurations.
    #----------------------------------------------------------------------

    print('Started agents... Close the figure to quit')

    with Fig(axpos=axpos, figtitle='Agent Cameras', figsize=(10,8)) as f:
        while sim.time <= sim_end_time:
            fnum += 1
            print(f'\n[[ Frame {fnum}, Time {sim.time:.2f} sec. ]]\n')
            f.fig.suptitle(f'⟦ Time: {sim.time:.2f} sec. ⟧', fontsize=10)

            # Update the configuartion of all agent sensors.
            config_agent_sensors(agent)

            # Update the 2D groundtruth map.
            mymap.Update()

            # Collect and display sensor data from all agents.
            collect_agent_sensor_data(agent, f, playaudio, verbose)

            if outfolder is not None:
                # Save figures to files.
                f.savefig(f'{outfolder}/agents_{fnum:07d}.png')
                mymap.mfig.savefig(f'{outfolder}/map_{fnum:07d}.png')

            sim.inc_time(dtime)
