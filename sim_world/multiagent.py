"""
Multiple agents view a common environment.

Author:
    Phil David, US Army Research Laboratory, December 5, 2020.

"""


from simworld import *
from camera import *
from microphone import *
from agent import *

audiorows = 0.5          # fraction of image rows (in [0,1]) over which to plot audio signals

if __name__ == '__main__':
    # Program parameters.
    numagents = 6             # number of agents to place in the environment
    imsize = (1280,720)       # all rendered images are this size
    dtime = 0.1               # time (seconds) between agent updates
    panstep = [0]*numagents   # step size depends on size of camera FOV
    fnum = 0                  # frame number of saved images
    prob_have_camera = 1.0    # probability that an agent has a camera
    prob_have_mic = 1.0       # probability that an agent has a microphone
    prob_cam_onoff = 0.0      # probability of switching a camera on/off on any step
    prob_mic_onoff = 0.0      # probability of switching a microphone on/off on any step
    prob_cam_reset = 0.1      # probability of resetting a camera pan direction on any step
    initsensors = True        # need to initialize sensors?
    verbose = True            # display a lot of output?

    # Create a random outdoor environmeent.
    sim = SimWorld(imsize=imsize, timeofday=[700,1800], env_radius=200,
                   bldg_density=1, road_density=2.0, clutter_density=0.5,
                   plant_density=1, people_density=0.1, animal_density=0.1,
                   vehicle_density=0.1, airborne_density=0.05,
                   bldg_plant_density=0.5, barrier_density=1, gndfeat_density=0.2,
                   lookouts=True, probwindowoccupied=0.25,
                   p_over_building={'person':0.5, 'clutter':0.1, 'animal':1.0},
                   p_over_road={'person':0.1, 'clutter':0.05, 'animal':0.2},
                   textures='textures', rand_seed=8370646)

    # Create and display a 2D map of the environment.
    mymap = Map2D(maps=sim.map3d, size=8, label_colors=sim.label_colors)

    # Create the agents, each with their own camera and/or microphone. Each
    # agent will have a least one of these sensors.
    print('Creating {:d} agents...'.format(numagents))
    agent = [[]]*numagents
    for k in range(numagents):
        camera = microphone = None

        if np.random.rand() <= prob_have_camera:
            camera = PTZCamera(imsize=imsize, rnghfov=(3,54),
                              rngpan=(-np.Inf,np.Inf), rngtilt=(-45,60),
                              pos=(0,0,1), pan=0, tilt=0, zoom=0)

        if camera is None or np.random.rand() <= prob_have_mic:
            microphone = Microphone(pos=(0,0,1))

        agent[k] = Agent(env=sim, cam=camera, mic=microphone,
                         map2d=mymap, objdet=None)

        if True:
            # Move agent to a random ground location.
            agent[k].move_random(to='ground')
        else:
            # User manually drives the agent into position.
            agent[k].you_drive()

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
        while True:
            fnum += 1
            print(f'\n[[ Frame {fnum}, Time {sim.time:.2f} sec. ]]\n')
            if sim.time > 46:
                exit(0)
            f.fig.suptitle(f'[[ Time: {sim.time:.2f} sec. ]]', fontsize=10)

            # Update the sensor configuartion of all agents.
            for k in range(numagents):
                if initsensors or (agent[k].cam and
                                   np.random.rand() < prob_cam_reset):
                    # Reset agent camera zoom and pan speed.
                    agent[k].cam.set(zoom=np.random.rand())
                    panstep[k] = np.deg2rad((1 if np.random.rand() > 0.5 else -1)
                                            *(5-3*agent[k].cam.zoom))
                    if verbose:
                        print(f"Reset agent {k}'s camera")

                # Increment camera pan (rotation about z axis).
                if agent[k].cam:
                    agent[k].inc(orient=[0,0,panstep[k]])

            # Update the 2D groundtruth map.
            mymap.Update()

            # Collect sensor data from all agents.
            for k in range(numagents):
                if verbose:
                    print(f'Agent {k+1}')

                f.clearaxis(axisnum=k, keepimage=True)   # clear agent's display

                if agent[k].cam:
                    imgs = agent[k].get_images(imlist=['color'])
                    f.set(axisnum=k, image=imgs['color'],
                          axistitle=agent[k].name, axisoff=True)

                if agent[k].mic:
                    audio = agent[k].get_audio(maxdist=250, verbose=verbose)

                    # Plot the audio signal.
                    nrows=imgs['color'].shape[0]
                    ncols=imgs['color'].shape[1]
                    sig = audio['signal']
                    s = np.ceil(len(sig)/ncols).astype(int)  # resample interval
                    sig2 = sig[0:-1:s]
                    x = np.arange(len(sig2))
                    y = nrows/2 + audiorows*nrows*sig2/WorldObj.max_audio
                    f.set(axisnum=k)
                    plt.plot(x, y, 'r', linewidth=0.5)
                    plt.pause(0.1)

            if True:
                # Save figures to files.
                f.savefig('tmp/originals/agents_{:07d}.png'.format(fnum))
                mymap.mfig.savefig('tmp/originals/map_{:07d}.png'.format(fnum))

            sim.inc_time(dtime)

            if initsensors: initsensors = False
