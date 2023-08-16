"""
Functions for user interaction with the VTK renderer window.

Author:
    Phil David, US Army Research Laboratory.
"""

import time
import os
import vtk
import numpy as np
import vtkutils as vtu
import imageio
from fig import *
from phutils import *
from speed import *
import matplotlib.pyplot as plt


# Global variables for the keypress callback function.
myworld = None              # the active world (SimWorld object)
mycamera = None             # the external camera (not the renderer's)
camerahome = None           # the camera's home position, orientation, etc.
vehiclehome = None          # the vehicle's home position and orientation
holdposition = False        # hold camera position (no translation)?
viewdirzero = None          # view direction for pan == 0 when holdposition is on
renderers = []              # list of renderers to view
vtkcamera = None            # VTK renderer's camera
renwin_idx = None           # index of render window to interact with
time_last_cmd = None        # time that last window interaction command was completed
save_images = None          # should displayed images be saved to files?
followers = None            # list of actors that follow the camera
recording = False           # recording images?
record_folder = None        # folder to save recording in
record_frame = 0            # frame number of recording
my2dmap = None              # 2D map of environment
gh_cur_pos1 = None          # graphic handle of current position marker in 2D map
gh_cur_pos2 = None          # graphic handle of current position marker in 2D map
speed = None                # speed of camera params (trans, rot, fps, zoom)
speed_index = 0             # what parameter to adjust?
freeze_objs = False         # do not update the positions of dynamic objects?


def callback_save(event):
    global save_images
    save_images = True


def callback_close(event):
    global save_images
    save_images = False


def init_wininteract(world, cam, renderer_index, map2d=None):
    """
    Initialize the window interaction functions: copy some data about the world
    to this module's global variables.

    Arguments:
        world: The SimWorld object that the user will operate in.

        cam: The external camera (not the renderer's) to mimick.

        map2d: (Map2D) If not None, this is the 2D map on which to display the
        camera position. Default is None.

        renderer_index: (int) Index of renderer to display during the
        interaction.
    """
    global renderers, vtkcamera, renwin_idx, myworld, holdposition, mycamera, \
           time_last_cmd, followers, my2dmap, camerahome, vehiclehome, speed

    myworld = world
    mycamera = cam
    followers = world.followers
    my2dmap = map2d

    if type(world.renderers) is not list:
        renderers = [world.renderers]
    else:
        renderers = world.renderers
    time_last_cmd = time.time()

    # Save current camera and vehicle pose in case the user wants to return to home.
    camerahome = mycamera.copy()
    if mycamera.mountedon:
        vehiclehome = (mycamera.mountedon.pos, mycamera.mountedon.orient)

    # Setup the VTK camera to mimick the external camera.
    vtkcamera = renderers[0].GetActiveCamera()
    pos, fp = mycamera.get_pos_fp()
    vtkcamera.SetPosition(pos)
    vtkcamera.SetFocalPoint(fp)
    vtkcamera.SetViewUp(0, 0, 1)
    vtkcamera.SetViewAngle(mycamera.vfov)
    vtkcamera.SetDistance(0.05)
    for ren in renderers:
        renwin = ren.GetRenderWindow()
        renwin.SetSize(mycamera.ncols, mycamera.nrows)

    # Set which render windows to use to acquire different images.
    renwin_idx = renderer_index

    speed = Speed()

    if my2dmap:
        my2dmap.Update()


def set_hold_position(holdpos):
    """
    Turn on or off "hold camera position." When turned on, the camera will not
    translate, but it can pan, tilt, and zoom up to the camera's pan, tilt, and
    zoom limits. Also, when on, report camera pan angles relative to initial
    camera focal point. Since the ground is flat, tilt angles are always
    relaitve to horizontal.
    """
    global holdposition, vtkcamera, viewdirzero
    holdposition = holdpos
    if holdposition:
        # Report pan angles relative to current camera focal point.
        pos = vtkcamera.GetPosition()
        fp = vtkcamera.GetFocalPoint()
        viewdirzero = np.array(fp) - np.array(pos)


def print_cam_pose(vtkcamera):
    """
    Print the camera's position and orientation.
    """
    global mycamera
    # pos = np.around(np.array(vtkcamera.GetPosition()), decimals=3)

    if mycamera.mountedon:
        pos, orient = mycamera.mountedon.camera_pos()
        pan = orient[2]
        tilt = orient[1]
    else:
        pos = mycamera.pos
        pan = mycamera.pan
        tilt = mycamera.tilt

    pos = np.around(pos, decimals=3)
    pan = np.around(pan, decimals=3)
    tilt = np.around(tilt, decimals=3)
    hfov = np.around(mycamera.hfov, decimals=3)
    print('Pos:({:.2f},{:0.2f},{:0.2f}) Pan:{:0.2f}º Tilt:{:0.2f}º Hfov:{:0.2f}º'.
          format(pos[0], pos[1], pos[2], pan, tilt, hfov))


def render_views():
    """
    Make objects face the camera and then render the scene.

    Notes:
        Actors that are set to follow the camera will be rotated about their Z
        axis so that their forward facing side faces the camera. The "forward
        facing side" of an actor is the side of the actor that pointed in the
        direction of the positive Y axis when it was created.
    """
    global renderers, myworld

    myworld.look_at_camera()

    # Update all renderers.
    for ren in renderers:
        # ren.ResetCameraClippingRange()  # this may cause objects to be wrongly clipped
        ren.GetRenderWindow().Render()


def print_ctrl_menu():
    """
    Print the interactor control menu.
    """
    print('-----------------------------------------------')
    print('                CONTROL MENU')
    print('-----------------------------------------------')
    print('=/- ................ Zoom camera in/out')
    print('A .................. Acquire some images')
    print('Ctrl-R ............. Start/stop recording')
    print('H .................. Camera to home')
    print('Keypad +/- ......... Increase/decrease speeds')
    print('L .................. Level camera tilt')
    print('Left/right arrow ... Pan camera left/right')
    print('Page up/down ....... Tilt camera up/down')
    print('S .................. Select paramater to adjust')
    print('Up/down arrow ...... Move camera forward/back')
    print('X .................. Return to main program')
    print('Z .................. Toggle freeze object motion')
    print('Space .............. Next frame')
    print('Esc ................ End all rendering')
    print('? .................. Print this menu')
    print('-----------------------------------------------')


def flush_input():
    pass


def my_keypress_callback(renderer, interactor, key):
    """
    This function is called by the VTK interactor when a key is pressed.

    Description:
        A variety of functions are implemented to move the camera around the
        scene and view various data.

        See print_ctrl_menu() for a list of the currently implemented user
        control commands.
    """

    global renderers, renwin_idx, myworld, mycamera, time_last_cmd, \
           save_images, followers, recording, record_folder, \
           record_frame, my2dmap, speed, speed_index, freeze_objs

    if time.time()-time_last_cmd < 0.05:
        # Event too soon after last event. Ignore it.
        return

    xptr, yptr = interactor.GetEventPosition()          # mouse pointer position
    key = key.lower()
    camera_motion = True

    cur_renderer = renderers[renwin_idx]
    cur_renwin = cur_renderer.GetRenderWindow()

    if key == 'escape':
        # Exit interactor's callback loop and end rendering.
        interactor.EnableRenderOff()
        interactor.ExitCallback()
        return
    elif key == 'x':
        # Exit interactor's callback loop and return control to the parent process.
        interactor.ExitCallback()
        return
    elif key == 'r':
        # Start or stop recording of images.
        if recording:
            ans = input('Stop recording images? (y/n) ')
        else:
            ans = input('Start recording images? (y/n) ')
        if ans.lower() == 'y' or ans.lower() == 'yes':
            recording = not recording
            if recording:
                while True:
                    record_folder = input('Enter name of folder to save images to: ')
                    if not os.path.isdir(record_folder):
                        try:
                            os.makedirs(record_folder)
                            break
                        except:
                            print('Unable to create that folder. Try again.')
                    record_frame = 0
                print('Recording...')
            else:
                print('Recording stopped')
        return
    elif key == 'question':
        # Print interactor control menu.
        print_ctrl_menu()
        return
    elif key == 'a':
        # Acquire images and groundtruth.
        cur = myworld.get_images()

        # Display the images.
        imnames = list(cur.keys())
        numimgs = len(imnames)
        nrows = 1 if numimgs < 3 else 2         # num rows & cols in figure grid
        ncols = int(np.ceil(numimgs/nrows))
        start = nrows*100 + ncols*10 + 1
        axpos = [k for k in range(start, start+numimgs)]   # figure axis positions: three-digit integers
        axlink = [k for k in range(numimgs)]               # link all axes in figure
        with Fig(figsize=(10,8.5), axpos=axpos, figtitle='Images', link=axlink) as f:
            for imname in imnames:
                curimage = cur[imname]
                axnum = imnames.index(imname)
                vmin, vmax = None, None
                if imname == 'imdepth':
                    vmin, vmax = 0, 2*myworld.env_radius+1
                elif imname == 'targetgt':
                    curimage = cur[imnames[0]]      # draw ground truth over 1st image
                f.set(image=curimage, axisnum=axnum, axistitle=imname.upper(),
                      vmin=vmin, vmax=vmax, axisfontsize=10)
                if imname == 'targetgt':
                    for t in cur[imname]:
                        f.draw(axisnum=axnum, rect=t, edgecolor='r', linewidth=1, linestyle='-')

            save_images = None
            f2 = Fig(figsize=(3,0.75), grid=[(1,4),(0,0,1,2),(0,2,1,2)],
                     figtitle='Select to continue...')
            f2.set(axisnum=0, button=('Save Images', callback_save, None, None, 10))
            f2.set(axisnum=1, button=('Close', callback_close, None, None, 10))
            while save_images == None:
                plt.pause(0.1)
            plt.close(f2.fig)

            if save_images == True:
                # Save all images.
                fpath = input('Enter file path/name (without extension): ')
                axnum = -1
                for imname in imnames:
                    fname = '{}_{}.png'.format(fpath,imname)
                    curimage = cur[imname]
                    axnum += 1
                    if type(curimage) == np.ndarray:
                        if curimage.ndim == 2:
                            # Convert monochrome images to RGB.
                            if curimage.max() == np.Inf:
                                # Remove Inf values from this image.
                                maxval = curimage[curimage < np.Inf].max()
                                curimage[curimage == np.Inf] = maxval + 1
                            ncolors = int(np.ceil(curimage.max())) + 1
                            if ncolors > 256:
                                # Scale image to [0,255].
                                curimage = (255*curimage/ncolors + 0.5).astype(np.uint8)
                                ncolors = 256
                            curimage = (255*cmjet1(numcolors=ncolors)[np.round(curimage).astype(int)]).astype(np.uint8)
                        imageio.imwrite(fname, curimage)

                    else:
                        f.saveaxis(axisnum=axnum, filename=fname)
                    print('Saved {} to "{}"'.format(imname, fname))
        return

    elif key == 'z':
        # Toggle object freeze motion.
        freeze_objs ^= True
        print('Objects', 'frozen' if freeze_objs else 'unfrozen')
        return
    elif key == 's':
        # Select the next speed parameter to adjust.
        speed.next()
        print("Adjust... ", end="")
        print('{:s} speed = {:.2f} {:s}'.format(speed.name, speed.value, speed.units))
        return
    elif key == 'kp_add' or key == 'period':
        # Increase the camera speed.
        speed.inc()
        print('{:s} speed = {:.2f} {:s}'.format(speed.name, speed.value, speed.units))
        return
    elif key == 'kp_subtract' or key == 'comma':
        # Decrease the camera speed.
        speed.dec()
        print('{:s} speed = {:.2f} {:s}'.format(speed.name, speed.value, speed.units))
        return
    elif key == 'd':
        # Get the 3D world coordinates of the point under the mouse.
        # The origin (0,0) for mouse coordinates is lower left corner.
        picker = vtk.vtkWorldPointPicker()
        picker.Pick([xptr, yptr, 0.0], cur_renderer)
        pos = picker.GetPickPosition()
        print('R,C = ({},{})  X,Y,Z = ({:.1f},{:.1f},{:.1f})'.
              format(yptr, xptr, pos[0], pos[1], pos[2]))
        zbuf = vtu.zbuffer2numpy(cur_renwin)
        yptr = zbuf.shape[0] - yptr - 1       # zbuf[0,0] is upper left corner.
        z = zbuf[yptr, xptr]
        print('Z-buffer[{},{}] = {:.4f}'.format(yptr, xptr, z))
        return
    elif key == 'v':
        # List visible objects.
        print('This function is not implemented')
        # cur_renderer.ResetCameraClippingRange()
        # print('{} visible actors'.format(cur_renderer.VisibleActorCount()))
        # VN.vtk_to_numpy(sel_node.GetSelectionList()).tolist()

        # actors = cur_renderer.GetActors()
        # print('{} actors'.format(actors.GetNumberOfItems()))
        # while True:
            # actor = actors.GetNextActor()
            # if actor is None:
                # break

        # picker = vtk.vtkRenderedAreaPicker() # vtk.vtkAreaPicker() vtkRenderedAreaPicker()
        # picker.SetPickCoords(0,0,1,1)
        # picker.SetRenderer(cur_renderer)
        # props = picker.GetProp3Ds()
        # print('Number of picked items = {}'.format(props.GetNumberOfItems()))
        # while True:
            # act = props.GetNextProp3D()
            # if act is None:
                # break
            # print('Actor position = {}'.format(act.GetPosition()))
        return
    elif key == 'prior':   # Page Up
        # Tilt up
        drot = speed.rot/speed.fps
        if mycamera.tilt + drot <= mycamera.maxtilt:
            vtkcamera.Pitch(drot)
            mycamera.inc(dtilt=drot)
        print_cam_pose(vtkcamera)
    elif key == 'next':  # Page Down
        # Tilt down
        drot = speed.rot/speed.fps
        if mycamera.tilt - drot >= mycamera.mintilt:
            vtkcamera.Pitch(-drot)
            mycamera.inc(dtilt=-drot)
        print_cam_pose(vtkcamera)
    elif key == 'left':
        # Pan left (increase pan angle)
        drot = speed.rot/speed.fps
        if holdposition:
            # Rotate mycamera if possible.
            if mycamera.pan + drot <= mycamera.maxpan:
                mycamera.inc(dpan=drot)
                vtkcamera.Yaw(drot)
        else:
            if mycamera.mountedon:
                # Rotate vehcile instead of mycamera.
                mycamera.mountedon.inc(orient=[0, 0, np.deg2rad(drot)])
                vtkcamera.Yaw(drot)
            else:
                # Rotate mycamera.
                mycamera.inc(dpan=drot)
                vtkcamera.Yaw(drot)
        print_cam_pose(vtkcamera)
    elif key == 'right':
        # Pan right (decrease pan angle)
        drot = speed.rot/speed.fps
        if holdposition:
            # Rotate mycamera if possible.
            if mycamera.pan - drot >= mycamera.minpan:
                mycamera.inc(dpan=-drot)
                vtkcamera.Yaw(-drot)
        else:
            if mycamera.mountedon:
                # Rotate vehcile instead of mycamera.
                mycamera.mountedon.inc(orient=[0, 0, np.deg2rad(-drot)])
                vtkcamera.Yaw(-drot)
            else:
                # Rotate mycamera.
                mycamera.inc(dpan=-drot)
                vtkcamera.Yaw(-drot)
        print_cam_pose(vtkcamera)
    elif key == 'minus':
        # Zoom out
        hfov = min(mycamera.maxhfov, mycamera.hfov+speed.zoom)
        mycamera.set(hfov=hfov)
        vtkcamera.SetViewAngle(mycamera.vfov)
        print_cam_pose(vtkcamera)
    elif key == 'equal':
        # Zoom in
        hfov = max(mycamera.minhfov, mycamera.hfov-speed.zoom)
        mycamera.set(hfov=hfov)
        vtkcamera.SetViewAngle(mycamera.vfov)
        print_cam_pose(vtkcamera)
    elif key == 'l':
        # Level the vtkcamera: set the tilt to 0.
        p = vtkcamera.GetPosition()
        fp = np.array(vtkcamera.GetFocalPoint())
        fp[2] = p[2]
        vtkcamera.SetFocalPoint(fp)
        mycamera.set(tilt=0)
        print_cam_pose(vtkcamera)
    elif key == 'h':
        # Return to home.
        if holdposition: return
        mycamera.identical(camerahome)
        if mycamera.mountedon:
            mycamera.mountedon.set(pos=vehiclehome[0][0:2], orient=vehiclehome[1])
        pos, fp = mycamera.get_pos_fp()    # camera position includes vehicle position
        vtkcamera.SetPosition(pos)
        vtkcamera.SetFocalPoint(fp)
        vtkcamera.SetViewUp(0, 0, 1)
        vtkcamera.SetViewAngle(mycamera.vfov)
        print_cam_pose(vtkcamera)
    elif key == 'up':
        # Move forward
        if holdposition: return
        p = vtkcamera.GetPosition()
        fp = np.array(vtkcamera.GetFocalPoint())
        dir = np.array(fp) - np.array(p)
        t = (speed.trans/speed.fps)*np.array(dir)/np.linalg.norm(dir)
        p = np.array(p) + t
        fp = fp + t
        vtkcamera.SetPosition(p)
        vtkcamera.SetFocalPoint(fp)
        if mycamera.mountedon:
            mycamera.mountedon.inc(pos=t)
        else:
            mycamera.pos += t
        print_cam_pose(vtkcamera)
    elif key == 'down':
        # Move backward
        if holdposition: return
        p = vtkcamera.GetPosition()
        fp = np.array(vtkcamera.GetFocalPoint())
        dir = np.array(fp) - np.array(p)
        t = (speed.trans/speed.fps)*np.array(dir)/np.linalg.norm(dir)
        p = np.array(p) - t
        fp = fp - t
        vtkcamera.SetPosition(p)
        vtkcamera.SetFocalPoint(fp)
        if mycamera.mountedon:
            mycamera.mountedon.inc(pos=-t)
        else:
            mycamera.pos -= t
        print_cam_pose(vtkcamera)
    elif key == 'space':
        # Refresh images.
        camera_motion = False
    else:
        # Unknown command.
        # print('Unknown command: "{}"'.format(key))
        return

    if not freeze_objs:
        myworld.inc_time(1/speed.fps)
    render_views()
    if my2dmap:
        my2dmap.Update()

    if camera_motion and recording:
        record_frame += 1
        imlist = myworld.get_images()
        write_images({'imcolor':imlist['color']},
                      path=record_folder, framenum=record_frame)

    time_last_cmd = time.time()

    return



def write_images(imlist, path='', framenum=None):
    """
    Write all images in the dictionary to files.
    """
    for imname in imlist.keys():
        if framenum:
            assert type(framenum) == int, '"framenum" must be an int'
            fname = '{}_{:06d}.png'.format(imname, framenum)
        else:
            fname = '{}.png'.format(imname)
        fname =  os.path.join(path, fname)
        img = imlist[imname]
        if type(img) == np.ndarray:
            if img.ndim == 2:
                # Convert monochrome images to RGB.
                if img.max() == np.Inf:
                    # Remove Inf values from this image.
                    maxval = img[img < np.Inf].max()
                    img[img == np.Inf] = maxval + 1
                ncolors = int(np.ceil(img.max())) + 1
                if ncolors > 256:
                    # Scale image to [0,255].
                    img = (255*img/ncolors + 0.5).astype(np.uint8)
                    ncolors = 256
                img = (255*cmjet1(numcolors=ncolors)[img]).astype(np.uint8)
            imageio.imwrite(fname, img)
            # print('Saved {} to "{}"'.format(imname, fname))

