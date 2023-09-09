"""
This is a class for displaying a 2D map of a SimWorld environemnt.

Author:
    Phil David, US Army Research Laboratory, December 2021.

Change History:
    P. David, Parsons Corp., Added microphone sensors. Added code to handle
    sensors that may be on or off.
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import map3d
from camera import *
from microphone import *
from fig import *


class Map2D:

    def __init__(self,  maps=None, label_colors=None, ptzcam=None,
                 camid=None, mic=None, micid=None, size=7, pos=(20,1200)):
        """
        Create and display a 2D map of an environment.

        Usage:
            map = Map2D(maps=None, label_colors=None, ptzcam=None, camid=None,
                        size=7, pos=(20,1200), fig=None)

        Arguments:
            maps: (Map3D) The 3D map that the 2D map is derived from.

            label_colors: (list) List of RGB values (3-tuples), one RGB value
            for each possible map label. All RGB values are in the range
            [0,255]. A map pixel with value N is assigned the color
            `label_colors[N]`

            ptzcam: (list or PTZCamera) If not None, one or more PTZCameras to
            display on the map. The position and pan angle of each camera will
            be overlayed on the 2D map. Default is None. PTZCameras may also be
            added to the map display via Map2D.AddCamera().

            camid: (int or list) An integer camera ID or a list of integer
            camera IDs to display. If None, then no IDs are displayed. Default
            is None. If `camid` is not None, then it must be a single integer
            when `ptzcam` is a single PTZCamera, and must be a list of integers
            of the same length as `ptzcam` when `ptzcam` is a list of PTZCamera.

            mic: (list or Microphone) If not None, one or more Microphone to
            display on the map. The position of each microphone will be
            overlayed on the 2D map. Default is None.

            micid: (int or list) An integer microphone ID or a list of integer
            microphone IDs to display on top of microphone markers in the map.
            If None, then no IDs are displayed. Default is None. If `micid` is
            not None, then it must be a single integer when `mic` is a single
            Microphone, and must be a list of integers of the same length as
            `mic` when `mic` is a list of Microphone.

            size: (float) Width and height of map window on screen (inches).

            pos: (float array-like) The (row, col) position on the screen
            to place the upper left corner of the map's window.

            fig: (fig.Fig) If not None, draw the map into the existing figure.
            Default is None.

        Description:
            If maps.dynamic is True, then the 2D map will be redrawn on each
            call to Map2D.Update. When maps.dynamic is False, only the camera
            overlays are updated when Map2D.Update is invoked. The map is
            dynamic if there are objects, such as people or vehicles, in the
            environment whose positions change over time and should be updated
            on the map.
        """

        assert type(maps) == map3d.Map3D, 'Argument "maps" must be a Map3D'
        assert type(label_colors) is list, 'Argument "label_colors" must be a list of RGB tuples'
        if ptzcam and camid:
            if type(ptzcam) is PTZCamera:
                assert type(camid) is int, 'Argument "camid" must be an int when "ptzcam" is a PTZCamera'
            else:
                assert len(ptzcam) == len(camid), 'Arguments "ptzcam" and "camid" must be same length'

        self.setup(maps=maps, label_colors=label_colors, ptzcam=ptzcam,
                   camid=camid, mic=mic, micid=micid, size=size, pos=pos)


    def setup(self, maps=None, label_colors=None, ptzcam=None,
              camid=None, mic=None, micid=None, size=7, pos=(20,1200), fig=None):
        """
        Create and display a 2D map of an environment.

        Usage:
            Map2d.setup(maps=None, label_colors=None, ptzcam=None, camid=None,
                        size=7, pos=(20,1200), fig=None)

        Arguments:
            maps: (Map3D) The 3D map that the 2D map is derived from.

            label_colors: (list) List of RGB values (3-tuples), one RGB value
            for each possible map label. All RGB values are in the range
            [0,255]. A map pixel with value N is assigned the color
            `label_colors[N]`

            ptzcam: (list or PTZCamera) If not None, one or more PTZCameras to
            display on the map. The position and pan angle of each camera will
            be overlayed on the 2D map. Default is None. PTZCameras may also be
            added to the map display via Map2D.AddCamera().

            camid: (int or list) An integer camera ID or a list of integer
            camera IDs to display. If None, then no IDs are displayed. Default
            is None. If `camid` is not None, then it must be a single integer
            when `ptzcam` is a single PTZCamera, and must be a list of integers
            of the same length as `ptzcam` when `ptzcam` is a list of PTZCamera.

            mic: (list or Microphone) If not None, one or more Microphone to
            display on the map. The position of each microphone will be
            overlayed on the 2D map. Default is None.

            micid: (int or list) An integer microphone ID or a list of integer
            microphone IDs to display on top of microphone markers in the map.
            If None, then no IDs are displayed. Default is None. If `micid` is
            not None, then it must be a single integer when `mic` is a single
            Microphone, and must be a list of integers of the same length as
            `mic` when `mic` is a list of Microphone.

            size: (float) Width and height of map window on screen (inches).

            pos: (float array-like) The (row, col) position on the screen
            to place the upper left corner of the map's window.

            fig: (fig.Fig) If not None, draw the map into the existing figure.
            Default is None, in which case a new figure is created.

        Description:
            If maps.dynamic is True, then the 2D map will be redrawn on each
            call to Map2D.Update. When maps.dynamic is False, only the camera
            overlays are updated when Map2D.Update is invoked. The map is
            dynamic if there are objects, such as people or vehicles, in the
            environment whose positions change over time and should be updated
            on the map.
        """

        assert type(maps) == map3d.Map3D, 'Argument "maps" must be a Map3D'
        assert label_colors is None or type(label_colors) is list, \
               'Argument "label_colors" must be a list of RGB tuples'

        if ptzcam and camid:
            if type(ptzcam) is PTZCamera:
                assert type(camid) is int, \
                       'Argument "camid" must be an int when "ptzcam" is a PTZCamera'
            else:
                assert len(ptzcam) == len(camid), \
                       'Arguments "ptzcam" and "camid" must be same length'

        if mic and micid:
            if type(mic) is Microphone:
                assert type(micid) is int, \
                       'Argument "micid" must be an int when "mic" is a Microphone'
            else:
                assert len(mic) == len(micid), \
                       'Arguments "mic" and "micid" must be same length'

        self.maps = maps
        self.cams = []
        self.cam_ids = []
        self.gh_cam_wedge = []
        self.gh_cam_circ1 = []
        self.gh_cam_line1 = []
        self.gh_cam_line2 = []
        self.gh_cam_text = []
        self.gh_lines = []
        self.num_cams = 0
        self.mics = []
        self.mic_ids = []
        self.gh_mic_circ1 = []
        self.gh_mic_circ2 = []
        self.gh_mic_text = []
        self.num_mics = 0
        self.radius = maps.map_radius
        self.mgridspc = maps.mgridspc
        if label_colors is not None: self.label_colors = label_colors

        if ptzcam:
            # Add cameras to the map.
            if type(ptzcam) is list:
                self.num_cams = len(ptzcam)
                for cam in ptzcam:
                    if type(cam) is not PTZCamera:
                        raise TypeError('Argument "ptzcam" must be a PTZCamera or a list of PTZCamera')
                self.cams.extend(ptzcam)
                if camid:
                    self.cam_ids.extend(camid)
                else:
                    self.cam_ids.extend([None]*self.num_cams)
            elif type(ptzcam) is PTZCamera:
                self.cams.extend([ptzcam])
                self.cam_ids.extend([camid])
                self.num_cams = 1
            else:
                raise TypeError('Argument "ptzcam" must be a PTZCamera or a list of PTZCamera')

        if mic:
            # Add microphones to the map.
            if type(mic) is list:
                self.num_mics = len(mic)
                for m in mic:
                    if type(m) is not Microphone:
                        raise TypeError('Argument "mic" must be a Microphone or a list of Microphone')
                self.mics.extend(mic)
                if micid:
                    self.mic_ids.extend(micid)
                else:
                    self.mic_ids.extend([None]*self.num_mics)
            elif type(mic) is Microphone:
                self.mics.extend([mic])
                self.mic_ids.extend([micid])
                self.num_mics = 1
            else:
                raise TypeError('Argument "mic" must be a Microphone or a list of Microphone')

        # Flatten the multi-channel label map into a single channel map.
        self.flatlabels = self.maps.flatlabels()

        # Get a figure to display the map in.
        if fig is None:
            self.mfig = Fig(figtitle='Map', figsize=(size,size), winpos=pos)
        else:
            self.mfig = fig

        # Get the colormap for map labels.
        self.numcolors = len(self.label_colors)
        self.cmap = colors.ListedColormap(np.array(self.label_colors)/255)

        # Display the flattened map.
        self.mfig.set(image=self.flatlabels, vmin=0, vmax=self.numcolors-1,
                      imextent=[-self.radius,self.radius,-self.radius,self.radius],
                      cmap=self.cmap, labelpos='bl', xlabel='X', ylabel='Y')

        if self.num_cams > 0:
            # Add cameras and create temporary markers at the origin. Markers
            # will be moved to their correct locations by the call to Update().
            r = 0.1*self.radius
            for k in range(self.num_cams):
                self.mfig.ax[0].add_patch(w)
                w = Wedge((0,0), r, 45, 135, zorder=1, facecolor=(1,0,0,0.6),
                          edgecolor=(1,1,1,1), linewidth=1)
                self.mfig.ax[0].add_patch(w)
                self.gh_cam_wedge.append(w)
                self.gh_cam_circ1.extend(plt.plot(0, 0, marker='o', markersize=9,
                                             markeredgecolor='w',
                                             markerfacecolor='k', zorder=2))
                if self.cam_ids[k]:
                    self.gh_cam_text.extend([plt.text(0, 0, str(self.cam_ids[k]),
                                                 fontsize=6, zorder=3)])
                else:
                    self.gh_cam_text.extend([None])

        if self.num_mics > 0:
            # Add microphones and create temporary markers at the origin. Markers
            # will be moved to their correct locations by the call to Update().
            for k in range(self.num_mics):
                self.gh_mic_circ1.extend(plt.plot(0, 0, marker='o', markersize=9,
                                             markeredgecolor='w',
                                             markerfacecolor='k', zorder=4))
                self.gh_mic_circ2.extend(plt.plot(0, 0, marker='o', markersize=14,
                                             markeredgecolor='w', markeredgewidth=2,
                                             markerfacecolor='r', zorder=3))
                if self.mic_ids[k]:
                    self.gh_mic_text.extend([plt.text(0, 0, str(self.mic_ids[k]),
                                                 fontsize=6, zorder=5)])
                else:
                    self.gh_mic_text.extend([None])

        # Update the sensor markers.
        self.Update()


    def new(self, maps):
        """
        Reset the 2D map using a new 3D map.

        Arguments:
            maps: (Map3D) The 3D map that the 2D map is derived from.
        """
        assert type(maps) == map3d.Map3D, 'Argument "maps" must be a Map3D'

        # Clear all the camera graphics from the current figure.
        for gh in self.gh_cam_circ1 + self.gh_mic_circ1 + self.gh_mic_circ2 + \
                  self.gh_cam_wedge + self.gh_cam_text + self.gh_mic_text:
            gh.remove()

        # Create a new 2D map in the existing figure.
        self.setup(maps=maps, fig=self.mfig)


    def AddCamera(self, ptzcam=None, camid=None):
        """
        Add one or more cameras to the map display.

        Usage:
            Map2D.AddCamera(ptzcam=None, camid=None)

        Arguments:
            ptzcam: (list or PTZCamera) If not None, one or more PTZCameras to
            display on the map. The position and pan angle of each camera will
            be overlayed on the 2D map. Default is None.

            camid: (int or list) An integer camera ID or a list of integer
            camera IDs to display on top of camera markers in the map. If None,
            then no IDs are displayed. Default is None. If `camid` is not None,
            then it must be a single integer when `ptzcam` is a single
            PTZCamera, and must be a list of integers of the same length as
            `ptzcam` when `ptzcam` is a list of PTZCamera.
        """

        if ptzcam is None:
            return

        numoldcams = self.num_cams

        if ptzcam and camid:
            if type(ptzcam) is PTZCamera:
                assert type(camid) is int, \
                       'Argument "camid" must be an int when "ptzcam" is a PTZCamera'
            else:
                assert len(ptzcam) == len(camid), \
                       'Arguments "ptzcam" and "camid" must be same length'

        # Add the cameras and camera IDs.
        if type(ptzcam) is list:
            numnewcams = len(ptzcam)
            for cam in ptzcam:
                if type(cam) is not PTZCamera:
                    raise TypeError('Argument "ptzcam" must be a PTZCamera or a list of PTZCamera')
            self.cams.extend(ptzcam)
            self.num_cams += numnewcams
            if camid:
                self.cam_ids.extend(camid)
            else:
                self.cam_ids.extend([None]*numnewcams)
        elif type(ptzcam) is PTZCamera:
            numnewcams = 1
            self.cams.extend([ptzcam])
            self.cam_ids.extend([camid])
            self.num_cams += 1
        else:
            raise TypeError('Argument "ptzcam" must be a PTZCamera or a list of PTZCamera')

        # Create temporary markers at origin. Markers will be moved to their
        # correct locations by the call to Update().
        r = 0.1*self.radius
        for k in range(numnewcams):
            w = Wedge((0,0), r, -45, 45, zorder=1, facecolor=(1,0,0,0.6),
                      edgecolor=(1,1,1,1), linewidth=1)
            self.mfig.ax[0].add_patch(w)
            self.gh_cam_wedge.append(w)
            self.gh_cam_circ1.extend(plt.plot(0, 0, marker='o', markersize=9,
                                markeredgecolor='w', markerfacecolor='k',
                                zorder=2))
            if self.cam_ids[numoldcams+k]:
                self.gh_cam_text.extend([plt.text(0, 0, str(self.cam_ids[numoldcams+k]),
                                              fontsize=6, color='w', weight='bold',
                                              zorder=3, ha='center', va='center')])
            else:
                self.gh_cam_text.extend([None])

        # Update the camera markers.
        self.Update()



    def AddMicrophone(self, mic=None, micid=None):
        """
        Add one or more microphones to the map display.

        Usage:
            Map2D.AddMicrophone(mic=None, micid=None)

        Arguments:
            mic: (list or Microphone) If not None, one or more Microphone to
            display on the map. The position of each microphone will be
            overlayed on the 2D map. Default is None.

            micid: (int or list) An integer microphone ID or a list of integer
            microphone IDs to display on top of microphone markers in the map. If None,
            then no IDs are displayed. Default is None. If `micid` is not None,
            then it must be a single integer when `mic` is a single
            Microphone, and must be a list of integers of the same length as
            `mic` when `mic` is a list of Microphone.
        """

        if mic is None:
            return

        numoldmics = self.num_mics

        if mic and micid:
            if type(mic) is Microphone:
                assert type(micid) is int, \
                       'Argument "micid" must be an int when "mic" is a Microphone'
            else:
                assert len(mic) == len(micid), \
                       'Arguments "mic" and "micid" must be same length'

        # Add the microphones and microphones IDs.
        if type(mic) is list:
            numnewmics = len(mic)
            for m in mic:
                if type(m) is not Microphone:
                    raise TypeError('Argument "mic" must be a Microphone or a list of Microphone')
            self.mics.extend(m)
            self.num_mics += numnewmics
            if micid:
                self.mic_ids.extend(micid)
            else:
                self.mic_ids.extend([None]*numnewmics)
        elif type(mic) is Microphone:
            numnewmics = 1
            self.mics.extend([mic])
            self.mic_ids.extend([micid])
            self.num_mics += 1
        else:
            raise TypeError('Argument "mic" must be a Microphone or a list of Microphone')

        # Create temporary markers at origin. Markers will be moved to their
        # correct locations by the call to Update().
        for k in range(numnewmics):
            self.gh_mic_circ1.extend(plt.plot(0, 0, marker='o', markersize=9,
                                markeredgecolor='w', markerfacecolor='k',
                                zorder=4))
            self.gh_mic_circ2.extend(plt.plot(0, 0, marker='o', markersize=14,
                                markeredgecolor='w', markerfacecolor='r',
                                zorder=3))
            if self.mic_ids[numoldmics+k]:
                self.gh_mic_text.extend([plt.text(0, 0, str(self.mic_ids[numoldmics+k]),
                                         fontsize=6, color='w', weight='bold',
                                         zorder=5, ha='center', va='center')])
            else:
                self.gh_mic_text.extend([None])

        # Update the microphone markers.
        self.Update()



    def Update(self, newmap=None):
        """
        Update the map positions of all sensors.

        Arguments:
            newmap: (Map3D) The 3D map that the 2D map is derived from.
        """

        for gh in self.gh_lines:
            gh.remove()
        self.gh_lines = []

        if self.maps.dynamic:
            # Update the flat labels map.
            self.flatlabels = self.maps.flatlabels()
            self.mfig.update_image(self.flatlabels)

        self.Update_cameras(newmap)
        self.Update_microphones(newmap)


    def Update_cameras(self, newmap=None):
        """
        Update the map positions of all cameras.

        Arguments:
            newmap: (Map3D) The 3D map that the 2D map is derived from.
        """

        if self.num_cams <= 0:
            return

        for cnum in range(self.num_cams):
            # Get position and direction of camera. This works whether or not
            # the camera is mounted on a vehicle.
            cpos, fp = self.cams[cnum].get_pos_fp()
            cpos = cpos[:2]       # (x, y)
            fp = fp[:2]           # (x, y)
            cdir = fp - cpos      # (dx, dy)
            cdir = cdir/np.linalg.norm(cdir)

            # Angles (in degrees) of camera HFOV.
            theta = np.rad2deg(np.arctan2(cdir[1],cdir[0]))
            theta1 = theta - self.cams[cnum].hfov/2
            theta2 = theta + self.cams[cnum].hfov/2

            # Update the camera marker.
            self.gh_cam_wedge[cnum].set(center=cpos,
                                        theta1=theta1,
                                        theta2=theta2)
            self.gh_cam_circ1[cnum].set_data(cpos[0], cpos[1])
            if self.cam_ids[cnum]:
                self.gh_cam_text[cnum].set_position((cpos[0], cpos[1]))

            if self.cams[cnum].power_on:
                self.gh_cam_wedge[cnum].set(facecolor=(1,0,0,0.6),
                                            edgecolor=(1,1,1,1))
            else:
                self.gh_cam_wedge[cnum].set(facecolor='none',
                                            edgecolor=(1,1,1,1))

        # Show changes to the figure.
        self.mfig.update()


    def Update_microphones(self, newmap=None):
        """
        Update the map positions of all microphones.

        Arguments:
            newmap: (Map3D) The 3D map that the 2D map is derived from.
        """

        if self.num_mics <= 0:
            return

        for mnum in range(self.num_mics):
            # Get position and direction of microphone. This works whether or not
            # the microphone is mounted on a vehicle.
            mpos = self.mics[mnum].get_pos()
            mpos = mpos[:2]

            # Update the microphone marker.
            plt.sca(self.mfig.ax[0])
            self.gh_mic_circ1[mnum].set_data(mpos[0], mpos[1])
            self.gh_mic_circ2[mnum].set_data(mpos[0], mpos[1])
            if self.mic_ids[mnum]:
                self.gh_mic_text[mnum].set_position((mpos[0], mpos[1]))

            if self.mics[mnum].power_on:
                self.gh_mic_circ2[mnum].set_markerfacecolor('r')
            else:
                self.gh_mic_circ2[mnum].set_markerfacecolor('k')

        # Show changes to the figure.
        self.mfig.update()


    def LineOfSight(self, p0, p1, maxdensity=10):
        """
        Check if there is a line of sight between two points in the 2D map.

        Usage:
            density = Map2D.LineOfSight(p0, p1)

        Arguments:
            p0: (2d array-like) First 2D endpoint  (x,y) of the proposed line of
            sight.

            p1: (2d array-like) Second 2D endpoint  (x,y) of the proposed line of
            sight.

        Returns:
            density: (float) Number of map cells that block the line of sight.
        """
        return


    def mark(self, x, y):
        """
        Mark a position in the map.
        """
        plt.sca(self.mfig.ax[0])
        plt.plot(x, y, marker='.', markersize=1, markeredgecolor='w',
                 zorder=0.5)    # fillstyle='none',


    def line(self, p1, p2, color=(0.01,0.46,1), linestyle='--', linewidth=1.5,
             zorder=1, alpha=0.6):
        """
        Draw  line on the map.

        Arguments:
            p1: (array-like) 2D position of the 1st endpoint of the line.

            p2: (array-like) 2D position of the 2nd endpoint of the line.
        """
        plt.sca(self.mfig.ax[0])
        gh = plt.plot((p1[0], p2[0]), (p1[1], p2[1]), linestyle,
                      color=color, linewidth=linewidth, zorder=zorder,
                      alpha=alpha)
        self.gh_lines.extend(gh)
        plt.pause(0.01)


    def WaitKeypress(self):
        """
        Wait for a user keypress.

        Usage:
            Map2D.WaitKeypress()

        Description:
            The program is paused until a key is pressed while the map figure
            is in focus.
        """
        self.mfig.wait(event='key_press_event')

