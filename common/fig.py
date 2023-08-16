
"""
Simplified interface to the Matplotlib plotting library.

Author:
    Phil David, Army Research Laboratory
"""

import numpy as np
import os.path
import matplotlib.pyplot as plt
from matplotlib import get_backend
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import multiprocessing as mp
import PIL
from phutils import *


class Formatter(object):
    """
    Class to display values under the mouse pointer in a displayed image.
    """
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        x1 = int(round(x))
        y1 = int(round(y))
        z = None
        if isinstance(self.im, PIL.Image.Image):
            nrows, ncols = self.im.size
            if x1 >= 0 and x1 < ncols and y1 >= 0 and y1 < nrows:
                z = self.im.getpixel((x1,y1))
                str = 'x={:.01f}, y={:.01f}, z={:.03f}'.format(x, y, z)
            else:
                str = ''
        elif isinstance(self.im, np.ndarray):
            nrows, ncols = self.im.shape
            if x1 >= 0 and x1 < ncols and y1 >= 0 and y1 < nrows:
                z = self.im[y1,x1]
                str = 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
            else:
                str = ''
        else:
            raise TypeError('Unable to display image type: {}'.format(type(im)))
        return str


class Fig():
    """
    A class to work with Matplotlib figures. This class is setup as a context
    manager (it has __enter__ and __exit__ methods).
    """

    def __init__(self, figsize=(6,6), winpos=None, axpos=None, grid=None,
                 link=None, fontsize=12, figtitle=None, cmapname=None,
                 facecolor=(0.93,0.88,0.83)):
        """
        Create an empty figure.

        Arguments:
            figsize: A tuple (width, height) of the figure size (inches).

            winpos: list -- The (row, col) position on the screen to place the
            upper left corner of the figure's window. The coordinates of the
            upper left corner of the screen are (0,0).

            axpos: Define the axis positions using three-digit integers. 'axpos'
            is a list of axes positions. Each item in the list is a three-digit
            integer or a three-element tuple where the first digit is the number
            of rows, the second the number of columns, and the third the index
            of the subplot. For example, axpos=[221, 222, 212] creates three
            subplots, two small subplots in the 1st row and one wide subplot in
            the 2nd row.

            grid: Define the axes positions using a grid layout. 'grid' is a list
                [(nrows, ncols), (r0, c0, rspan0, cspan0, label0),
                (r1, c1, rspan1, cspan1, label1), ...]
                where:
                    (nrows, ncols) is the number of rows and columns in the grid

                    (rK, cK) is the start position (indexed from 0) of the Kth axis

                    (rspanK, cspanK) is the number of rows and columns in the Kth
                    axis. If these are omitted, then rspanK = cspanK = 1.

                    labelK is 1 if the X and Y axes will be labeled, 0 otherwise.

                It is important than none of the grid cells overlap. If grid
                cells do overlap, an error will likely occur when the respective
                axis is accessed.

            fontsize: The figure font size, a number in [1,maxfontsize?].

            figtitle: The figure title, a number or a string. This appears in
            the title bar of the figure.

            cmapname: The name of the figure colormap, a string. For a list of
            different colormaps, see https://matplotlib.org/users/colormaps.html.

            link: A list of axes numbers to link.

        Description:
            Only one of arguments 'axpos' or 'grid' should be used. If both are
            omitted, then one axis is created by default.

        """
        self.facecolor = facecolor

        self.fig = plt.figure(num=figtitle, figsize=figsize, facecolor=facecolor,
                              edgecolor=None, frameon=True, clear=False, dpi=130)

        if winpos is not None:
            move_figure(self.fig, winpos[0], winpos[1])

        if axpos is None and grid is None:
            axpos = [111]
        elif axpos is not None and grid is not None:
            raise ValueError('Only one of arguments "axpos" and "grid" can be given')

        # Create the axes.
        self.ax = []
        if axpos is not None:
            if type(axpos) is not list:
                axpos = [axpos]
            for p in axpos:
                if type(p) in [list, tuple]:
                    self.ax += [plt.subplot(p[0],p[1],p[2])]
                else:
                    self.ax += [plt.subplot(p)]
        elif grid is not None:
            shape = grid[0]               # shape of grid: (nrows, ncols)
            for p in grid[1:]:
                try:
                    loc = p[0:2]          # start location of axis: (rnum, cnum)
                    nr = 1 if len(p) == 2 else p[2]     # num of rows in axis
                    nc = 1 if len(p) <= 3 else p[3]     # num of cols in axis
                except:
                    raise Exception('Grid positions must be (rnum, cnum [,rspan [,cspan]])')
                self.ax += [plt.subplot2grid(shape, loc , rowspan=nr, colspan=nc,
                                            fig=self.fig)]
                if len(p) > 4 and not p[4]:
                    # Turn off X and Y axis labels.
                    self.ax[-1].set_xticks([])
                    self.ax[-1].set_yticks([])
        else:
            raise Exception('Internal error: no axes are defined')

        self.numaxes = len(self.ax)
        self.numcuraxis = 0

        self.axistitle = [None]*self.numaxes
        self.image = [None]*self.numaxes
        self.limits = [None]*self.numaxes
        self.image_shape = [None]*self.numaxes
        self.image_dtype = [None]*self.numaxes
        self.scroll = [{'text':[], 'lines':3, 'fontsize':8, 'active':False,
                        'bkgndcolor': '0.8', 'gh':[]} for k in range(self.numaxes)]

        self.figfontsize = fontsize     # Size of figure title font
        self.axisfontsize = 10          # size of axis title font
        self.figtitle = figtitle        # Figure title

        # if self.figtitle is not None:
        #     # This places a title in the figure above all subplots (the supertitle).
        #     plt.suptitle(self.figtitle, fontsize=16)

        # Global colormap. Used if no other colormaps are specified for individual axes.
        if cmapname is None:
            self.cmapname = 'jet'
        else:
            self.cmapname = cmapname
        plt.set_cmap(self.cmapname)

        if link is not None:
            # Link a set of axes.
            for axnum in link:
                self.ax[axnum].set_adjustable('box')  # ('box-forced')
            ax0 = self.ax[link[0]]
            for axnum in link[1:]:
                ax0.get_shared_x_axes().join(ax0, self.ax[axnum])
                ax0.get_shared_y_axes().join(ax0, self.ax[axnum])

        self.cmap = [None]*self.numaxes
        self.vmin = [None]*self.numaxes
        self.vmax = [None]*self.numaxes
        self.imextent = [None]*self.numaxes
        self.xlabel = [None]*self.numaxes
        self.ylabel = [None]*self.numaxes
        self.btn = [None]*self.numaxes
        self.overlays = [[]]*self.numaxes
        for k in range(0, self.numaxes):
            self.cmap[k] = plt.get_cmap()
            # self.set(image=np.array([[0]]), axisnum=k)

        # Adjust layout whenever the figure is redrawn.
        # ==> This causes warning messages to be generated.
        # self.fig.set_tight_layout(True)

        def handle_close(event):
            # This function is called when a window close_event occurs.
            del self.fig
            self.fig = None

        self.fig.canvas.mpl_connect('close_event', handle_close)

        return


    def __enter__(self):
        return self


    def __exit__(self, *args):
        """
        Close the current figure.
        """
        if self.fig is not None:
            plt.close(self.fig)


    def close(self):
        """
        Close the current figure.
        """
        if self.fig is not None:
            plt.close(self.fig)


    def set(self, axisnum=None, image=None, axistitle=None, cmapname=None,
            vmin=None, vmax=None, axisfontsize=None, imextent=None, aspect=None,
            xlabel=None, ylabel=None, labelpos=None, clearoverlays=True,
            axiscolor=(0.93,0.88,0.83), axisoff=False, button=None, vminmax=None,
            titleoffset=1.0, limits=None, scroll=None, grid=None, shownow=True,
            xticks=None, yticks=None, cmap=None,
            **kwargs):
        """
        Set axis properties.

        Arguments:
            axisnum: The number [0,1,...] of the axis to change.

            image: A 2D array-like image to display in the axis.

            axistitle: The title of the axis, a string.

            axisfontsize: The font size of the axis title.

            axiscolor: The background (face) color of the axis. This may be a
            RGB or RGBA tuple of floats in [0,1], or a hex RGB or RGBA string,
            or a single character color code, or a color name as a string.
            Default is the same color as the figure background.

            cmapname: The name of the colormap, a string.

            vmin, vmax: The colors in the colormap are mapped to data values
            ranging from vmin to vmax.

            vminmax: The colors in the colormap are mapped to data values
            ranging from vminmax[0] to vminmax[1].

            imextent: The location, (left, right, bottom, top), in
            data-coordinates, of the lower-left and upper-right corners of image
            data. If None, the image is positioned such that the pixel centers
            fall on zero-based (row, column) indices. Default is None. By
            assigning these coordinates, the positions of graphic overlays may
            be given in a more meaningful coordinate sytem.

            limits: The range of visible data in the axis. This is a tuple or
            list (left, right, bottom, top). Default is None, in which case the
            display limits are determined automatically.

            xlabel: The label of the x axis.

            ylabel: The label of the y axis.

            xticks: (xtick_locs, xtick_labels) The X axis tick locations and
            tick labels. To turn off X axis ticks, use xticks=([],[]).

            yticks: (ytick_locs, ytick_labels) The Y axis tick locations and
            tick labels. To turn off Y axis ticks, use yticks=([],[]).

            labelpos: Position of axes labels. This is a string specifying the
            positions of the axes labels and tick labels. This string may
            include "l", "r", "t", and "b" for left, right, top, and bottom,
            respectively.

            button: A tuple (labelstr, btncallback, color, hovercolor, fontsize)
            defining a button GUI for axis 'axisnum'. 'color' and 'hovercolor'
            are optional color strings.  Use None for default values.

            cmap: A Matplotlib colormap.  Default is None.

            scroll: The properties of a scrolling text box. This is a dictionary
            with any of the following keys:
                    'text': string to display. This may contain '\n'.
                    'fontsize': size of text font, an int.
                    'bkgndcolor: string or tuple giving background color.

            aspect: Axis aspect ratio: 'equal' or 'auto'.

            clearoverlays: Should existing graphic overlays (text, boxes, etc.)
            be removed when new data is displayed? Default is True.

            axisoff: A boolean value indicating whether or not the x and y axis
            should be turned off. Default is False.

            grid: Set the axis grid on (True) or off (False). Default is off.
                To customize the grid, use Matplotlib.axes.Axes.grid() keyword
                arguments as in the following:
                    Fig.set(grid=True, which='major', color=(1,0,0),
                            linestyle='--', linewidth=0.75)
                    Fig.set(grid=True, which='minor', color=(0,0,1),
                            linestyle='--', linewidth=0.25)

            kpcallback: (function) Set the keypress callback function. This

        """

        if self.fig is None:
            raise Exception('Figure does not exist')

        if axisnum is None:
            axisnum = self.numcuraxis
        elif type(axisnum) is not int:
            raise TypeError('Axis number must be a single int')
        elif axisnum >= self.numaxes:
            raise ValueError('Axis number {} out of range: 0..{}'.
                            format(axisnum, self.numaxes-1))
        else:
            self.numcuraxis = axisnum
        plt.sca(self.ax[axisnum])

        if axiscolor is not None:
            # Set the axis background (face) color.
            self.ax[axisnum].set_facecolor(axiscolor)

        if button is not None:
            # Setup a button GUI. To guarantee that the button remains
            # responsive and not garbage-collected, a reference to the object
            # must be maintained.
            labelstr = button[0]
            callback = button[1]
            color1 = button[2] if len(button) > 2 and button[2] != None else (0.9,0.9,0.9)
            color2 = button[3] if len(button) > 3 and button[3] != None else (1,1,0.8)
            fontsize = button[4] if len(button) > 4 and button[4] != None else 7
            self.ax[axisnum].set_autoscale_on(True)
            self.ax[axisnum].set_frame_on(True)
            self.btn[axisnum] = Button(self.ax[axisnum], labelstr,
                                       color=color1, hovercolor=color2)
            self.btn[axisnum].label.set_fontsize(fontsize)
            self.btn[axisnum].label.set_verticalalignment('center_baseline')
            # self.btn[axisnum].label.set_fontweight('bold')
            self.btn[axisnum].on_clicked(callback)


        if scroll is not None:
            # The axis is a box with scrolling text.
            ax = self.ax[axisnum]
            if self.scroll[axisnum]['active'] == False:
                ax.axis(xmin=0, xmax=1, ymin=0, ymax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor(self.scroll[axisnum]['bkgndcolor'])
                self.scroll[axisnum]['active'] = True

            # Retrieve new input about the scroll box.
            for key in scroll.keys():
                if key == 'text':
                    txt = scroll['text'].split('\n')
                    self.scroll[axisnum]['text'] += txt
                else:
                    self.scroll[axisnum][key] = scroll[key]
                    if key == 'bkgndcolor':
                        ax.set_facecolor(self.scroll[axisnum]['bkgndcolor'])

            # Get axis height in pixels.
            bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            height = self.fig.dpi*bbox.height

            # Determine number of text lines and line spacing.
            fontsize = self.scroll[axisnum]['fontsize']
            interlinespace = 8
            numlines = int(np.floor((height-interlinespace)/(fontsize+interlinespace)))
            ypos = 1 - interlinespace/height
            dpos = (fontsize+interlinespace)/height

            # Keep only the last 'numlines' lines of text.
            self.scroll[axisnum]['text'] = self.scroll[axisnum]['text'][-numlines:]

            # Remove any previous text from the axis.
            for gh in self.scroll[axisnum]['gh']:
                gh.remove()
            gh = []

            # Display new text.
            for txt in self.scroll[axisnum]['text']:
                t = ax.text(0.005, ypos, txt, ha="left", va="top", fontsize=fontsize)
                gh += [t]
                ypos -= dpos
            self.scroll[axisnum]['gh'] = gh          # save text graphic handles

        if vminmax is not None:
            vmin = vminmax[0]
            vmax = vminmax[1]
        if vmin is not None:
            self.vmin[axisnum] = vmin
        if vmax is not None:
            self.vmax[axisnum] = vmax

        if imextent is not None:
            # Imextent == (left, right, bottom, top)
            self.imextent[axisnum] = imextent

        if xlabel is not None:
            self.xlabel[axisnum] = xlabel
            plt.xlabel(xlabel)
        if ylabel is not None:
            self.ylabel[axisnum] = ylabel
            plt.ylabel(ylabel)

        if xticks is not None:
            plt.xticks(xticks[0], xticks[1])

        if yticks is not None:
            plt.yticks(yticks[0], yticks[1])

        if cmapname is not None:
            self.cmap[axisnum] = plt.get_cmap(cmapname)
        elif cmap is not None:
            self.cmap[axisnum] = cmap

        if axistitle is not None:
            self.axistitle[axisnum] = axistitle

        if axisfontsize is not None:
            # Currently, all axis have the same font size.
            self.axisfontsize = axisfontsize

        if aspect is not None:
            self.ax[axisnum].set_aspect(aspect, adjustable='box')

        if axisoff is True:
            self.ax[axisnum].axis('off')

        if grid is not None:
            if grid == True:
                if 'which' not in kwargs:
                    # Display major and minor grid lines using default line colors and widths.
                    plt.minorticks_on()
                    self.ax[axisnum].grid(b=True, which='major', color=(0.1,0.1,0.1), linestyle='--', linewidth=0.75)
                    self.ax[axisnum].grid(b=True, which='minor', color=(0.1,0.1,0.1), linestyle='--', linewidth=0.25)
                elif kwargs['which'].lower() == 'both':
                    # Setup both major and minor grid lines.
                    plt.minorticks_on()
                    self.ax[axisnum].grid(b=True, **kwargs)
                elif kwargs['which'].lower() == 'minor':
                    # Setup minor grid lines.
                    plt.minorticks_on()
                    self.ax[axisnum].grid(b=True, **kwargs)
                else:
                    # Setup major grid lines. (`which` == 'major')
                    self.ax[axisnum].grid(b=True, **kwargs)
            else:
                # Turn off all grid lines.
                self.ax[axisnum].grid(b=False, **kwargs)

        if labelpos is not None:
            # Set the position of the axes ticks and tick labels.
            for p in labelpos:
                p = p.lower()
                if p == 't':
                    self.ax[axisnum].xaxis.tick_top()
                    self.ax[axisnum].xaxis.set_label_position('top')
                elif p == 'b':
                    self.ax[axisnum].xaxis.tick_bottom()
                    self.ax[axisnum].xaxis.set_label_position('bottom')
                elif p == 'l':
                    self.ax[axisnum].yaxis.tick_left()
                    self.ax[axisnum].yaxis.set_label_position('left')
                elif p == 'r':
                    self.ax[axisnum].yaxis.tick_right()
                    self.ax[axisnum].yaxis.set_label_position('right')
                else:
                    raise ValueError('Invalid tick position string:', p)

        if image is not None:
            # Check the type of image to be displayed.
            if isinstance(image, PIL.Image.Image):
                imshape = (image.size[1], image.size[0])     # (nrows, ncols)
            elif isinstance(image, np.ndarray):
                imshape = image.shape
            else:
                raise TypeError('Unable to display image type: {}'.format(type(image)))

            if clearoverlays:
                # Remove existing graphic overlays.
                for obj in self.overlays[axisnum]:
                    try:
                        obj.remove()
                    except:
                        pass
                self.overlays[axisnum] = []

            # Update the image data.
            if self.image[axisnum] is None or imshape != self.image_shape[axisnum] \
                        or image.dtype != self.image_dtype[axisnum]:
                # New image is different size than previous image. Reset axis.
                plt.sca(self.ax[axisnum])
                if self.image_shape[axisnum] != None and \
                               imshape[0:2] != self.image_shape[axisnum][0:2]:
                    self.ax[axisnum].clear()
                self.image[axisnum] = self.ax[axisnum].imshow(image,
                                                              cmap=self.cmap[axisnum],
                                                              vmin=self.vmin[axisnum],
                                                              vmax=self.vmax[axisnum],
                                                              interpolation='none',
                                                              extent=self.imextent[axisnum],
                                                              **kwargs)
                self.image_shape[axisnum] = imshape
                self.image_dtype[axisnum] = image.dtype
            else:
                # Update existing data.
                self.image[axisnum].set_data(image)    # .set_array(image)

            # Function to display values under mouse pointer.
            self.ax[axisnum].format_coord = Formatter(image)

        if limits is not None:
            self.limits[axisnum] = limits
            self.ax[axisnum].set_xlim(limits[0:2])
            self.ax[axisnum].set_ylim(limits[2:4])

        if axistitle is not None:
            if self.numaxes > 1:
                self.ax[axisnum].set_title(axistitle, fontsize=self.axisfontsize,
                                           y=titleoffset)
            else:
                self.fig.suptitle(axistitle, fontsize=self.axisfontsize,
                                  y=titleoffset)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        if shownow:
            plt.draw()
            try:
                plt.pause(0.01)
            except:
                pass
        else:
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # plt.draw()
            plt.show(block=False)
            # self.fig.canvas.manager.window.raise_()    # Raise figure
            # plt.pause(0.01)        # figure only draws when there is free CPU cycles

        return


    def update_image(self, image, ax=0):
        """
        Update the image in a figure without changing focus to it.

        Note:
            This function was created because I wasn't able to get Fig.set() to
            do this without changing the wondow focus.
        """
        self.image[ax].set_data(image)


    def text(self, x, y, s, axisnum=None, shownow=True, **kwargs):
        """
        Display text in the figure.

        Usage:
            text = fig.text(x, y, s, axisnum=None, **kwargs)

        Arguments:
            x, y: The location (in data coordinates) of the upper left corner of
            the string.

            s: The string to display.

            axisnum: The axis number to display the text in.

            kwargs: Keyword arguments passed to the Matplotlib text() function.
            Some popular keyword arguments include:

                color=COLOR

                backgroundcolor=COLOR

                fontfamily=FONTNAME, e.g., 'serif', 'monospace', etc.

                fontsize=SIZE, a float or string (e.g., 'small').

                fontweight=WEIGHT, a number in [0,1000] or a string (e.g.,
                'bold').

                fontstyle=STYLE, a string (e.g., 'normal', 'italic', etc.)

        Returns:
            The text object that was displayed.
        """

        # Get the axis to edit.
        if axisnum is None:
            axisnum = self.numcuraxis
        elif type(axisnum) is not int:
            raise TypeError('Axis number must be a single int')
        elif axisnum >= self.numaxes:
            raise ValueError('Axis number {} out of range: 0..{}'.
                            format(axisnum, self.numaxes-1))
        else:
            self.numcuraxis = axisnum
        ax = self.ax[axisnum]
        plt.sca(ax)

        t = ax.text(x, y, s, **kwargs)
        self.overlays[axisnum] += [t]

        if shownow:
            plt.draw()
            try:
                plt.pause(0.01)
            except:
                pass

        return t


    def draw(self, axisnum=None, rect=None, poly=None, circ=None, point=None,
             line=None, fill=False, edgecolor='None', facecolor='None', linewidth=1,
             linestyle='-', markersize=5, marker='.', markercolor='None',
             xlabel=None, ylabel=None, clearoverlays=False, shownow=True,
             **kwargs):
        """
        Draw a graphic object (point, line, polygon, circle) into an axis.

        Usage:
            gh = fig.draw(axisnum=None, rect=None, poly=None, circ=None,
                          point=None, line=None, fill=False, edgecolor='r',
                          facecolor='None', linewidth=1, linestyle='-',
                          markersize=5, marker='.', markercolor='b',
                          xlabel=None, ylabel=None, clearoverlays=False,
                          shownow=True, **kwargs)

        Arguments:
            axisnum: The axis to draw the object into. Default is None, in which
                case the current axis is used.
            point: A list [xdata, ydata], where 'xdata' and 'ydata' are lists of
                the x- and y-coordinates of a set of points to plot.
            line: A list of line segment endpoints: [xdata, ydata].
            rect: Parameters of a rectangle to draw. This may be a Blob object
                or a list/tuple [xmin, ymin, width, height]. These values are in
                "data coordinates" whenever an 'imextent' is specified for the
                axes (see fig.set()).
            poly: An Nx2 numpy array of the polygon verticies. The last vertex is
                connected to the first.
            circ: Parameters of a circle to draw. This is a list or tuple
                [xctr, yctr, radius].
            fill: Fill the object? True/False. Default is False.
            edgecolor: The color to draw the object lines and edges. Default
                is 'r'.
            facecolor: The color to draw the object face. 'None' is no color.
                None is the Matplotlib default face color. Default is 'None'.
            linewidth: The width of the object edges. Default is 1.
            linestyle: The style of the object edges. Default is '-'.
            xlabel: The label on the x-axis (a string).
            ylabel: The label on the y-axis (a string).
            marker: The marker to use for points. A string. E.g.: '.','o','s'.
            markercolor: Color of the marker. Default is 'b'.
            markersize: The size of the marker.
            clearoverlays: Should existing graphic overlays (text, boxes, etc.)
                be removed before drawing the new object? Default is False.
            shownow: If True, run plt.show() to immediately show the graphic
                object. Default is True.
            kwargs: Keyword arguments passed to Matplotlib's plotting functions.

        Returns:
            gh: The handle of the object drawn into the axis. To remove the
                graphic object from the axis, use: gh.remove()
        """

        patch = gh = None

        # Get the axis to edit.
        if axisnum is None:
            axisnum = self.numcuraxis
        elif type(axisnum) is not int:
            raise TypeError('Axis number must be a single int')
        elif axisnum >= self.numaxes:
            raise ValueError('Axis number {} out of range: 0..{}'.
                            format(axisnum, self.numaxes-1))
        else:
            self.numcuraxis = axisnum
        ax = self.ax[axisnum]
        plt.sca(ax)

        if clearoverlays:
            # Remove existing graphic overlays.
            for obj in self.overlays[axisnum]:
                try:
                    obj.remove()
                except:
                    pass
                    # print('Error removing', obj)
            self.overlays[axisnum] = []

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if type(facecolor) is str: facecolor = facecolor.lower()
        if type(edgecolor) is str: edgecolor = edgecolor.lower()
        if type(markercolor) is str: markercolor = markercolor.lower()
        if type(edgecolor) is str and type(facecolor) is str:
            if edgecolor == 'none' and facecolor == 'none':
                if type(markercolor) is str and markercolor == 'none':
                    markercolor = 'b'
                edgecolor = facecolor = markercolor
        if facecolor is None or (type(facecolor) is str and facecolor != 'none'):
            fill = True                      # draw filled object
        if fill is True and type(facecolor) is str and facecolor == 'none':
            facecolor = None                 # fill with default face color

        if point is not None:
            # Plot points as vertices on a line with 'linestyle' == None. This
            # allows the points to be removed using gh.remove(). Axis.plot()
            # doesn't seem to support this function.

            # gh = ax.plot(point[0], point[1], marker=marker,
                       # markersize=markersize, markerfacecolor=markercolor,
                       # markeredgecolor=markercolor, linestyle='none', **kwargs)
            x = point[0]
            y = point[1]
            if type(x) not in [list, tuple]:
                x = [x]     # individual point - must be provided in a sequence.
            if type(y) not in [list, tuple]:
                y = [y]
            ln = Line2D(x, y, marker=marker,
                        markersize=markersize, markerfacecolor=facecolor,
                        markeredgecolor=edgecolor, linestyle='none', **kwargs)
            gh = ax.add_line(ln)
        elif line is not None:
            ln = Line2D(line[0], line[1], linewidth=linewidth,
                        linestyle=linestyle, color=edgecolor, **kwargs)
            gh = ax.add_line(ln)
        elif rect is not None:
            # Draw a rectangle.
            if type(rect) is Blob:          # Get the ractangle's parameters.
                xmin = rect.xmin
                ymin = rect.ymin
                width = rect.width
                height = rect.height
            else:
                try:
                    xmin = rect[0]
                    ymin = rect[1]
                    width = rect[2]
                    height = rect[3]
                except:
                    raise Exception('Unknown data type for rect')

            # Create a Rectangle patch
            patch = patches.Rectangle((xmin, ymin), width, height, fill=fill,
                                      edgecolor=edgecolor, facecolor=facecolor,
                                      linewidth=linewidth, linestyle=linestyle,
                                      **kwargs)
        elif poly is not None:
            # Draw a polygon.
            patch = patches.Polygon(poly, closed=True, edgecolor=edgecolor,
                                    facecolor=facecolor, linewidth=linewidth,
                                    linestyle=linestyle, **kwargs)
        elif circ is not None:
            patch = patches.Circle(circ[0:2], circ[2], edgecolor=edgecolor,
                                   facecolor=facecolor, linewidth=linewidth,
                                   linestyle=linestyle, **kwargs)

        if patch is not None:
            # Add the patch to the Axes
            gh = ax.add_patch(patch)

        if gh is not None:
            self.overlays[axisnum] += [gh]

        if self.limits[axisnum] == None and self.imextent[axisnum] == None:
            ax.autoscale(enable=True, axis='both')

        if shownow:
            # plt.show(block=False)
            plt.draw()
            self.fig.canvas.flush_events()
            # self.fig.canvas.manager.window.raise_()    # Raise figure
            try:
                pass # plt.pause(0.01)
            except:
                pass

        return gh


    def update(self):
        """
        Update the figure without sending the user interface focus to it.
        """
        # plt.draw()
        # self.fig.canvas.manager.window.raise_()    # Raise figure
        # try:
            # plt.pause(0.01)
        # except:
            # pass
        try:
            self.fig.canvas.flush_events()
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.01)
        except:
            raise Exception('Unable to flush figure events')


    def savefig(self, filename=None, **kwargs):
        """
        Save the figure as an image file.

        Usage:
            Fig.savefig(filename, **kwargs)

        Arguments:
            filename: (str) Name of image file to save figure to. The image
                format is deduced from the extension of the filename. If
                'filename' includes a path to a nonexisting directory, then
                this function will try to create that directory prior to saving
                the image.
            kwargs: See matplotlib.pyplot.savefig documentaion for additional
                keyword arguments.
        """
        fpath, fname = os.path.split(filename)
        if fpath != '' and not os.path.exists(fpath):
            os.mkdir(fpath)
        self.fig.savefig(filename, **kwargs)


    def clearaxis(self, axisnum:int=0, shownow:bool=True, keepimage=False):
        """
        Clear images and graphics from an axis without changing axis labels or
        limits.

        Usage:
            Fig.clearaxis( axisnum:int=0, shownow:bool=True)

        Arguments:
            axisnum: (int) The number of the axis to clear. Default is 0.

            shownow: (bool) If True, show the updated figure immediately.
            Default is True.

            keepimage: (bool) If True, do not clear the axis image, if there is
            one. Dafault is False.
        """
        glist = self.ax[axisnum].patches + self.ax[axisnum].lines + \
                self.ax[axisnum].texts + self.ax[axisnum].collections
        if not keepimage:
            glist = glist + self.ax[axisnum].images
        for o in glist:
            o.remove()
        if shownow:
            plt.draw()
            plt.pause(0.01)

    def saveaxis(self, axisnum=0, filename=None, **kwargs):
        """
        Save one axis of a figure to an image file.

        Usage:
            Fig.saveaxis(axisnum=0, filename=None, **kwargs)

        Arguments:
            axisnum: (int) The number of the axis to save.
            filename: (str) Name of image file to save figure to. The image
                format is deduced from the extension of the filename. If
                'filename' includes a path to a nonexisting directory, then
                this function will try to create that directory prior to saving
                the image.
            kwargs: See matplotlib.pyplot.savefig documentaion for additional
                keyword arguments.
        """
        extent = self.ax[axisnum].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        fpath, fname = os.path.split(filename)
        if fpath != '' and not os.path.exists(fpath):
            os.mkdir(fpath)
        self.fig.savefig(filename, bbox_inches=extent, **kwargs)


    def set_callback(self, event='key_press_event', callback=None):
        """
        Set the figure callback function.

        Arguments:
            event: (str) The event to trigger the callback function. This may be
            any of the following events.

                Event name 	            Class and description
                'button_press_event' 	MouseEvent - mouse button is pressed
                'button_release_event' 	MouseEvent - mouse button is released
                'draw_event' 	        DrawEvent - canvas draw (but before screen update)
                'key_press_event' 	    KeyEvent - key is pressed
                'key_release_event' 	KeyEvent - key is released
                'motion_notify_event' 	MouseEvent - mouse motion
                'pick_event' 	        PickEvent - an object in the canvas is selected
                'resize_event' 	        ResizeEvent - figure canvas is resized
                'scroll_event'      	MouseEvent - mouse scroll wheel is rolled
                'figure_enter_event' 	LocationEvent - mouse enters a new figure
                'figure_leave_event' 	LocationEvent - mouse leaves a figure
                'axes_enter_event' 	    LocationEvent - mouse enters a new axes
                'axes_leave_event'   	LocationEvent - mouse leaves an axes

            callback: (function) The callback function to invoke when the event
            occurs. This function should take a single argument, event, as in
            the following example.

                def on_key_press(event):
                    print('Pressed "{}"'.format(event.key))
                    plt.show(block=False)
                    return
        """
        connectid = self.fig.canvas.mpl_connect(event, callback)


    def wait(self, event=None, key=None):
        """
        Wait for an event.

        Arguments:
            event: A string naming the event to wait for. Recognized events are the
                following:
                    'button_press_event'
                    'key_press_event'
        """

        if self.fig is None:
            raise Exception('Figure does not exist')

        if event is None:
            # Wait for the user to close the figure.
            plt.show(block=True)
        elif event == 'button_press_event':
            plt.waitforbuttonpress(timeout=None)
        elif event == 'key_press_event':
            while True:
                if plt.waitforbuttonpress(timeout=1e5):
                    break
            # def on_key_press(event):
                # print('key pressed')
                # plt.show(block=False)
                # return
            # connectid = self.fig.canvas.mpl_connect(event, on_key_press)
            # print('Blocking...')
            # plt.show()
            # # Wait for a key press...
            # print('Event has occured')
            # plt.show(block=False)
        else:
            raise ValueError('Unrecognized event type: ', event)

        return


    def set_linear_colormap(self, rgbcolors, nbins=256, maxrgb=1.0,
                            axisnums=None, cmapname='mycolors'):
        """
        Assign a linear colormap to the figure.

        'rgbcolors' is an Nx3, 2D array-like, color map. For 'rgbcolors'[K,:] == [R,
        G, B], the colormap maps the integer value K (K = 0,...,N-1) in the figure's
        axes to the RGB value [R, G, B]/'maxrgb'. Floating point values between K and
        K+1 are mapped to the linearly interpolated color between 'rgbcolors'[K,:]
        and 'rgbcolors'[K+1,:]. Objects in the figure should be assigned colors in
        the continuous range [0, N-1].

        'nbins' is the number of color bins to create. This determines the smoothness
        of the transition between the colors.
        """

        rgbcolors = np.asarray(rgbcolors)/maxrgb
        self.numcolors = rgbcolors.shape[0]
        self.cmapname = cmapname
        newcmap = LinearSegmentedColormap.from_list(cmapname, rgbcolors, N=nbins)

        if axisnums is None:
            axisnums = range(0,self.numaxes)   # All axes will use this colormap.
        for anum in axisnums:
            self.cmap[anum] = newcmap
            self.vmin[anum] = 0
            self.vmax[anum] = self.numcolors - 1

        return


    def spawn(self):
        """
        EXPERIMENTAL -- Spawn the figure as a new process to keep it awake.

        Notes:
            This doesn't always work!
        """
        # def keep_awake(im):
            # while True:
                # plt.draw()
                # #fig.canvas.draw()
                # time.sleep(0.1)
        # show_image(self.image[0].get_array())
        p = mp.Process(target=show_image, args=(self.image[0].get_array(),))
        p.start()


def move_figure(f, r, c):
    """
    Move figure's upper left corner to pixel (r, c).
    """
    backend = get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (c, r))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((c, r))
    elif backend in {'Qt5Agg', 'GTK'}:
        f.canvas.manager.window.move(c, r)  # can also use window.setGeometry
    else:
        print("move_figure: Don't know how to move figure window for "\
              "backend {:s}".format(backend))


def show_image(im):
    imax = plt.imshow(np.zeros((1,1),dtype=np.uint8))
    imax.set_array(im)
    # plt.imshow(im)
    plt.draw()
    # print('should not get here!')


def btncallback(event):
    print('Button was pressed')


if __name__ == '__main__':
    """
    Test the figure class.
    """

    def quitcallback(event):
        print('Closing all figures')
        plt.close(f1.fig)
        plt.close(f2.fig)
        exit()

    def on_key_press(event):
        print('Pressed key "{}"'.format(event.key))
        plt.show(block=False)
        return

    def callback_save(event):
        print('Saving images...')

    def callback_close(event):
        print('Closing images...')
        plt.close(f3.fig)

    f1 = Fig(figsize=(3,3), winpos=(10,20))
    f1.set(image=100*np.random.rand(100,100), labelpos='tr',
           imextent=[-100,100,20,200])     #, limits=[-50, 50, 50, 150])
    f1.draw(rect=[-60,110,25,50], edgecolor='r', facecolor='k', linewidth=5, zorder=4)
    f1.draw(line=[[0,-100],[50,200]], edgecolor='b', linewidth=5, zorder=5)
    gh1 = f1.draw(line=[[-100,0],[50,200]], edgecolor='g', linewidth=5, zorder=2)
    gh2 = f1.draw(point=[(0, 75, 50, -80),(100, 100, 50, 80)], markersize=20,
            markercolor='k', xlabel='X Label', ylabel='Y Label', zorder=1)
    f1.set_callback(event='key_press_event', callback=on_key_press)
    print('Press Q to close this figure...')
    plt.show()

    colors = [(0,0,0),           # 0 = Black
              (255,255,255),     # 1 = White
              (255,0,0),         # 2 = Red
              (255,255,0),       # 3 = Yellow
              (163,252,255),     # 4 = Sky blue
              (55,223,102),      # 5 = Green 1
              (178,223,191),     # 6 = Green 2 (low saturation)
              (191,155,112),     # 7 = Tan 1
              (191,180,167)]     # 8 = Tan 2 (low saturation)
    ncolors = len(colors)

    naxes = 3
    # f2 = Fig(figsize=(10,5), axpos=[221, 222, 212], figtitle='Test', link=[0,1,2])
    f2 = Fig(figsize=(5,7), grid=[(9,4),(0,0,2,2),(0,2,2,2),(2,0,2,4),(4,0,2,4),
                                  (7,0,2,2),(7,2,2,2)],
             figtitle='Test', link=[0,1,2])
    f2.set(axisnum=4, button=('Next', btncallback, 'tan', 'grey'))
    f2.set(axisnum=5, button=('Quit', quitcallback))
    f2.set(axisnum=3, scroll={'bkgndcolor':'#ffe9a7', 'fontsize':10}, axistitle='Status')
    f2.set_linear_colormap(colors, maxrgb=255)

    for k in range(20):
        axnum = k % naxes
        f2.set(axisnum=3, scroll={'text':'Axis number {}'.format(axnum)})
        sz = np.random.randint(low=10, high=500, dtype=np.int)  # Size of image.
        im = np.random.rand(sz, sz)*(ncolors-1)                 # Float image.

        f2.set(axisnum=axnum, image=im, axistitle='Frame {}'.format(k),
               xlabel='columns', ylabel='rows', labelpos='bl')
        f2.text(5, 5, 'Image {}'.format(k), axisnum=axnum,
                color='w', backgroundcolor='k', fontsize=6, fontweight='bold')
        print('Size of image {} = {}'.format(k, sz))

        f3 = Fig(figsize=(3,0.75), grid=[(1,4),(0,0,1,2),(0,2,1,2)],
                 figtitle='Select to continue...')
        f3.set(axisnum=0, button=('[ Save Images ]', callback_save, 'None', None, 12))
        f3.set(axisnum=1, button=('[ Close ]', callback_close, None, None, 12))

        f2.wait(event='key_press_event') # Wait for user to press a key

    plt.show()

    print('done')
