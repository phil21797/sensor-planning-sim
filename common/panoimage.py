"""
Panoramic image creation.

Author:
    P. David, Army Research Laboratory.
"""


import numpy as np
from numbers import Number
from skimage.transform import warp
from phutils import *


class PanoImage():

    def __init__(self, prange, trange, ares=1, numchan=3, dtype=int, init=0):
        """
        Initialize a panoramic image.

        Usage:
            panoim = PanoImage(prange, trange, ares=1, numchan=3, dtype=int)

        Arguments:
            prange: A list or tuple (minpan, maxpan) giving the range of pan
            angles.

            trange: A list or tuple (mintilt, maxtilt) giving the range of tile
            angles.

            ares: Angular resolution of the image, in pixels per degree, a
            float. Default is 1.0.

            numchan: The number of channels (depth) of the panoramic image.
            Default 3.

            dtype: The type of the image pixels. Default is int.

            init: Initial value to assign to each channel of the image. Default
            is 0.
        """

        if not isinstance(prange, (list, tuple)) or len(prange) != 2:
            raise Exception('Argument PRANGE must be like (MINPAN, MAXPAN)')
        if not isinstance(trange, (list, tuple)) or len(trange) != 2:
            raise Exception('Argument TRANGE must be like (MINTILT, MAXTILT)')
        if not isinstance(ares, Number) or ares <= 0:
            raise Exception('Argument ARES must be a positive number')
        if dtype != int and dtype != float and dtype != np.uint8:
            raise Exception('Argument DTYPE must be int or float or numpy.uint8')
        if not isinstance(init, Number):
            raise Exception('Argument INIT must be a number')

        self.pmin = max(-180, prange[0])
        self.pmax = min(180, prange[1])
        self.tmin = max(-90, trange[0])
        self.tmax = min(90,trange[1])
        self.ares = ares                            # pixels per degree
        self.nrows = np.ceil((self.tmax - self.tmin)*self.ares+1).astype(int)
        self.ncols = np.ceil((self.pmax - self.pmin)*self.ares+1).astype(int)
        self.numchan = numchan
        self.image = np.squeeze(init*np.ones((self.nrows, self.ncols, numchan),
                                             dtype=dtype))
        self.patchcurfov = []   # patches in panoramic image showing current FOV

        # Get mapping from row/column numbers to tilt/pan angles, respectively.
        self.col2pan = np.linspace(self.pmax, self.pmin, num=self.ncols)
        self.row2tilt = np.linspace(self.tmax, self.tmin, num=self.nrows)


    def update(self, newimage, pan, tilt, foclen, op=None, panomaskimage=None,
               **kwargs):
        """
        Update a panoramic image.

        Usage:
            PanoImage().update(newimage, pan, tilt, foclen, op=None, kwargs)

        Arguments:
            newimage: The image (a Numpy array) to be inserted into the
            panoramic image.

            pan: The pan angle at the center of the image (degrees).

            tilt: The tilt angle at the center of the image (degrees).

            foclen: The camera focal length (in units of pixels).

            op: (str) Operation to use in updating the panoramic
            image. Default is None. Allowed values for "op" are the following:

                None -- All of the pixels in "newimage" are copied into the
                panoramic image.

                'max' -- The maximum of the old and new pixels is stored. This
                may be used only for single-channel images.

                'mask' -- Pixels in "newimage" are copied wherever corresponding
                pixels in "panomaskimage" have value True. "panomaskimage" must
                be a binary image the same size as the base panoramic image.

            kwargs: Keyword arguments passed to the warp() function. The main
                warp() argument of concern is:
                    order: The order of interpolation. 'order' is an int in the
                        range 0-5:
                            0: Nearest-neighbor
                            1: Bi-linear (default)
                            2: Bi-quadratic
                            3: Bi-cubic
                            4: Bi-quartic
                            5: Bi-quintic

        Description:
            This function currently assumes that image pixels are square.
        """

        if newimage is None:
            raise Exception('Argument IMAGE is missing')
        if pan is None or tilt is None or foclen is None:
            raise Exception('Argument PAN, TILT, or FOCLEN is missing')

        inshape = newimage.shape
        nrows = inshape[0]
        ncols = inshape[1]
        nchan = 1 if len(inshape) == 2 else inshape[2]
        if nchan != self.numchan:
            raise Exception('Number of channels in input image does not match pano image')

        # Project the new image onto the inside of a cylinder.
        map_args = {'prange':(self.pmin, self.pmax), 'trange':(self.tmin, self.tmax),
                    'nrows':nrows, 'ncols':ncols, 'foclen':foclen, 'pan':pan,
                    'tilt':tilt}
        pano_shape = (self.nrows, self.ncols)

        if newimage.dtype in ['int32', 'int16']:
            newimage = newimage.astype(np.uint8)

        imwarped = warp(newimage, inv_cylindrical_proj, output_shape=pano_shape,
                        map_args=map_args, **kwargs)  # output is float image
        if self.image.dtype != float:
            imwarped = (255*imwarped).astype(self.image.dtype)  # convert to correct data type

        # The mask image will be warped and will indicate which pixels in the
        # similarly warped input image to update.
        mask = np.ones(newimage.shape[:2], dtype=np.uint8)
        warpedmask = warp(mask, inv_cylindrical_proj, output_shape=pano_shape,
                          map_args=map_args, **kwargs)  # output is float image
        warpedmask = (255*warpedmask).astype(self.image.dtype)  # convert to correct data type

        # Get index of pixels to copy into base panoramic image. "idx[:,0]" are
        # the row indicies and "idx[:,1]" are the column indicies.
        if op != None and op.lower() == 'mask':
            idx = np.argwhere(np.logical_and(warpedmask > 0,
                                             panomaskimage == True))
        else:
            idx = np.argwhere(warpedmask > 0)

        # Update pixels in panoramic image.
        if nchan == 1:
            if op != None and op.lower() == 'max':
                    self.image[idx[:,0],idx[:,1]] = \
                        np.maximum(self.image[idx[:,0],idx[:,1]],
                                   imwarped[idx[:,0],idx[:,1]])
            elif op == None or op.lower() == 'mask':
                self.image[idx[:,0],idx[:,1]] = imwarped[idx[:,0],idx[:,1]]
            else:
                raise ValueError('Invalid "op" argument: {}'.format(op))
        else:
            self.image[idx[:,0],idx[:,1],:] = imwarped[idx[:,0],idx[:,1],:]


    def pt2xy(self, pan, tilt):
        """
        Map pan and tilt angles to (x,y)-pixel positions in the panoramic image.

        Usage:
            x, y = PanoImage.pt2xy(pan, tilt)

        Arguments:
            pan: (float or np.ndarray) One or more pan angles to convert.

            tilt: (float or np.ndarray) One or more tilt angles to convert.
        """
        x = np.argmin(abs(self.col2pan - pan))
        y = np.argmin(abs(self.row2tilt - tilt))
        return x, y


    def draw_fov(self, cam, f, axisnum=0):
        """
        Display the current camera FOV in a figure of the panoraminc image.
        """
        try:
            # Remove the previous FOV graphic.
            for p in self.patchcurfov:
                p.remove()
        except:
            pass
        self.patchcurfov = []

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
            pt1 = np.vstack(([self.pmax,top],pt[p1+1:p2+1,:],[self.pmax,bot]))
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt1, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt1, linewidth=1,
                                           edgecolor='b', linestyle=':'))
            pt2 = np.vstack((pt[:p1+1,:], [self.pmin,top],
                             [self.pmin,bot], pt[p2+1:,:]))
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt2, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt2, linewidth=1,
                                           edgecolor='b', linestyle=':'))
        else:
            # Draw a dashed white rectangle on top of a solid blue rectangle.
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt, linewidth=1.5,
                                           edgecolor='w', linestyle='-'))
            self.patchcurfov.append(f.draw(axisnum=axisnum, poly=pt, linewidth=1,
                                           edgecolor='b', linestyle=':'))


def inv_cylindrical_proj(ptcr, **kwargs):
    """
    This function computes the inverse cylindrical projection coordinate map,
    which transforms (col,row) coordinates in the cylindrical (output) image to
    their corresponding (col,row) coordinates in the planar (input) image, where
    the planar image was acquired by a pan-tilt-zoom camera as described by the
    keyword arguments (see below).

    Usage:
        imcr = inv_cylindrical_proj(ptcr, **kwargs)

    Arguments:
        ptcr: An Nx2 array of (column, row) image coordinates in the pan/tilt
        (aka., cylindrical) output image. These always start at (0,0).

        kwargs: The keyword arguments, which must include the following:
            nrows: Number of rows in planar (input) image.
            ncols: Number of columns in planar (input) image.
            foclen: Camera focal length (pixels) -- determines HFOV & VFOV.
            pan: Pan angle (degrees) of the camera.
            tilt: Tilt angle (degrees) of the camera.
            prange: Range of pan angles represented in the pan/tilt image, a
                tuple (minpan, maxpan).
            trange: Range of tilt angles represented in the pan/tilt image, a
                tuple (mintilt, maxtilt).

    Returns:
        imcr: An Nx2 array of (column, row) pixel coordinates in the planar
            input image corresponding to each of the coordinates in 'ptcr'.
    """

    # Get parameters of the transformation.
    pmin = kwargs['prange'][0]   # minimum pan angle in the pan/tilt image
    pmax = kwargs['prange'][1]   # maximum pan angle in the pan/tilt image
    tmin = kwargs['trange'][0]   # minimum tilt angle in the pan/tilt image
    tmax = kwargs['trange'][1]   # maximum tilt angle in the pan/tilt image
    nrows = kwargs['nrows']      # number of rows in input image
    ncols = kwargs['ncols']      # number of columns in input image
    # foclen = kwargs['foclen']    # camera focal length (pixels)
    pan = kwargs['pan']          # pan angle (degrees) of the camera
    tilt = kwargs['tilt']        # tilt angle (degrees) of the camera

    if 'foclen' in kwargs.keys():
        # Same horizontal and vertical focal lengths (units in pixels).
        hfoclen = kwargs['foclen']
        vfoclen = hfoclen
    else:
        # Seperate horizontal and vertical focal lengths (units in pixels).
        hfoclen = kwargs['hfoclen']
        vfoclen = kwargs['vfoclen']

    # Range of row and column coordinates in cylindrical (pan/tilt) output image.
    colmax, rowmax = np.max(ptcr, axis=0)

    # Map pan/tilt image coordinates to pan/tilt angles.
    p = pmax + (pmin-pmax)*ptcr[:,0]/colmax     # pan angles (degrees)
    t = tmax + (tmin-tmax)*ptcr[:,1]/rowmax     # tilt angles (degrees)

    # Map pan/tilt angles to 3D world points on a cylinder of radius 1.
    x = -np.sin(np.deg2rad(p))
    y = np.cos(np.deg2rad(p))
    z = np.tan(np.deg2rad(t))
    w = np.vstack((x,y,z))            # 3x1 vector needed for matrix multiplications

    # Rotate the camera. Camera rotation is accomplished by rotating the world
    # points in the opposite direction. The camera is first tilted and then
    # panned. The opposite rotation of the world points must therefore first
    # inverse pan, and then inverse tilt.
    w = Rot3d(rx=0, ry=0, rz=np.deg2rad(-pan))*w        # inv. pan about z axis
    w = Rot3d(rx=np.deg2rad(-tilt), ry=0, rz=0)*w      # inv. tilt about x axis

    # Get coordinates of world points in the rotated camera coordinate system.
    # The camera coordinate system is a left-handed system with its optical
    # axis aligned with its Z axis, its Y axis pointing up, and its X axis to
    # the right. The camera's image plane is in the X-Y plane. The camera's
    # optical (Z) axis is aligned with the world Y axis. Thus, the depth of a
    # point (projection on optical axis) is its Y coordinate.
    x = np.squeeze(np.asarray(w[0,:]))   # camera X coord. is world X coord.
    y = np.squeeze(np.asarray(w[2,:]))   # camera Y coord. is world Z coord.
    z = np.squeeze(np.asarray(w[1,:]))   # camera Z coord. is world Y coord.

    # Project points into the image. Only points that are in front of the camera
    # (z > 0) are projected. The image's Y axis is reveresed so that Y increases
    # from top to bottom, and the origin is moved to the image's upper left corner.
    imx = np.full_like(x, np.NaN)
    imy = np.full_like(y, np.NaN)
    infront = np.argwhere(z > 0)
    imx[infront] = hfoclen*x[infront]/z[infront] + (ncols-1)/2
    imy[infront] = -vfoclen*y[infront]/z[infront] + (nrows-1)/2

    imcr = np.column_stack((imx, imy))              # form an Nx2 array
    return imcr