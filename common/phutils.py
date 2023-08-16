"""
Miscellaneous utility classes and functions.

AUTHOR:
    Phil David, Army Research Laboratory

HISTORY:
    2017-11-27: P. David, Created.
    2018-01-06: Added class Blob.
    2021-02-11: Add imwrite().

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import imageio
from matplotlib.path import Path
from time import sleep
from scipy import ndimage as nd
from matplotlib.colors import hsv_to_rgb
from numpy.linalg import norm


class Blob():
    """
    Class for working with 2D rectangles.
    """

    blob_count = 0

    def __init__(self, props=None, xs=None, ys=None, xys=None, yxs=None,
                 id=None, label='', pixels=False):
        """
        Create a new Bolb.

        Usage:
            b = Blob(props=None, xs=None, ys=None, xys=None, yxs=None,
                     id=None, pixels=False)

        Arguments:
            props: A dictionary giving the blob's properties. The blob's
            position must be included in these properties if it is not specified
            through the named arguments. Recognized properties include the
            following:
                'label': A string.
                'id': An integer.
                'confidence': A float.
                'topleft': A dictionary with keys ['x', 'y'].
                'bottomright': A dictionary with keys ['x', 'y'].
                'dist': A float.
            Any other key/value pairs may be included in the 'props' dictionary.
            These key/value pairs are inserted into the blob exactly as given.
            If a props key is identical to a named argument, the value of the
            named argument takes presedence over the corresponding value of the
            props key.

            xs: A slice defining the X extent of the blob.

            ys: A slice defining the Y extent of the blob.

            xys: A list or tuple of two slices (xslice, yslice).

            yxs: A list or tuple of two slices (yslice, xslice).

            id: An integer ID to assign to the blob. If None, then assign the
            sequential blob count. If a negative integer, then reset the Blob
            count to zero and return without creating a new Blob. Default is
            None.

            label: The label of the blob, a string. Default is the empty string.

            pixels: Does the blob originate from pixels in an image? If True,
            then the blob's positional properties are assumed to give the
            location of the centers of pixels, so that the boundaries of the
            blob extend half a pixel in all directions past the given locations.
            Default is False.

        Returns:
            b: A new blob object. The blob will have at least these properties:
                xmin, ymin, width, height, xctr, yctr, id

        Description:
            Slices use the standard python slice function with increment 1:
            slice(start, stop). The stop index is *not* included in the extent
            of the blob.
        """

        if type(id) is int and id < 0:
            # Reset the blob_count.
            Blob.blob_count = 0
            return

        Blob.blob_count += 1
        self.xmin = self.ymin = None
        self.xmax = self.ymax = None
        self.width = self.height = self.id = None
        self.confidence = 0
        self.dist = 0
        self.res = 0
        self.gt = False
        self.label = ''

        if props is not None:
            if type(props) is not dict:
                raise TypeError('Argument PROPS must be a dictionary')
            for key in props:
                if key == 'label':
                    self.label = props['label']
                elif key == 'confidence':
                    self.confidence = props['confidence']
                elif key == 'topleft':
                    try:
                        self.xmin = props['topleft']['x']
                        self.ymin = props['topleft']['y']
                    except:
                        raise ValueError('Property "topleft" must be dictionary with keys "x" and "y"')
                elif key == 'bottomright':
                    try:
                        self.xmax = props['bottomright']['x']
                        self.ymax = props['bottomright']['y']
                    except:
                        raise ValueError('Property "bottomright" must be dictionary with keys "x" and "y"')
                elif key == 'width':
                    self.width = props['width']
                elif key == 'height':
                    self.height = props['height']
                elif key == 'id':
                    self.id = props['id']
                elif key == 'dist':
                    self.dist = props['dist']
                else:
                    setattr(self, key, props[key])

        if xys is not None:
            try:
                xs = xys[0]
                ys = xys[1]
            except:
                raise TypeError('Argument XYS must be list or tuple of slices.')

        if yxs is not None:
            try:
                xs = yxs[1]
                ys = yxs[0]
            except:
                raise TypeError('Argument YXS must be list or tuple of slices.')

        if xs is not None:
            try:
                self.xmin = xs.start
                self.xmax = xs.stop - 1
            except:
                raise TypeError('XS must be a slice.')

        if ys is not None:
            try:
                self.ymin = ys.start
                self.ymax = ys.stop - 1
            except:
                raise TypeError('YS must be a slice.')

        if pixels:
            # Make the edge of the blob to be halfway between pixels.
            self.xmin -= 0.5
            self.ymin -= 0.5

        if self.xmax == None or self.ymax == None:
            self.xmax = self.xmin + self.width - 1
            self.ymax = self.ymin + self.height - 1

        if self.width == None or self.height == None:
            self.width = self.xmax - self.xmin + 1
            self.height = self.ymax - self.ymin + 1

        self.xctr = self.xmin + self.width/2
        self.yctr = self.ymin + self.height/2

        if self.xmin is None or self.ymin is None or self.width is None or self.height is None:
            raise ValueError('Position of blob is not fully defined')

        if self.id is None:
            if id is None:
                self.id = Blob.blob_count
            else:
                self.id = id

        if label != '':
            self.label = label


    def inside(self, x, y):
        """
        Check if the point (x,y) is inside the bounding box of the blob.

        Usage:
            tf = Blob().inside(x, y)

        Arguments:
            x, y: The coordinates of the point to check.

        Returns:
            tf: True or False
        """
        if x >= self.xmin and x <= self.xmin + self.width and \
           y >= self.ymin and y <= self.ymin + self.height:
            return True
        else:
            return False


    def overlapswith(self, b2):
        """
        Check if the blob overlaps with another blob. Blobs overlap if the
        centroid of one blob is inside the bounding box of another.
        """
        if self.xctr >= b2.xmin and self.xctr <= b2.xmin + b2.width and \
               self.yctr >= b2.ymin and self.yctr <= b2.ymin + b2.height:
            return True
        elif b2.xctr >= self.xmin and b2.xctr <= self.xmin + self.width and \
                   b2.yctr >= self.ymin and b2.yctr <= self.ymin + self.height:
            return True
        else:
            return False


    def __str__(self):
        return '{} {}, conf={:.2f}, x,y={:.1f},{:.1f}, w,h={:.1f},{:.1f}, dist={:.1f}, res={:.1f}, gt={}'. \
               format(self.label, self.id, self.confidence, self.xmin,
                           self.ymin, self.width, self.height, self.dist,
                           self.res, self.gt)


    def __repr__(self):
        return 'Blob: ' + str(self)


def distPntLineSeg(pt, ep1, ep2, tol=1e-12):
    """
    Get the distance bewteen a point and a line segment.

    Usage:
        dist = distPntLineSeg(pt, ep1, ep2, tol=1e-12)

    Arguments:
        pt: (array-like) The point whose distance to the line segement is to be
        found. This may be a 2D or 3D point.

        ep1, ep2: (array-like) The two endpoints of the line segment. These may
        be 2D or 3D points, but must be the same dimenion as "pt".

        tol: (float) The tolerance on inter-point distance used to determine if
        two points are identical. This must be a positive number. Defailt is
        1e-12.

    Returns:
        dist: (float) The distance of the point to the line segment.

    Description:
        The distance between the point "pt" and the line segment is computed as
        the minimum of three distances: (1) The distance of "pt" to the 1st
        endpoint, "ep1", of the line segment, (2) the distance of "pt" the the
        2nd endpoint, "ep2" of the line segment, (3) The distance of "pt" to the
        closest point on the infinite line passing through the two line segment
        endpoints provided this point lies on the finite line segment.
    """
    ep1 = np.array(ep1)
    ep2 = np.array(ep2)
    pt = np.array(pt)
    ndim = ep1.shape[0]

    assert ep1.shape == ep2.shape == pt.shape
    assert ndim == 2 or ndim == 3
    assert tol > 0

    n12 = norm(ep1 - ep2)
    if n12 < tol:
        raise ValueError('The line segment endpoints are identical')

    # Get distances to the two line segment endpoints.
    dp1 = norm(ep1 - pt)
    dp2 = norm(ep2 - pt)

    if dp1 < tol or dp2 < tol:
        # The point is one of the line segment endpoints.
        return 0

    # Determine if the closest point on the infinite line is inside the line
    # segment. It is if the angles linept1--linept2--pt and linept2--linept1--pt
    # are both less than 90 degrees. The cosine of these angles must be
    # nonnegative.
    v1 = ep1 - ep2
    v2 = pt - ep2
    n = norm(v1)*norm(v2)               # This must be > 0 due to above checks.
    cos1 = (v1*v2).sum()/n              # dot(v1,v2)/n
    v1 = ep2 - ep1
    v2 = pt - ep1
    n = norm(v1)*norm(v2)               # This must be > 0 due to above checks.
    cos2 = (v1*v2).sum()/n

    if cos1 < 0 or cos2 < 0:
        # The closest point on the infinite line is not on the line segment. So,
        # one of the line segment endpoints is closest to the point.
        dist = min(dp1, dp2)
        return dist

    # The closest point on the infinite line is on the line segment.  Find this
    # distance.

    # Determine the unit-length normal to the line.
    d = ep2 - ep1
    if ndim == 2:
        n = np.array([-d[1], d[0]])
    else:
        n = np.array([d[1]+d[2], -d[0]-d[2], -d[0]+d[1]])
    l = norm(n)
    if l == 0:
        raise ValueError('The line endpoints are identical!')
    n = n/l

    # Get the distance to the closest point on the infinite line.
    d = abs(n*(pt-ep1)).sum()           # abs(dot(n,pt-ep1))

    # Find the minimum of these three distances.
    dist = min(dp1, dp2, d)

    return dist


def rect_iou(rect1, rect2):
    """
    Compute the intersection over union (IoU) score of two rectangles.

    Usage:
        iou = rect_iou(rect1, rect2)

    Arguments:
        rect1, rect2: The two rectangles to compute the IoU score for. Each is
            either 4-element array-like:
                [xmin, ymin, width, height]
            or a Blob object with attributes xmin, ymin, width, height.

    Returns:
        iou: The IoU score, a float in [0,1].
    """

    if type(rect1) is Blob:
        x1 = rect1.xmin
        y1 = rect1.ymin
        w1 = rect1.width
        h1 = rect1.height
    else:
        x1 = rect1[0]
        y1 = rect1[1]
        w1 = rect1[2]
        h1 = rect1[3]
    if type(rect2) is Blob:
        x2 = rect2.xmin
        y2 = rect2.ymin
        w2 = rect2.width
        h2 = rect2.height
    else:
        x2 = rect2[0]
        y2 = rect2[1]
        w2 = rect2[2]
        h2 = rect2[3]

    # Determine the coordinates of the intersection rectangle.
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1+w1, x2+w2)
    ymax = min(y1+h1, y2+h2)

    if xmin >= xmax or ymin >= ymax:
        # The rectangles do not intersect.
        return 0.0

    # Get areas of rectangles.
    areainter = (xmax-xmin)*(ymax-ymin)    # intersecting rectangle
    arearect1 = w1*h1
    arearect2 = w2*h2

    # IoU = area of intersection / area of union.
    iou = areainter / float(arearect1 + arearect2 - areainter)

    return iou


def iou_mat(blobs1, blobs2):
    """
    Compute a matrix of IoU scores for all pairs of blobs from two sets.

    Usage:
        iou = iou_mat(blobs1, blobs2)

    Arguments:
        blobs1, blobs2: The lists of Blob objects.

    Returns:
        iou: A 2D Numpy.ndarray, where iou[m,n] = IoU(blobs1[m], blobs2[n]).
    """

    numb1 = len(blobs1)
    numb2 = len(blobs2)

    b1xmin = np.empty((numb1, numb2))
    b1ymin = np.empty((numb1, numb2))
    b1w = np.empty((numb1, numb2))
    b1h = np.empty((numb1, numb2))
    b2xmin = np.empty((numb1, numb2))
    b2ymin = np.empty((numb1, numb2))
    b2w = np.empty((numb1, numb2))
    b2h = np.empty((numb1, numb2))

    # Extract blob info into initial matrices.
    for tnum, t in enumerate(blobs1):
        b1xmin[tnum,:] = t.xmin
        b1ymin[tnum,:] = t.ymin
        b1w[tnum,:] = t.width
        b1h[tnum,:] = t.height
    for tnum, t in enumerate(blobs2):
        b2xmin[:,tnum] = t.xmin
        b2ymin[:,tnum] = t.ymin
        b2w[:,tnum] = t.width
        b2h[:,tnum] = t.height

    # Maximum X & Y extent, and area, of blobs.
    b1xmax = b1xmin + b1w
    b1ymax = b1ymin + b1h
    b1area = b1w*b1h
    b2xmax = b2xmin + b2w
    b2ymax = b2ymin + b2h
    b2area = b2w*b2h

    # Properties of the intersecting rectangles.
    intxmin = np.maximum(b1xmin, b2xmin)
    intymin = np.maximum(b1ymin, b2ymin)
    intxmax = np.minimum(b1xmax, b2xmax)
    intymax = np.minimum(b1ymax, b2ymax)
    nonempty = np.logical_and(np.less(intxmin,intxmax),np.less(intymin,intymax)) # nonempty rectangle intersections
    intarea = (intxmax-intxmin)*(intymax-intymin)      # areas of intersections
    unionarea = b1area + b2area - intarea              # area of unions

    iou = np.zeros((numb1, numb2))
    iou[nonempty] = intarea[nonempty]/unionarea[nonempty]
    return iou


def ios_mat(blobs1, blobs2):
    """
    Compute a matrix of IoS (Intersection over Smaller) scores for all pairs of
    blobs from two sets.

    Usage:
        ios = ios_mat(blobs1, blobs2)

    Arguments:
        blobs1, blobs2: The lists of Blob objects.

    Returns:
        ios: A 2D Numpy.ndarray, where ios[m,n] = IoS(blobs1[m], blobs2[n]).

    Description:
        The IoS metric for two rectangles is the area of the intersection of the
        rectangles divided by the smaller of the two areas.
    """

    numb1 = len(blobs1)
    numb2 = len(blobs2)

    b1xmin = np.empty((numb1, numb2))
    b1ymin = np.empty((numb1, numb2))
    b1w = np.empty((numb1, numb2))
    b1h = np.empty((numb1, numb2))
    b2xmin = np.empty((numb1, numb2))
    b2ymin = np.empty((numb1, numb2))
    b2w = np.empty((numb1, numb2))
    b2h = np.empty((numb1, numb2))

    # Extract blob info into initial matrices.
    for tnum, t in enumerate(blobs1):
        b1xmin[tnum,:] = t.xmin
        b1ymin[tnum,:] = t.ymin
        b1w[tnum,:] = t.width
        b1h[tnum,:] = t.height
    for tnum, t in enumerate(blobs2):
        b2xmin[:,tnum] = t.xmin
        b2ymin[:,tnum] = t.ymin
        b2w[:,tnum] = t.width
        b2h[:,tnum] = t.height

    # Maximum X & Y extent, and area, of blobs.
    b1xmax = b1xmin + b1w
    b1ymax = b1ymin + b1h
    b1area = b1w*b1h
    b2xmax = b2xmin + b2w
    b2ymax = b2ymin + b2h
    b2area = b2w*b2h

    # Properties of the intersecting rectangles.
    intxmin = np.maximum(b1xmin, b2xmin)
    intymin = np.maximum(b1ymin, b2ymin)
    intxmax = np.minimum(b1xmax, b2xmax)
    intymax = np.minimum(b1ymax, b2ymax)
    nonempty = np.logical_and(np.less(intxmin,intxmax),np.less(intymin,intymax)) # nonempty rectangle intersections
    intarea = (intxmax-intxmin)*(intymax-intymin)      # areas of intersections
    minarea = np.minimum(b1area, b2area)               # area of smaller rectangle

    ios = np.zeros((numb1, numb2))
    ios[nonempty] = intarea[nonempty]/minarea[nonempty]
    return ios


def Rot3d(rx=None, ry=None, rz=None, angles=None):
    """
    Create a 3D rotation matrix.

    Usage:
        R = Rot3d(rx=theta1, ry=theta2, rz=theta3)
        R = Rot3d(angles=vec)

    Arguments:
        rx: The rotation angle about the X axis, in radians.
        ry: The rotation angle about the Y axis, in radians.
        rz: The rotation angle about the Z axis, in radians.
        angles: A 3-element tuple or Numpy array: (RX, RY, RZ). The 'angles'
            argument may be used to pass RX, RY, and RZ in through one argument.

    Returns:
        R: A 3x3 rotation matrix. A 3D point P (a column 3-vector) is rotated
        about the origin to a new point P' by left multiplication of P with R:
        P' = R*P. The rotation R is first about the X axis, then the new Y axis,
        and finally about the new Z axis.
    """

    if angles is None:
        assert rx is not None, 'only one of angles or (rx,ry,rz) allowed'
        assert ry is not None, 'only one of angles or (rx,ry,rz) allowed'
        assert rz is not None, 'only one of angles or (rx,ry,rz) allowed'
    else:
        assert rx is None, 'only one of angles or (rx,ry,rz) allowed'
        assert ry is None, 'only one of angles or (rx,ry,rz) allowed'
        assert rz is None, 'only one of angles or (rx,ry,rz) allowed'
        rx = angles[0]
        ry = angles[1]
        rz = angles[2]

    cx = np.cos(rx)
    sx = np.sin(rx)
    cy = np.cos(ry)
    sy = np.sin(ry)
    cz = np.cos(rz)
    sz = np.sin(rz)

    Rotx = np.matrix([[1, 0, 0],
                      [0, cx, -sx],
                      [0, sx, cx]])
    Roty = np.matrix([[cy, 0, sy],
                      [0, 1, 0],
                      [-sy, 0, cy]])
    Rotz = np.matrix([[cz, -sz, 0],
                      [sz, cz, 0],
                      [0, 0, 1]])
    rmat = Rotz*Roty*Rotx

    return rmat


def findxy(axy, xy, tol=(0,0)):
    """
    Find the locations of (x,y) in an Nx2 array.

    Usage:
        rows = findxy(axy, xy, tol=(0,0))

    Arguments:
        axy: The Nx2 array to search.
        xy: The value to find, a tuple (x, y).
        tol: The x and y tolerance, a tuple (xtol, ytol). Default is (0,0).

    Returns:
        rows: A 1D array of the row locations, or [] if (x,y) is not found.
    """
    xloc = np.squeeze(np.argwhere(np.abs(axy[:,0] - xy[0]) <= tol[0]))
    yloc = np.squeeze(np.argwhere(np.abs(axy[:,1] - xy[1]) <= tol[1]))
    rows = np.intersect1d(xloc, yloc)
    return rows


def dangle(a1, a2):
    """
    Get the absolute difference between two angles (both in degrees).
    """
    da = abs(a1 - a2);
    if da >= 180:
        da = 360 - da
    return da


def vang(v1, v2):
    """
    Returns the signed angle, in radians, *from* vector v1 *to* vector v2.
    Counter-clockwise rotations are considered positive angles.
    """
    n1 = norm(v1)
    if n1 < 1e-10:
        raise ValueError('Argument V1 is a zero vector')
    n2 = norm(v2)
    if n2 < 1e-10:
        raise ValueError('Argument V2 is a zero vector')
    v1 = v1/n1
    v2 = v2/n2
    ang = np.arccos(np.dot(v1, v2))
    sgn = np.sign(np.reshape(np.cross(v1, v2),-1)[-1])  # sign of cross product gives +/- direction
    return ang if sgn == 0 else sgn*ang


def polyfill(im, vert, val=1):
    """
    Fill a polygon in a 2D array. The polygon is defined by a list of vertices.
    Vertex coordinates are given with respect to the image coordinate systems where
    the top left has coordinate [0,0].

    Usage:
        im_out = polyfill(im, vert, val)

    Arguments:
        'im' is the 2D array to be filled.
        'vert' is a Nx2 array of vertices. Each row gives one vertex as [X, Y].
        'val' is the value to fill the polygon with. Default = 1.

    Returns:
        'im_out' is the array with the filled polygon.

    Author:
        Phil David, ARL, 2017-12-01.
    """

    nrows, ncols = im.shape

    # Create coordinates for each cell of the image.  [0,0] is the top left.
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    cols, rows = cols.flatten(), rows.flatten()

    # Create the polygon path.
    points = np.vstack((cols, rows)).T
    path = Path(vert)

    # Get an array which is True where the path contains the corresponding points.
    mask = path.contains_points(points, radius=0.5)
    mask = mask.reshape((nrows, ncols))

    im[mask] = val
    return im


def progress_bar(percent, barLen = 20):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


def np2str(a, fmt=''):
    """
    Convert a Numpy array to a string using a specified format.

    Usage:
        str = np2str(a, fmt='')

    Arguments:
        a: The Numpy array
        fmt: The format string to pass to format(). This does not include the
            curly braces or the colon.

    Returns:
        str: The string representation of array 'a'.

    Example:
        x = 100*np.random.rand(4,5)
        print('X =\n', np2str(x, fmt='>7.2f'))
    """
    fmt = '{:' + fmt + '}'
    return np.array2string(a, formatter={'all':fmt.format}, separator=', ',
                           prefix=' ')


def status_bar():
    for i in range(21):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
        sys.stdout.flush()
        sleep(0.25)


def dtfill(data, invalid=None):
    """
    Replace the values of invalid 'data' cells (indicated by 'invalid') by the
    value of the nearest valid data cell

    Arguments:
        data: Numpy array of any dimension.
        invalid: A binary array of same shape as 'data'. 'data' values are
            replaced where 'invalid' is True. If None (default), then invalid is
            set to np.isnan('data')

    Returns:
        The filled array.

    Source:
        https://stackoverflow.com
    """
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


def sigmoid_weights(numpts, start=0.1, end=0.9, xval=7.0):
    """
    Return a Sigmoid weight vector.

    Usage:
        w = sigmoid_weights(numpts, start=0.1, end=0.9, xval=7.0)

    Arguments:
        numpts: The number of weights (i.e., size of the returned vector).
        start: Float in [0,1]. The first 100*'start' percent of the weights
            will be approximately zero.
        end: Float in [0,1]. The last 100*'end' percent of weights will be
            approximately 1.0.
        xval: The returned weight w[round(start*numpts)] is mapped to
            Sigmoid(-'xval'), and the returned weight w[round(end*numpts)] is
            mapped to Sigmoid('xval'). Default is 7.0. Weight values between
            these two points are uniformly sampled from the Sigmoid function.

    Returns:
        w: The weight vector, a length 'Numpts' Numpy array.

    Description:
        This function essentially horizontally shifts and compresses a specified
        region of the Sigmoid function.
    """

    pstart = np.round(start*numpts).astype(int)
    pend = np.round(end*numpts).astype(int)
    x = np.zeros(numpts)
    x[:pstart] = np.linspace(-500, -xval, num=pstart, endpoint=True,
                             dtype=np.float)
    x[pstart-1:pend] = np.linspace(-xval, xval, num=pend-pstart+1,
                                   endpoint=True, dtype=np.float)
    x[pend-1:] = np.linspace(xval, 500, num=numpts-pend+1, endpoint=True,
                             dtype=np.float)
    w = 1.0/(1.0 + np.exp(-x))       # w = sigmoid(x)
    return w


def goodpath(path):
    """
    Make the path correct for either Windows of Linux. Tilde (~) is replaced
    by the path to the user's home directory.

    Usage:
        path = goodpath(path)

    Arguments:
        path: A path string (may include file names), list of paths, or dictionary
        containing paths. All strings (except dictionary keys) are checked.

    Returns:
        path: The same type of object as input, but with path strings correct for the
        current operating system.
    """
    p = path
    if type(p) is str:
        p = os.path.normpath(os.path.expanduser(p))
    elif type(p) is dict:
        for key in p.keys():
            if type(p[key]) is str:
                p[key] = os.path.normpath(os.path.expanduser(p[key]))
    elif type(p) is list:
        for k, elm in enumerate(p, 0):
            if type(elm) is str:
                p[k] = os.path.normpath(os.path.expanduser(p[k]))
    else:
        raise TypeError('Unable to process argument of type: ', type(p))
    return p


def mimshow(imlist, link=None, title=None):
    """
    Spawn a new procees that shows one or more images in a single figure.

    Usage:
        mimshow(imlist, link, title)

    Arguments:
        imlist: A single image (Numpy array) or list of images.
        link: A list of image numbers (starting at 0) whose axes should be
            linked.
        title: Title string for the figure.
    """

    import multiprocessing

    def show_images(imlist, link, title):

        if type(imlist) not in [list, tuple]:
            imlist = [imlist]

        numimg = len(imlist)
        nrows = int(np.ceil(np.sqrt(numimg)))
        ncols = int(np.ceil(numimg/nrows))

        # Create a figure with an axis for each image.
        f, axarray = plt.subplots(nrows, ncols)
        if type(axarray) is not list and type(axarray) is not np.ndarray:
            axarray = [axarray]
        else:
            axarray = axarray.ravel()

        for axnum, im in enumerate(imlist):
            axarray[axnum].imshow(im, aspect='equal')

        if link is not None:
            # Link a set of axes.
            for axnum in link:
                axarray[axnum].set_adjustable('box-forced')
            ax0 = axarray[link[0]]
            for axnum in link[1:]:
                ax0.get_shared_x_axes().join(ax0, axarray[axnum])
                ax0.get_shared_y_axes().join(ax0, axarray[axnum])

        if title is not None:
            f.suptitle(title, fontsize=16)

        plt.show()

    if os.name == 'nt':
        # The multiprocessing code doesn't yet work in Windows.
        show_images(imlist, link, title)
    else:
        # Create a new process.
        p1 = multiprocessing.Process(target=show_images, args=(imlist,link,title))
        p1.start()
        # p1.join()     # wait for spawned process to finish


def cmjet1(numcolors=101):
    """
    Create a "hot" colormap that goes from blue at index 0 to red at index
    'numcolors'-1.

    Usage:
        rgb = cmjet1(numcolors)

    Returns:
        rgb: A Numpy 'numcolors'x3 Numpy array giving the RGB colormap. Each
            color value is in [0,1]x3.
    """
    hsv = np.zeros((numcolors,3))
    hsv[:,0] = np.linspace(240/360, 0, numcolors)
    hsv[:,1:3] = 1
    rgb = hsv_to_rgb(hsv)
    return rgb


def cmjet2(numcolors=101):
    """
    Create a colormap that goes from blue at index 0 to magenta at index
    'numcolors'-1.

    Usage:
        rgb = cmjet2(numcolors)

    Returns:
        rgb: A Numpy 'numcolors'x3 Numpy array giving the RGB colormap. Each
            color value is in [0,1]x3.
    """
    hsv = np.zeros((numcolors,3))
    hsv[:,0] = np.linspace(300/360, 0, numcolors)
    hsv[:,1:3] = 1
    rgb = hsv_to_rgb(hsv)
    return rgb



def cmdistinct(numcolors=101, amp=1.0):
    """
    Create a colormap with `numcolors` maximally distinct RGB colors indexed
    from 0 to 'numcolors'-1. `amp` is the amplitude of the color values, which
    is typically either 1.0 or 255.0. For a given `numcolors` and `amp`, the
    same colormap (colors and order of colors) is always returned.

    Usage:
        rgb = cmdistinct(numcolors=101, amp=1.0)

    Returns:
        rgb: (numpy.ndarray) A NUMCOLORSx3 array giving the RGB colormap. Each
        color value is in [0, AMP]x3.

    Credits:
        This code is based on Matlab code by Tim Holy:

            Tim Holy (2020). Generate maximally perceptually-distinct colors.
            https://www.mathworks.com/matlabcentral/fileexchange/29702-generate-maximally-perceptually-distinct-colors.
            MATLAB Central File Exchange. Retrieved September 10, 2020.

    """

    ndiv = 30                          # number of samples in each RGB dimension
    nsamp = ndiv**3                    # total number of RGB samples

    if numcolors > nsamp/3:
        raise ValueError("Requested too many colors: can't distinguish {:d} colors.".format
                         (numcolors))

    # Generate a large sample of RGB color space.  Shape is [NSAMP, 3].
    x = np.linspace(0, 1, num=ndiv)
    r, g, b = np.meshgrid(x, x, x)
    rgb = np.concatenate([r.reshape((nsamp,1)), g.reshape((nsamp,1)),
                          b.reshape((nsamp,1))], axis=1)

    # List of background colors. Try to be distinct from these. Shape is [N, 3]
    bg = np.array([[0,0,0], [0.5,0.5,0.5], [1,1,1]])                     # greys

    # Convert RGB colors to the Lab color space.
    from skimage import color
    clab = color.rgb2lab(rgb.reshape((1,rgb.shape[0],3)))     # shape is [1, NSAMP, 3]
    bglab = color.rgb2lab(bg.reshape((1,bg.shape[0],3)))      # shape is [1, N, 3]

    # Get squared distances of candidate colors to background colors.
    mindist2 = np.inf*np.ones((nsamp,1))
    for k in range(bglab.shape[1]):
        dx = clab[0,:,:] - bglab[0,k,:]
        dist2 = (dx*dx).sum(axis=1).reshape(nsamp,1)
        mindist2 = np.minimum(dist2, mindist2)

    # Pick colors that maximize distnaces to nearest previously picked colors.
    colors = np.zeros((numcolors, 3))                     # RGB colors to return
    lastlab = bglab[0,-1,:]                              # last Lab color chosen
    for k in range(numcolors):
        dx = clab[0, :, :] - lastlab
        dist2 = (dx*dx).sum(axis=1).reshape(nsamp,1)
        mindist2 = np.minimum(dist2, mindist2)   # min squared dists to chosen colors
        idx = np.argmax(mindist2)                # index of furthest unchosen color
        colors[k, :] = rgb[idx, :]
        lastlab = clab[0, idx, :]

    return amp*colors


def cmrand(numcolors=101, amp=1.0):
    """
    Create a colormap with `numcolors` random RGB colors indexed from 0 to
    'numcolors'-1. `amp` is the amplitude of the color values, which is
    typically either 1.0 or 255.0. For a given `numcolors` and `amp`, the same
    colormap (colors and order of colors) is always returned.

    Usage:
        rgb = cmrand(numcolors, amp)

    Returns:
        rgb: A Numpy 'numcolors'x3 Numpy array giving the RGB colormap. Each
        color value is in [0,`amp`]x3.
    """
    rstate = np.random.get_state()   # must restore random state before returning to calling routine
    cm = plt.get_cmap(name='nipy_spectral', lut=numcolors)  # 'gist_rainbow' & 'nipy_spectral' are good
    colors = [cm(k) for k in range(numcolors)]
    np.random.seed(20)      # always want the same colors and ordering of colors (20)
    colors = [colors[k] for k in np.random.permutation(numcolors)]
    rgb = amp*np.array(colors)[:,0:3]
    np.random.set_state(rstate)
    return rgb


# From http://ascii-table.com/ansi-escape-sequences.php
#
# These sequences define functions that change display graphics, control cursor
# movement, and reassign keys.
#
# ANSI escape sequence is a sequence of ASCII characters, the first two of which
# are the ASCII "Escape" character 27 (1Bh) and the left-bracket character "["
# (5Bh). The character or characters following the escape and left-bracket
# characters specify an alphanumeric code that controls a keyboard or display
# function. ANSI escape sequences distinguish between uppercase and lowercase
# letters.
#
# <Esc>[<Value>;...;<Value>m --- Set Graphics Mode. Calls the graphics functions
# specified by the following <value>s. These specified functions remain active
# until the next occurrence of this escape sequence. Graphics mode changes the
# colors and attributes of text (such as bold and underline) displayed on the
# screen.
#
# TEXT ATTRIBUTES
#   0	All attributes off
#   1	Bold on
#   4	Underscore (on monochrome display adapter only)
#   5	Blink on
#   7	Reverse video on
#   8	Concealed on
#
# FOREGROUND COLORS
#   30	Black
#   31	Red
#   32	Green
#   33	Yellow
#   34	Blue
#   35	Magenta
#   36	Cyan
#   37	White
#
# BACKGROUND COLORS
#   40	Black
#   41	Red
#   42	Green
#   43	Yellow
#   44	Blue
#   45	Magenta
#   46	Cyan
#   47	White

# Get the name of the python executable running this code.  Some executables do
# not support changing the text attributes.
import sys
exename = sys.executable.split('/')[-1].lower()

# Codes for foreground colors, background colors, and text attributes.
fc = {'k':'30', 'r':'31', 'g':'32', 'y':'33', 'b':'34', 'm':'35', 'c':'36', 'w':'37'}
bc = {'k':'40', 'r':'41', 'g':'42', 'y':'43', 'b':'44', 'm':'45', 'c':'46', 'w':'47'}
tc  = {'o':'0', 'b':'1', 'u':'4', 'l':'5', 'r':'7', 'c':'8'}


def font(text, fbt=None):
    '''
    Insert escape sequences into a text string to change the text color and
    other attributes when the string is printed in aterminal.

    Usage:
        esctext = font(text, fbt=None)

    Arguments:
        text: (str) The text string to change the font of when printed.

        fbt: (str) A string of the form "<fc>;<bc>;<ta>" where
            <fc> is a one-character color code for the foreground color.
            <bc> is a one-character color code for the background color.
            <ta> is a multi-character code for the text attributes.

            <fc> and <bc> are empty or one of the following:
                k: black, r: red, g: green, y: yellow, b: blue, m: magenta,
                c: cyan, w: white

            <ta> is empty or one or more of the following:
                o: turn off all, b: bold, u: underline, l: blink, r: reverse,
                c: conceal

    Returns:
        esctext: (str) The original text with special escape characters
        embedded to produce the desired colors and fonts when printed.

    Example:
        print(font('This is a test!','r;b;bu'))

    Notes:
        There are certain environments (e.g., some debuggers) where the text
        formatting escape sequences do not work. This function attempts to
        detect those environments and return the original, unescaped text in
        those environments.

    Author:
        Phil David, U.S. Army Research Labortory, 2019-04-04.
    '''

    if fbt is None or exename == 'wingdb':
        return text

    escseq = '\033['

    cmd = fbt.split(';')

    if len(cmd) > 0:
        for c in cmd[0]:
            # Foreground color.
            if c in fc.keys():
                escseq += ';' + fc[c]
            else:
                print('Invalid foreground color: "{:s}"'.format(c))

        if len(cmd) > 1:
            # Background color.
            for c in cmd[1]:
                if c in bc.keys():
                    escseq += ';' + bc[c]
                else:
                    print('Invalid background color: "{:s}"'.format(c))

            if len(cmd) > 2:
                # Other text attributes.
                for c in cmd[2]:
                    if c in tc.keys():
                        escseq += ';' + tc[c]
                    else:
                        print('Invalid text code: "{:s}"'.format(c))

    return escseq + 'm' + text + '\033[0m'


def mymatmul(m1, m2, blocksize=10000):
    """
    My matrix multiplication, for some pairs of large 2D matricies that cause
    Numpy.matmul() to freeze (Numpy version 1.18.1 as of 2020-120-07). This is
    intended to compute the product P of two matrices, P = A*B, where one of the
    outer dimensions of the product (the number of rows of A or the number of
    columns of B) is larger than any of the inner dimensions (the number of
    columns of A or number of rows of B).

    Usage:
        cmat = mymatmul(amat, bmat, blocksize=4)

    Arguments:
        amat: (Numpy.array) A 2D Numpy array.

        bmat: (Numpy.array) A 2D Numpy array.

        blocksize: (int) Maximum dimension of a single matrix block used in
        contructing the full matrix product. The larger this number, the more
        efficient the code will be, but also the more likely that Numpy will
        freeze.

    Description:
        Some pairs of large 2D matricies cause Numpy.matmul() to freeze (Numpy
        version 1.18.1 as of 2020-120-07). This function is intended to compute
        the product P of two matrices, P = A*B, where one of the outer
        dimensions of the product (the number of rows of A or the number of
        columns of B) is larger than any of the inner dimensions (the number of
        columns of A or number of rows of B). The product matrix is computed as
        a collection of smaller matrices.

    Example:
        a = np.random.rand(100,100)
        b = np.random.rand(100,1000000)
        c = np.mymatmul(a,b)
    """

    nrow1, ncol1 = m1.shape
    nrow2, ncol2 = m2.shape
    if ncol1 != nrow2:
        raise Exception('Matrix shapes are incompatable for multiplication')

    mprod = np.empty((nrow1, ncol2))

    if ncol2 >= max(nrow1, ncol1, nrow2):
        numblocks = int(np.ceil(ncol2/blocksize))
        col = 0
        for _ in range(numblocks):
            mprod[:,col:col+blocksize] = np.matmul(m1, m2[:,col:col+blocksize])
            col += blocksize
    elif nrow1 >= max(ncol1, nrow2, ncol2):
        numblocks = int(np.ceil(nrow1/blocksize))
        row = 0
        for _ in range(numblocks):
            mprod[row:row+blocksize,:] = np.matmul(m1[row:row+blocksize,:], m2)
            row += blocksize
    else:
        raise Exception('Not implemented for these shape matrices')

    return mprod


def imwrite(filename, image):
    """
    Write an image to a file.

    Usage:
        imwrite(filename, image)

    Arguments:
        filename: (str) Destination path of image file. This may be a simple
        file name or a folder path and a file name. The file extension in
        "filename" determines the image format. If a path to a nonexistant
        folder is specified, then this function attempts to create that folder.

        image: (numpy.ndarray) The data to write to the image file.
    """
    fpath, fname = os.path.split(filename)
    if fpath != '' and not os.path.exists(fpath):
        try:
            os.mkdir(fpath)
        except:
            raise Exception('Unable to create the folder "{}"'.format(fpath))
    imageio.imwrite(filename, image)


class Transcript(object):
    """
    Transcript - direct print output to a file, in addition to terminal.

    Usage:
        import transcript
        transcript.start('logfile.log')
        print("inside file")
        transcript.stop()
        print("outside file")

    Source of original code:
        https://stackoverflow.com/questions/14906764/
                 how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        """
        This flush method is needed for python 3 compatibility.

        OLD: This handles the flush command by doing nothing. you might want to
        specify some extra behavior here.
        """
        self.logfile.flush()

def start_logging(filename):
    """
    Start transcript, appending print output to given filename.  If a folder is
    specified that doesn't exist, then try to create that folder.
    """
    if hasattr(sys.stdout,'logging') and sys.stdout.logging:
        stop_logging()
    fpath, fname = os.path.split(filename)
    if fpath != '' and not os.path.exists(fpath):
        try:
            os.mkdir(fpath)
        except:
            raise Exception('Unable to create the folder "{}"'.format(fpath))
    sys.stdout = Transcript(filename)
    sys.stdout.logging = True

def stop_logging():
    """
    Stop transcript and return print functionality to normal.
    """
    if not sys.stdout.logging:
        return
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
    sys.stdout.logging = False


if __name__ == '__main__':

    assert distPntLineSeg([2,3,10], [0,-3,10], [1,0,10]) == 3.1622776601683795

    rgb = cmdistinct()

    print(font('This is a test!','r;b;bu'))

    a = 100*(np.random.rand(5,3)-0.5)
    print('A =\n',np2str(a,fmt='>5.1f'))

    a = {'name':'bob', 'age':23, 'dir':'~/test/space'}
    p = goodpath(a)
    print(p)

    b = Blob(xs=slice(10,20),ys=slice(100,150), props={'mykey':123})
    print('Blob', b)

    im1 = np.random.randint(0,high=100,size=(100,100))*100
    im2 = np.random.randint(0,high=100,size=(500,400))*200
    im3 = np.random.randint(0,high=100,size=(100,100))*500
    im4 = np.random.randint(0,high=100,size=(200,200))*30
    mimshow(im1, title='Image 1')
    mimshow([im1, im2, im3, im4], link=[0,2], title='Four Images')
    mimshow([im1, im3], link=[0,1])
    print('Main is done')
