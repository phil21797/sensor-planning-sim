
"""
This is a class for working with a 3D (multichannel) map of a SimWorld
environemnt. The goal is to enable fast lookup of world objects given 2D spatial
coordinates. Maps identify the types of objects, the object IDs, and object
heights at each (x,y) location. Each object type has its own set of map layers
for storing the object labels, IDs, and elevations.

Author:
    Phil David, US Army Research Laboratory, January 2022.
"""

import numpy as np
import simworld as sim
from numbers import Number

class Map3D:

    def __init__(self, gridspacing=0.25, radius=None, dynamic=True):
        """
        Initialize the 3D map.

        Usage:
            map3d = Map3D(gridspacing=0.25, radius=None, dynamic=True)

        Arguments:
            gridspacing: (float) Space, in meters, between map sample points.
            Default is 0.25 meters.

            radius: (float) Radius, in meters, of simulated world.

            dynamic: (bool) True if objects in the simulated world are dynamic.
            Default is True.

        Description:
            Each object label (from simworld.label2id) has its own set of layers
            for storing the positions, IDs, and elevations of objects with that
            label. This implies that there should be at most one object of a
            given type at each map location.
        """
        assert radius != None, '"radius" argument must be defined'

        # Create a buffer around the environment to give dynamic objects some
        # leeway in navigating around each other.
        self.map_radius = radius + 25

        self.mgridspc = gridspacing                           # grid spacing, in meters
        self.mdim = int(np.ceil(2*self.map_radius/self.mgridspc)) # map dimension (X and Y)
        self.dynamic = dynamic
        self.map_nchan = len(sim.label2id)

        # Maps: one channel for each class of objects. "mapelev" gives the
        # elevation at the top most (highest) point in each object class.
        # "mapceiling" gives the elevation of the top most (highest) point in
        # ceilings (empty space) under object surface.
        self.maplabel = np.zeros((self.mdim, self.mdim, self.map_nchan), dtype=int)
        self.mapid = np.zeros((self.mdim, self.mdim, self.map_nchan), dtype=int)
        self.mapelev = np.zeros((self.mdim, self.mdim, self.map_nchan), dtype=float)
        self.mapceiling = np.zeros((self.mdim, self.mdim, self.map_nchan), dtype=float)

        # Constants for mapping from world coordinates to map coordinates:
        #     col = self.mca*x + self.mcb
        #     row = self.mcc*y + self.mcb
        self.mca = self.mdim/(2.0*self.map_radius)
        self.mcb = self.mdim/2.0 - 0.5
        self.mcc = -self.mdim/(2.0*self.map_radius)


    def get(self, rect=None, out='EIL'):
        """
        Get all object labels, IDs, and/or elevations from a region in the map.

        Usage:
            oelevs, oids, olabels = Map3d.get(rect, out='EIL')

        Arguments:
            rect: (list) [xcenter, ycenter, xhalfwidth, yhalfwidth] defines
            the region, in world coordinates.

            out: (str) This string specifies what information is to be returned.
            Default is "EIL". This string should include one or more of the
            letters "E", "I", and "L", with no repetitoins, where "E" stands for
            elevations, "I" stands for IDs, and "L" stand for labels. The order
            of the returned values will be the same as the order of the letters
            in "out". For example:
                ids, labels, elev = Map3D.get(rect, out="ILE")
                labels = Map3D.get(rect, out="L")

        Returns:
            olabels: (list) List of labels (strings) of WorlObj objects in RECT.

            oids: (list) List of IDs (ints) of WorlObj objects in RECT.

            oelevs: (list) List of elevations (floats) of WorlObj objects in RECT.

        Description:
            The returned lists will all have the same lengths. There is always a
            one-to-one correspondence between entries in these lists. I.e.,
            olabels[K], oids[K], and oelevs[K] will all describe properties of
            the same point in the environment. If there are multiple different
            objects (with different IDs) in the map region that are the same
            type (have the same labels), then the returned object labels will
            not be unique.
        """
        if rect == None or len(rect) != 4:
            raise ValueError('"rect" argument must be a 4-element rectangle definition')
        if type(out) != str or len(out) not in {1,2,3}:
            raise ValueError('"out" must be a string of 1 to 3 of the chars EIL')

        # Get world coordinates at two opposite corners of the rectangle.
        x0 = rect[0] - rect[2]
        x1 = rect[0] + rect[2]
        y0 = rect[1] - rect[3]
        y1 = rect[1] + rect[3]

        # Map world corrdinates to array coordinates.
        c0 = self.mca*x0 + self.mcb
        c1 = self.mca*x1 + self.mcb
        r0 = self.mcc*y0 + self.mcb
        r1 = self.mcc*y1 + self.mcb

        # Get number of rows & columns in rectangle.
        nc = int(round(abs(c0 - c1)))
        nr = int(round(abs(r0 - r1)))

        # Get integer range of rectangle.
        if c0 > c1: c0 = c1
        if r0 > r1: r0 = r1
        c0 = int(round(c0))                                # 1st column
        c1 = c0 + nc                                       # last column + 1
        r0 = int(round(r0))                                # 1st row
        r1 = r0 + nr                                       # last row + 1

        # Make sure all coordinates are in the range [0, self.mdim-1].
        md = self.mdim - 1
        c0 = max(0, min(md, c0))
        c1 = max(0, min(md, c1))
        r0 = max(0, min(md, r0))
        r1 = max(0, min(md, r1))

        # Get indices into the ROI of unique object IDs > 0.
        v, indices = np.unique(self.mapid[r0:r1, c0:c1, :], return_index=True)
        indices = indices[v > 0]

        out = out.upper()

        if "E" in out:
            # Get object elevations.
            oelevs = np.ravel(self.mapelev[r0:r1, c0:c1, :])[indices]

        if "I" in out:
            # Get object IDs.
            oids = np.ravel(self.mapid[r0:r1, c0:c1, :])[indices]

        if "L" in out:
            # Get object integer labels.
            olabels = np.ravel(self.maplabel[r0:r1, c0:c1, :])[indices]

        # Put return data into the correct order.
        ret = []
        for w in out:
            if w == 'E':
                ret.append(list(oelevs))
            elif w == 'I':
                ret.append(list(oids))
            elif w == 'L':
                ret.append([sim.id2label[v] for v in olabels])

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


    def set(self, rect, olabel, oid, oelev, ceiling=None):
        """
        Set object labels, IDs, and elevation for a region of the map.

        Usage:
            mrect = Map3d.set(rect, olabel, oid, oelev, ceiling=None)

        Arguments:
            rect: (list) [xcenter, ycenter, xhalfwidth, yhalfwidth] defines
            the map region, in world coordinates, to set the label/ID of.

            olabel: (string) The object label to write into the map region.

            oid: (int) The object ID to write into the map region.

            oelev: (float) The object elevation to write into the map region.
            Each object label ("olabel") has its own elevation layer.

            ceiling: (type, p1, p2, ...). A tuple defining a ceiling in the
            elevation data.  "type" is a string defining the type of ceiling,
            which must be one of the following:
                "post" - There is a post of diameter "p1" in the center of the
                rectangle "rect" with a ceiling height of "p2" around the post.
                "p1" and "p2" are both in world coordinates. For example, a post
                may be created for a tree trunk: everywhere in the rectangle
                except where the tree trunk is located (of diameter "p1") has
                open space of height "p2" (the space from the ground up to the
                lower limbs of the tree, or the ceiling).

        Returns:
            mrect: (4-array-like) The rectangle in the 3D map that identifies
            the labeled pixels of current object. This is a length 4 array
            [xctr, yctr, xhwidth, yhwidth] where all components are in world
            coordinates. Due to the discrete grid spacing of the 3D map, this is
            usually not the same as "rect" argument.

        Description:
            Each object label (from simworld.label2id) has its own set of layers
            for storing the positions, IDs, and elevations of objects with that
            label.
        """
        if type(olabel) != str:
            raise ValueError('"olabel" must be a str')
        if type(oid) != int:
            raise ValueError('"oid" must be an int')
        if not isinstance(oelev, Number):
            raise ValueError('"oelev" must be a float or int')
        olabel = olabel.lower()
        if olabel not in sim.label2id.keys():
            raise ValueError('Invalid object label: "{}"'.format(olabel))

        # Get world coordinates at two opposite corners of the rectangle.
        x0 = rect[0] - rect[2]
        x1 = rect[0] + rect[2]
        y0 = rect[1] - rect[3]
        y1 = rect[1] + rect[3]

        # Map world corrdinates to array coordinates.
        c0 = self.mca*x0 + self.mcb
        c1 = self.mca*x1 + self.mcb
        r0 = self.mcc*y0 + self.mcb
        r1 = self.mcc*y1 + self.mcb

        # Get number of rows & columns in rectangle.
        nc = int(round(abs(c0 - c1)))
        nr = int(round(abs(r0 - r1)))

        # Get integer range of rectangle.
        if c0 > c1: c0 = c1
        if r0 > r1: r0 = r1
        c0 = int(round(c0))                                # 1st column
        c1 = c0 + nc                                       # last column + 1
        r0 = int(round(r0))                                # 1st row
        r1 = r0 + nr                                       # last row + 1

        # Make sure all coordinates are in the range [0, self.mdim-1].
        md = self.mdim - 1
        c0 = max(0, min(md, c0))
        c1 = max(0, min(md, c1))
        r0 = max(0, min(md, r0))
        r1 = max(0, min(md, r1))

        intlabel = sim.label2id[olabel]
        self.maplabel[r0:r1, c0:c1, intlabel] = intlabel
        self.mapid[r0:r1, c0:c1, intlabel] = oid
        self.mapelev[r0:r1, c0:c1, intlabel] = oelev

        # Get the discrete-pixeled rectangle in the map that the original
        # rectangle is mapped to.
        xctr = ((c0+c1-1)/2 - self.mcb)/self.mca
        yctr = ((r0+r1-1)/2 - self.mcb)/self.mcc
        mrect = [xctr, yctr, (c1-c0)*self.mgridspc/2, (r1-r0)*self.mgridspc/2]

        if ceiling is not None:
            assert type(ceiling[0]) is str
            if ceiling[0].lower() == 'post':
                # Create open space around the object center post. Everwhere in
                # the object's rectangle except the center post will have a
                # fixed ceiling height.
                diam = ceiling[1]
                hgt = ceiling[2]
                self.mapceiling[r0:r1,c0:c1,intlabel] = hgt  # ceiling height around post
                cctr = int(np.round((c0+c1)/2))              # column center of rectangle
                rctr = int(np.round((r0+r1)/2))              # row center of rectangle
                chw = int(np.round((c1-c0+1)*(diam/(2*rect[2])))) # column half width
                rhw = int(np.round((r1-r0+1)*(diam/(2*rect[3])))) # row half width
                rows = slice(rctr-rhw, rctr+rhw)             # rows occupied by post
                cols = slice(cctr-chw, cctr+chw)             # columns occupied by post
                self.mapceiling[rows,cols,intlabel] = 0      # no open space inside post
            else:
                raise ValueError('Invalid ceiling type: "{:s}"'.format(ceiling[0]))

        return mrect


    def flatlabels(self):
        """
        Create a flat (single-channel) copy of the labels map. Smaller label
        values, other than zero, take precedence (overwrite) over larger label
        values.
        """
        labels = self.maplabel.copy()
        labels[labels == 0] = 1000
        flatlabels = np.min(labels, axis=2)
        return flatlabels


    def isclear(self, rect, myid, mytype, myelev):
        """
        Check if a rectangular area in the map is clear of obstacles (objects
        other than the current object).

        Usage:
            tf = Map3D.isclear(rect, myid, mytype, myelev)

        Arguments:
            rect: (list) The rectangle to check: [xcenter, ycenter, xhalfwidth,
            yhalfwidth].

            myid: (int) ID of current object (which potentially wants to move
            into the rectangle).

            mytype: (str) Class label (a string) of current object.

            myelev: (float) Elevation of current object. For ground-based
            objects, this should be the height of the object. For airborne
            objects, this should be the elevation at its base (i.e., at the
            bottom of the object).

        Returns:
            tf: (bool) True if the rectangle is clear to move into.

        Description:
            Different conditions are required depending on whether the current
            object (as specified by myid, mytype, and myelev) is ground-based or
            airborne. Two ground-based objects or two airborne objects cannot
            occupy the same map cells. It is possible for a ground-based and
            airborne object to occupy the same map cells if the airborne object
            is above the ground-based object.
        """

        oelevs, oids, olabels = self.get(rect=rect, out="EIL")

        if mytype == 'airborne':
            # The current object is airborne. There should be no other airborne
            # objects in the rectangle, and the current object should be higher
            # in elevation than all ground-based objects in the rectangle except
            # for plants.
            for k in range(len(olabels)):
                if oids[k] != myid:
                    if olabels[k] == 'airborne' or myelev <= oelevs[k]:
                        return False     # multiple airborne or airborne too low
        else:
            # The current object is ground-based. There should be no other
            # ground-based objects in the rectangle. Airborne objects are
            # allowed if they are above the current ground object.
            for k in range(len(olabels)):
                if oids[k] != myid:
                    if olabels[k] in sim.gnd_obstacles:
                        return False      # two different ground objects in rect
                    elif olabels[k] == 'airborne' and myelev >= oelevs[k]:
                        return False    # ground obj will intersect airborne obj

        return True


    def move(self, srect, drect, olabel):
        """
        Move one layer of map data from a source rectangle to a destination
        rectangle.

        Usage:
            mrect = Map3d.move(self, srect, drect, olabel)

        Arguments:
            srect: (list) [xcenter, ycenter, xhalfwidth, yhalfwidth] defines
            the source rectangle, in world coordinates.

            drect: (list) [xcenter, ycenter, xhalfwidth, yhalfwidth] defines
            the destination rectangle, in world coordinates.

            olabel: (string) The object label determines what layer (channel) of
            the 3D map to move.


        Returns:
            mrect: (4-array-like) The rectangle in the 3D map that identifies
            the pixels of destination rectangle. This is a length 4 array
            [xctr, yctr, xhwidth, yhwidth] where all components are in world
            coordinates. Due to the discrete grid spacing of the 3D map, this is
            usually not the same as "drect" argument.
        """

        if type(olabel) != str:
            raise ValueError('"olabel" must be a str')
        olabel = olabel.lower()
        if olabel not in sim.label2id.keys():
            raise ValueError('Invalid object label: "{}"'.format(olabel))

        for k,rect in enumerate([srect, drect]):
            # Get world coordinates at two opposite corners of the rectangle.
            x0 = rect[0] - rect[2]
            x1 = rect[0] + rect[2]
            y0 = rect[1] - rect[3]
            y1 = rect[1] + rect[3]

            # Map world corrdinates to array coordinates.
            c0 = self.mca*x0 + self.mcb
            c1 = self.mca*x1 + self.mcb
            r0 = self.mcc*y0 + self.mcb
            r1 = self.mcc*y1 + self.mcb

            # Get number of rows & columns in rectangle.
            nc = int(round(abs(c0 - c1)))
            nr = int(round(abs(r0 - r1)))

            # Get integer range of rectangle.
            if c0 > c1: c0 = c1
            if r0 > r1: r0 = r1
            c0 = int(round(c0))                                # 1st column
            c1 = c0 + nc                                       # last column + 1
            r0 = int(round(r0))                                # 1st row
            r1 = r0 + nr                                       # last row + 1

            # Make sure all coordinates are in the range [0, self.mdim-1].
            md = self.mdim - 1
            c0 = max(0, min(md, c0))
            c1 = max(0, min(md, c1))
            r0 = max(0, min(md, r0))
            r1 = max(0, min(md, r1))

            if k == 0:
                # Source rectangle array coordinates.
                sc0 = c0
                sc1 = c1
                sr0 = r0
                sr1 = r1
            else:
                # Destination rectangle array coordinates.
                dc0 = c0
                dc1 = c1
                dr0 = r0
                dr1 = r1

        # Make a copy of the source data.
        intlabel = sim.label2id[olabel]
        slabels = self.maplabel[sr0:sr1, sc0:sc1, intlabel].copy()
        sids = self.mapid[sr0:sr1, sc0:sc1, intlabel].copy()
        selevs = self.mapelev[sr0:sr1, sc0:sc1, intlabel].copy()

        # Zero out the source rectangle data.
        self.maplabel[sr0:sr1, sc0:sc1, intlabel] = 0
        self.mapid[sr0:sr1, sc0:sc1, intlabel] = 0
        self.mapelev[sr0:sr1, sc0:sc1, intlabel] = 0

        # Copy source data into destination rectangle.
        self.maplabel[dr0:dr1, dc0:dc1, intlabel] = slabels
        self.mapid[dr0:dr1, dc0:dc1, intlabel] = sids
        self.mapelev[dr0:dr1, dc0:dc1, intlabel] = selevs

        # Get the discrete-pixeled rectangle in the map that the destination
        # rectangle is mapped to.
        xctr = ((dc0+dc1-1)/2 - self.mcb)/self.mca
        yctr = ((dr0+dr1-1)/2 - self.mcb)/self.mcc
        mrect = [xctr, yctr, (dc1-dc0)*self.mgridspc/2, (dr1-dr0)*self.mgridspc/2]

        return mrect


    def IntersectCount(self, p0, p1, olabel):
        """
        Count the number map cells of a specific label that a line intersects
        with.

        Usage:
            cnt = Map3d.IntersectCount(p0, p1, olabel, oid)

        Arguments:
            p0: (2d array-like) First 2D endpoint  (x,y) of the line segment.

            p1: (2d array-like) Second 2D endpoint  (x,y) of the line segment.

            olabel: (string) The object label to count intersections with.

        Returns:
            cnt: (int) The number of map cells of type `olabel` that the line
            intersects.

        Description:

        """
        if type(olabel) != str:
            raise ValueError('"olabel" must be a str')
        olabel = olabel.lower()
        if olabel not in sim.label2id.keys():
            raise ValueError('Invalid object label: "{}"'.format(olabel))

        # Get world coordinates at two line endpoints.
        x0, y0 = p0[0:2]
        x1, y1 = p1[0:2]

        # Map world corrdinates to map cell coordinates.
        c0 = self.mca*x0 + self.mcb
        r0 = self.mcc*y0 + self.mcb
        c1 = self.mca*x1 + self.mcb
        r1 = self.mcc*y1 + self.mcb

        # Get number of rows & columns in bounding rectangle.
        nc = int(round(abs(c0 - c1))) + 1
        nr = int(round(abs(r0 - r1))) + 1
        maxrc = max(nr, nc)

        # Make sure all coordinates are in the range [0, self.mdim-1].
        md = self.mdim - 1
        assert c0 >= 0 and c0 <= md, "Column of 1st line endpoint is outside of map"
        assert c1 >= 0 and c1 <= md, "Column of 2nd line endpoint is outside of map"
        assert r0 >= 0 and r0 <= md, "Row of 1s line endpoint is outside of map"
        assert r1 >= 0 and r1 <= md, "Row of 2nd line endpoint is outside of map"

        # Row and column coordinates for line following.
        rows = np.round(np.linspace(r0, r1, num=maxrc, endpoint=True)).astype(int)
        cols = np.round(np.linspace(c0, c1, num=maxrc, endpoint=True)).astype(int)

        # Count the number of cells that intersect the specified object label.
        intlabel = sim.label2id[olabel]
        linelabels = self.maplabel[rows, cols, intlabel]
        cnt = sum(linelabels > 0)

        return cnt

