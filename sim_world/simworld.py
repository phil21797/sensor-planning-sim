"""
Create random 3D outdoor environments.

Author:
    Phil David, US Army Research Laboratory, December 2017.

Description:
    The 3D outdoor environment is created using the Visualization Toolkit (VTK,
    www.vtk.org).

    Run this as a main program to see a demonstration of the simulation code.

    A number of geometrically identical, but photometrically different, parallel
    models are created. This allows for easier generation of groundtruth data.

History:
    2023-06-13: P. David -- Added simple acoustic sensor model.
    2017-12-07: P. David -- Created from Phil's cylinder_world.py.
    2017-12-25: P. David -- Added texture mapping of objects.
    2018-01-03: P. David -- Added multiple, parallel worlds.
    2020-05-11: P. David -- Improved photorealism; improve model generation
                            speed; added time-of-day rendering; added people
                            looking out windows.
    2020-07-28: P. David -- Added fences and brick/stone walls to scene elements.
    2020-12-22: P. David -- Allow objects in the scene to move.
    2021-01-31: P. David -- Create shadows of some static objects.
    2021-02-04: P. David -- All objects, except airbornes, generate
                            time-appropriate shadows.
    2021-04-01: P. David -- Added "object ID" world view.
"""

import numpy as np
import time
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import vtk
import vtk.util.colors
import os
import vtkutils as vtu
import vtkwininteract as vwi
import skimage
import pathlib
import imageio
import wavio
from enum import Enum
import scipy.signal as sig
import skimage.measure as skm
from skimage.morphology import disk
from skimage import draw
# from skimage.filters.rank import modal
from scipy import ndimage as ndi
from fig import *
from map3d import *
from map2d import *


# The basic categories of texture image tags. The "clutter" tag can include
# any type of object that is not included in the remaining tags.
base_tags = {'person', 'clutter', 'sky', 'ground', 'plant', 'airborne',
             'building', 'road', 'sign', 'barrier', 'gndfeat', 'animal',
             'vehicle', 'background', 'intersection'}

# Map object names to integer IDs. The list `label_colors` must be updated
# whenever this dictionary (`label2id`) is changed. The integer ID assigned to
# an object name is mapped to a color via 'id2color1' and 'id2color255'. The
# numeric value assigned to each label should roughly correspond to the order
# for which objects can be placed on top of other objects, where lower numbered
# labels may generally appear on top of higher number labels. This property of
# the labels is used only for the 2D map display.
label2id = {'unknown':0, 'person':1, 'clutter':2, 'animal':3, 'sign':4,
            'vehicle':5, 'airborne':6, 'barrier':7, 'plant':8, 'building':9,
            'gndfeat':10, 'road':11, 'water':12, 'ground':13, 'sky':14}

# Map SimWorld semantic labels to MIT's SceneParse150 semantic labels.
simworld2mit150 = {'unknown':'arcade', 'person':'person', 'clutter':'plaything',
                   'animal':'animal', 'sign':'signboard', 'vehicle':'car',
                   'barrier':'fence', 'plant':'tree', 'building':'building',
                   'gndfeat':'grass', 'road':'road', 'airborne':'airplane',
                   'water':'water', 'ground':'earth', 'sky':'sky'}

# List of obstacles to movement of ground-based objects.
gnd_obstacles = {'person', 'clutter', 'animal', 'sign', 'vehicle', 'barrier',
                 'plant', 'building', 'water'}

# Map specific object types to general object classes recognized by SimWorld.
objclass = {c:c for c in base_tags}
objclass = {**objclass, 'dog':'animal', 'cat':'animal', 'cow':'animal',
            'bird':'animal'}

# Map object IDs to object labels.
id2label = {id:label for (label,id) in label2id.items()}

# RGB colors of object semantic labels. The order of the colors must correspond
# to the label IDs given in "label2id". Also, these colors must correspond to
# the colors in the texture label images.
label_colors = [(255,255,255),     #  0 = Unknown
                (255,0,0),         #  1 = Person
                (255,217,0),       #  2 = Clutter
                (152,125,88),      #  3 = Animal
                (255,160,0),       #  4 = Sign
                (140,60,200),      #  5 = Vehicle
                (0,255,255),       #  6 = Airborne
                (0,0,0),           #  7 = barrier
                (0,255,144),       #  8 = Plant
                (240,240,200),     #  9 = Building
                (178,223,191),     # 10 = Ground feature
                (130,130,130),     # 11 = Road (was (80,80,80))
                (0,0,255),         # 12 = Water
                (160,155,115),     # 13 = Ground
                (0,156,255)]       # 14 = Sky

# Map object IDs to RGB label colors.
id2color255 = np.array(label_colors)   # range is [0:255]x3
id2color1 = id2color255/255            # range is [0:1]x3

# Map directly from object names to object colors.
otype2color1 = {key:id2color1[label2id[key]] for key in label2id.keys()}
otype2color255 = {key:id2color255[label2id[key]] for key in label2id.keys()}

# Height (meters) of various near-ground layers in the model.
hgt_ground = 0.000
hgt_road = 0.001
hgt_intersection = 0.002
hgt_gndfeat = 0.003
hgt_bldg_shadow = 0.01
hgt_plant_shadow = 0.01
hgt_obj_shadow = 0.01      # obj in {person, animal, clutter, sign}

audio_sample_rate = 16000  # resample all audio files at this rate (Hz)
audio_record_dist = 10     # distance (meters) from source at which sounds are
                           #    assumed to be recorded (was 3)

# How should audio signals be normalized relative to the maximum, `audio_max`?
# This value must be in the range [0,1]. The default for any object is 0.5 when
# not specified. `audio_max` is the initial maximum absolute amplitude (volume)
# of any audio signal assigned to an object prior to that object's sound being
# measured by a microphone in a simulated environment. An object's measured
# audio signal may be greater than this value if the object is closer than
# `audio_record_dist` to a microphone. The maximum absolute measured audio
# signal is limited to `audio_max_scale` x `audio_max`.
audio_max = float(2**14)
audio_max_scale = 2.0
audio_normalize = {'bird':0.2, 'car':0.7, 'cat':0.3, 'cow':0.4, 'dog':0.4,
                   'drone':0.3, 'motorcycle':1.0, 'person':0.3, 'robot':0.3,
                   'sheep':0.4, 'traffic':0.6, 'truck':1.0}
audio_cmap = plt.get_cmap('rainbow')


# Different types of object motion.
class Motion(Enum):
    Static = 1        # do not move
    Random = 2        # choose new random direction when encountering obstacle
    DefinedPath = 3   # follow a predefined path


class WorldObj:
    """
    Class for individual objects (person, clutter, building, signs, etc.) in
    the world.
    """
    id_counter = 0         # used to assign a unique ID to each object.
    textures = []          # list of textures, each is a dict
    numtextures = 0        # number of texture images
    sounds = []            # list of sounds, each is a dict
    numsounds = 0          # number of sounds
    tag2txtid = {}         # dict that maps tags to sets of texture IDs
    tag2soundid = {}       # dict that maps tags to sets of sound IDs
    lightcoef = (1,1,0)    # lighting coefficients: (ambient, diffuse, specular)
    audio_max = audio_max  # used to normalize audio signals
    audio_max_scale = audio_max_scale  # scale factor of max measured audio
    verbosity = 0          # level of normal output: [0,1,2,3]

    # The measure audio volume is allowed to be greater than `audio_max` when
    # the object (sound source) is very close to the microphone.
    audio_max_measured = audio_max_scale*audio_max


    def __init__(self, renderer, worldviews, objtags=None, pos=None, rot=(0,0,0),
                 axis1=None, axis2=None, rect2d=None, rect3d=None, rot90=False,
                 cylinder=None, height=None, orient=None, ellipsoid=None,
                 sphere=None, labelobject=False, motiontype:Motion=Motion.Random):
        """
        Create a new world object.

        Usage:
            obj = WorldObj(renderer, worldviews, objtags=None, pos=None,
                           rot=(0,0,0), axis1=None, axis2=None, rect2d=None,
                           rect3d=None, rot90=False, cylinder=None, ellipsoid=None,
                           sphere=None, labelobject=False,
                           motiontype:Motion=Motion.Random)

        Arguments:
            renderer: A list of VTK renderers to which the object should be added.

            worldviews: A list of the names of all world views (each corresponding
            to one renderer).

            objtags: The tags of the object (string or set of strings). E.g.,
            'sign.stop' or {'sign', 'stop', '123'}.

            motiontype: (Motion) The type of motion that the object may exhibit.
            The default is Motion.Random motion.

            One or more geometric object descriptors must be provided:
                pos: [x, y, z]
                axis1: [dx, dy, dz]
                axis2: [dx, dy, dz]
                rect2d: [xcenter, ycenter, xhalfwidth, yhalfwidth]
                rect3d: [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
                cylinder: [xcenter, xcenter, zcenter, radius, zhalfwidth]
                ellispoid: [xcenter, xcenter, zcenter, xyhalfwidth, zhalfwidth]
                sphere: [xcenter, ycenter, zcenter, radius]
                rot: [xrot, yrot, zrot]

        Description:
            To create a 'person', 'clutter', 'plant', 'animal', 'vehicle', or
            'barrier', the following arguments are required:

                pos: (3-array-like) The position (x,y,z) of one corner of the
                quadrilateral. This should correspond to location of the
                lower left corner (the origin, below) of the texture image.

                axis1: (3-array-like) The 3-vector (dx,dy,dz) from the
                quadrilateral's 1st corner (defined by `pos`) that defines
                the 1st edge of the quadrilateral (both its direction and
                length). This should correspond to the X-axis (horizontal)
                of the texture image, where the origin is taken to be the
                lower left corner.

                axis2: (3-array-like) The 3-vector (dx,dy,dz) from the
                quadrilateral's 1st corner (defined by `pos`) that defines
                the 2nd edge of the quadrilateral (both its direction and
                length). This should correspond to the Y-axis (vertical) of
                the texture image, where the origin is taken to be the lower
                left corner.
        """
        assert objtags != None, 'Missing argument "objtags"'
        assert type(objtags) == str or type(objtags) == set
        if type(objtags) == str: objtags = set(objtags.split('_')) # convert string to set

        WorldObj.id_counter += 1
        self.id = WorldObj.id_counter       # Unique ID
        self.tags = objtags                 # set of object tags
        self.actors = []                    # actors associated with this object
        self.minspeed = 0                   # minimum speed of a moving object
        self.maxspeed = 0                   # maximum speed of a moving object
        self.speed = 0                      # initial speed, if object is moving
        self.shadow = True                  # create a shadow for the object
        self.motion_type = motiontype       # type of this object's motion
        numworlds = len(renderer)

        lcflat = (1.0, 0.0, 0.0)  # Lighting coef. (ambient, diffuse, specular) for flat colors.

        # Get the texture type and ID.
        otype, objid = TagTypeID(objtags)
        keytags = {otype} if objid == '' else {otype, objid}
        self.type = otype                   # class label (a string)
        self.keytags = keytags

        # Create a unique name for this object.
        tmp = list(self.keytags)
        tmp.sort(reverse=True)
        tmp.append(str(self.id))
        self.name = '_'.join(tmp)

        if otype in ['ground', 'road', 'gndfeat']:

            assert rect3d is not None, 'missing argument "rect3d"'
            # rect3d: [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
            self.xctr = rect3d[0]
            self.yctr = rect3d[1]
            self.zctr = rect3d[2]
            self.xhalflen = rect3d[3]
            self.yhalflen = rect3d[4]
            self.zhalflen = rect3d[5]
            ctr = rect3d[0:3]

            txt = WorldObj.GetTexture(inctags=keytags, exctags='label')
            assert txt != None, 'There are no texture photos for tags {}'.format(
                                 keytags)
            otype, oid = TagTypeID(txt['tags'])
            self.class_id = int(oid)                    # within-class object ID
            self.shadow = False           # do not make a shadow for this object

            labeltxt = WorldObj.GetTexture(inctags=keytags|{'label'})
            if labeltxt == None:
                # No label image. Color the entire quadrilateral.
                labelcolor = otype2color1[otype]
                labelim = None
            else:
                labelim = labeltxt['im']
                labeltxt = labeltxt['vtk']
                labelcolor = None

            if rot90 or {'road','ew'}.issubset(objtags):
                # Texture images for roads are assumed to be for roads in the
                # north-south (vertical) orientation. For east-west roads, the
                # texture coordinates must be rotated 90 degrees.
                corner = (-rect3d[3], rect3d[4], rect3d[5])
                axis1 = (0, -2*rect3d[4], 0)
                axis2 = (2*rect3d[3], 0, 0)
                reps = (-axis1[1]/txt['hsize'], axis2[0]/txt['vsize'])
            else:
                # Do not rotate the texture coordinates.
                corner = (-rect3d[3], -rect3d[4], rect3d[5])
                axis1 = (2*rect3d[3], 0, 0)
                axis2 = (0, 2*rect3d[4], 0)
                reps = (axis1[0]/txt['hsize'], axis2[1]/txt['vsize'])

            for w in range(numworlds):
                if worldviews[w] == 'color camera':
                    vtu.make_quad(renderer[w], origin=ctr, pos=corner,
                                  v1=axis1, v2=axis2, orient=rot, reps=reps,
                                  texture=txt['vtk'], lightcoef=WorldObj.lightcoef,
                                  opacity=1.0, tags=objtags)
                elif worldviews[w] == 'objectid':
                    # We don't record this object's ID, so recolor its semantic
                    # label image with (0,0,0) values.
                    if labeltxt == None:
                        idtxt = None
                        c = (0,0,0)
                    else:
                        idtxt = semlabel2objid(labelim, otype2color255[otype], 0, 0)
                        c = None
                    vtu.make_quad(renderer[w], origin=ctr, pos=corner, v1=axis1,
                                  v2=axis2, orient=rot, texture=idtxt, color=c,
                                  opacity=1.0, lightcoef=lcflat, tags=objtags)
                elif worldviews[w] == 'semantic labels':
                    vtu.make_quad(renderer[w], origin=ctr, pos=corner, v1=axis1,
                                  v2=axis2, orient=rot, texture=labeltxt,
                                  color=labelcolor, opacity=1.0,
                                  lightcoef=lcflat, tags=objtags)
                else:
                    raise Exception('Unrecognized worldview: "{:s}"'.format(worldviews[w]))

        elif otype == 'sign':
            # Create a street sign.

            assert height is not None, 'Missing argument "height"'
            assert pos is not None, 'Missing argument "pos"'
            assert orient is not None, 'Missing argument "orient"'

            # Get texture maps for front, back, and label of a particular sign.
            ftxt = WorldObj.GetTexture(inctags=objtags|{'front'})
            ftags = ftxt['tags']
            otype, oid = TagTypeID(ftags)
            self.class_id = int(oid)                    # within-class object ID
            tags = set([otype, oid, 'noflip'])
            btxt = WorldObj.GetTexture(inctags={otype,oid,'back'})
            mtxt = WorldObj.GetTexture(inctags={otype,oid,'label'})
            w = ftxt['hsize']   # width and height of sign, not including the post
            h = ftxt['vsize']

            if 'p' in ftags:
                # Use the pole tag ("p=IDxWIDTHxHEIGHT") to setup the sign post.
                pID, pW, pH = [n for n in TagValue(ftags,'p').split('x')]
                pW = float(pW)           # post width (meters)
                pH = float(pH)           # height of post to bottom of sign (meters)
                ptxt = WorldObj.GetTexture(inctags=set(['pole',pID]), exctags='label')
                if ptxt is None:
                    raise Exception('No texture for object {}'.format(['pole',pID]))
                post_type = 1 if 's' in ptxt['tags'] else 2  # 1 == square, 2 == round
                pRad = pW/2              # radius of post (meters)
                totalheight = pH+h+0.05  # height at top of sign post
                height = pH              # height at bottom of sign
            else:
                # Use a uniformly-colored cylinderical sign post.
                post_type = 3
                pRad = 0.04
                totalheight = height+h+0.05   # height at bottom of sign + height of sign + 5cm

            # Post position is [xctr, yctr, zctr, radius, halfheight]:
            ppos = [pos[0], pos[1], totalheight/2, pRad, totalheight/2]
            porigin = [pos[0], pos[1], 0]       # post origin is at base of post.

            # Set the sign position. The back of the sign should be touching the
            # sign post. The front of the sign needs to be offset from the back
            # by a small distance for correct rendering. 0.004 is a good offset:
            # small enough that the gap between the front and back is very
            # small, large enough so that VTK consistently renders the front
            # over the back when viewing the sign from the front.
            sorg = [ppos[0], ppos[1], height]   # origin: center of rotation
            sposf = [-w/2, -pRad-0.004, 0]      # front of sign
            sposb = [-w/2, -pRad, 0]            # back of sign

            sv1 = [w, 0, 0]
            sv2 = [0, 0, h]

            self.xctr = ppos[0]
            self.yctr = ppos[1]
            self.zctr = ppos[2]
            self.xhalflen = w/2
            self.yhalflen = w/2
            self.zhalflen = ppos[4]

            for w in range(numworlds):
                if worldviews[w] == 'color camera':
                    if post_type == 1:
                        # Square, texture-mapped, sign post
                        vtu.make_cyl(renderer[w], origin=porigin, width=pW,
                                     height=totalheight, radius=pW/2, textures=[ptxt['vtk']],
                                     lightcoef=WorldObj.lightcoef, tags=ptxt['tags'])
                    elif post_type == 2:
                        # Round, texture-mapped, sign post.
                        vtu.make_cylinder(renderer[w], ppos, texture=ptxt['vtk'],
                                          lightcoef=WorldObj.lightcoef, res=20,
                                          tags=ptxt['tags'])
                    else:
                        # Round, uniformly-colored, sign post.
                        vtu.make_cylinder(renderer[w], ppos, color=(0.7,0.7,0.7),
                                          lightcoef=WorldObj.lightcoef, res=20)

                    # Front of sign.
                    vtu.make_quad(renderer[w], origin=sorg,
                                  pos=sposf, v1=sv1, v2=sv2, orient=orient,
                                  reps=(1,1), texture=ftxt['vtk'], tags=tags,
                                  lightcoef=WorldObj.lightcoef)

                    # Back of sign.
                    vtu.make_quad(renderer[w], origin=sorg,
                                  pos=sposb, v1=sv1, v2=sv2, orient=orient,
                                  reps=(1,1), texture=btxt['vtk'], tags=tags,
                                  lightcoef=WorldObj.lightcoef)
                elif worldviews[w] == 'objectid':
                    # We don't record the sign's ID, so recolor the sign's
                    # semantic label image with (0,0,0) values.
                    vtu.make_cylinder(renderer[w], ppos, color=(0,0,0),
                                      lightcoef=lcflat, res=20)
                    idtxt = semlabel2objid(mtxt['im'], otype2color255[otype], 0, 0)
                    vtu.make_quad(renderer[w], origin=sorg, pos=sposf, v1=sv1,
                                  v2=sv2, orient=orient, reps=(1,1),
                                  texture=idtxt, tags=tags, lightcoef=lcflat)
                elif worldviews[w] == 'semantic labels':
                    vtu.make_cylinder(renderer[w], ppos, color=otype2color1['sign'],
                                      lightcoef=lcflat, res=20)
                    vtu.make_quad(renderer[w], origin=sorg,
                                  pos=sposf, v1=sv1, v2=sv2, orient=orient,
                                  reps=(1,1), texture=mtxt['vtk'], tags=tags,
                                  lightcoef=lcflat)
                else:
                    raise Exception('Unrecognized worldview: "{:s}"'.format(worldviews[w]))

        elif otype == 'building':

            # Create a building.
            # rect3d: [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
            assert rect3d is not None, 'missing argument "rect3d"'

            # Color texture image.
            ctxt = WorldObj.GetTexture(inctags=keytags, exctags='label')
            if ctxt == None:
                raise Exception('No color texture image matches tags {}'.format(keytags))
            tags = ctxt['tags']
            otype, oid = TagTypeID(tags)
            self.class_id = int(oid)                    # within-class object ID

            # Semantic label image.
            ltxt = WorldObj.GetTexture(inctags=keytags|{'label'})

            self.xctr = rect3d[0]
            self.yctr = rect3d[1]
            self.zctr = rect3d[2]
            self.xhalflen = rect3d[3]
            self.yhalflen = rect3d[4]
            self.zhalflen = rect3d[5]
            self.height = 2*rect3d[5]
            vec1 = np.array((2*rect3d[3], 0, 0))         # X axis of building
            vec2 = np.array((0, 2*rect3d[4], 0))         # Y axis of building
            vec3 = np.array((0, 0, 2*rect3d[5]))         # Z axis of building
            origin = np.array((rect3d[0],rect3d[1],0))   # center of building at ground level
            pos = -np.array((rect3d[3],rect3d[4],0))     # one corner relative to origin
            reps = (vec1[0]/ctxt['hsize'], vec2[1]/ctxt['hsize'], vec3[2]/ctxt['vsize'])
            self.txt_reps = reps

            # Get a random texture for the roof, it any exist.
            ttxt = WorldObj.GetTexture(inctags='roof', exctags='label')
            if ttxt:
                treps = (vec1[0]/ttxt['hsize'], vec2[1]/ttxt['vsize'])
                ttxt = ttxt['vtk']
            else:
                treps = None

            for w in range(numworlds):
                if worldviews[w] == 'color camera':
                    # Create texture mapped building with random diffuse lighting
                    # coefficient.
                    rg = np.random.uniform(0.1, 0.4)     # random roof greylevel
                    act = vtu.make_box(renderer[w], origin=origin, tags=tags,
                                 pos=pos, v1=vec1, v2=vec2, v3=vec3, reps=reps,
                                 texture=ctxt['vtk'], color=(rg, rg, rg),
                                 toptexture=ttxt, topreps=treps, opacity=1.0,
                                 lightcoef=WorldObj.lightcoef)
                elif worldviews[w] == 'objectid':
                    # We don't record the building's ID, so recolor the
                    # building's semantic label image with (0,0,0) values.
                    idtxt = semlabel2objid(ltxt['im'], otype2color255[otype], 0, 0)
                    act = vtu.make_box(renderer[w], origin=origin, tags=tags,
                                 pos=pos, v1=vec1, v2=vec2, v3=vec3, reps=reps,
                                 texture=idtxt, color=(0,0,0), lightcoef=lcflat,
                                 opacity=1.0)
                elif worldviews[w] == 'semantic labels':
                    act = vtu.make_box(renderer[w], origin=origin, tags=tags,
                                 pos=pos, v1=vec1, v2=vec2, v3=vec3, reps=reps,
                                 texture=ltxt['vtk'], color=otype2color1['building'],
                                 lightcoef=lcflat, opacity=1.0)
                else:
                    raise Exception('Unrecognized worldview: "{:s}"'.format(worldviews[w]))
                for a in act:
                    a.ForceOpaqueOff()        # not sure why this is needed

        elif otype == 'barrier':
            # Create a barrier. The long edge of the barrier is assumed to be
            # parallel to the ground plane. Arguments are:
            #    pos: one bottom corner (3D) of barrier quadrilateral.
            #    axis1: 3D vector from bottom corner parallel to barrier
            #           horizontal direction (parallel to ground plane).
            #    axis2: 3D vector from bottom corner parallel to barrier vertical
            #           direction.

            assert rect3d is not None, 'Missing argument "rect3d"'

            ftxt = WorldObj.GetTexture(inctags=keytags, exctags='label')
            tags = ftxt['tags']
            otype, oid = TagTypeID(tags)
            self.class_id = int(oid)                    # within-class object ID

            if 'wall' in tags:
                self.make_wall(renderer, worldviews, rect3d, ftxt, tags)
            elif 'fence' in tags:
                self.make_fence(renderer, worldviews, rect3d, ftxt, tags)

        elif otype in ['person','clutter','plant','animal','airborne','vehicle']:

            # Draw the object onto a planar surface that automatically reorients
            # toward the camera.
            assert pos is not None, 'Missing argument "pos"'
            assert axis1 is not None, 'Missing argument "axis1"'
            assert axis2 is not None, 'Missing argument "axis2"'

            # Enclose the flat rectangle, which the texture is mapped onto, in a
            # 3D, axis-aligned, bounding box.
            ctr = np.array(pos) + 0.5*(np.array(axis1) + np.array(axis2))  # center of 3D cuboid
            w2 = np.linalg.norm(axis1)/2      # 1/2 width in both X and Y directions
            h2 = np.linalg.norm(axis2)/2      # 1/2 height in Z direction
            self.xctr = ctr[0]
            self.yctr = ctr[1]
            self.zctr = ctr[2]
            self.xhalflen = w2
            self.yhalflen = w2
            self.zhalflen = h2

            # Position (relative to ctr) of lower left corner of texture image.
            corner = np.array([-w2, 0, -h2])

            ftxt = WorldObj.GetTexture(inctags=set(TagTypeID(objtags)), exctags='label')  # front texture
            if ftxt is None:
                raise Exception('No color texture for object {}'.format(objtags))
            ftags = ftxt['tags']
            otype, oid = TagTypeID(ftags)
            self.class_id = int(oid)                    # within-class object ID

            if 'noshdw' in ftags or self.zctr - h2 > 0.3:
                # No shadow tag, or object is off the ground.
                self.shadow = False

            if 'cr' in ftags:
                # Adjust the horizontal center of rotation.
                cr = float(TagValue(ftags, 'cr'))
                dx = 2*(cr - 0.5)*w2
                ctr[0] += dx
                corner[0] -= dx
            if motiontype != Motion.Static and 'spd' in ftags:
                # Save the range of allowed speeds (in meters/sec.) of the object.
                speedrange = TagValue(ftags, 'spd')
                if ',' in speedrange:
                    smin, smax = speedrange.split(',')
                else:
                    smin = smax = speedrange
                self.minspeed = float(smin)
                self.maxspeed = float(smax)
            if 'noflip' not in ftags and 'f' not in ftags and np.random.rand() > 0.5:
                tflip = True               # horizontally flip the texture image
            else:
                tflip = False

            mtxt = WorldObj.GetTexture(inctags=set(TagTypeID(objtags))|{'label'})         # label texture
            if mtxt is None:
                raise Exception('No label texture for object {}'.format(objtags))
            self.pct_bb_area = mtxt['pct_bb_area']   # area as % of bounding box

            for w in range(numworlds):
                if worldviews[w] == 'color camera':
                    actor = vtu.make_quad(renderer[w], origin=ctr,
                                       pos=corner, v1=axis1, v2=axis2,
                                       reps=(1,1), texture=ftxt['vtk'], hflip=tflip,
                                       lightcoef=WorldObj.lightcoef, opacity=1.0,
                                       tags=ftags)
                elif worldviews[w] == 'semantic labels':
                    actor = vtu.make_quad(renderer[w], origin=ctr,
                                       pos=corner, v1=axis1, v2=axis2,
                                       reps=(1,1), texture=mtxt['vtk'], hflip=tflip,
                                       tags=objtags, lightcoef=lcflat)
                elif worldviews[w] == 'objectid':
                    # Create a texture image with the object ID only for "person" objects.
                    if otype == 'person':
                        idtxt = semlabel2objid(mtxt['im'], otype2color255[otype], self.id, 0)
                    else:
                        idtxt = semlabel2objid(mtxt['im'], otype2color255[otype], 0, 0)
                    actor = vtu.make_quad(renderer[w], origin=ctr,
                                       pos=corner, v1=axis1, v2=axis2,
                                       reps=(1,1), texture=idtxt, hflip=tflip,
                                       tags=objtags, lightcoef=lcflat)
                else:
                    raise Exception('Unrecognized worldview: "{:s}"'.format(worldviews[w]))
                # actor.ForceOpaqueOn()
                self.actors.append(actor)

        else:
            raise Exception("Don't know how to make object: {}".format(otype))

        if labelobject:
            for w in range(numworlds):
                z = self.zctr + self.zhalflen + 0.1 + np.random.uniform(low=0,high=1)
                vtu.make_text(renderer[w], otype+'-'+str(self.id),
                              textscale=0.3, pos=(self.xctr, self.yctr, z))


    def read_textures(self, texturepath=None, verbosity=0, loadslim=False):
        """
        Read texture images.

        Arguments:
            texturepath: (str) Path to folder containing texture images.

            verbosity: (int) Level of normal output, an int in [0,1,2,3].
            0 is no output. 3 is all output. Default is 0.

            loadslim: (bool) Should the semantic label images be loaded into
            memory for later use in generating object ID world views? Default
            is False.

        Description:
            Texture images may be JPG, JPEG, or PNG. PNG images may have an
            alpha (transparency) channel that enables parts of the image to be
            partially or fully transparent.

            The texture file naming convention is:
                <object tags>.<ext>
            where
                <object tags> is a list of strings, each separated by a "_". For
                example: "person_05_sitting_n=john' has tags "person",
                "05", "sitting", and "n=john".  The tag "n=john" is a "keyed"
                tag with key "n" and value "john".

                <ext> may be any of "jpg", "jpeg", or "png".

            All texture image files are required to include the following two tags.

                "wh=WxH" -- This gives the width and height of the texture in
                world coordinates (meters). This tag is required of all
                textures. These give the size of the texture image when mapped
                onto a surface whose size is specified in the same units.
                Usually, these give the horizontal and vertical size of the
                corresponding world rectangle in meters.

                "NUM" -- The integer NUM gives the texture ID. This is used to
                group related textures. It is saved as the ID attribute of the
                texture.

            See README.md in the texture image folder for a desription of the
            texture images and their file tags.
        """

        WorldObj.verbosity = verbosity

        if texturepath[0] != '/':
            sim_home = os.path.dirname(os.path.realpath(__file__))
            texturepath = pathlib.Path(sim_home, texturepath)

        if not os.path.isdir(texturepath):
            raise Exception('The texture directory "{}" does not exist.'.format(texturepath))

        if WorldObj.verbosity > 1: print('Reading textures from {}...'.format(texturepath))
        if loadslim: print('Loading images for groundtruth IDs...')
        filelist = os.listdir(texturepath)
        filelist.sort()

        for fullname in filelist:
            fname, fext = os.path.splitext(fullname)
            if fext.lower() in ['.jpg', '.jpeg', '.png']:
                fullpath = os.path.join(texturepath, fullname)
                tags = set(fname.lower().split('_'))
                wh = TagValue(tags, 'wh')      # width and height of texture (m)
                if wh == '':
                    # raise Exception('Texture "{}" is missing wh= tag'.format(fullname))
                    continue
                hsize, vsize = [float(n) for n in wh.split('x')]
                if hsize <= 0 or vsize <= 0:
                    raise Exception('Texture horiz./vert. sizes must be > 0: {}'.format(fullpath))
                texture = vtk.vtkTexture()
                # texture.SetMipmap(True)    # Anti-aliasing, available in VTK 8.1.0
                texture.SetInterpolate(False if "label" in tags else True)

                if fext in ['.jpeg', '.jpg']:
                    jpgreader = vtk.vtkJPEGReader()
                    jpgreader.SetFileName(fullpath)
                    texture.SetInputConnection(jpgreader.GetOutputPort())
                else:
                    pngreader = vtk.vtkPNGReader()
                    pngreader.SetFileName(fullpath)
                    texture.SetInputConnection(pngreader.GetOutputPort())

                # Process the texture tags.
                #    1. Update mappings from tags to texture IDs. Each tag is
                #       mapped to a set of texture IDs (ints) that index into
                #       the list of all textures.
                #    2. Each texture may be given a special ID tag used to link
                #       the same object to different texture files.
                #    3. Extract key names of key tags and add the names to the
                #       tags list.
                #    4. Get the width and height of the texture image in world
                #       coordinates (meters).
                #    5. Add a "noflip" tag to building textures with transparent
                #       windows. Flipping would cause the window position to be
                #       incorrect.
                tid = ''
                keytags = set()
                if "building" in tags and TagValue(tags, 'wp') != '':
                    # Don't flip building textures with transparent windows.
                    tags = tags | {"noflip"}
                for t in tags:
                    if "=" in t:
                        t = t[0:t.index("=")]    # get the key name of a key tag
                        keytags = keytags | {t}
                    elif t.isdigit():
                        tid = t                  # special tag: texture ID
                    if t in WorldObj.tag2txtid.keys():
                        WorldObj.tag2txtid[t].add(WorldObj.numtextures)
                    else:
                        WorldObj.tag2txtid[t] = {WorldObj.numtextures}


                if loadslim and 'label' in tags:
                    # Store semantic label images with the texture. These can be
                    # used to create groundtruth data for each object by
                    # replacing the semantic labels with the object's ID. Also,
                    # get the frontal surface area (as a percentage of the
                    # object's bounding box) of the object. This area can be
                    # used to estimate the degree of object occlusion.
                    im = np.array(imageio.imread(fullpath))
                    objclass = TagTypeID(tags)[0]
                    if objclass in label2id.keys():
                        c = np.array(label_colors[label2id[objclass]])
                        numpix = np.sum(np.all(im[:,:,0:3] == c, 2))
                        if numpix == 0:
                            # Object is labeled some other class.
                            numpix = np.sum(np.any(im[:,:,0:3] != 0, 2))
                        area = numpix/(im.shape[0]*im.shape[1])
                    else:
                        # No recognizable object class label in image.
                        numpix = np.sum(np.any(im[:,:,0:3] != 0, 2))
                        area = numpix/(im.shape[0]*im.shape[1])
                else:
                    im = np.zeros([5,5,3],dtype=np.uint8)
                    area = 0

                WorldObj.textures.append({'vtk': texture,
                                          'tags': tags | keytags,
                                          'id': tid,
                                          'numoccur': 0,
                                          'hsize': hsize, 'vsize': vsize,
                                          'file': fullpath,
                                          'im': im,
                                          'pct_bb_area': area})
                WorldObj.numtextures += 1

        if WorldObj.numtextures == 0:
            raise Exception('No texture images were found in {}'.format(texturepath))
        elif WorldObj.verbosity > 0:
            numlabel = len(WorldObj.GetTexture(inctags='label', getall=True))
            numrgb = len(WorldObj.GetTexture(exctags='label', getall=True))
            print('Loaded {} texture images ({:d} color, {:d} label) from {}'.format(
                  WorldObj.numtextures, numrgb, numlabel, texturepath))

        return


    def read_sounds(self, soundpath=None, verbosity=0):
        """
        Read sound files.

        Arguments:
            soundpath: (str) Path to folder containing sound files.

            verbosity: (int) Level of normal output, an int in [0,1,2,3].
            0 is no output. 3 is all output. Default is 0.

        Description:
            Sound files must be in the WAV format.

            The sound file naming convention is:
                <object tags>.<ext>
            where
                <object tags> is a list of strings, each separated by a "_". For
                example: "drone_quadrotor_01.wav" has tags "drone", "quadrotor",
                and "01".

                <ext> must be "wav".
        """
        WorldObj.verbosity = verbosity

        if soundpath[0] != '/':
            sim_home = os.path.dirname(os.path.realpath(__file__))
            soundpath = pathlib.Path(sim_home, soundpath)

        if not os.path.isdir(soundpath):
            raise Exception('The sound directory "{}" does not exist.'.format(soundpath))

        if WorldObj.verbosity > 1:
            print('Reading sounds from {}...'.format(soundpath))

        filelist = os.listdir(soundpath)
        filelist.sort()

        for fullname in filelist:
            fname, fext = os.path.splitext(fullname)
            if fext.lower() in ['.wav']:
                fullpath = os.path.join(soundpath, fullname)
                tags = set(fname.lower().split('_'))

                wav = wavio.read(fullpath)
                signal = wav.data.squeeze()
                nsamples = signal.shape[0]           # number of samples
                nchannels = signal.ndim              # number of channels (mono or stereo?)
                samplerate = wav.rate                # samples/sec (Hz)
                samplebits = 8*wav.sampwidth         # bits/sample
                amp_max = 2**samplebits              # max amplitude
                duration = nsamples/samplerate       # duration of signal (seconds)
                signal = signal.astype(float)
                if nchannels > 1:
                    signal = signal[:,0]             # keep only one channel
                    nchannels = 1

                if False:
                    # Plot the original signal.
                    plt.figure(figsize=(12,4))
                    # plt.cla()
                    plt.title(f'{samplerate} Hz (blue) and {audio_sample_rate} Hz (red)')
                    plt.plot(signal, 'b-')
                    t_max = max(nsamples, int(np.ceil(duration*audio_sample_rate)))
                    plt.xlim(0, t_max)
                    plt.ylim(-amp_max, amp_max)
                    plt.draw()
                    plt.pause(0.1)

                # Normalize the signal.
                soundtags = tags & set(audio_normalize.keys())
                if len(soundtags) > 0:
                    s = audio_normalize[list(soundtags)[0]]
                else:
                    s = 0.5
                mag = max(abs(signal.min()), abs(signal.max()))
                signal = s*WorldObj.audio_max*(signal/mag)

                # Resample the signal to a fixed sample rate.
                xp = np.arange(0, nsamples)    # coordinates of original samples
                nsamples_new = int(np.ceil(duration*audio_sample_rate))
                x = np.linspace(0, nsamples-1, num=nsamples_new)     # coordinates of new samples
                signal = np.interp(x, xp, signal)
                nsamples = nsamples_new
                samplerate = audio_sample_rate

                if False:
                    # Plot the resampled signal.
                    plt.plot(signal, 'r-', alpha=0.5)
                    plt.draw()
                    plt.show()
                    plt.close()

                # Process the sound tags.
                #    1. Update mappings from tags to sound IDs. Each tag is
                #       mapped to a set of sound IDs (ints) that index into
                #       the list of all sounds.
                #    2. Each sound may be given a special ID tag used to link
                #       the same object to different sound files.
                #    3. Extract key names of key tags and add the names to the
                #       tags list.
                sid = ''
                keytags = set()
                for t in tags:
                    if t.isdigit():
                        continue               # don't need to store number tags
                    elif "=" in t:
                        t = t[0:t.index("=")]    # get the key name of a key tag
                        keytags = keytags | {t}
                    elif t.isdigit():
                        sid = t                  # special tag: sound ID
                    if t in WorldObj.tag2soundid.keys():
                        WorldObj.tag2soundid[t].add(WorldObj.numsounds)
                    else:
                        WorldObj.tag2soundid[t] = {WorldObj.numsounds}

                WorldObj.sounds.append({'signal': signal,
                                        'tags': tags | keytags,
                                        'id': sid,
                                        'numoccur': 0,
                                        'file': fullpath,
                                        'nsamples': nsamples,
                                        'nchannels': nchannels,
                                        'samplerate': samplerate,
                                        'samplebits': samplebits,
                                        'duration': duration})
                WorldObj.numsounds += 1

        if WorldObj.numsounds == 0:
            raise Exception('No sound files were found in {}'.format(soundpath))
        elif WorldObj.verbosity > 0:
            print('Loaded {} sound files from {}'.format(WorldObj.numsounds,
                                                         soundpath))
            print(f'Resampled audio signals to {audio_sample_rate} Hz')

        return




    def GetTexture(inctags=None, exctags=None, getall=False):
        """
        Get a random texture that includes or excludes specific tags.

        Usage:
            txt = GetTexture(inctags=None, exctags=None, getall=False)

        Arguments:
            inctags: (set or str) Set or string of tags that must be included.
            If a string, tags must be seperated by an underscore ('_').

            exctags: (set or str) Set or string of tags that must be excluded.
            If a string, tags must be seperated by an underscore ('_').

            getall: (bool) Get all matching textures?  Default is False. If True,
            then a list of textures is returned instead of a single texture.

        Returns:
            txt: (texture or list) Returns a single texture, or a list of
            textures, or None if no textures match the tag conditions. Each
            texture is a dict with keys ['vtk', 'tags', 'id', 'hsize', 'vsize'].
        """

        matched = set(range(WorldObj.numtextures))
        if type(inctags) == str: inctags = inctags.lower().split('_')
        if type(exctags) == str: exctags = exctags.lower().split('_')

        # Get IDs of textures that include all tags in `inctags`.
        if inctags != None:
            for tag in inctags:
                if tag == '':
                    continue
                elif tag in WorldObj.tag2txtid:
                    matched = matched & WorldObj.tag2txtid[tag]
                else:
                    return None

        # Remove IDs of all textures that have tags in `exctags`.
        if exctags != None:
            for tag in exctags:
                if tag == '':
                    continue
                elif tag in WorldObj.tag2txtid:
                    matched = matched - WorldObj.tag2txtid[tag]

        if matched == set():
            # No textures matched all conditions.
            return None
        elif getall:
            # Return all matching textures in a list.
            return [WorldObj.textures[k] for k in matched]
        else:
            # Return one randomly selected matching texture.
            if len(matched) == 1:
                return WorldObj.textures[list(matched)[0]]
            else:
                idx = np.random.randint(0, len(matched))
                return WorldObj.textures[list(matched)[idx]]


    def dist2d(self, rect):
        """
        Get the 2D Manhatten distance from object to the given rectangle.

        Arguments:
            'rect': A tuple or list, (xctr, yctr, xhalflen, yhalflen),
                describing the rectangle.
        """
        xc, yc, xr, yr = rect
        if abs(self.xctr - xc) <= self.xhalflen + xr:
            dx = 0
        else:
            dx = abs(self.xctr - xc) - self.xhalflen - xr
        if abs(self.yctr - yc) <= self.yhalflen + yr:
            dy = 0
        else:
            dy = abs(self.yctr - yc) - self.yhalflen - yr
        return dx + dy


    def dist2w(self, obj):
        """
        Get the 2D Manhatten distance between the current object's rectangle and
        a different object's rectangle.

        Arguments:
            obj: Description of the 2nd object. This may be a WorldObj or a
            list or tuple. A list or tuple should be of the form (xctr, yctr,
            xhalflen, yhalflen) describing the rectangle.
        """
        if type(obj) is WorldObj:
            xc = obj.xctr
            yc = obj.yctr
            xr = obj.xhalflen
            yr = obj.yhalflen
        else:
            xc, yc, xr, yr = obj

        if abs(self.xctr - xc) <= self.xhalflen + xr:
            dx = 0
        else:
            dx = abs(self.xctr - xc) - self.xhalflen - xr
        if abs(self.yctr - yc) <= self.yhalflen + yr:
            dy = 0
        else:
            dy = abs(self.yctr - yc) - self.yhalflen - yr
        return dx + dy


    def closestBBPnt(self, x, y, dst2edge=0):
        """
        Get the closest 2D point on the XY-bounding box of the object.

        Usage:
            xy = obj.closestBBPnt(x, y, dst2edge=0)

        Arguments:
            x, y: (x,y) is the point inside or outside the object's XY bounding
            box to which the closest point on the bounding box is desired.

            dst2edge: (float) The distance of the returned position from the
            edge of the bounding box. Positive values give positions outside the
            box and negative values give positions inside the box.

        Returns:
            xy: A 1D Numpy array giving the point (x',y') on the object's XY
            bounding box closest to (x, y).
        """

        # There are 9 possible regions around the rectangle (like a tic-tac-toe
        # board) for the query point to lie. (The notes below assume a
        # right-handed coordinate system.)
        if x < self.xctr - self.xhalflen:
            x2 = self.xctr - self.xhalflen
            if y < self.yctr - self.yhalflen:
                y2 = self.yctr - self.yhalflen    # lower left
            elif y > self.yctr + self.yhalflen:
                y2 = self.yctr + self.yhalflen    # upper left
            else:
                y2 = y                            # vert. center left
        elif x > self.xctr + self.xhalflen:
            x2 = self.xctr + self.xhalflen
            if y < self.yctr - self.yhalflen:
                y2 = self.yctr - self.yhalflen    # lower right
            elif y > self.yctr + self.yhalflen:
                y2 = self.yctr + self.yhalflen    # upper right
            else:
                y2 = y                            # center right
        else:
            x2 = x
            if y < self.yctr - self.yhalflen:
                y2 = self.yctr - self.yhalflen    # lower horiz. center
            elif y > self.yctr + self.yhalflen:
                y2 = self.yctr + self.yhalflen    # upper horiz. center
            else:
                # Point is inside bounding box.
                dx = min(abs(self.xctr-self.xhalflen-x), abs(self.xctr+self.xhalflen-x))
                dy = min(abs(self.yctr-self.yhalflen-y), abs(self.yctr+self.yhalflen-y))
                if dx <= dy:
                    # Move in x-direction to closest edge.
                    y2 = y
                    if x <= self.xctr:
                        x2 = self.xctr - self.xhalflen
                    else:
                        x2 = self.xctr + self.xhalflen
                else:
                    # Move in y-direction to closest edge.
                    x2 = x
                    if y <= self.yctr:
                        y2 = self.yctr - self.yhalflen
                    else:
                        y2 = self.yctr + self.yhalflen

        xy = np.array((x2, y2))          # the point on the edge of the box

        if abs(dst2edge) > 0:
            # Move inside or outside the box by `dst2edge`.
            v = xy - np.array((self.xctr, self.yctr))
            xy = xy + dst2edge*v/np.linalg.norm(v)

        return xy


    def occlusion(self, campos:numpy.ndarray, objpos:numpy.ndarray,
                  hfov:float, vfov:float, imshape:tuple, objarea:float) -> float:
        """
        Estimate the degree of occlusion (in [0,1]) of an object.

        Usage:
            ocl = WorldObj.occlusion(cam:PTZCamera, depth:float, area:float,
                                     perimlen:float)

        Arguments:
            cam: (PTZCamera) The camera that generated the image.

            depth: (float) True depth (in meters) of the object from the camera.

            area: (float) The area in the image (in pixels) of the object.

            imshape: (tuple) The shape, (nrows, ncols), of the image.

        Returns:
            ocl: (float) The fraction of the object that is occluded. This is
            estimated as:
                1 - (actual area of object)/(predicted area of object)

        Notes:
            This method uses the approxi
            mation that all pixels in an image have
            the same angular size.
        """

        viewdir = objpos - campos          # vector from camera to object center
        dist3d = np.linalg.norm(viewdir)   # 3D camera-to-object distance
        if dist3d < 1e-3:
            print('Object {:s}_{:d} is too close to camera'.format(self.type,
                                                                   self.class_id))
            return 1.0

        # Get the true width and height of the object (in meters).
        w = 2*self.xhalflen
        h = 2*self.zhalflen

        # Get the horizontal & vertical angles (in degrees) subtended by the
        # object, assuming no occlusion.
        dist2d = np.linalg.norm(viewdir[:2])  # camera-to-object distance on ground plane
        hangle = 2*np.rad2deg(np.arctan(w/(2*dist3d)))
        theta1 = np.rad2deg(np.arctan((self.zctr-self.zhalflen)/dist2d))
        theta2 = np.rad2deg(np.arctan((self.zctr+self.zhalflen)/dist2d))
        vangle = theta2 - theta1

        assert self.zctr-self.zhalflen >= 0
        assert theta2 > theta1

        # Get the expected size in the image (in pixels) of the object's
        # bounding box, assuming no occlusion.
        nrows = (vangle/vfov)*imshape[0]
        ncols = (hangle/hfov)*imshape[1]

        # Predicted area (in pixels) of object in image.
        predarea = max(1, self.pct_bb_area*nrows*ncols)

        # Visibility of the object is the ratio of actual to predicted area.
        vis = min(1, objarea/predarea)

        return 1 - vis


    def occlusion_old(self, cam:PTZCamera, depth:float, area:float,
                  perimlen:float) -> float:
        """
        Estimate the degree of occlusion of an object.

        Usage:
            ocl = WorldObj.occlusion(cam:PTZCamera, depth:float, area:float,
                                     perimlen:float)

        Arguments:
            cam: (PTZCamera) The camera that generated the image.

            depth: (float) True depth (in meters) of the object from the camera.

            area: (float) The area in the image (in pixels) of the object.

            perimlen: (float) The length in the image (in pixels) of the
            object's perimeter.

        Returns:
            ocl: (float) The fraction of the object that is occluded. This is
            estimated as:
                1 - (actual area of object)/(predicted area of object)

        Notes:
            This method uses the approximation that all pixels in an image have
            the same angular size.
        """
        raise Exception('You should be using the newer version of this function')

        assert depth > 0, "Object depth must be > 0"
        assert area > 0, "Object area must be > 0"

        # Get the true width and height of the object (in meters).
        w = 2*self.xhalflen
        h = 2*self.zhalflen

        # Get the horizontal & vertical angles (in degrees) subtended by the
        # object, assuming no occlusion.
        hangle = np.rad2deg(np.arctan(w/depth))
        vangle = np.rad2deg(np.arctan(h/depth))

        # Get the size in the image (in pixels) of the object's predicted
        # bounding box, assuming no occlusion.
        ncols = hangle*cam.horizres
        nrows = vangle*cam.vertres

        # Predicted area (in pixels) of object in image.
        predarea = max(1, self.pct_bb_area*nrows*ncols)

        # Visibility of the object is the ratio of actual to predicted area. Add
        # the object's perimeter length to its area to account for object ID
        # filtering (in rgbid2uint()) and discretization of the image.
        vis = min(1, (area + 2*perimlen)/predarea)

        return 1 - vis


    def make_fence(self, renderer, worldviews, rect3d, ftxt, tags):
        # Create a fence. The long edge of the fence is assumed to be
        # parallel to the ground plane. Arguments are:
        #    pos: one bottom corner (3D) of fence quadrilateral.
        #    axis1: 3D vector from bottom corner parallel to fence
        #           horizontal direction (parallel to ground plane).
        #    axis2: 3D vector from bottom corner parallel to fence vertical
        #           direction.
        # The fence texture image is mapped onto the fence quadrilateral in
        # a left-to-right direction. If the fence texture image has a p=
        # tag, it is interpreted as p=ID_W_H_Dparll_Dperp_S where
        #    ID is the texture ID of a pole texture image (i.e., pole_ID.png),
        #    W and W are the width and height of the fence poles,
        #    Dparll is the relative pole center in the fence parallel direction,
        #    Dperp is the relative pole center in the fence perpindicular direction,
        #    S is the inter-pole spacing.

        numworlds = len(renderer)
        lcflat = (1.0, 0.0, 0.0)  # Lighting coef. (ambient, diffuse, specular) for flat colors.

        mtxt = WorldObj.GetTexture(inctags=set(TagTypeID(tags))|{'label'})  # texture label
        if mtxt is None:
            raise Exception('No label texture for object {}'.format(tags))

        # Enclose the quadrilateral that the texture is mapped onto in a
        # 3D, axis-aligned, bounding box.
        self.xctr = rect3d[0]
        self.yctr = rect3d[1]
        self.zctr = rect3d[2]
        self.xhalflen = rect3d[3]
        self.yhalflen = rect3d[4]
        self.zhalflen = rect3d[5]

        if rect3d[3] > rect3d[4]:
            # Long edge of fence is aligned with x-axis.
            length = 2*rect3d[3]
            pos = (rect3d[0]-rect3d[3], rect3d[1], 0)   # position of botm. ctr. of 1st pole
            axis1 = np.array([length, 0, 0])            # aligns w/ long edge of fence
        else:
            # Long edge of fence is aligned with y-axis.
            length = 2*rect3d[4]
            pos = (rect3d[0], rect3d[1]-rect3d[4], 0)   # position of botm. ctr. of 1st pole
            axis1 = np.array([0, length, 0])            # aligns w/ long edge of fence

        hgt = 2*rect3d[5]
        axis2 = np.array([0, 0, hgt])
        reps = (length/ftxt['hsize'], hgt/ftxt['vsize'])
        self.txt_reps = reps

        if 'p' in tags:
            # Get fence pole position and spacing.  The pole tag ("p=") on the
            # fence is ID x WIDTH x HEIGHT x Dparll x Dperp x SPACING
            numpoles = int(reps[0]) + 1
            pID, pW, pH, pDparll, pDperp, pS = [n for n in TagValue(tags,'p').split('x')]
            pW, pH, pDparll, pDperp, pS = [float(n) for n in [pW, pH, pDparll, pDperp, pS]]
            ptxt = WorldObj.GetTexture(inctags=set(['pole',pID]), exctags='label')
            if ptxt is None:
                raise Exception('No texture for object {}'.format(['pole',pID]))
            square_pole = True if 's' in ptxt['tags'] else False
            ptW = ptxt['hsize']
            ppos = np.array([*pos[0:2],0])   # center bottom of 1st pole centered on end of fence
            if abs(axis1[0]) > abs(axis1[1]):
                # Fence parallel to x-axis.
                ppos[1] += pDperp*np.random.choice([1,-1])   # perpindicular offset
                ppos[0] += pDparll                           # parallel offset
                pspace = np.array([pS,0,0])                  # inter-pole spacing
            else:
                # Fence parallel to y-axis.
                ppos[0] += pDperp*np.random.choice([1,-1])   # perpindicular offset
                ppos[1] += pDparll                           # parallel offset
                pspace = np.array([0,pS,0])                  # inter-pole spacing
        else:
            # No fence poles.
            numpoles = 0

        for w in range(numworlds):
            if worldviews[w] == 'color camera':
                # Insert the fence as one quadrilateral.
                vtu.make_quad(renderer[w], origin=pos, v1=axis1, v2=axis2,
                              reps=reps, texture=ftxt['vtk'], tags=tags,
                              lightcoef=WorldObj.lightcoef, opacity=1.0)

                # Insert fence poles.
                pp = ppos.copy()
                if square_pole:
                    for _ in range(numpoles):
                        vtu.make_cyl(renderer[w], origin=pp, width=ptW, height=pH,
                                     radius=pW/2,
                                     textures=[ptxt['vtk']], reps=(1,1,1),
                                     lightcoef=WorldObj.lightcoef, opacity=1.0,
                                     tags=ptxt['tags'])
                        pp += pspace
                else:
                    for _ in range(numpoles):
                        p = [pp[0], pp[1], pH/2, pW/2, pH/2]
                        vtu.make_cylinder(renderer[w], param=p, color=None,
                                          texture=ptxt['vtk'], scale=(1,1,1),
                                          lightcoef=WorldObj.lightcoef, cap=False,
                                          res=20, camera=None, tags=ptxt['tags'])
                        pp += pspace
            elif worldviews[w] == 'semantic labels':
                # Insert the fence as one quadrilateral.
                vtu.make_quad(renderer[w], origin=pos, v1=axis1, v2=axis2,
                              reps=reps, texture=mtxt['vtk'], tags=tags,
                              lightcoef=lcflat)

                # Insert fence poles.
                pp = ppos.copy()
                if square_pole:
                    for _ in range(numpoles):
                        vtu.make_cyl(renderer[w], origin=pp, radius=pW/2,
                                     width=ptW, height=pH, colors = [otype2color1['barrier']],
                                     lightcoef=lcflat, opacity=1.0, tags=ptxt['tags'])
                        pp += pspace
                else:
                    for _ in range(numpoles):
                        p = [pp[0], pp[1], pH/2, pW/2, pH/2]
                        vtu.make_cylinder(renderer[w], param=p, color=otype2color1['barrier'],
                                          lightcoef=lcflat, cap=False,
                                          res=20, camera=None, tags=ptxt['tags'])
                        pp += pspace
            elif worldviews[w] == 'objectid':
                # We don't record the fence's ID, so recolor the fence's semantic label
                # image with (0,0,0) values.
                idtxt = semlabel2objid(mtxt['im'], otype2color255['barrier'], 0, 0)

                # Insert the fence as one quadrilateral.
                vtu.make_quad(renderer[w], origin=pos, v1=axis1, v2=axis2,
                              reps=reps, texture=idtxt, tags=tags, lightcoef=lcflat)

                # Insert fence poles.
                pp = ppos.copy()
                if square_pole:
                    for _ in range(numpoles):
                        vtu.make_cyl(renderer[w], origin=pp, radius=pW/2,
                                     width=ptW, height=pH, colors = [(0,0,0)],
                                     lightcoef=lcflat, opacity=1.0, tags=ptxt['tags'])
                        pp += pspace
                else:
                    for _ in range(numpoles):
                        p = [pp[0], pp[1], pH/2, pW/2, pH/2]
                        vtu.make_cylinder(renderer[w], param=p, color=(0,0,0),
                                          lightcoef=lcflat, cap=False,
                                          res=20, camera=None, tags=ptxt['tags'])
                        pp += pspace
            else:
                raise Exception('Unrecognized worldview:'.format(worldviews[w]))


    def make_wall(self, renderer, worldviews, rect3d, ftxt, tags):

        # Create a wall. The long edge of the wall is assumed to be
        # parallel to the ground plane. Arguments are:
        #    pos: one bottom corner (3D) of wall quadrilateral.
        #    axis1: 3D vector from bottom corner parallel to wall
        #           horizontal direction (parallel to ground plane).
        #    axis2: 3D vector from bottom corner parallel to wall vertical
        #           direction.
        # The wall texture image is mapped onto the quadrilateral in
        # a left-to-right direction.

        numworlds = len(renderer)
        lcflat = (1.0, 0.0, 0.0)  # Lighting coef. (ambient, diffuse, specular) for flat colors.

        mtxt = WorldObj.GetTexture(inctags=set(TagTypeID(tags))|{'label'})  # texture label
        if mtxt is None:
            raise Exception('No label texture for object {}'.format(tags))

        # Enclose the quadrilateral that the texture is mapped onto in a
        # 3D, axis-aligned, bounding box.
        self.xctr = rect3d[0]
        self.yctr = rect3d[1]
        self.zctr = rect3d[2]
        self.xhalflen = rect3d[3]
        self.yhalflen = rect3d[4]
        self.zhalflen = rect3d[5]

        # One corner of wall and vectors in 3 perpindicular directions.
        corner = [rect3d[0]-rect3d[3], rect3d[1]-rect3d[4], rect3d[2]-rect3d[5]]
        v1 = [2*rect3d[3], 0, 0]
        v2 = [0, 2*rect3d[4], 0]
        v3 = [0, 0, 2*rect3d[5]]

        # Texture repitition count in the V1, V2, and V3 directions.
        reps = (2*rect3d[3]/ftxt['hsize'], 2*rect3d[4]/ftxt['hsize'], 2*rect3d[5]/ftxt['vsize'])
        self.txt_reps = reps

        for w in range(numworlds):
            if worldviews[w] == 'color camera':
                vtu.make_box(renderer[w], origin=corner, v1=v1, v2=v2, v3=v3,
                             reps=reps, color=None, texture=ftxt['vtk'],
                             lightcoef=WorldObj.lightcoef, opacity=1.0, tags=tags)
            elif worldviews[w] == 'semantic labels':
                vtu.make_box(renderer[w], origin=corner, v1=v1, v2=v2, v3=v3,
                             reps=reps, color=None, texture=mtxt['vtk'],
                             lightcoef=lcflat, opacity=1.0, tags=tags)
            elif worldviews[w] == 'objectid':
                # We don't record the wall's ID, so recolor the wall's semantic label
                # image with (0,0,0) values.
                idtxt = semlabel2objid(mtxt['im'], otype2color255['barrier'], 0, 0)
                vtu.make_box(renderer[w], origin=corner, v1=v1, v2=v2, v3=v3,
                             reps=reps, color=None, texture=idtxt,
                             lightcoef=lcflat, opacity=1.0, tags=tags)
            else:
                raise Exception('Unrecognized worldview:'.format(worldviews[w]))


class SimWorld:
    """
    Class to simulate 3D outdoor environments.
    """

    def __init__(self, env_radius=100, bkgtags='background', gndtags='ground',
                 skytags='sky', roadtags='road',
                 bldg_density=0.5, road_density=5, plant_density=0.5,
                 people_density=0.5, animal_density=0.2, clutter_density=0.2,
                 vehicle_density=0.5, airborne_density=0.5, bldg_plant_density=0.5,
                 barrier_density=0.5, gndfeat_density=0.5, lookouts=True,
                 probwindowoccupied=0.25, verbosity='low', views={'color'},
                 bldg_maxheight=30, road_min_sep=100, textures=None, sounds=None,
                 timeofday=None, offscreenrender=False, model_tweeks=None,
                 rand_seed=None, origin_axes=False, imsize=(800,600),
                 dynamic_env=True, pathsfile=None, show_obj_ids=False, envtype='urban',
                 p_over_road={'person':0.1, 'clutter':0.1, 'animal':0.2, 'vehicle':0.5},
                 p_over_building={'person':0.5, 'clutter':0.2, 'animal':1.0, 'vehicle':0}):

        """
        Create a new 3D world.

        Arguments:
            envtype: (str) A string specifying the type of environment to
            create. Possible values are {'urban', 'forest', 'random'}. The
            default value is 'urban'.

            env_radius: The radius of the environment to create (meters).

            road_density: (float) The density of roads in the created world.
            This value must be non-negative. The number of roads that the
            simulator attempts to insert into the world model is proportional,
            by the value of this parameter, to the area of the world model. A
            value of 0 will result in no roads being created. A value of 1 is
            considered normal road density. If this value is too high, road
            creation may take a long time as the process of searching for places
            to locate roads is not very sophisticated.

            building_density: (float) The density of buildings in the created
            world. This value must be non-negative. The number of buildings
            that the simulator attempts to insert into the world model is
            proportional, by the value of this parameter, to the total length of
            the roads inserted into the world model. A value of 0 will result in
            no buildings being created. A value of 1 is considered normal
            building density. If this value is too high, building creation may
            take a long time as the process of searching for places to locate
            buildings is not very sophisticated.

            vehicle_density: (float) The density of vehicles in the created world.
            This value must be non-negative. The number of vehicles that the
            simulator attempts to insert into the world model is proportional,
            by the value of this parameter, to the area of the world model. A
            value of 0 will result in no vehicles being created. A value of 1 is
            considered normal vehicles density.

            plant_density: (float) The density of plants in the created world.
            This value must be non-negative. The number of plants that the
            simulator attempts to insert into the world model is proportional,
            by the value of this parameter, to the area of the world model. A
            value of 0 will result in no plants being created. A value of 1 is
            considered normal plant density.

            bldg_plant_density: (float) The density of "landscaping" plants
            around buildings in the created world. This value must be
            non-negative. Building plants are relatively small, landscaping
            plants (e.g., flowers and bushes) that are typically found around
            buildings. The number of plants next to buildings that the simulator
            attempts to insert into the world model is proportional, by the
            value of this parameter, to the total perimeter of the buildings
            previously inserted into the world model. A value of 0 will result
            in no building plants being created. A value of 1 is considered
            normal building plant density.

            people_density: (float) The density of people in the created world.
            This value must be non-negative. The number of people that the
            simulator attempts to insert into the world model is proportional,
            by the value of this parameter, to the area of the world model.
            People may appear on the ground, or on tops of buildings, or inside
            of buildings looking out windows (see below). A value of 0 will
            result in no people being created. A value of 1 is considered normal
            people density. Some buildings may have transparent windows (as
            described by texture file tags) which people can look out of. The
            people that look out of windows are not included in the number of
            people created according to the people_density parameter. Currently,
            for each building with transparent windows that is inserted into the
            world, one person will look out a window on each of the four sides
            of the building.

            barrier_density: (float) The density of barriers in the created
            world. This value must be non-negative. A barrier is a fence or
            wall. Fences are thin and include fence poles while walls don't
            include poles but have greater thickness. The number of barriers
            that the simulator attempts to insert into the world model is
            proportional, by the value of this parameter, to the area of the
            world model. A value of 0 will result in no barriers being created.
            A value of 1 is considered normal barrier density.

            gndfeat_density: (float) The density of ground features in the
            created world. This value must be non-negative. Ground features
            are textures (e.g., ponds, leaves and sticks, tire tracks, etc.)
            that are overlayed on the ground plane to add variety to the ground
            plane. The number of ground features that the simulator attempts to
            insert into the world model is proportional, by the value of this
            parameter, to the area of the world model. A value of 0 will result
            in no ground features being created. A value of 1 is considered
            normal ground feature density.

            airborne_density: (float) The density of airborne objects (e.g.,
            drones, birds, butterflies, etc.) in the created world. This value
            must be non-negative. The number of airborne objects that the
            simulator attempts to insert into the world model is proportional,
            by the value of this parameter, to the area of the world model. A
            value of 0 will result in no airbornes being created. A value of 1
            is considered normal airborne density.

            clutter_density: (float) The density of clutter in the created
            world. This value must be non-negative. Clutter can be any type of
            object that can be supported by a horizontal surface. The number of
            clutter objects that the simulator attempts to insert into the world
            model is proportional, by the value of this parameter, to the area
            of the world model. A value of 0 will result in no clutter being
            created. A value of 1 is considered normal clutter density.

            lookouts: (bool) Should the environement include people looking out
            of building windows? Default is True.

            probwindowoccupied: (float) Probability (in [0,1]) that any
            particular building window is occupied by a person looking out.
            Default is 0.25.

            bldg_maxheight: (float) The maximum height (in meters) of any
            building.

            road_min_sep: (float) The minimum distance (in meters) between
            parallel roads. Currently, all roads are parallel to either the X or
            Y axis of the world model.

            p_over_building: (dict) For each type of object, OBJ (e.g.,
            "person", "clutter"), p_over_building[OBJ] gives the probability
            that an object of that type appears over or on a building given that
            the specific object is allowed (via texture file tags) to be over a
            building.

            p_over_road: (dict) For each type of object, OBJ (e.g., "person",
            "clutter"), p_over_road[OBJ] gives the probability that an object of
            that type appears over or on a road given that the specific object
            is allowed (via texture file tags) to be over a road.

            rand_seed: (int) Seed for random number generator. This can be used
            to create a specific world model, which may be useful for debugging.
            Default is None.

            vertobj_rand_seed: (int) Seed to restart random number generator
            prior to insertion of vertical objects. This enables the user to
            independently reset the static and dynamic parts of the
            environement. Default value is None.

            origin_axes: (bool) Create an axes at the origin of the world?
            Default is False.

            textures: (str) Path to folder containing texture images. If this is
            None, then the folder "textures", if it exists in the simworld
            module, will be used.

            sounds: (str) Path to folder containing sound files. If this is
            None, then the folder "sounds", if it exists in the simworld
            module, will be used.

            bkgtags: (str) Tags to identify the image that will be used to
            create a background scene (panoramic). Default is 'background', in
            which case no background scene will be created.

            roadtags: (str) Tags to identify the image that will be used to
            texture map roads. Default is 'road', in which case a random road
            texture will be used.

            gndtags: (str) Tags to identify the image that will be used to
            texture map the ground plane. Default is 'ground', in which case a
            random ground texture will be used.

            skytags: (str) Tags to identify the image that will be used to
            texture map the sky. Default is 'sky', in which case a random
            sky texture will be used.

            timeofday: (int) Time of day in 24-hour time (0-2399). Default is
            None, in which case a random time of day between 0500 and 2100 is
            selected. If TIMEOFDAY is a list/tuple of two floats, [TIME1, TIME2],
            then the time of day is randomly chosen between those two values.

            imsize: (tuple) Size of rendered images, (width, height), in pixels.
            Due to VTK limitaions, the size of rendered images cannot exceed the
            machine screen size, unless VTK uses off-screen rendering.

            offscreenrender: (bool) Should VTK render the environment
            off-screen? Default is False. With on-screen rendering, the maximum
            size of a rendered image is limited to the screen resolution (minus
            the window border). With off-screen rendering, there are no image
            size limitations (within reason). Also, off-screen rendering may be
            desirable when there are multiple cameras, to avoid screen clutter.
            With off-screen rendering, VTK still creates rendering windows, but
            these windows are empty.

            dynamic_env: (bool) Is the environment dynamic? Default is True.
            Some objects, such as people, animals, and airborne, may move if
            their speed tags (spd=min,max) are greater than zero. The
            "dynamic_env" argument must be True to allow these objects to move.

            pathsfile: (str) Name of text file that defines objects moving along
            fixed paths. The format of this file is:
                <object_1_tags>
                <Time_0> <X_0> <Y_0> <Z_0>
                <Time_1> <X_1> <Y_1> <Z_1>
                ...
                <Time_N> <X_N> <Y_N <Z_N>
                END

            show_obj_ids: (bool) Should object IDs be overlayed on the color
            images? Default is False. This may be useful for debugging code, but
            will probably cause problems for the object detector if it's
            running.

            verbosity: (str) Level of information provided during normal
            operations, one of {'off', 'low', 'medium', 'high'}. Default is
            'low'

            model_tweeks: (dict) Dictionary of random seeds to use for each part
            of the model generation process. Recognized keys include the
            following: 'light', 'backgnds', 'roads', 'signs', 'buildings',
            'barriers', 'gndfeats', 'plants', 'persons', 'lookouts', 'clutter',
            'animals', 'bldgplants', 'urbananimals', 'airborne', 'shadows'.
            These enable parts of the model to be changed independently of
            other parts. Defauls is None.

            views: (list/set) List or set of world views, each a string, that
            should be rendered. Valid world views include 'color', 'label', and
            'objid'. 'color' is the view from the color camera. 'label' is the
            corresponding semantic scene labels. 'objid' is the corresponding
            object ID groundtruth. The more different views that are rendered,
            the slower the overall processing pipeline will be. Default is
            {'color'}
        """

        # Save environment parameters.
        self.env_radius = env_radius
        self.road_density = road_density
        self.plant_density = plant_density
        self.bldg_plant_density = bldg_plant_density
        self.bldg_density = bldg_density
        self.gndfeat_density = gndfeat_density
        self.barrier_density = barrier_density
        self.airborne_density = airborne_density
        self.people_density = people_density
        self.animal_density = animal_density
        self.clutter_density = clutter_density
        self.vehicle_density = vehicle_density
        self.lookouts = lookouts
        self.probwindowoccupied = probwindowoccupied
        self.rand_seed = rand_seed
        self.offscreenrender = offscreenrender
        self.dynamic_env = dynamic_env
        self.pathsfile = pathsfile
        self.show_obj_ids = show_obj_ids
        self.envtype = envtype.lower()
        self.bkgtags = bkgtags
        self.gndtags = gndtags
        self.skytags = skytags
        self.roadtags = roadtags
        self.building_maxheight = bldg_maxheight
        self.road_min_sep = road_min_sep
        self.textures = textures
        self.sounds = sounds
        self.timeofday = timeofday
        self.origin_axes = origin_axes
        self.imsize = imsize
        self.p_over_road = p_over_road
        self.p_over_building = p_over_building
        self.verbosity = verbosity
        self.model_tweeks = model_tweeks
        self.views = views
        self.audio_lines = []

        # Convert the verbosity string to a verbosity level in [0, 1, 2, 3].
        if self.verbosity.lower() not in {'off','low','medium','high'}:
            raise ValueError('**Verbosity**' + \
                             ' must be one of {"off", "low", "medium", "high"}')
        self.verbosity = ['off','low','medium','high'].index(self.verbosity.lower())

        if self.envtype not in {'urban', 'forest'}:
            raise ValueError('Unrecognized ENVTYPE: "{:s}"'.format(self.envtype))

        self.time = 0.0                  # time (sec.) since start of simulation
        self.time_last = 0.0             # time (sec.) of last simulation update
        self.audio_sample_rate = audio_sample_rate

        # The target number (not always achievable) of each type of object.
        self.num_plants = int(self.plant_density*np.pi*(self.env_radius**2)/100)
        self.num_people = int(self.people_density*np.pi*(self.env_radius**2)/500)
        self.num_animals = int(self.animal_density*np.pi*(self.env_radius**2)/500)
        self.num_clutter = int(self.clutter_density*np.pi*(self.env_radius**2)/500)
        self.num_gnd_feat = int(self.gndfeat_density*np.pi*(self.env_radius**2)/500)
        self.num_vehicles = int(self.vehicle_density*np.pi*(self.env_radius**2)/500)
        self.num_airborne = int(self.airborne_density*np.pi*(self.env_radius**2)/500)

        # 3D (multi-channel) map of the environemnt identifies the types of
        # objects, the object IDs, and object heights at each (x,y) location.
        self.map3d = Map3D(gridspacing=0.25, radius=self.env_radius, dynamic=self.dynamic_env)

        # Define the probability of an object being above or on a road or a
        # building given that the object is allowed to be in such a position.
        # These values needn't sum to 1. The class arguments are used to change
        # the default values.
        p_over_road = self.p_over_road
        p_over_building = self.p_over_building
        self.p_over_road = {'airborne':0.5, 'animal':0.2, 'clutter':0.1,
                            'person':0.1,  'plant':0.05, 'vehicle':0.5}
        self.p_over_building = {'airborne':0.5, 'animal':0.2, 'clutter':0.1,
                                'person':0.5, 'plant':0.05, 'vehicle':0}
        for obj in {'person', 'clutter', 'animal', 'airborne'}:
            if obj in p_over_road:
                self.p_over_road[obj] = p_over_road[obj]
            if obj in p_over_building:
                self.p_over_building[obj] = p_over_building[obj]

        # Properties of generic buildings (units in meters)
        self.building_interfloor = 3.6      # Standard interfloor spacing
        self.building_minheight = 3
        self.building_minlen = 5
        self.building_maxlen = 30
        self.building_meanlen = 10
        self.building_mindist2road = 5      # min distance to closest road
        self.building_maxdist2road = 10     # max distance to closest road
        self.building_minsep = 1e-6         # min distance between buildings
        self.building_maxhgt2base = 4       # max aspect ratio of height to base
        assert self.building_minheight >= self.building_minheight

        # Properties of generic roads (units in meters)
        self.road_lane_width = 3.5          # width of one lane of a road
        self.road_sidewalk_width = 2        # width of sidewalk and grass median

        # Properties of generic trees (units in meters).
        self.plant_std = 0.5             # std. from texture file height & width
        self.plant_osc_freq = 0.2        # plant oscillation frequency (cycles/sec)

        # Properties of barriers.
        self.barrier_minlen = 5
        self.barrier_maxlen = 60

        # The world consists of a list of WorldObj objects.
        self.num_objs = 0
        self.objs = []                     # list of all objects
        self.noise_makers = []             # list of objects that can make noise
        self.object_labels = list(label2id.values())

        # Setup the random number generator. Also assign random seeds for each
        # part of the model generation so that each part of the model can be
        # tweeked independently of the other parts.
        if self.rand_seed is None:
            self.rand_seed = int((time.time()*1e7)%1e7) # Generate a new random seed.
        if self.verbosity > 0: print('Random seed = {}'.format(self.rand_seed))
        np.random.seed(self.rand_seed)
        modparts = ['lights', 'backgnds', 'roads', 'signs', 'buildings',
                    'barriers', 'gndfeats', 'plants', 'persons', 'lookouts',
                    'clutter', 'vehicles', 'animals', 'bldgplants',
                    'followpaths', 'urbananimals', 'airborne', 'shadows']
        self.model_seed = dict()
        for name in modparts:
            self.model_seed[name] = np.random.randint(1, 1e7)
        if self.model_tweeks:
            if self.verbosity > 0: print('Model tweeks =', self.model_tweeks)
            for key in self.model_tweeks.keys():
                self.model_seed[key] = self.model_tweeks[key]

        # A number of geometrically identical, but photometrically different,
        # parallel world views may be created. This allows for easier generation
        # of groundtruth data (semantic labels, object IDs).
        self.worldviews = []
        self.view_obj_ids = False
        for v in self.views:
            if v.lower() == 'color':
                self.worldviews.append('color camera')
            elif v.lower() == 'label':
                self.worldviews.append('semantic labels')
            elif v.lower() == 'objid':
                self.worldviews.append('objectid')
                self.view_obj_ids = True
            else:
                raise ValueError('Unrecognized world view: "{}"'.format(v))
        self.numworlds = len(self.worldviews)
        self.idx_labels = self.worldviews.index('semantic labels') if 'semantic labels' in self.worldviews else None
        self.idx_color = self.worldviews.index('color camera') if 'color camera' in self.worldviews else None
        self.idx_objid = self.worldviews.index('objectid') if 'objectid' in self.worldviews else None

        # The timeofday argument is a float in [0,2400) and is converted to a
        # float in [0,24). E.g., the timeofday 1630 is represented as 16.5.
        if self.timeofday:
            if type(self.timeofday) in [list, tuple]:
                if len(self.timeofday) != 2:
                    raise ValueError("TIMEOFDAY argument must be a single float or a length 2 tuple")
                t1 = (self.timeofday[0] // 100) + (self.timeofday[0] % 100)/60
                t2 = (self.timeofday[1] // 100) + (self.timeofday[1] % 100)/60
                self.time_of_day = np.random.uniform(t1, t2)     # 24-hour time
            else:
                self.time_of_day = (self.timeofday // 100) + (self.timeofday % 100)/60
        else:
            self.time_of_day = np.random.uniform(5,21)     # 24-hour time
        if self.verbosity > 0: print('Time of day is {:02d}{:02d}'.format(
                                     int(self.time_of_day),
                                     round(60*(self.time_of_day % 1))))
        r = np.exp(-(self.time_of_day - 12)**4/3000)
        self.amb_light = 0.8*r                         # ambient light coef. (0.8*r)
        self.dif_light = 0.8*r                         # diffuse light coef.
        WorldObj.lightcoef = (self.amb_light, self.dif_light, 0)

        # Read the object texture images.
        if self.textures is None:
            import os
            self.textures = os.path.dirname(__file__) + '/textures'
        WorldObj.read_textures(None, texturepath=self.textures,
                               verbosity=self.verbosity, loadslim=self.view_obj_ids)

        # Read the object sounds.
        if self.sounds is None:
            import os
            self.sounds = os.path.dirname(__file__) + '/sounds'
        WorldObj.read_sounds(None, soundpath=self.sounds, verbosity=self.verbosity)

        self.followers = []               # actors that are following the camera
        self.movers = []                  # objects that may move
        self.renderers = None
        self.init_renderer()

        # Create the world model.

        np.random.seed(self.model_seed["lights"])
        self.insert_lights()

        np.random.seed(self.model_seed["backgnds"])
        self.insert_backgrounds(bkgtags=bkgtags, gndtags=gndtags, skytags=skytags)

        np.random.seed(self.model_seed["roads"])
        self.insert_roads(roadtags=roadtags)

        np.random.seed(self.model_seed["signs"])
        self.insert_signs()

        np.random.seed(self.model_seed["buildings"])
        self.insert_buildings()

        np.random.seed(self.model_seed["barriers"])
        self.insert_barriers(self.barrier_density)

        np.random.seed(self.model_seed["bldgplants"])
        self.num_bldg_plants = int(self.bldg_plant_density*self.bldg_perim)
        self.insert_vert_objs(self.num_bldg_plants, "bldg_plant")

        np.random.seed(self.model_seed["gndfeats"])
        self.insert_horiz_objs(self.num_gnd_feat, "gndfeat")

        np.random.seed(self.model_seed["plants"])
        self.insert_vert_objs(self.num_plants, "plant")

        np.random.seed(self.model_seed["vehicles"])
        self.insert_vert_objs(self.num_vehicles, "vehicle")

        np.random.seed(self.model_seed["clutter"])
        self.insert_vert_objs(self.num_clutter, "clutter")

        np.random.seed(self.model_seed["persons"])
        self.insert_vert_objs(self.num_people, "person")

        np.random.seed(self.model_seed["lookouts"])
        if self.lookouts: self.insert_lookouts(pwoccupied=self.probwindowoccupied)

        np.random.seed(self.model_seed["animals"])
        self.insert_vert_objs(self.num_animals, "animal")

        np.random.seed(self.model_seed["urbananimals"])
        self.insert_vert_objs(2*self.num_animals, "urban_animal")

        np.random.seed(self.model_seed["airborne"])
        self.insert_vert_objs(self.num_airborne, "airborne")

        np.random.seed(self.model_seed["shadows"])
        self.insert_shadows()

        if self.pathsfile:
            np.random.seed(self.model_seed["followpaths"])
            self.insert_fixed_path_objs(self.pathsfile)

        self.interact = True         # allow user interaction with the world?
        self.label_colors = label_colors   # may be needed by window interaction code

        if self.origin_axes:
            # Create a cartesian axis at the origin of the world.
            for w in range(self.numworlds):
                vtu.make_axes(self.renderers[w], self.vtkcamera)


    def new(self, env_radius=None, bkgtags=None, gndtags=None, skytags=None,
            roadtags=None, bldg_density=None, road_density=None,
            plant_density=None, people_density=None, animal_density=None,
            clutter_density=None, vehicle_density=0.5, airborne_density=None,
            bldg_plant_density=None, barrier_density=None, gndfeat_density=None,
            lookouts=None, probwindowoccupied=None, verbosity=None, views=None,
            bldg_maxheight=None, road_min_sep=None, timeofday=None,
            offscreenrender=None, model_tweeks=None, rand_seed=None,
            origin_axes=False, imsize=None, dynamic_env=None, paths=None,
            show_obj_ids=None, envtype=None, p_over_road=None,
            p_over_building=None):
        """
        Create a new environment.

        Usage:
            SimWorld.new(env_radius=None, bkgtags=None, gndtags=None,
                         skytags=None, roadtags=None, bldg_density=None,
                         road_density=None, plant_density=None,
                         people_density=None, animal_density=None,
                         clutter_density=None, vehicle_density=0.5,
                         airborne_density=None, bldg_plant_density=None,
                         barrier_density=None, gndfeat_density=None,
                         lookouts=None, verbosity=None, bldg_maxheight=None,
                         road_min_sep=None, timeofday=None,
                         offscreenrender=None, model_tweeks=None,
                         rand_seed=None, origin_axes=False, imsize=None,
                         dynamic_env=None, show_obj_ids=None, envtype=None,
                         p_over_road=None, p_over_building=None)

        Description:
            The parameters of the new environment are identical to the
            previously created environment except where arguments to
            SimWorld.new() make changes. This does not mean that the
            environments will be identical (see "random_seed" below).

            If the "random_seed" argument is not provided, then the random
            number generator is not reinitialized, so the new environment will
            be different from the previous environment, even if the environment
            parameters are identical.

            There is no "textures" argument to SimWorld.new() as the new
            environment is created using the same set of texture images.
        """

        # Update environment parameters.
        if env_radius: self.env_radius = env_radius
        if road_density: self.road_density = road_density
        if plant_density: self.plant_density = plant_density
        if bldg_plant_density: self.bldg_plant_density = bldg_plant_density
        if bldg_density: self.bldg_density = bldg_density
        if gndfeat_density: self.gndfeat_density = gndfeat_density
        if barrier_density: self.barrier_density = barrier_density
        if airborne_density: self.airborne_density = airborne_density
        if people_density: self.people_density = people_density
        if animal_density: self.animal_density = animal_density
        if clutter_density: self.clutter_density = clutter_density
        if vehicle_density: self.vehicle_density = vehicle_density
        if lookouts: self.lookouts = lookouts
        if rand_seed: self.rand_seed = rand_seed
        if offscreenrender: self.offscreenrender = offscreenrender
        if dynamic_env: self.dynamic_env = dynamic_env
        if paths: self.paths = paths
        if show_obj_ids: self.show_obj_ids = show_obj_ids
        if envtype: self.envtype = envtype.lower()
        if bkgtags: self.bkgtags = bkgtags
        if gndtags: self.gndtags = gndtags
        if skytags: self.skytags = skytags
        if roadtags: self.roadtags = roadtags
        if bldg_maxheight: self.building_maxheight = bldg_maxheight
        if road_min_sep: self.road_min_sep = road_min_sep
        if timeofday: self.timeofday = timeofday
        if origin_axes: self.origin_axes = origin_axes
        if imsize: self.imsize = imsize
        if p_over_road: self.p_over_road = p_over_road
        if p_over_building: self.p_over_building = p_over_building
        if verbosity: self.verbosity = verbosity
        if model_tweeks: self.model_tweeks = model_tweeks
        if views: self.views = views

        # Convert the verbosity string to a verbosity level in [0, 1, 2, 3].
        if verbosity:
            if self.verbosity.lower() not in {'off','low','medium','high'}:
                raise ValueError('**Verbosity**' + \
                                 ' must be one of {"off", "low", "medium", "high"}')
            self.verbosity = ['off','low','medium','high'].index(self.verbosity.lower())

        if self.envtype not in {'urban', 'forest'}:
            raise ValueError('Unrecognized ENVTYPE: "{:s}"'.format(self.envtype))

        self.time = 0.0                  # time (sec.) since start of simulation
        self.time_last = 0.0             # time (sec.) of last simulation update

        # The target number (not always achievable) of each type of object.
        self.num_plants = int(self.plant_density*np.pi*(self.env_radius**2)/100)
        self.num_people = int(self.people_density*np.pi*(self.env_radius**2)/500)
        self.num_animals = int(self.animal_density*np.pi*(self.env_radius**2)/500)
        self.num_clutter = int(self.clutter_density*np.pi*(self.env_radius**2)/500)
        self.num_gnd_feat = int(self.gndfeat_density*np.pi*(self.env_radius**2)/500)
        self.num_vehicles = int(self.vehicle_density*np.pi*(self.env_radius**2)/500)
        self.num_airborne = int(self.airborne_density*np.pi*(self.env_radius**2)/500)

        # 3D (multi-channel) map of the environemnt identifies the types of
        # objects, the object IDs, and object heights at each (x,y) location.
        self.map3d = Map3D(gridspacing=0.25, radius=self.env_radius, dynamic=self.dynamic_env)

        # Define the probability of an object being above or on a road or a
        # building given that the object is allowed to be in such a position.
        # These values needn't sum to 1. The class arguments are used to change
        # the default values.
        p_over_road = self.p_over_road
        p_over_building = self.p_over_building
        self.p_over_road = {'airborne':0.5, 'animal':0.2, 'clutter':0.1,
                            'person':0.1,  'plant':0.05, 'vehicle':0.5}
        self.p_over_building = {'airborne':0.5, 'animal':0.2, 'clutter':0.1,
                                'person':0.5, 'plant':0.05, 'vehicle':0}
        for obj in {'person', 'clutter', 'animal', 'airborne', 'vehicle'}:
            if obj in p_over_road:
                self.p_over_road[obj] = p_over_road[obj]
            if obj in p_over_building:
                self.p_over_building[obj] = p_over_building[obj]

        # Properties of generic buildings (units in meters)
        self.building_interfloor = 3.6      # Standard interfloor spacing
        self.building_minheight = 3
        self.building_minlen = 5
        self.building_maxlen = 30
        self.building_meanlen = 10
        self.building_mindist2road = 5      # min distance to closest road
        self.building_maxdist2road = 10     # max distance to closest road
        self.building_minsep = 1e-6         # min distance between buildings
        self.building_maxhgt2base = 4       # max aspect ratio of height to base
        assert self.building_minheight >= self.building_minheight

        # Properties of generic roads (units in meters)
        self.road_lane_width = 3.5          # width of one lane of a road
        self.road_sidewalk_width = 2        # width of sidewalk and grass median

        # Properties of generic trees (units in meters).
        self.plant_std = 0.5 #0.1              # std. from texture file height & width

        # Properties of barriers.
        self.barrier_minlen = 5
        self.barrier_maxlen = 60

        # The world consists of a list of WorldObj objects.
        WorldObj.id_counter = 0
        self.num_objs = 0
        self.objs = []
        self.object_labels = list(label2id.values())

        if rand_seed:
            # Initialize the random number generator.
            if self.verbosity > 0: print('SimWorld: Random seed = {}'.format(self.rand_seed))
            np.random.seed(self.rand_seed)

        # Assign random seeds for each part of the model generation so that each
        # part of the model can be tweeked independently of the other parts.
        for key in self.model_seed.keys():
            self.model_seed[key] = np.random.randint(1, 1e7)
        if self.model_tweeks:
            if self.verbosity > 0: print('Model tweeks =', self.model_tweeks)
            for key in self.model_tweeks.keys():
                self.model_seed[key] = self.model_tweeks[key]

        # A number of geometrically identical, but photometrically different,
        # parallel world views may be created. This allows for easier generation
        # of groundtruth data (semantic labels, object IDs).
        self.worldviews = []
        self.view_obj_ids = False
        for v in self.views:
            if v.lower() == 'color':
                self.worldviews.append('color camera')
            elif v.lower() == 'label':
                self.worldviews.append('semantic labels')
            elif v.lower() == 'objid':
                self.worldviews.append('objectid')
                self.view_obj_ids = True
            else:
                raise ValueError('Unrecognized world view: "{}"'.format(v))
        self.numworlds = len(self.worldviews)
        self.idx_labels = self.worldviews.index('semantic labels') if 'semantic labels' in self.worldviews else None
        self.idx_color = self.worldviews.index('color camera') if 'color camera' in self.worldviews else None
        self.idx_objid = self.worldviews.index('objectid') if 'objectid' in self.worldviews else None

        # The timeofday argument is a float in [0,2400) and is converted to a
        # float in [0,24). E.g., the timeofday 1630 is represented as 16.5.
        if self.timeofday:
            if type(self.timeofday) in [list, tuple]:
                if len(self.timeofday) != 2:
                    raise ValueError("TIMEOFDAY argument must be a single float or a length 2 tuple")
                t1 = (self.timeofday[0] // 100) + (self.timeofday[0] % 100)/60
                t2 = (self.timeofday[1] // 100) + (self.timeofday[1] % 100)/60
                self.time_of_day = np.random.uniform(t1, t2)     # 24-hour time
            else:
                self.time_of_day = (self.timeofday // 100) + (self.timeofday % 100)/60
        else:
            self.time_of_day = np.random.uniform(5,21)     # 24-hour time
        if self.verbosity > 0: print('Time of day is {:02d}{:02d}'.format(
                                     int(self.time_of_day),
                                     round(60*(self.time_of_day % 1))))
        r = np.exp(-(self.time_of_day - 12)**4/3000)
        self.amb_light = 0.8*r                         # ambient light coef. (0.8*r)
        self.dif_light = 0.8*r                         # diffuse light coef.
        WorldObj.lightcoef = (self.amb_light, self.dif_light, 0)

        self.followers = []               # actors that are following the camera
        self.movers = []                  # objects that may move
        self.init_renderer()

        # Create the world model.

        np.random.seed(self.model_seed["lights"])
        self.insert_lights()

        np.random.seed(self.model_seed["backgnds"])
        self.insert_backgrounds(bkgtags=self.bkgtags, gndtags=self.gndtags,
                                skytags=self.skytags)

        np.random.seed(self.model_seed["roads"])
        self.insert_roads(roadtags=self.roadtags)

        np.random.seed(self.model_seed["signs"])
        self.insert_signs()

        np.random.seed(self.model_seed["buildings"])
        self.insert_buildings()

        np.random.seed(self.model_seed["barriers"])
        self.insert_barriers(self.barrier_density)

        np.random.seed(self.model_seed["bldgplants"])
        self.num_bldg_plants = int(self.bldg_plant_density*self.bldg_perim)
        self.insert_vert_objs(self.num_bldg_plants, "bldg_plant")

        np.random.seed(self.model_seed["gndfeats"])
        self.insert_horiz_objs(self.num_gnd_feat, "gndfeat")

        np.random.seed(self.model_seed["plants"])
        self.insert_vert_objs(self.num_plants, "plant")

        np.random.seed(self.model_seed["vehicles"])
        self.insert_vert_objs(self.num_vehicles, "vehicle")

        np.random.seed(self.model_seed["clutter"])
        self.insert_vert_objs(self.num_clutter, "clutter")

        np.random.seed(self.model_seed["persons"])
        self.insert_vert_objs(self.num_people, "person")

        np.random.seed(self.model_seed["lookouts"])
        if self.lookouts: self.insert_lookouts()

        np.random.seed(self.model_seed["animals"])
        self.insert_vert_objs(self.num_animals, "animal")

        np.random.seed(self.model_seed["urbananimals"])
        self.insert_vert_objs(2*self.num_animals, "urban_animal")

        np.random.seed(self.model_seed["airborne"])
        self.insert_vert_objs(self.num_airborne, "airborne")

        np.random.seed(self.model_seed["shadows"])
        self.insert_shadows()

        if self.pathsfile:
            np.random.seed(self.model_seed["followpaths"])
            self.insert_fixed_path_objs(self.pathsfile)

        self.interact = True         # allow user interaction with the world?
        self.label_colors = label_colors   # may be needed by window interaction code

        if self.origin_axes:
            # Create a cartesian axis at the origin of the world.
            for w in range(self.numworlds):
                vtu.make_axes(self.renderers[w], self.vtkcamera)


    def init_renderer(self):
        """
        Initialize the world renderer. Most of the parameters of the camera will
        need to be reset later for a specific camera model.

        Notes:
            As of 2020-05-01, the depth peeling works properly only when the
            following environment variable is set:
                VTK_USE_LEGACY_DEPTH_PEELING=1

            Off screen rendering (via renderwindow.SetOffScreenRendering(True))
            is necessary whenever a rendered window needs to have greater size
            (rows and columns) than the computer's screen resolution.
        """

        if self.renderers:
            # The renderers already exist. Clear existing actors and return.
            for r in self.renderers:
                r.RemoveAllViewProps()
            return

        # A number of geometrically identical, but photometrically different,
        # parallel worlds may be created. This allows for easier simulation of
        # some computer vision algorithms such as image semantic labeling.
        self.renderers = [None]*self.numworlds
        self.renwindow = [None]*self.numworlds
        self.interactor = [None]*self.numworlds

        self.interactor_ready = False              # ready for user interaction?

        # Setup the camera. One camera is used for all worlds.
        self.vtkcamera = vtk.vtkCamera()
        self.vtkcamera.SetPosition(0, 0, 2)
        self.vtkcamera.SetFocalPoint(0, 1, 2)
        self.vtkcamera.SetViewUp(0, 0, 1)
        self.vtkcamera.SetViewAngle(40)
        # self.vtkcamera.SetDistance(0.05)
        self.vtkcamera.SetClippingRange(0.1, 10*self.env_radius)
        self.last_camera_pos = np.inf*np.ones(3, dtype=float)

        for w in range(self.numworlds):
            # The renderer renders into the render window.
            self.renderers[w] = vtk.vtkRenderer()
            if self.worldviews[w] == 'color camera':
                # self.renderers[w].UseShadowsOn()   # shadows is not working (2020-10-29)
                self.renderers[w].SetUseFXAA(False)  # Using FXAA is worse than no antialiasing!
                # options = self.renderers[w].GetFXAAOptions()
                # options.SetSubpixelBlendLimit(1.0)         # (0.75) higher = more smoothing
                # options.SetHardContrastThreshold(0.01)     # (0.0625) lower = more smoothing
                # options.SetRelativeContrastThreshold(0.01) # (0.125) lower = more smoothing
                # options.SetEndpointSearchIterations(12)    # (12) higher = detect longer edges
                # options.SetSubpixelContrastThreshold(0.0)  # (0.25) lower = more smoothing
                # options.SetUseHighQualityEndpoints(True)   # (True) True = improved alias detection
            else:
                # Renderer for world views that provide groundtruth information:
                # semantic labels and object IDs.
                # self.renderers[w].UseShadowsOff()
                self.renderers[w].SetUseFXAA(False)
            # antialias = False if self.worldviews[w] == 'semantic labels' else True
            # self.renderers[w].SetUseFXAA(antialias)   # Use FXAA anti-aliasing, if supported.

            # The render window interactor captures mouse events and will
            # perform appropriate camera or actor manipulation depending on the
            # nature of the events.
            self.renwindow[w] = vtk.vtkRenderWindow()
            self.renwindow[w].AddRenderer(self.renderers[w])
            self.renwindow[w].SetWindowName(self.worldviews[w].title())
            self.interactor[w] = vtk.vtkRenderWindowInteractor()
            self.interactor[w].SetRenderWindow(self.renwindow[w])
            # self.interactor[w].Disable() # disable standard keyboard/mouse controls

            # Set the background color and camera image size.
            if self.worldviews[w] == 'objectid':
                self.renderers[w].SetBackground((0,0,0))
            else:
                self.renderers[w].SetBackground(otype2color1['sky'])  # vtk.util.colors.sky_blue
            self.renwindow[w].SetSize(self.imsize)         # image size, (width, height) in pixels
            self.renwindow[w].SetAlphaBitPlanes(True)      # turn on/off use of alpha bitplanes
            self.renwindow[w].SetMultiSamples(16)          # (0) num. of multisamples for antialiasing

            if self.offscreenrender:
                # Do not use an on-screen window to render the scene. The
                # rendered image size will not be limited to the screen
                # resolution. Also, this may be desired when there are multiple
                # cameras, to avoid screen clutter.
                self.renwindow[w].SetOffScreenRendering(True)

            # Options for rendering partially transparent surfaces.
            self.renderers[w].SetUseDepthPeeling(8)        # 8
            self.renderers[w].SetMaximumNumberOfPeels(100) # 20 layers of translucency
            self.renderers[w].SetOcclusionRatio(0.002)     # 0.002 == 2 out of 1000 pixels

            # self.renderers[w].SetPreserveDepthBuffer(False)  # use existing depth buffer for renderinf
            # self.renderers[w].SetUseDepthPeelingForVolumes(True)

            self.renderers[w].SetActiveCamera(self.vtkcamera)


    def insert_lights(self):
        """
        Insert light sources into the world.

        From www.vtk.org:
            vtkLight is a virtual light for 3D rendering. It provides methods to
            locate and point the light, turn it on and off, and set its brightness
            and color. In addition to the basic infinite distance point light
            source attributes, you also can specify the light attenuation values
            and cone angle. These attributes are only used if the light is a
            positional light. The default is a directional light (e.g. infinite
            point light source).

            Lights have a type that describes how the light should move with respect
            to the camera. A Headlight is always located at the current camera
            position and shines on the camera's focal point. A CameraLight also
            moves with the camera, but may not be coincident to it. CameraLights are
            defined in a normalized coordinate space where the camera is located at
            (0, 0, 1), the camera is looking at (0, 0, 0), and up is (0, 1, 0).
            Finally, a SceneLight is part of the scene itself and does not move with
            the camera. (Renderers are responsible for moving the light based on its
            type.)

            There are two subclasses of lights: Positional and Directional (Spot
            or non-Positional). A Positional light (set with PositionalOn()) is
            a point light that illuminates in all directions (360 degrees). A
            Directional light (set with PositionalOff()) has a cone of emitted
            light whose cone angle is less than 90 degrees.
        """

        # Get the position of the sun. The vertical angle is the angle from the
        # XY plane. 90 degrees is straight up. The horizontal angle is the angle
        # from the Y axis in the XY plane, with positive values moving in the
        # counterclockwise direction. Both angles are in radians.
        self.light_vangle = np.deg2rad(max(0, 90-11.25*abs(self.time_of_day-12)))
        self.light_hangle = np.deg2rad(360*np.random.rand())
        x = np.sin(self.light_hangle)
        y = np.cos(self.light_hangle)
        z = np.tan(self.light_vangle)
        lightpos = 1e6*np.array([x, y, z])
        # lightpos = 1e6*np.array([-np.sin(np.deg2rad(self.light_vangle)), 1,
                                 # np.cos(np.deg2rad(self.light_vangle))])

        if self.verbosity > 0: print("Angle of Sun: azimuth = {:.1f}º from north, elevation = {:.1f}º".format(
                                     np.rad2deg(self.light_hangle),
                                     np.rad2deg(self.light_vangle)))

        for w in range(self.numworlds):
            self.renderers[w].RemoveAllLights()     # remove any previous lights
            if self.worldviews[w] in {'semantic labels', 'objectid'}:
                # Use only ambient light. Set SceneLight intensity to 0 (turn
                # off). Create two light sources opposite of each other, so that
                # all objects are illuminated on both sides.
                self.renderers[w].SetAmbient(1, 1, 1)
                light = vtk.vtkLight()
                light.SetLightTypeToSceneLight()
                light.SetColor(1.0, 1.0, 1.0)
                light.SetPosition(1e10, 1e10, 1e10)
                light.SetFocalPoint(0,0,0)
                light.SetIntensity(1.0)
                light.PositionalOff()
                self.renderers[w].AddLight(light)
                light = vtk.vtkLight()
                light.SetLightTypeToSceneLight()
                light.SetColor(1.0, 1.0, 1.0)
                light.SetPosition(1e10, 1e10, -1e10)
                light.SetFocalPoint(0,0,0)
                light.SetIntensity(1.0)
                light.PositionalOff()
                self.renderers[w].AddLight(light)
            else:
                # Create a SceneLight at infinity (diffuse light).
                light = vtk.vtkLight()
                if self.time_of_day < 8:
                    # Sunrise - yellow tint
                    c = (1, 0.9, 0.5+(8-self.time_of_day)/8)
                elif self.time_of_day > 16:
                    # Sunset - red tint
                    c = (1, 1-(self.time_of_day-16)/16, 1-(self.time_of_day-16)/16)
                else:
                    # All other times - white light
                    c = (1, 1, 1)
                light.SetLightTypeToSceneLight()
                light.SetColor(c)
                light.SetPosition(lightpos)
                light.SetFocalPoint(0,0,0)
                light.SetIntensity(1.0)
                light.SetAmbientColor(1, 1, 1)                        ## (0,0,0)
                light.PositionalOn()                # illuminates in all directions
                light.SetShadowAttenuation(1.0)  # 0.1  ??
                self.renderers[w].AddLight(light)
                self.renderers[w].SetAmbient(1,1,1)       # ???


    def insert_backgrounds(self, gndtags='ground', skytags='sky',
                           bkgtags='background'):
        """
        Insert a ground plane, background scenery, and sky into the world model.

        Arguments:
            gndtags: (str) Tags to use in selecting the ground cover. If empty,
            then a random ground cover is chosen. Default is ''.

            skytags: (str) Tags to use in selecting the sky texture. If empty,
            then a random sky texture is chosen. Default is ''.

            bkgtags: (str) Tags to use in selecting the backgound scene. If
            empty, then a random background scene is chosen. Default is ''.
        """

        bkgrad = self.env_radius+10   # radius of background scene cylinder (m)
        gndrad = 5*bkgrad             # radius of ground plane
        skyrad = 5*bkgrad             # radius of sky scene cylinder
        lcflat = (1.0, 0.0, 0.0)      # lighting coef. (ambient, diffuse, specular) for flat colors
        self.sky_actors = []

        #-----------------------------------------------------------------------
        # Insert the ground plane.
        #-----------------------------------------------------------------------

        txt = WorldObj.GetTexture(inctags=gndtags, exctags='label')
        if txt == None:
            raise Exception('There is no ground texture with tags "{}"'.format(gndtags))
        gndtags = set(TagTypeID(txt['tags']))
        if self.verbosity > 0: print("Using", "_".join(sorted(gndtags, reverse=True)))
        rect3d = [0, 0, 0, gndrad, gndrad, hgt_ground]   # [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
        self.objs.append(WorldObj(self.renderers, self.worldviews,
                                  objtags=gndtags, rect3d=rect3d))
        self.map3d.set([0,0,gndrad,gndrad], oid=self.objs[-1].id, olabel='ground',
                       oelev=0)

        #-----------------------------------------------------------------------
        # Insert a sky into the color world view. The sky is a panoramic image
        # mapped onto a cylinder. The background color of all renderers has
        # already been set to otype2color1['sky'] in init_renderer(), so we
        # don't need to semantic label the sky. The lighting coefficient for sky
        # images uses a diffuse coefficient of zero so that all tiles in the sky
        # have the same illumination, and so there should be no lighting
        # discontinuities due to image plane orientation differences.
        #-----------------------------------------------------------------------

        if self.idx_color != None:
            # Get the dimensions of the sky cylinder's texture images.
            txt = WorldObj.GetTexture(inctags=skytags, exctags='label')
            if txt == None:
                raise Exception('There is no sky texture with tags "{}"'.format(skytags))
            keytags = set(TagTypeID(txt['tags']))
            if self.verbosity > 0: print("Using", "_".join(sorted(keytags, reverse=True)))
            height = txt['vsize']
            width = txt['hsize']

            # Load all of the sky texture images.
            txt = WorldObj.GetTexture(inctags=keytags, exctags='label', getall=True)
            textures = []
            for k in range(1,len(txt)+1):
                t = WorldObj.GetTexture(inctags=keytags|{'t'+str(k)}, exctags='label')
                if t == None:
                    raise Exception(
                        'Loading sky image tiles 1..{}...\n'.format(len(txt)+1) + \
                        'There is no image with tags {}'.format(keytags|{'t'+str(k)}))
                textures.append(t['vtk'])

            # Make the texture mapped sky cylinder.
            self.sky_actors = vtu.make_cyl(self.renderers[self.idx_color], origin=(0,0,0),
                                           radius=skyrad, width=width, height=height,
                                           seamless=True, textures=textures,
                                           reps=(1,1,1), opacity=1.0,
                                           lightcoef=(WorldObj.lightcoef[0],0,0))

            # Rotational speed (degrees/second) of sky background.
            self.sky_speed = np.random.uniform(0.05,0.3)
            if self.verbosity > 0: print("Sky speed = {:.3f}º/sec".format(self.sky_speed))

        #-----------------------------------------------------------------------
        # Insert a background scene. This is a panoramic image mapped onto a
        # cylinder. The lighting coefficient for background images uses a
        # diffuse coefficient of zero so that all tiles in the background have
        # the same illumination, and so that there should be no lighting
        # discontinuities due to image plane orientation differences.
        #-----------------------------------------------------------------------

        # Get the dimensions of the background cylinder's texture images.
        txt = WorldObj.GetTexture(inctags=bkgtags, exctags='label')
        if txt == None:
            raise Exception('There is no background texture with tags "{}"'.format(bkgtags))
        keytags = set(TagTypeID(txt['tags']))
        if self.verbosity > 0: print("Using", "_".join(sorted(keytags, reverse=True)))
        height = txt['vsize']
        width = txt['hsize']

        if self.idx_color != None:
            # Make the color background scene.
            txt = WorldObj.GetTexture(inctags=keytags, exctags='label', getall=True)
            textures = []
            for k in range(1,len(txt)+1):
                t = WorldObj.GetTexture(inctags=keytags|{'t'+str(k)}, exctags='label')
                if t == None:
                    raise Exception(
                        'Loading background image tiles 1..{}...\n'.format(len(txt)+1) + \
                        'There is no image with tags {}'.format(keytags|{'t'+str(k)}))
                textures.append(t['vtk'])
            vtu.make_cyl(self.renderers[self.idx_color], origin=(0,0,0), radius=bkgrad,
                         width=width, height=height, seamless=True,
                         textures=textures, reps=(1,1,1),
                         lightcoef=(WorldObj.lightcoef[0],0,0), opacity=1.0)

        if self.idx_labels != None:
            # Make the semanticly labeled background scene.
            keytags |= {'label'}
            txt = WorldObj.GetTexture(inctags=keytags, getall=True)
            textures = []
            for k in range(1,len(txt)+1):
                t = WorldObj.GetTexture(inctags=keytags|{'t'+str(k)})
                textures.append(t['vtk'])
            vtu.make_cyl(self.renderers[self.idx_labels], origin=(0,0,0),
                         radius=bkgrad, width=width, height=height,
                         seamless=True, textures=textures, reps=(1,1,1),
                         lightcoef=lcflat, opacity=1.0)


    def insert_roads(self, roadtags='road'):
        """
        Insert roads into the world model.
        """
        def randmult(step, rmin, rmax):
            ''' Get a random float in [rmin, rmax] that is a multiple of step. '''
            return step*np.fix(np.random.uniform(rmin,rmax)/step)

        s = np.random.uniform(0.7,1.3)
        roadminsep = s*self.road_min_sep
        self.num_roads = int(np.floor(3*self.road_density*self.env_radius/roadminsep))
        if self.verbosity > 0: print('Creating {} roads...'.format(self.num_roads))
        self.total_road_length = 0
        map_is_full = False

        # Roads extend out to a maximum of `roadpct` of the city radius. This is
        # to make room around the outside of the road network for other stuff
        # (targets, buildings, trees, etc.)
        roadpct = 0.99
        maxp1 = roadpct*self.env_radius

        # Group roads into north-south and east-west.
        self.nsroads = []
        self.ewroads = []

        if self.num_roads == 0: return

        # To make texture mapping road intersections easier, all road textures
        # are taken from the same class of roads. Pick a random road class.
        roadtags = set(roadtags.split('_'))
        txt = WorldObj.GetTexture(inctags=roadtags)
        if txt == None:
            raise Exception('No road textures were found.')
        _, roadclass = TagTypeID(txt['tags'])
        if self.verbosity > 0: print("Using road_{:s}".format(roadclass))
        self.road_class = roadclass
        roadclass = {'road', roadclass}
        rhw = txt['hsize']/2
        self.road_lane_width = rhw
        self.sign_pos = rhw*float(TagValue(txt['tags'], 'sp'))

        # Create roads on a grid with the following spacing for road centers.
        roadstep = roadminsep + 2*rhw
        roadstep = min(roadstep, roadpct*self.env_radius/2)

        for rnum in range(self.num_roads):
            if map_is_full:
                # We sometimes can't fit enough roads.
                break

            goodlocation = False
            numtries = 0

            while not goodlocation:
                numtries += 1
                if numtries > 1000:
                    numtries = 0
                    roadminsep /= 2
                    roadstep /= 2

                # Choose a random road direction.
                roadtype = 'ns' if np.random.rand() > 0.5 else 'ew'

                # Pick a random location within in the city radius for the
                # center of the road's rectangle and its length. The center and
                # length are both mutliples of the minimum inter-road distance.
                # Below, (p1,p2) is the center of the road rectangle.
                p1 = randmult(roadstep, -maxp1, maxp1)
                maxp2 = roadpct*self.env_radius*np.cos((np.pi/2)*(p1/self.env_radius))
                length = 2*randmult(roadstep, roadstep, max(maxp2,roadstep))
                length += 2*rhw                     # make ends of roads line up
                assert length > roadstep
                rhl = length/2                                # road half length
                p2 = randmult(roadstep, -maxp2+rhl, maxp2-rhl)

                # The road rectangle is [xcenter, ycenter, xhalfwidth, yhalfwidth].
                rect = [p1, p2, rhw, rhl] if roadtype == 'ns' else [p2, p1, rhl, rhw]

                # Make sure the proposed road is not too close to parallel roads
                # and that it connects with at least one perpindecular road.
                goodlocation = True
                connected = True
                if roadtype == 'ns':
                    parallel = self.nsroads
                    perpindicular = self.ewroads
                else:
                    parallel = self.ewroads
                    perpindicular = self.nsroads
                for road in parallel:
                    if road.dist2d(rect) < roadminsep - self.road_lane_width:
                        # Too close to a parallel road.
                        goodlocation = False
                        break
                if goodlocation and rnum > 0:
                    connected = False
                    for road in perpindicular:
                        if road.dist2d(rect) <= 0:
                            # Connected to a perpindicular road.
                            connected = True
                            break
                    if not connected:
                        # The new road does not connect to any existing roads.
                        goodlocation = False

            # Create the road.
            self.total_road_length += length
            rect3d = [rect[0], rect[1], 0, rect[2], rect[3], hgt_road]
            road = WorldObj(self.renderers, self.worldviews,
                            objtags=roadtags|roadclass|{roadtype}, rect3d=rect3d)
            road.roadtype = roadtype
            self.objs.append(road)
            self.map3d.set(rect, oid=road.id, olabel='road', oelev=0)

            # Save roads in two differnt lists for easier access later.
            if roadtype == 'ns':
                self.nsroads.append(road)
            else:
                self.ewroads.append(road)

        if 'color camera' in self.worldviews:
            # Texture map road intersections with crosswalks, etc.
            # v = self.worldviews.index('color camera')
            renderer = self.renderers[self.worldviews.index("color camera")]
            for nsroad in self.nsroads:
                for ewroad in self.ewroads:
                    if nsroad.dist2w(ewroad) == 0:
                        # Roads intersect. Insert an intersection image.
                        txt = WorldObj.GetTexture(inctags='intersection_'+
                                                           self.road_class)
                        if txt == None:
                            raise Exception('No intersection textures found for ' \
                                            'road class {}.'.format(self.road_class))
                        txtr = txt['vtk']
                        dx = txt['hsize']
                        dy = txt['vsize']
                        org = (nsroad.xctr,ewroad.yctr,hgt_intersection) # quadrilateral origin
                        corner = (-dx/2,-dy/2,hgt_intersection)          # one corner of quad
                        vtu.make_quad(renderer, origin=org, pos=corner,
                                      v1=(dx,0,0), v2=(0,dy,0),
                                      texture=txtr, lightcoef=WorldObj.lightcoef)


    def insert_lookouts(self, pwoccupied=0.5):
        """
        Insert people that look out of building windows.

        Arguments:
            pwoccupied: (float) Probability (in [0,1]) that any particular
            building window is occupied by a person looking out.
        """
        assert pwoccupied >= 0 and pwoccupied <= 1.0

        cnt = 0
        for bldg in self.buildings:
            if bldg.window:
                for wallnum in range(4):
                    if np.random.rand() > pwoccupied:
                        continue

                    # What repetition of texture should the person look out?
                    rep = np.random.randint(0,bldg.txt_reps,3)

                    # Get a person texture to look out the window.
                    txt = WorldObj.GetTexture(inctags="person_w", exctags="label")

                    if txt is None:
                        print('*** There are no people to look out of windows')
                        return               # no people can look out of windows

                    tags = txt['tags']
                    width = txt['hsize']
                    height = txt['vsize']

                    if wallnum == 0:
                        # Wall parallel to X axis facing south.
                        x = bldg.xctr - bldg.xhalflen + \
                            (rep[0] + bldg.window_pos[0])*bldg.txt_size[0] - width/2
                        y = bldg.yctr - bldg.yhalflen + width/2 + 0.01
                    elif wallnum == 1:
                        # Wall parallel to Y axis facing east.
                        x = bldg.xctr + bldg.xhalflen - width - 0.01
                        y = bldg.yctr - bldg.yhalflen + \
                            (rep[1] + bldg.window_pos[0])*bldg.txt_size[0]
                    elif wallnum == 2:
                        # Wall parallel to X axis facing north.
                        x = bldg.xctr + bldg.xhalflen - \
                            (rep[0] + bldg.window_pos[0])*bldg.txt_size[0] - width/2
                        y = bldg.yctr + bldg.yhalflen - width/2 - 0.01
                    else:
                        # Wall parallel to Y axis facing west.
                        x = bldg.xctr - bldg.xhalflen + 0.01
                        y = bldg.yctr + bldg.yhalflen - \
                            (rep[1] + bldg.window_pos[0])*bldg.txt_size[0]

                    z = (rep[2] + bldg.window_pos[1])*bldg.txt_size[1]
                    ax1=[width,0,0]
                    ax2=[0,0,height]

                    if False:
                        print(f"Created lookout person at pos ({x:.1f},{y:.1f})")

                    soundtags = tags & set(WorldObj.tag2soundid.keys())
                    if len(soundtags) > 0:
                        # Create a sound for the object. First choose a single
                        # random sound tag from the set of sound tags that the
                        # object may generate, then choose a random sound
                        # matching that tag. The object is also assigned a
                        # random position in its audio signal to be the sample
                        # at time zero (the start of the simulation).
                        t = np.random.choice(list(soundtags))
                        sid = np.random.choice(list(WorldObj.tag2soundid[t]))
                        sound = WorldObj.sounds[sid].copy()
                        WorldObj.sounds[sid]['numoccur'] += 1
                        dvol = 1 + 0.2*(2*np.random.rand() - 1)   # perturb volume
                        signal = dvol*sound['signal']             # amplitude scale
                        tscale = 2*np.random.rand() + 0.5         # time scale
                        sp0 = np.arange(0, sound['nsamples'])     # original sample points
                        nsamples_new = int(np.ceil(tscale*sound['nsamples']))
                        sp1 = np.linspace(0, sound['nsamples']-1, num=nsamples_new) # new sample points
                        sound['signal'] = np.interp(sp1, sp0, signal)
                        sound['nsamples'] = nsamples_new
                        sound['duration'] = tscale*sound['duration']
                        sound['samplezero'] = np.random.randint(nsamples_new)  # start position of signal
                        sound['dvolume'] = dvol
                    else:
                        sound = None

                    # Add the person to the model. The person is rendered on a flat
                    # 3D rectangle that orients towards the viewer. The position of
                    # the rectangle is defined by one 3D corner (pos) and two 3D
                    # vectors (axis1, axis2) for the horizontal and vertical axis,
                    # respectively, of the texture image.
                    newobj = WorldObj(self.renderers, self.worldviews, objtags=tags,
                                      pos=[x,y,z], axis1=ax1, axis2=ax2,
                                      motiontype=Motion.Static)
                    newobj.sound = sound
                    newobj.dynamic = False    # lookouts are not allowed to move
                    self.objs.append(newobj)
                    self.followers.extend(newobj.actors)
                    newobj.maprect = self.map3d.set([x, y, width/2, width/2],
                                                    oid=self.objs[-1].id,
                                                    olabel='person', oelev=z+height)
                    if sound != None:
                        # Keep a list of all objects that can make noise.
                        self.noise_makers.append(newobj)

                    # Add the object's ID to the object's actors. This is used to make
                    # moving actors face the right direction while following the camera.
                    for actor in newobj.actors:
                        p = actor.GetProperty()
                        p.id = newobj.id
                        actor.SetProperty(p)

                    if self.show_obj_ids:
                        # Display each object's ID overlayed on the color image.
                        actor = vtu.make_text(self.renderers[1], text=str(newobj.id),
                                              textscale=0.3, pos=(x, y, z))
                        newobj.actors.append(actor)

                    cnt += 1

        if self.verbosity > 0: print('Creating {} lookouts...'.format(cnt))


    def insert_signs(self):
        """
        Insert street signs into the world model.
        """
        if self.verbosity > 0: print("Creating street signs...")
        botheight = 1.83                     # height at bottom of sign (meters)
        txt = WorldObj.GetTexture(inctags='sign_stop_front', exctags='label')
        got_stop_signs = txt is not None
        if not got_stop_signs:
            print('There are no stop sign textures')

        # Initialize the list of X or Y coordinates of road intersections. This
        # will be used to delineate the city blocks when inserting miscellaneous
        # street signs between road intersections.
        for rd in self.nsroads:
            rd.yintersects = [rd.yctr-rd.yhalflen, rd.yctr+rd.yhalflen]
        for rd in self.ewroads:
            rd.xintersects = [rd.xctr-rd.xhalflen, rd.xctr+rd.xhalflen]

        # Insert stop signs at road intersections.
        for rd_ns in self.nsroads:
            for rd_ew in self.ewroads:
                d = rd_ns.dist2w(rd_ew)
                if d <= 0:
                    # Roads intersect.
                    ctr = np.array([rd_ns.xctr,rd_ew.yctr])   # 2D center of intersection
                    rd_ns.yintersects.append(ctr[1])
                    rd_ew.xintersects.append(ctr[0])

                    if got_stop_signs:
                        # Insert stop signs at road intersections.
                        if np.random.rand() < 0.5:
                            locs = [(1,1,90), (-1,1,180), (-1,-1,270), (1,-1,0)]   # 4-ways stop
                        elif np.random.rand() < 0.5:
                            locs = [(1,1,90), (-1,-1,270)]   # 2-way stop
                        else:
                            locs = [(-1,1,180), (1,-1,0)]    # 2-way stop
                        for d in locs:
                            pos = ctr + self.sign_pos*np.array(d[0:2])
                            steetsign = WorldObj(self.renderers, self.worldviews,
                                                 'sign_stop', height=botheight,
                                                 pos=pos, orient=(0,0,d[2]))
                            self.objs.append(steetsign)
                            rect = [pos[0], pos[1], txt['hsize'], txt['hsize']]
                            self.map3d.set(rect, oid=steetsign.id, olabel='sign',
                                           oelev=botheight+txt['vsize'])

        # Insert signs between road intersections.
        for rd in self.nsroads + self.ewroads:
            xcoords = []
            ycoords = []
            orient = []

            if rd.roadtype == 'ns':
                # North-south road (parallel to Y-axis).
                rd.yintersects.sort()
                for k in range(len(rd.yintersects)-1):
                    numsigns = int(0.05*(rd.yintersects[k+1]-rd.yintersects[k]))
                    if numsigns > 0:
                        y = np.linspace(rd.yintersects[k], rd.yintersects[k+1],
                                        num=numsigns+2, endpoint=True)
                        x = rd.xctr*np.ones(numsigns)
                        pmone = np.random.rand(numsigns)         # random +/- 1
                        pmone = (pmone < 0.5).astype(float) - (pmone >= 0.5).astype(float)
                        x += self.sign_pos*pmone
                        xcoords.extend(list(x))
                        ycoords.extend(list(y[1:-1]))
                        orient.extend(list((pmone<0)*180))
            else:
                # East-west road (parallel to X-axis).
                rd.xintersects.sort()
                for k in range(len(rd.xintersects)-1):
                    numsigns = int(0.05*(rd.xintersects[k+1]-rd.xintersects[k]))
                    if numsigns > 0:
                        x = np.linspace(rd.xintersects[k], rd.xintersects[k+1],
                                        num=numsigns+2, endpoint=True)
                        y = rd.yctr*np.ones(numsigns)
                        pmone = np.random.rand(numsigns)          # random +/- 1
                        pmone = (pmone < 0.5).astype(float) - (pmone >= 0.5).astype(float)
                        y += self.sign_pos*pmone
                        xcoords.extend(list(x[1:-1]))
                        ycoords.extend(list(y))
                        orient.extend(list(pmone*90))

            for k in range(len(xcoords)):
                txt = WorldObj.GetTexture(inctags='sign_front', exctags='stop')
                if txt == None:
                    print('There are no non-stop street signs')
                    return
                idtags = 'sign_'+txt['id']    # tags to identify this particular object
                pos = [xcoords[k], ycoords[k]]
                botheight = 1.5
                steetsign = WorldObj(self.renderers, self.worldviews,
                                     objtags=idtags, height=botheight,
                                     pos=pos, orient=(0,0,orient[k]))
                self.objs.append(steetsign)
                rect = [pos[0], pos[1], txt['hsize'], txt['hsize']]
                self.map3d.set(rect, oid=steetsign.id, olabel='sign',
                               oelev=botheight+txt['vsize'])



    def insert_buildings(self):
        """
        Insert buildings into the world model.
        """
        if self.envtype == 'forest':
            numbuildings = 0
        else:
            numbuildings = int(2*0.7*self.bldg_density*self.total_road_length
                                   /self.building_maxlen)

        if self.verbosity > 0: print('Creating {} buildings... '.format(numbuildings))
        self.buildings = []
        self.num_buildings = 0
        self.bldg_perim = 0                        # sum of building perimeters
        bldgcnt = 0

        if 'building' in WorldObj.tag2txtid.keys():
            # Determine building sizes from available texture maps.
            use_building_textures = True
        else:
            use_building_textures = False

        xylocs = self.FindMapPts(dist=[('road', 0.5*self.building_maxlen+2,
                                                0.5*self.building_maxlen+10)],
                                 numpts=100*numbuildings,
                                 sample=0.5, dispfig=False)

        numlocs = xylocs.shape[0]
        nextloc = 0

        while bldgcnt < numbuildings:
            # Get size of new building.
            if use_building_textures:
                # Define building size based on a random building texture.
                txt = WorldObj.GetTexture(inctags='building', exctags='label')
                tags = txt['tags']
                txtwidth = txt['hsize']
                txtheight = txt['vsize']

                # Pick a random width, length, and height in the range of
                # valid values for buildings. Then round it to the nearest
                # increment of the texture's width and height.
                width = np.random.uniform(low=self.building_minlen, high=self.building_maxlen)
                length = np.random.uniform(low=self.building_minlen, high=self.building_maxlen)
                height = np.random.uniform(low=self.building_minheight, high=self.building_maxheight)
                width = max(txtwidth, txtwidth*round(width/txtwidth))
                length = max(txtwidth, txtwidth*round(length/txtwidth))
                height = max(txtheight, txtheight*round(height/txtheight))
                if width > self.building_maxlen or length > self.building_maxlen:
                    # This building texture is too large. Choose a different texture.
                    continue
            else:
                # Building size is based on pre-defined parameters.
                tags = 'building'
                height = np.random.uniform(low=self.building_minheight, high=self.building_maxheight)
                height = self.building_interfloor*np.floor(height/self.building_interfloor)
                minbase = max(height/self.building_maxhgt2base, self.building_minlen)
                maxbase = min(height*self.building_maxhgt2base, self.building_maxlen)
                length = np.random.random()*(maxbase - minbase) + minbase
                width = np.random.random()*(maxbase - minbase) + minbase
            xhalflen = width/2
            yhalflen = length/2

            # Get a building location that is unoccupied by other objects except
            # ground.
            while True:
                if nextloc >= numlocs:
                    if self.verbosity > 0: print('Environment is full with {} buildings'.format(bldgcnt))
                    self.num_buildings = bldgcnt
                    return
                xy = xylocs[nextloc,:]
                nextloc += 1
                olabels = self.map3d.get([xy[0],xy[1],xhalflen,yhalflen], out="L")
                if len(olabels) == 1 and olabels[0] == 'ground':
                    break                       # found a good building location

            rect = [xy[0], xy[1], height/2, xhalflen, yhalflen, height/2]
            self.objs.append(WorldObj(self.renderers, self.worldviews,
                                      objtags=tags, rect3d=rect))
            self.map3d.set([xy[0], xy[1], xhalflen, yhalflen], olabel='building',
                           oid=self.objs[-1].id, oelev=height)

            bldg = self.objs[-1]
            bldg.window = True if 'wp' in tags else False
            self.bldg_perim += 2*(length + width)
            if bldg.window:
                # Record info about the building's window location.
                bldg.window_pos = np.array([float(n) for n in TagValue(tags,'wp').split('x')])
                bldg.txt_size = [txt['hsize'], txt['vsize']]

            self.buildings.append(bldg)
            bldgcnt += 1

        self.num_buildings = bldgcnt
        return


    def insert_barriers(self, barrier_density=1.0):
        """
        Insert barriers (fences and walls) into the world model.
        """

        numbarriers = int(barrier_density*np.pi*(self.env_radius**2)/5000)
        if self.verbosity > 0: print('Creating {} barriers... '.format(numbarriers))
        self.barriers = []
        barriercnt = 0

        if 'barrier' not in WorldObj.tag2txtid.keys():
            print('There are no "barrier" textures.')
            return

        while barriercnt < numbarriers:
            # Define barrier size based on a random barrier texture.
            txt = WorldObj.GetTexture(inctags='barrier', exctags='label')
            if txt is None:
                raise Exception("No texture images match barrier tags")
            tags = txt['tags']
            txtwidth = txt['hsize']
            height = txt['vsize']

            # How thick is the barrier?
            thickness = self.map3d.mgridspc
            if 't' in tags:
                thickness = max(thickness, float(TagValue(tags,'t')))

            # Pick a random barrier length in the range of valid values for
            # barriers. Then round it to the nearest increment of the texture's
            # width and height.
            length = np.random.uniform(low=self.barrier_minlen, high=self.barrier_maxlen)
            length = max(txtwidth, txtwidth*round(length/txtwidth))
            if length > self.barrier_maxlen:
                # This barrier texture is too large. Choose a different texture.
                continue

            if np.random.rand() > 0.5:
                # Barrier parallel to the x-axis.
                xhalflen = length/2
                yhalflen = thickness/2
            else:
                # Barrier parallel to the y-axis.
                yhalflen = length/2
                xhalflen = thickness/2

            # Get a location that is unoccupied by other objects except ground.
            # Since barriers are created before ground features (gndfeat), they
            # can be on top of ground features.
            while True:
                xy = np.random.uniform(-self.env_radius, self.env_radius,2)
                if np.linalg.norm(xy) > 0.95*self.env_radius:
                    continue                     # center is outside city radius
                olabels = self.map3d.get([xy[0],xy[1],xhalflen,yhalflen], out="L")
                if len(olabels) == 1 and olabels[0] == 'ground':
                    break                                # found a good location

            # rect3d: [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
            rect3d = [xy[0], xy[1], height/2, xhalflen, yhalflen, height/2]
            newobj = WorldObj(self.renderers, self.worldviews, objtags=tags, rect3d=rect3d)
            self.objs.append(newobj)
            self.map3d.set([xy[0], xy[1], xhalflen, yhalflen], olabel='barrier',
                           oid=newobj.id, oelev=height)
            self.barriers.append(newobj)
            barriercnt += 1

        self.num_barriers = barriercnt
        return


    def insert_shadows(self):
        """
        Insert shadows of some static objects onto the ground plane.

        Description:
            Shadows are currently created for all plants, builidings, and
            barriers (fences and walls). All buildings are assumed to be
            rectangular prisms.

            Shadows do not exist as WorldObj objects in the scene or in the 2D
            or 3D maps. Shadows are just semi-transparent surfaces just above
            the ground in the color camera world view. These shadows are not
            cast onto any objects above the ground plane.

            Shadows of other objects, such as people and vehicles, must be part
            of the objects texture image.
        """

        if self.idx_color == None: return
        if self.verbosity > 0: print('Creating shadows... ')

        renderer = self.renderers[self.idx_color]
        opacity = 0.4*abs(self.light_vangle/1.57079) # max opacity at 90 degrees

        # The distance between the vertical projection onto the ground plane of
        # a point at height H and the projection of the same point along the
        # angle of the sun's rays, VANGLE, is D = H*tan(VANGLE). DXSCALE and
        # DYSCALE are used to get the X and Y components of this distance.
        hscale = 1/np.tan(self.light_vangle)
        dxscale = hscale*np.sin(self.light_hangle + np.pi)
        dyscale = hscale*np.cos(self.light_hangle + np.pi)

        ##################################################################
        # Create shadows for animals, clutter, persons, plants, and signs.
        ##################################################################

        txt_plants = WorldObj.GetTexture(inctags={"shadow","plants"})
        if txt_plants == None:
            print('*** There are no textures for "shadow_plants"')
            return
        txt_misc = WorldObj.GetTexture(inctags={"shadow","misc"})
        if txt_misc == None:
            print('*** There are no textures for "shadow_misc"')
            return
        txt_signs = WorldObj.GetTexture(inctags={"shadow","signs"})
        if txt_signs == None:
            print('*** There are no textures for "shadow_signs"')
            return

        # 3D rotation angles to orient shadow away from light source.
        rot = (0,0,-np.rad2deg(self.light_hangle+np.pi))
        for obj in self.objs:
            if obj.shadow and obj.type in {"person", "plant", "animal", "clutter", "sign"}:
                assert obj.xhalflen == obj.yhalflen

                if obj.type == "plant":
                    txt = txt_plants
                    shdw_hgt = hgt_plant_shadow
                    s = 1                 # include shadow directly below object
                elif obj.type == "sign":
                    txt = txt_signs
                    shdw_hgt = hgt_obj_shadow
                    s = 0                 # do not include shadow below object
                else:
                    txt = txt_misc
                    shdw_hgt = hgt_obj_shadow
                    s = 1                 # include shadow directly below object
                shdw_hgt += np.random.rand()/100

                width = 2*obj.xhalflen
                width = 0.75*width           # narrower shadows look better
                idtags = "shadow_"+txt['id'] # tags to identify this particular object
                txt['numoccur'] += 1     # number of occrances of this texture image

                # Project the object's center high point parallel to the angle
                # of the sun onto the ground plane. (DX,DY) is the position of
                # the projection relative to the center at the ground level. We
                # limit the distance of the shadow in order to keep it near the
                # object that generated it.
                hgt = 2*obj.zhalflen            # height of object
                slen = hgt*hscale + s*width     # length of shadow on ground

                org = (obj.xctr,obj.yctr,shdw_hgt)        # quad origin in world
                corner = (-width/2,-s*width/2,shdw_hgt)  # loc of one corner of quad
                axis1 = (width,0,0)    # vector from corner along texture X axis
                axis2 = (0,slen,0)     # vector from corner along texture Y axis
                reps = (1,1)                       # num. repetitions of texture

                actor = vtu.make_quad(renderer, origin=org, pos=corner,
                                      v1=axis1, v2=axis2, orient=rot, reps=reps,
                                      texture=txt['vtk'], opacity=opacity,
                                      lightcoef=WorldObj.lightcoef, tags=idtags)

                obj.actors.append(actor)

        ########################################################
        # Create shadows for 3D prisms (barriers and buildings).
        ########################################################

        for obj in self.objs:
            if obj.shadow and obj.type in {"barrier", "building"}:
                width = 2*obj.xhalflen
                length = 2*obj.yhalflen
                height = 2*obj.zhalflen
                idtags = "shadow_"+txt['id']    # tags to identify this particular object
                txt['numoccur'] += 1     # number of occrances of this texture image

                # The location of the shadow depends on the projection, parallel
                # to the rays from the light source, of the building's roof onto
                # the ground plane. (DX,DY) is the position of the projection
                # relative to the building's base.
                dx = height*dxscale
                dy = height*dyscale
                dxyz = np.array([dx, dy, 0])

                # Get the four coorners of the base of the building.
                x0 = obj.xctr + obj.xhalflen
                y0 = obj.yctr + obj.yhalflen
                corners = np.array([(x0, y0, hgt_bldg_shadow),               # 0:+,+
                                    (x0, y0-length, hgt_bldg_shadow),        # 1:+,-
                                    (x0-width, y0-length, hgt_bldg_shadow),  # 2:-,-
                                    (x0-width, y0, hgt_bldg_shadow)])        # 3:-,+

                # Get the corners of the polygonal shadow. This is a 6-sided
                # polygon that includes the base of the building.
                if dx >= 0:
                    if dy >= 0:
                        s = np.vstack((corners[[1,2,3],:],corners[[3,0,1],:]+dxyz))
                    else:
                        s = np.vstack((corners[[2,3,0],:],corners[[0,1,2],:]+dxyz))
                else:
                    if dy < 0:
                        s = np.vstack((corners[[3,0,1],:],corners[[1,2,3],:]+dxyz))
                    else:
                        s = np.vstack((corners[[0,1,2],:],corners[[2,3,0],:]+dxyz))

                # Render the shadow in the color world view.
                vtu.make_poly(renderer, origin=(0,0,0), pts=s, orient=(0,0,0),
                              color=(0,0,0), lightcoef=(1,1,1), opacity=opacity,
                              tags='')

        return


    def insert_horiz_objs(self, numobjects, objtype):
        """
        Insert horizontal objects into the world model.

        Description:
            These objects are parallel to the ground plane and sit on or just
            above the ground plane. Their purpose is to provide interesting
            textures on the ground plane. Examples include water, tire tracks,
            dirt spots, and leaves. These objects do not follow the camera.
        """
        objtype = objtype.lower()
        if self.verbosity > 0: print('Creating {} {}s... '.format(numobjects, objtype))

        # Get 2D center locations for objects. We assume object centers may be
        # as close as 2 m (sample == 2 below) to other objects of the same type.
        if objtype == "gndfeat":
            xy_locs = self.FindMapPts(dist=[('road', 2, None),
                                            ('water', 2, None),
                                            ('building', 2, None)],
                                     numpts=100*numobjects, sample=2)
            baseheight = hgt_gndfeat
        else:
            raise Exception('Unknown horizontal object type: {}'.format(objtype))

        numlocs = 0
        maxlocs = xy_locs.shape[0]

        for onum in range(numobjects):
            # Pick a random texture image.
            while True:
                txt = WorldObj.GetTexture(inctags=objtype, exctags='label')
                if txt == None:
                    print('*** There are no textures for "{}"'.format(objtype))
                    return
                tags = txt['tags']
                water_feature = 'water' in tags
                if 'mo' in tags:
                    maxoccur = int(TagValue(tags, 'mo'))
                    if txt['numoccur'] >= maxoccur:
                        # This texture image has already been used to its maximum.
                        continue
                if 'rp' in tags:
                    # Accept this texture with probability "rp".
                    rp = float(TagValue(tags, 'rp'))
                    if np.random.rand() > rp:
                        continue
                break                                   # use this texture image

            width = txt['hsize']
            length = txt['vsize']
            height = baseheight
            idtags = objtype+'_'+txt['id']    # tags to identify this particular object

            # Get an XY location to place the object.
            good_location = False
            while not good_location:
                if numlocs == maxlocs:
                    if self.verbosity > 0:
                        print('*** Ran out of locations on object #', onum)
                    return
                xy = xy_locs[numlocs]
                numlocs += 1
                olabels = self.map3d.get([xy[0],xy[1],width/2,width/2], out="L")
                if water_feature:
                    if set(olabels) - {'ground'} != set():
                        continue    # more than ground here - get new location
                elif set(olabels) - {'ground','barrier'} != set():
                    continue        # more than ground/barrier here - get new location
                good_location = True

            if "z" in tags:
                height = baseheight + float(TagValue(tags, 'z'))

            maplabel = 'water' if 'water' in tags else 'gndfeat'

            # Randomly rotate some objects by 90 degrees.
            rot90 = np.random.rand() > 0.5  # np.random.choice([0,0,1,1,1]) == 1
            if rot90:
                width, length = length, width

            # Add the object to the model.
            txt['numoccur'] += 1     # number of occrances of this texture image
            # rect3d: [xcenter, ycenter, zcenter, xhalfwidth, yhalfwidth, zhalfwidth]
            rect3d = [xy[0], xy[1], height, width/2, length/2, 0.001]
            self.objs.append(WorldObj(self.renderers, self.worldviews, rot90=rot90,
                                      objtags=idtags, rect3d=rect3d))
            self.map3d.set([xy[0], xy[1], width/2, length/2],
                           oid=self.objs[-1].id, olabel=maplabel, oelev=0)

        return


    def insert_vert_objs(self, numobjects=1, objtype='', inctags='', exctags=''):
        """
        Insert vertical objects into the world model.

        Arguments:
            numobjects: (int) Number of objects to insert into the environment.

            objtype: (str) A string specifying the type of object to insert into
            the environment. Default is ''. This determines the class of object
            to insert into the environment and the location constraints.

            inctags: (str) Texture image tags required to be present on objects
            to be inserted into the environment. Default is ''.

            exctags: (str) Texture image tags required to be absent on objects
            to be inserted into the environment. Default is ''.

        Description:
            "p_over_building" and "p_over_road" are parameters that define for
            each class of objects that are allowed to be placed over a building
            or road, the probability that they will be placed over some building
            or road.
        """
        objtype = objtype.lower()
        motiontype = Motion.Static
        if self.verbosity > 0: print('Creating {} {}s...'.format(numobjects, objtype), end="")
        if numobjects <= 0:
            print()
            return

        inctags = objtype+"_"+inctags  # tags to include in texture image search
        exctags += "_label"          # tags to exclude from texture image search

        # "mapscale" is the scale factor (divisor) that determines how large an
        # object will appear in the 2D map. The 2D map is used to make sure
        # objects don't intersect. Objects are drawn actual size when "mapscale"
        # == 2 (since two arguments to Map3D.mapset() are the x and y
        # halfwidths). Some vertical objects, however, are drawn into the map
        # smaller ("mapscale" > 2) than their actual sizes so that these objects
        # can be closer together. Trees, for example, use a large "mapscale" so
        # that their trunks can be close together while their canopies may be
        # overlapping and intermixed.
        mapscale = 2

        # Get 2D center locations for objects. We assume object centers may be
        # as close as 2 m (sample == 2 below) to other objects of the same type.
        if objtype == "airborne":
            # Airborne objects. Tag "z=" gives height above surface below. Note
            # that airborne objects may intersect with plants. This has not, so
            # far, been seen as a problem.
            xy_gnd = self.FindMapPts(eq=['ground'],
                                     neq=['road', 'building'],
                                     numpts=100*numobjects, sample=2)
            xy_road = self.FindMapPts(eq=['road'],
                                     numpts=100*numobjects, sample=2)
            xy_bldg = self.FindMapPts(eq=['building'],
                                      numpts=100*numobjects, sample=4)
            p_over_building = p_over_road = 1/3
        elif objtype == "animal":
            # Animals (non-airborne)
            xy_gnd = self.FindMapPts(dist=[((0,0), 0.75*self.env_radius, None),
                                           ('plant', 1, None),
                                           ('barrier', 0.5, None),
                                           ('clutter', 1, None),
                                           ('water', 1, None),
                                           ('road', 5, None),
                                           ('person', 3, None),
                                           ('building', 20, None)],
                                     numpts=100*numobjects, sample=2, dispfig=False)
            xy_road = self.FindMapPts(eq=['road'],
                                     neq=['person','clutter'],
                                     numpts=100*numobjects, sample=2)
            xy_bldg = self.FindMapPts(eq=['building'],
                                      dist=[('ground', 0, 0.2)],
                                      numpts=100*numobjects, sample=None)
        elif objtype == "urban_animal":
            # Small, urban animals (non-airborne)
            objtype = "animal"
            inctags += "_urban"
            xy_gnd = self.FindMapPts(dist=[((0,0), None, 0.75*self.env_radius),
                                           ('plant', 0.25, None),
                                           ('barrier', 0.25, None),
                                           ('clutter', 0.25, None),
                                           ('water', 0.25, None),
                                           ('road', 0.25, None),
                                           ('person', 1, None),
                                           ('building', 0.25, None)],
                                     numpts=100*numobjects, sample=2)
            xy_road = self.FindMapPts(eq=['road'],
                                     neq=['person','clutter'],
                                     numpts=100*numobjects, sample=2)
            xy_bldg = self.FindMapPts(eq=['building'],
                                      dist=[('ground', 0, 0.2)],
                                      numpts=100*numobjects)
        elif objtype == "plant":
            # Any type of plant.
            # mapscale = 8
            xy_gnd = self.FindMapPts(dist=[('road', 1, None),
                                           ('barrier', 0.5, None),
                                           ('person', 0.5, None),
                                           ('animal', 0.5, None),
                                           ('clutter', 0.5, None),
                                           ('water', 0.5, None),
                                           ('building', 3, None)],
                                     numpts=10*numobjects, sample=2)
            xy_road = self.FindMapPts(eq=['road'],
                                     neq=['person','clutter'],
                                     numpts=100*numobjects, sample=2)
            xy_bldg = self.FindMapPts(eq=['building'],
                                      dist=[('ground', 0, 0.2)],
                                      numpts=100*numobjects, sample=None)
        elif objtype == "bldg_plant":
            # Plants (but not trees) around buildings.
            objtype = "plant"
            inctags = "plant"
            exctags += "_tree"
            # mapscale = 8
            bldgmaxdist = 3 if self.num_buildings > 0 else None
            xy_gnd = self.FindMapPts(dist=[('water', 1, None),
                                           ('barrier', 0.5, None),
                                           ('person', 0.5, None),
                                           ('animal', 0.5, None),
                                           ('clutter', 0.5, None),
                                           ('road', 2, None),
                                           ('building', 1, bldgmaxdist)],
                                     numpts=100*numobjects, sample=1, dispfig=False)
            xy_road = np.zeros((0,2))                # no locations on roads
            xy_bldg = np.zeros((0,2))                # no locations on buildings
        elif objtype in ["person", "vehicle", "clutter"]:
            # People, vehicles and clutter
            bldgmaxdist = 20 if self.num_buildings > 0 else None
            xy_gnd = self.FindMapPts(dist=[('plant', 1, None),
                                           ('barrier', 0.5, None),
                                           ('person', 0.5, None),
                                           ('clutter', 1, None),
                                           ('water', 1, None),
                                           ('road', 2, None),
                                           ('building', 2.5, bldgmaxdist)],
                                     numpts=100*numobjects, sample=3, dispfig=False)
            xy_road = self.FindMapPts(eq=['road'],
                                     neq=['animal'],
                                     dist=[('plant', 0.5, None),
                                           ('barrier', 0.5, None),
                                           ('person', 0.5, None),
                                           ('clutter', 0.5, None)],
                                     numpts=100*numobjects, sample=2)
            xy_bldg = self.FindMapPts(eq=['building'],
                                      dist=[('ground', 0.5, 2)],
                                      numpts=100*numobjects, sample=2)
        else:
            raise ValueError('Unrecognized object type: "{}"'.format(objtype))

        p_over_building = self.p_over_building[objtype]
        p_over_road = self.p_over_road[objtype]

        numroadlocs = xy_road.shape[0]
        numbuildinglocs = xy_bldg.shape[0]

        xy_gnd = list(xy_gnd)
        onum = 0
        numfailures = 0

        while onum < numobjects:
            # Pick a random texture image.
            while True:
                txt = WorldObj.GetTexture(inctags=inctags, exctags=exctags)
                if txt == None:
                    print('\n*** There are no textures for "{}"'.format(objtype))
                    return
                tags = txt['tags']
                if 'mo' in tags:
                    maxoccur = int(TagValue(tags, 'mo'))
                    if txt['numoccur'] >= maxoccur:
                        # This texture image has already been used to its maximum.
                        continue
                if 'rp' in tags:
                    # Accept this texture with probability "rp".
                    rp = float(TagValue(tags, 'rp'))
                    if np.random.rand() > rp:
                        continue
                break

            # Get the size of the object.
            if objtype == "plant":
                sizescale = 1 + min(0.25,max(-0.25,np.random.normal(scale=self.plant_std)))
            else:
                sizescale = 1.0
            width = txt['hsize']*sizescale     # assume that length == width
            height = txt['vsize']*sizescale

            idtags = objtype+'_'+txt['id']     # tags to identify this particular object
            zctr = height/2                    # vertical center of object

            if objtype == 'airborne':
                # Get height of this airborne object above objects below it.
                hgt_above = TagValue(tags, "z")
                if hgt_above == '':
                    # No height specified. Use default value.
                    hgt_above = 1.0
                elif ',' in hgt_above:
                    # Pick a random height in the given range.
                    hmin, hmax = hgt_above.split(',')
                    hgt_above = np.random.uniform(float(hmin), float(hmax))
                else:
                    # Use the specified height.
                    hgt_above = float(hgt_above)

            # Get an XY location to place the object.
            elev = height                           # elevation at top of object
            good_location = False
            num_gnd = len(xy_gnd)
            gnd_index = 0
            motiontype = Motion.Random
            while not good_location:
                if 'b' in tags and numbuildinglocs > 0 and np.random.rand() <= p_over_building:
                    # Place above or on top of a building.
                    xy = xy_bldg[numbuildinglocs-1]
                    numbuildinglocs -= 1
                    olabels, oids = self.map3d.get([xy[0],xy[1],width/mapscale,
                                                    width/mapscale], out="LI")
                    if 'building' not in olabels:
                        numfailures += 1
                        continue                      # try a different location
                    oindex = oids[olabels.index('building')] - 1
                    assert self.objs[oindex].type == 'building'
                    if objtype == "animal":
                        # Put animal on edge of the building.
                        xy = self.objs[oindex].closestBBPnt(xy[0], xy[1], dst2edge=0)
                    bldghght = 2*self.objs[oindex].zhalflen    # building height
                    elev += bldghght
                    zctr += bldghght              # place object on top of building
                    if objtype != "airborne":
                        motiontype = Motion.Static
                elif 'r' in tags and numroadlocs > 0 and np.random.rand() <= p_over_road:
                    # Place above or on a road.
                    xy = xy_road[numroadlocs-1]
                    numroadlocs -= 1
                    olabels, oids = self.map3d.get([xy[0],xy[1],width/mapscale,
                                                    width/mapscale], out="LI")
                    if 'road' not in olabels or \
                             set(olabels)-{'road','gndfeat','ground'} != set():
                        # Conflicts here. Try a different location.
                        numfailures += 1
                        continue
                else:
                    # Place object above or on the ground.
                    if gnd_index == num_gnd:
                        numfailures += 1
                        break             # have run out of ground locations
                    xy = xy_gnd[gnd_index]
                    olabels, oids = self.map3d.get([xy[0],xy[1],width/mapscale,
                                                    width/mapscale], out="LI")
                    if objtype == 'airborne':
                        z = max([2*self.objs[oids[k]-1].zhalflen for k in range(len(oids))])
                        if 'airborne' in olabels or z >= hgt_above:
                            # Already an airborne object here, or new airborne
                            # object is not high enough to clear objects below.
                            numfailures += 1
                            gnd_index += 1
                            continue      # there's already an airborne object here
                    elif set(olabels)-{'gndfeat', 'ground', 'plant'} != set():
                        numfailures += 1
                        gnd_index += 1
                        continue          # more than ground here - try next location
                    xy_gnd.pop(gnd_index) # found a good ground location

                good_location = True

            if not good_location:
                if self.verbosity > 0 and numfailures >= numobjects:
                    print('\n*** Could place only', onum, objtype, 'objects.', end="")
                    break
                continue                     # try again with a different object
            onum += 1

            if objtype == 'airborne':
                # Elevate the object above the surface below it. The elevation
                # of an airborne object is the elevation of its bottom, not
                # its top as for ground-based objects. This is needed to detect
                # collisions between ground and airborne objects.
                elev += hgt_above - height
                zctr += hgt_above
            else:
                zctr += hgt_obj_shadow    # this raises object just above shadow

            if False:
                print(f"Created {objtype} at pos ({xy[0]},{xy[1]})")

            self.insert_obj(txt, xy[0], xy[1], zctr, elev, width, height,
                            motiontype=motiontype)

            if False:
                if "clutter" in tags:
                    sound = None           # assume clutter is stationary and silent
                else:
                    soundtags = tags & set(WorldObj.tag2soundid.keys())
                    if len(soundtags) > 0:
                        # Create a sound for the object. First choose a single random
                        # sound tag from the set of sound tags that the object may
                        # generate, then choose a random sound matching that tag. The
                        # object is also assigned a random position in its audio signal
                        # to be the sample at time zero (the start of the simulation).
                        t = np.random.choice(list(soundtags))
                        sid = np.random.choice(list(WorldObj.tag2soundid[t]))
                        sound = WorldObj.sounds[sid].copy()
                        WorldObj.sounds[sid]['numoccur'] += 1
                        dvol = 1 - 0.25*np.random.rand()        # perturb volume
                        signal = dvol*sound['signal']           # amplitude scale
                        tscale = 2*np.random.rand() + 0.5       # time scale
                        xp = np.arange(0, sound['nsamples'])    # original sample points
                        nsamples_new = int(np.ceil(tscale*sound['nsamples']))
                        x = np.linspace(0, sound['nsamples']-1, num=nsamples_new)
                        sound['signal'] = np.interp(x, xp, signal)
                        sound['nsamples'] = nsamples_new
                        sound['duration'] = tscale*sound['duration']
                        sound['samplezero'] = np.random.randint(nsamples_new)  # start position of signal
                        sound['dvolume'] = dvol
                    else:
                        sound = None

                # Add the object to the model. The object is rendered on a flat 3D
                # rectangle that orients towards the viewer. The position of the
                # rectangle is defined by the lower left 3D corner (pos) and two 3D
                # vectors (axis1, axis2) for the horizontal and vertical axis,
                # respectively, of the texture image.
                txt['numoccur'] += 1     # number of occrances of this texture image
                newobj = WorldObj(self.renderers, self.worldviews, objtags=idtags,
                                  pos=[xy[0]-width/2,xy[1],zctr-height/2],
                                  static=static_obj, axis1=[width,0,0], axis2=[0,0,height])
                newobj.mapscale = mapscale    # draw objs in map smaller than actual size
                newobj.rect = [xy[0], xy[1], width/mapscale, width/mapscale]
                newobj.elev = elev
                newobj.tags = tags
                newobj.sound = sound
                self.objs.append(newobj)

                if sound != None:
                    # Keep a list of all objects that can make noise.
                    self.noise_makers.append(newobj)

                if 'tr' in tags:
                    # Create a ceiling around the tree trunk underneith the tree branches.
                    pos = np.array([float(n) for n in TagValue(tags,'tr').split('x')])
                    ceil = ('post', *pos)
                else:
                    ceil = None

                newobj.maprect = self.map3d.set(newobj.rect, oid=newobj.id,
                                                olabel=objtype, oelev=newobj.elev,
                                                ceiling=ceil)

                # Add the object's ID to the object's actors. This is used to make
                # moving actors face the right direction while following the camera.
                for actor in newobj.actors:
                    p = actor.GetProperty()
                    p.id = newobj.id
                    actor.SetProperty(p)

                self.followers.extend(newobj.actors)
                newobj.dynamic = self.dynamic_env and newobj.maxspeed > 0
                if newobj.dynamic:
                    newobj.forward_dir = TagValue(newobj.tags, 'f')
                    self.movers.append(newobj)

                if self.show_obj_ids:
                    # Display each object's ID overlayed on the color image.
                    actor = vtu.make_text(self.renderers[1], text=str(newobj.id),
                                          textscale=0.3, pos=(xy[0], xy[1], zctr))
                    newobj.actors.append(actor)

        if self.verbosity > 0: print(" ({})".format(numfailures))
        return


    def insert_obj(self, txt:dict, x: float, y:float, zctr:float,
                   elev:float, width:float, height:float,
                   mapscale:int=2, motiontype:Motion=Motion.Random,
                   path=None):
        """
        Arguments:
            txt:dict - Dictionary holding all information about the object's
            texture.

            path: numpy.ndarry - An Nx4 Numpy arrary that specifies a predefined
            trajectory for the object. Each row is [t, x, y, z] giving the time
            and positon (x,y,z) of the object at that time. This is only used
            when the object's `motiontype` is Motion.DefinedPath.
        """

        tags = txt['tags']
        objtype, objid = TagTypeID(tags)
        idtags = objtype + '_' + txt['id']     # tags to identify this particular object

        if "clutter" in tags:
            sound = None           # assume clutter is stationary and silent
        else:
            soundtags = tags & set(WorldObj.tag2soundid.keys())
            if len(soundtags) > 0:
                # Create a sound for the object. First choose a single random
                # sound tag from the set of sound tags that the object may
                # generate, then choose a random sound matching that tag. The
                # object is also assigned a random position in its audio signal
                # to be the sample at time zero (the start of the simulation).
                t = np.random.choice(list(soundtags))
                sid = np.random.choice(list(WorldObj.tag2soundid[t]))
                sound = WorldObj.sounds[sid].copy()
                WorldObj.sounds[sid]['numoccur'] += 1
                dvol = 1 - 0.25*np.random.rand()        # perturb volume
                signal = dvol*sound['signal']           # amplitude scale
                tscale = 2*np.random.rand() + 0.5       # time scale
                xp = np.arange(0, sound['nsamples'])    # original sample points
                nsamples_new = int(np.ceil(tscale*sound['nsamples']))
                xnew = np.linspace(0, sound['nsamples']-1, num=nsamples_new)
                sound['signal'] = np.interp(xnew, xp, signal)
                sound['nsamples'] = nsamples_new
                sound['duration'] = tscale*sound['duration']
                sound['samplezero'] = np.random.randint(nsamples_new)  # start position of signal
                sound['dvolume'] = dvol
            else:
                sound = None

        # Add the object to the model. The object is rendered on a flat 3D
        # rectangle that orients towards the viewer. The position of the
        # rectangle is defined by the lower left 3D corner (pos) and two 3D
        # vectors (axis1, axis2) for the horizontal and vertical axis,
        # respectively, of the texture image.
        txt['numoccur'] += 1     # number of occrances of this texture image
        newobj = WorldObj(self.renderers, self.worldviews, objtags=idtags,
                          pos=[x-width/2,y,zctr-height/2], motiontype=motiontype,
                          axis1=[width,0,0], axis2=[0,0,height])
        newobj.mapscale = mapscale    # draw objs in map smaller than actual size
        newobj.rect = [x, y, width/mapscale, width/mapscale]
        newobj.elev = elev
        newobj.tags = tags
        newobj.sound = sound
        self.objs.append(newobj)

        if sound != None:
            # Keep a list of all objects that can make noise.
            self.noise_makers.append(newobj)

        if 'tr' in tags:
            # Create a ceiling around the tree trunk underneith the tree branches.
            pos = np.array([float(n) for n in TagValue(tags,'tr').split('x')])
            ceil = ('post', *pos)
        else:
            ceil = None

        newobj.maprect = self.map3d.set(newobj.rect, oid=newobj.id,
                                        olabel=objtype, oelev=newobj.elev,
                                        ceiling=ceil)

        # Add the object's ID to the object's actors. This is used to make
        # moving actors face the right direction while following the camera.
        for actor in newobj.actors:
            p = actor.GetProperty()
            p.id = newobj.id
            actor.SetProperty(p)

        self.followers.extend(newobj.actors)
        newobj.dynamic = (motiontype == Motion.DefinedPath) or \
                         (self.dynamic_env and newobj.maxspeed > 0)
        if newobj.dynamic:
            newobj.forward_dir = TagValue(newobj.tags, 'f')
            newobj.defined_path = path
            self.movers.append(newobj)

        if self.show_obj_ids:
            # Display each object's ID overlayed on the color image.
            actor = vtu.make_text(self.renderers[1], text=str(newobj.id),
                                  textscale=0.3, pos=(xy[0], xy[1], zctr))
            newobj.actors.append(actor)


    def insert_fixed_path_objs(self, obj_defs):
        """
        Insert moving objects that follow fixed paths.

        Arguments:
            obj_defs:list -- A list of object definitions. Each object
            definition is a list [tags, txyz]. `tags` is a string giving the
            texture tags of the object and `txyz` is a list of the positions of
            the object at various times throughout its trajectory. Each element
            of txyz is a list [t, x, y, z] giving the position (x,y,z) of the
            object at a time t. The elements of the list `txyz` must be sorted
            in order of increasing time.

        """

        cnt = 0
        self.obj_defs = []

        for k in range(len(obj_defs)):
            tags1 = obj_defs[k][0]
            txyz = obj_defs[k][1]

            txt = WorldObj.GetTexture(inctags=tags1, exctags='label')
            if txt == None:
                raise ValueError(f'There are no textures for "{tags}"')
            tags = txt['tags']

            if txyz == []:
                raise ValueError(f'Missing time and position data for object'\
                                 f'#{cnt+1}, {tags1}')

            t, x, y, z = txyz[0]
            width = txt['hsize']
            height = txt['vsize']
            zctr = z + height/2        # center of object in z-direction
            elev = z + height               # elevation at top of object

            self.insert_obj(txt, x, y, zctr, elev, width, height,
                            motiontype=Motion.DefinedPath,
                            path=np.array(txyz, dtype=float))
            cnt += 1


        print(f'Created {cnt} objects with predefined trajectories')


    def get_audio(self, micpos, duration=3.0, maxdist=300, map2d=None,
                  verbose=False):
        """
        Get an audio recording of the environment from a specific location.

        Usage:
            audio = SimWorld.get_audio(micpos, duration=3.0, maxdist=500,
                                       map2d=None, verbose=False)

        Arguments:
            micpos: (3D array-like) 3D position of microphone.

            duration: (float) Duration (seconds) of the recording to fetch from
            the microphone. The time period of the recording will begin at
            `duration` seconds prior to the current time and end at the current
            time.

            maxdist: (float) The maximum distance (meters) of any object from
            the microphone for its audio signal to be heard by the microphone.
            This is used to filter out objects that are too far away to
            be heard, to speed up the computation.

            map2d: (Map2D) The 2D map of the environment. This is used only
            for displaying the ground truth audio tracks on the 2D map display.
            If `verbose` is False or `map2d` is None, then the ground truth
            audio tracks are not displayed.

            verbose: (bool) If True, print information about ground truth audio
            tracks and, if `map2d` is not None, display these tracks on the 2D
            map.

        Returns:
            audio: (dict) Dictionary with the following keys:
                'signal': The audio signalrecorded by the microphone during the
                previous `duration` seconds. The sample rate of this signal is
                the default sample rate for the simulation.

                'samplerate': The sample frequency (Hz) of the signal.

                'psd': The power spectal density of the recorded audio signal.

                'freq': The frequency (Hz) at each sample in `psd`.

        Description:
            The amplitude of a sound wave is scaled by 1/d when the wave has
            traveled a distance d through the air.
        """

        show_plots = False                    # plot all measured audio signals?

        curtime = self.time              # time (sec.) since start of simulation
        sample_rate = self.audio_sample_rate
        rec_samples = np.ceil(duration*sample_rate).astype(int)
        sum_signal = np.zeros(rec_samples)
        cnt = 0
        amp_max = 0
        f = None
        anum = 0

        if verbose:
            print(f'  Mic pos=({micpos[0]:.1f},{micpos[1]:.1f}), rec_samples={rec_samples}')
            showed_heading = False

        for obj in self.noise_makers:
            objpos = [obj.xctr, obj.yctr, obj.zctr]
            d = np.array(objpos) - micpos
            d = np.sqrt(d.dot(d))                     # distance of sound source
            d = max(d, 1e-4)                           # prevents divide by zero
            # print(f'  >> {"_".join(sorted(list(obj.keytags), reverse=True))}: '
                  # f'micpos=({obj.xctr:.1f},{obj.yctr:.1f}), dist={d:.1f}')
            if d > maxdist:
                continue

            # Number of building cells in 3D map that line-of-sight intersects with.
            bldgcnt = self.map3d.IntersectCount(micpos, objpos, 'building')

            if bldgcnt >= 2:
                continue

            # Scale factor for the audio signal's amplitude based on the
            # distance of the object relative to that which the original
            # audio was recorded.
            scale = min(WorldObj.audio_max_scale, audio_record_dist/d)

            sound = obj.sound
            num_src_samples = sound['nsamples']

            # Each object that produces audio was previously assigned a random
            # position in its signal to be the sample at time zero (the start of
            # the simulation). Start the current recording `duration` seconds
            # prior to "time zero" + "current time".
            p0 = sound['samplezero'] + round(sample_rate*curtime) # end of recording
            if p0 >= num_src_samples:
                p0 -= int(p0/num_src_samples)*num_src_samples
            p0 -= rec_samples                    # back up to start of recording
            if p0 < 0:
                p0 += (int(-p0/num_src_samples) + 1)*num_src_samples
            assert p0 >= 0 and p0 < num_src_samples

            nrecorded = 0
            nremainig = rec_samples
            src_signal = sound['signal']

            if verbose:
                if not showed_heading:
                    # Show the heading only one time.
                    print(f'    {"Obj":<15s} {"Sound":<20s} {"ObjPos":<16s}'
                          f' {"Dist":>6s} {"Scale":>8s} {"MaxSAmp":>9s}')
                    showed_heading = True

                t = list(sound['tags'])
                t.sort(reverse=True)
                vmax = 0
                # vmax = scale*max(abs(src_signal.min()), abs(src_signal.max()))
                print(f'    {"_".join(sorted(list(obj.keytags), reverse=True)):15s}'
                      f' {"_".join(t):19s} '
                      f' ({obj.xctr:<6.1f} {obj.yctr:>6.1f}) {d:>7.1f} '
                      f' {scale:>8.1e} ', end='')
                      # f' {scale:>8.1e} {vmax:>7.1f}')
                if map2d:
                    c = audio_cmap(float(min(1.0, 2*scale)))
                    map2d.line(micpos, [obj.xctr, obj.yctr], color=c, alpha=1,
                               linestyle=':', linewidth=1)

            if show_plots:
                # Plot the original signal.
                cnt += 1
                tags = "_".join(sound['tags'])
                t_max = max(rec_samples, sound['nsamples'])
                amp_max = max(amp_max, np.abs(src_signal).max())
                f = plotsignal(f, sig=src_signal, anum=anum, tmax=t_max, amax=amp_max,
                               title=f'#{cnt}: {tags}, d={d:.1f}, scale={scale:.1e}',
                               xmarks=[p0,p0+rec_samples])
                anum = (anum + 1) % 3

            while nrecorded < rec_samples:
                p1 = min(p0+nremainig-1, num_src_samples-1)
                num = p1 - p0 + 1
                sum_signal[nrecorded:nrecorded+num] += scale*src_signal[p0:p0+num]
                if verbose:
                    vmax = max(vmax, abs(scale*src_signal[p0:p0+num]).max())
                nrecorded += num
                nremainig -= num
                p0 += num
                if p0 >= num_src_samples:
                    p0 = 0

            if verbose:
                print(f'{vmax:>7.1f}')

            if show_plots:
                # Plot the summed signal.
                amp_max = max(amp_max, np.abs(sum_signal).max())
                f = plotsignal(f, sig=sum_signal, anum=anum, tmax=rec_samples,
                               amax=amp_max, title=f'Sum {cnt}')
                print(f'Cnt = {cnt}')
                anum = (anum + 1) % 3

        # Estimate power spectral density using a periodogram.
        freq, psd = sig.periodogram(sum_signal, sample_rate)

        # Save the recorded sound.
        audio = {}
        audio['signal'] = sum_signal
        audio['samplerate'] = audio_sample_rate
        audio['psd'] = psd
        audio['freq'] = freq

        if False:
            # Show final audio signal and its PSD.
            amp_max = np.abs(sum_signal).max()
            f = plotsignal([2,1], sig=sum_signal, anum=0, tmax=rec_samples,
                           amax=amp_max, title='Audio Signal')
            f = plotsignal(f, anum=1, psd=psd, freq=freq, title='Power Spectrum')

        return audio


    def NewAgent(self, x, y):
        """
        Insert a mobile agent into the world model. Since the agent is mobile,
        it is not added to the 2D map.
        """
        inc_tags = {'vehicle','robot'}
        exc_tags = {}

        # Get a texture image for the agent.
        txt = WorldObj.GetTexture(inctags=inc_tags, exctags=exc_tags)
        if txt == None:
            raise Exception('There are no textures for "{}"'.format(inc_tags))

        width = txt['hsize']               # assume that length == width
        height = txt['vsize']
        zctr = height/2
        idtags = inc_tags | {txt['id']}    # tags to identify this particular object

        newobj = WorldObj(self.renderers, self.worldviews, objtags=idtags,
                          pos=[x-width/2,y,zctr-height/2],
                          axis1=[width,0,0], axis2=[0,0,height])
        self.objs.append(newobj)
        self.followers.extend(newobj.actors)

        # self.map3d.set([x, y, width, width], oid=newobj.id, olabel='vehicle')

        return newobj


    def FindMapPts(self, eq=[], neq=[], dist=[], numpts=100, sample=None,
                   dispfig=False):
        """
        Find map points that satisfy certain conditions.

        Usage:
            xy = SimWorld().GetMapLocs(conds=None, numpts=100, combine='Intersect')

        Arguments:
            eq: (list) List of labels (strings such as 'ground', 'building',
            'plant', etc.) that must be true.

            neq: (list) List of labels that must be false.

            dist: (list) List of (LABEL, LOW, HIGH) tuples which describe
            distance conditions. LABEL is a str. LOW and HIGH are nonnegative
            floats or ints. A map cell satisfies the condition (LABEL, LOW,
            HIGH) if the distance of the map cell to the closest cell with label
            LABEL is in the range [LOW, HIGH]. At most one of LOW or HIGH may be
            None. If LOW is None, this represents the condition distance <=
            HIGH. If HIGH is None, this represents the condition distance >=
            LOW. The "DIST" conditions assumes that map cells with label ==
            LABEL never satisfy the search condition.

            numpts: (int) Number of map points to return in XY.  If more than
            NUMPTS satisfy all conditions, then a random selection of NUMPTS of
            these are returned to the calling routine.

            sample: (int or None) If not None, then subsample final satisfying
            map points with a grid spacing of SAMPLE. Default is None.

            dispfig: (bool) Display a figure showing intermediate steps. Default
            is False.

        Returns:
            xy: (numpy.ndarray) A Nx2 Numpy array of map points. Each point
            gives the world coordinates, [X,Y], of one point that satisfies all
            conditions in the conds= argument.
        """

        maplabel = self.map3d.maplabel
        mgridspc = self.map3d.mgridspc
        mdim = self.map3d.mdim

        # Initialize the set of map points that satisfy all conditions.
        satall = np.ones_like(maplabel[:,:,0], dtype=bool)

        if dispfig:
            # Flatten the multi-channel label map into a single channel map.
            flatlmap = self.map3d.flatlabels()
            fig = Fig(figsize=(6,8), axpos=[321,322,323,324,325,326],
                      link=[0,1,2,3,4],
                      figtitle='Map points: red is go, blue is no-go')
            cmap = colors.ListedColormap(np.array(label_colors)/255)
            fig.set(axisnum=0, image=flatlmap, axistitle='Map', axisfontsize=6,
                    vmin=0, vmax=self.map3d.map_nchan-1, cmap=cmap)
            print('\n==> Press any key on figure to continue')
            fig.wait(event='key_press_event')

        # Find map cells that satify all equality conditions.
        eqneqcond = False
        if eq == []:
            satall = np.ones_like(maplabel[:,:,0])
        else:
            intlabels = [label2id[l] for l in eq]
            satall = np.all(maplabel[:,:,intlabels], axis=2)
            eqneqcond = True
            if dispfig:
                fig.set(axisnum=1, image=satall, axisfontsize=6,
                        axistitle='EQ:'+' '.join(eq))
                fig.set(axisnum=4, image=satall, axistitle='SatAll', axisfontsize=6)
                fig.wait(event='key_press_event')

        # Find map cells that satify all inequality conditions.
        if neq != []:
            intlabels = [label2id[l] for l in neq]
            satcur = np.all(maplabel[:,:,intlabels]==0, axis=2)
            satall = np.logical_and(satcur, satall)
            eqneqcond = True
            if dispfig:
                fig.set(axisnum=1, image=satcur, axisfontsize=6,
                        axistitle='NEQ:'+','.join(neq))
                fig.set(axisnum=4, image=satall, axistitle='SatAll',
                        axisfontsize=6)
                fig.wait(event='key_press_event')

        for cond in dist:
            assert type(cond[0]) == tuple or cond[0] in label2id, \
                   '"Distance from" must be a 2D point or string: {}'.format(cond[0])
            low = cond[1]
            high = cond[2]
            assert low != None or high != None, 'One of LOW or HIGH must be non-None'

            # The function "distance_transform_edt()" computes the distance of
            # each nonzero (True) point to the closest zero (False) point. We
            # need the inverse, the distance of points to the closest point with
            # label "intlabel".
            if type(cond[0]) == str:
                # Get the distance to a specified map label.
                intlabel = label2id[cond[0]]
                im1 = maplabel[:,:,intlabel] == 0       # inverse image
            else:
                # Get the distance to a specified 2D world point. The world
                # coordinate system has the origin at the center of the map,
                # X increases moving to the right, and Y increases moving up.
                im1 = np.ones((mdim, mdim))
                im1[int(np.ceil(mdim/2 + cond[0][0]/mgridspc)),
                    int(np.ceil(mdim/2 - cond[0][1]/mgridspc))] = 0

            if eqneqcond:
                im1 = np.logical_or(im1, satall)
                eqneqcond = False
            dt = ndi.distance_transform_edt(im1) # distance to closest zero value

            if low == None:
                dmax = int(np.ceil(high/mgridspc))  # max distance in map coords
                satcur = dt <= dmax
            elif high == None:
                dmin = int(np.ceil(low/mgridspc))   # min distance in map coords
                satcur = dt >= dmin
            else:
                dmin = int(np.ceil(low/mgridspc))   # min distance in map coords
                dmax = int(np.ceil(high/mgridspc))  # max distance in map coords
                satcur = np.logical_and(dt >= dmin, dt <= dmax)

            satall = np.logical_and(satcur, satall)

            if dispfig:
                fig.set(axisnum=1, image=im1.astype(int), axisfontsize=6,
                        axistitle='DT Input:'+str(cond[0]))
                fig.set(axisnum=2, image=dt, axistitle='DT', axisfontsize=6)
                fig.set(axisnum=3, image=satcur.astype(int), axisfontsize=6,
                        axistitle='Dist=['+str(low)+','+str(high)+']')
                fig.set(axisnum=4, image=satall.astype(int),
                        axistitle='SatAll', axisfontsize=6)
                fig.wait(event='key_press_event')

        if sample == None:
            sample = 1
        else:
            # Subsample the map of satified points to help spread out the points.
            sample = int(np.ceil(sample/mgridspc))  # subsample distance in map coords.
            satall = satall[::sample,::sample]

        # Remove points that are outside the radius of environement model.
        ctr = int(round(mdim/(2*sample)))        # coordinates of subsampled center cell
        r = 0.95*self.env_radius/(mgridspc*sample) # radius of valid locations in subsampled cells
        rr, cc = draw.ellipse(ctr,ctr,r,r,shape=satall.shape)  # coordinates of pts in disk
        im2 = np.zeros_like(satall)              # create image to mask out ...
        im2[rr,cc] = 1                           # ... points outside of city radius
        satall[im2 == 0] = 0                     # pts outside of disk are invalid
        if dispfig:
            fig.set(axisnum=5, image=satall.astype(int),
                    axistitle='Sampled SatAll', axisfontsize=6)

        rc = sample*np.argwhere(satall > 0)      # [R,C] indicies of valid object centers

        # Convert map cell row/col coordinates to world coordinates
        yx = rc*np.array([1/self.map3d.mcc, 1/self.map3d.mca]) \
               - np.array([self.map3d.mcb/self.map3d.mcc, self.map3d.mcb/self.map3d.mca])
        xy = yx[:,[1,0]]                 # swap columns so coordinates are [X,Y]
        numpos = xy.shape[0]
        numpts = min(numpts, numpos)
        samples = np.random.choice(numpos, numpts, replace=False)
        xy = xy[samples,:]

        if dispfig:
            print('==> Close the figure to continue')
            fig.wait()
            fig.close()

        return xy


    def pantiltpos(self, pos, orient, objtype='person', pan=(-180,180),
                   tilt=(-90,90)):
        """
        Get the range of pan/tilt viewing angles of a set of objects.

        Usage:
            objlist = pantiltpos(self, pos, orient, objtype='person',
                                 pan=(-180,180), tilt=(-90,90))

        Arguments:
            pos: The observer's position, a tuple (X,Y,Z).
            orient: The orientation of the observer's zero pan/tilt, a
                tuple (rx, ry, rz), of rotation angles.
            objtype: The object type, a string.
            pan: Minimum and maximum pan angles, in degrees, a 2-tuple.
            tilt: Minimum and maximum tilt angles, in degrees, a 2-tuple.

        Description:
            Get a list Blob() objects that identify the pan/tilt view angles of
            the 3D bounding boxes of the specified object type from the
            specified position and orientation. Only objects in the specified
            range of pan/tilt angles are returned.

        Usage:
            objlist = SimWorld().pantiltpos(obspos, obsorient, objtype='person',
                                            pan=(-180,180), tilt=(-90,90))

        Returns:
            objlist: List of Blob() objects.
        """

        """
        objlist = []

        pos = np.array(pos)
        invrot = Rot3d(angles=-np.array(orient))

        for obj in self.objs:
            if basetype(obj.type) == objtype:

                # Translate and then rotate the world points into the agent's coordinate
                # system.
                w = np.vstack((x,y,z))    # 3xN vector needed for matrix multiplications
                w = w - np.array(self.pos)
                w = self.invrot*w

                wx = np.squeeze(np.asarray(w[0,:]))        # final world coordinates
                wy = np.squeeze(np.asarray(w[1,:]))
                wz = np.squeeze(np.asarray(w[2,:]))

                # Get the pan/tilt angle from the world coordinates. Note: np.arctan2()
                # returns positive values for clockwise angles. Camera pan angles are
                # positive for counter-clockwise angles, so we negate the pan angle
                # below.
                d = np.sqrt(wx**2 + wy**2)
                tilt = np.rad2deg(np.arctan(wz/d))        # tilt in [-90,90] degrees
                pan = -np.rad2deg(np.arctan2(wx,wy))      # pan range [-180,180] degrees

        return objlist
        """
        raise Exception('Function SimWorld.pantiltpos() is incomplete.')


    def close(self):
        """
        Close the SimWorld() object.
        """
        for renwin in self.renwindow:
            renwin.Finalize()
            del renwin
        for iren in self.interactor:
            iren.TerminateApp()
            del iren
        del self.objs


    def reset_time(self):
        """
        Reset the simulation time to 0.

        Usage:
            SimWorld.reset_time()
        """
        self.time = 0
        self.time_last = 0


    def inc_time(self, dtime:float, movedynobj:bool=True):
        """
        Increment the current simulation time by a set amount.

        Usage:
            SimWorld.inc_time(dtime)

        Arguments:
            dtime: (float) Number of seconds to increment the time by.

            movedynobj:(bool) Should the position of all dynamic objects be
            updated? Default is True. Sometimes, it may be desirable for
            efficiency reasons, to postpone dynamic object updates to a later
            time.

        Description:
            Dynamic objects are updated based on the new time.
        """
        assert isinstance(dtime, Number) and dtime >= 0, \
               'Argument "dtime" must be a nonnegative number'
        self.time += dtime
        if movedynobj:
            self.update_dyn_objs()


    def set_time(self, curtime):
        """
        Set the current time of the simulation.

        Usage:
            SimWorld.set_time(curtime)

        Arguments:
            curtime: (float) Time in seconds since start of simulation.

        Description:
            Dynamic objects are updated based on the new time.
        """
        if curtime < self.time:
            raise ValueError('Cannot move back in time: cur={} < last={}'.format(
                              curtime, self.time_last))
        self.time = curtime
        self.update_dyn_objs()


    def timestr(self):
        """
        Return the current time as a string HH:MM:SS.RR.
        """
        t = self.time
        hh = int(t/3600)
        t -= hh*3600
        mm = int(t/60)
        t -= mm*60
        ss = int(t)
        rr = int(round(100*(t - ss)))
        return '{:02d}:{:02d}:{:02d}.{:02d}'.format(hh,mm,ss,rr)





    def you_drive(self, cam, holdpos=False, map2d=None):
        """
        The user interactively moves around the world while rendering the scene.

        Usage:
            SimWorld.you_drive(cam, holdpos=False, map2d=None)

        Arguments:
            cam: The camera model to use in rendering the scene. This gives the
            initial position, orientation, and viewing angle of the camera.

            holdpos: (bool) If True, do not allow camera translation, allow only
            camera pan, tilt, and zoom.

            map2d: (Map2D) If not None, this is the 2D map on which to display
            the camera position. Default is None.

        Description:
            Camera control is via the vtkRenderWindowInteractor that is
            associated with the 'color camera' world view. So, this window must
            be on top of others.
        """

        if self.offscreenrender:
            raise Exception('Cannot interactively move camera with off-screen rendering.')

        if self.verbosity > 0: print('Rendering...')

        # Setup the VTK renderer's camera to match the input camera.
        pos, fp = cam.get_pos_fp()
        self.vtkcamera.SetPosition(pos)
        self.vtkcamera.SetFocalPoint(fp)
        self.vtkcamera.SetViewAngle(cam.vfov)

        # Choose which render window will control the camera.
        if 'color camera' in self.worldviews:
            rwctrl = self.worldviews.index('color camera')
            print('Drive using color camera...')
        elif 'semantic labels' in self.worldviews:
            rwctrl = self.worldviews.index('semantic labels')
            print('Drive using semantic labels...')
        elif 'objectid' in self.worldviews:
            rwctrl = self.worldviews.index('objectid')
            print('Drive using object IDs...')
        else:
            raise Exception('There is no renderer to drive with')

        # if not self.interactor[rwctrl].GetEnabled():
        vwi.init_wininteract(self, cam, rwctrl, map2d=map2d)
        self.update_dyn_objs()
        self.look_at_camera()
        vtu.set_keypress_callback(self.renderers[rwctrl], self.interactor[rwctrl],
                                  vwi.my_keypress_callback)
        for w in range(len(self.renwindow)):
            self.renwindow[w].Render()
        self.interactor[rwctrl].Initialize()     # initialize for the event loop
        self.interactor_ready = True

        # Interact with the environment.
        vwi.set_hold_position(holdpos)
        vwi.print_ctrl_menu()
        if self.verbosity > 1: print('Start drving...')
        self.interactor[rwctrl].Start()

        # Save the final camera pose.
        pos = np.array(self.vtkcamera.GetPosition())
        if cam.mountedon:
            cam.mountedon.pos = pos
        else:
            cam.pos = pos
        fp = np.array(self.vtkcamera.GetFocalPoint())
        va = self.vtkcamera.GetViewAngle()         # vertical FOV (degrees)
        p, t = cam.get_pt(fp-pos)                  # absolute pan & tilt

        if cam.mountedon:
            # Get pan and tilt relative to object that camera is mounted on.
            p -= np.rad2deg(cam.mountedon.orient[2])
            t -= np.rad2deg(cam.mountedon.orient[1])
        p = max(cam.minpan, min(cam.maxpan, p))
        t = max(cam.mintilt, min(cam.maxtilt, t))

        cam.set(pan=p, tilt=t, vfov=va)

        if not self.interactor[rwctrl].GetEnableRender():
            # Renderer has been disabled. User wants to quit.
            self.interact = False
            self.interactor_ready = False


    def update_dyn_objs(self):
        """
        Update the poses of all dynamic objects.

        Description:
            Dynamic objects include those that can translate (on the ground or
            in the air) and the sky background. The sky background slowly
            rotates about the origin so that clouds appear to move.
            SimWorld.dynamic_env must be True for update dynamic objects, but
            not to make them face the camera.

            Objects may move in random directions or may follow a predefined
            path (see insert_fixed_path_objs()). For objects that move in random
            directions, the translation is in a straight line (with no change in
            elevation) until an ostacle is encountered, and then a new random
            direction is selected.
        """

        curtime = self.time
        dt = curtime - self.time_last

        # Translate movable objects if the environment is dynamic.
        if self.dynamic_env and dt > 0:
            for obj in self.movers:
                if obj.motion_type == Motion.Random:
                    # Move object along its random direction.
                    if obj.speed <= 0:
                        # Choose a new random direction and speed. Speed and
                        # velocity are meters/sec.
                        obj.speed = np.random.uniform(obj.minspeed, obj.maxspeed)
                        theta = np.random.uniform(0,2*np.pi)   # translation direction
                        obj.vel = obj.speed*np.array([np.cos(theta),np.sin(theta),0])

                    newpos = np.array(obj.rect[0:2]) + dt*obj.vel[0:2]   # (x,y) obj center
                    newdist = np.linalg.norm(newpos)           # dist from center of env

                    if newdist > self.map3d.map_radius - obj.xhalflen - 1:
                        # Object is about to move outside the environment. Don't
                        # make the move. Try a different direction next frame.
                        obj.speed = 0
                    else:
                        # The object's position is specified by the rectangle
                        # [xcenter, ycenter, xhalflen, yhalflen].
                        newrect = [*newpos, obj.xhalflen, obj.yhalflen]
                        if self.map3d.isclear(newrect, obj.id, obj.type, obj.elev):
                            # Move all actors for this object.
                            obj.maprect = self.map3d.move(obj.rect, newrect, obj.type)
                            obj.rect = newrect
                            obj.xctr = newrect[0]
                            obj.yctr = newrect[1]
                            for act in obj.actors:
                                apos = np.array(act.GetPosition())
                                apos = apos + dt*obj.vel
                                act.SetPosition(apos)
                        else:
                            # Can't move object at this time. Try again next frame.
                            obj.speed = 0
                elif obj.motion_type == Motion.DefinedPath:
                    # Move object along its predefined path.
                    # print(f'Moving object {obj.name} on defined path')
                    t = curtime % obj.defined_path[-1,0]
                    x = np.interp(t, obj.defined_path[:,0], obj.defined_path[:,1])
                    y = np.interp(t, obj.defined_path[:,0], obj.defined_path[:,2])
                    z = np.interp(t, obj.defined_path[:,0], obj.defined_path[:,3])
                    newrect = [x, y, obj.xhalflen, obj.yhalflen]
                    obj.maprect = self.map3d.move(obj.rect, newrect, obj.type)
                    obj.rect = newrect
                    pos = np.array([x, y, z + obj.zhalflen])
                    obj.xctr = pos[0]
                    obj.yctr = pos[1]
                    obj.zctr = pos[2]
                    for act in obj.actors:     # Move all actors for this object
                        act.SetPosition(pos)
                else:
                    raise ValueError(f'Invalid motion type: {obj.motion_type}')

            # Rotate the planar facets of the sky cylinder about the origin.
            for act in self.sky_actors:
                o = act.GetOrientation()
                act.SetOrientation(0,0,o[2]+dt*self.sky_speed)

        self.time_last = curtime


    def look_at_camera(self):
        """
        Update the orientations of all objects that must always face the camera.
        """

        cam = self.renderers[0].GetActiveCamera()
        cpos = np.array(cam.GetPosition())

        for act in self.followers:
            apos = np.array(act.GetPosition())                # actor's position
            v = apos - cpos                 # vector from camera to actor center

            # Orientation that makes actor face the camera.
            theta = 180*np.arctan2(-v[0],v[1])/np.pi

            if self.dynamic_env:

                p = act.GetProperty()
                if hasattr(p,'id'):
                    obj = self.objs[p.id-1]
                    if obj.dynamic and hasattr(obj, 'vel'):
                        # Make sure the forward direction (left or right) of the
                        # actor's image faces the direction of motion.
                        vel = obj.vel
                        perpproj = -v[1]*vel[0] + v[0]*vel[1]
                        if (perpproj > 0 and obj.forward_dir == 'r') or \
                                (perpproj < 0 and obj.forward_dir == 'l'):
                            theta += 180

                    # Add small random orientaion changes to some objects. This,
                    # e.g., makes plants appear to "move in the wind".
                    # tags = act.GetProperty().tags
                    # if 'plant' in tags and np.random.rand() < 0.4:
                        # theta += np.random.uniform(-0.5,0.5)         # was (-1,1)
                    # if ({'person', 'animal'} & tags) and np.random.rand() < 0.4:
                        # theta += np.random.uniform(-3,3)
                    if obj.type == 'plant':
                        theta += np.sin(2*np.pi*self.time*self.plant_osc_freq)

            # Set the actor's orientation.
            act.SetOrientation(0,0,theta)                    # angles in degrees


    def render_scene(self, pos, fp, vfov):
        """
        Invoke VTK to render the scene for the given camera parameters.

        Usage:
            SimWorld.render_scene(pos, fp, vfov)

            Arguments:
                pos: (array-like) 3D position of camera.

                fp: (array-like) 3D focal point of camera.

                vfov: (float) Vertical field of view (in degrees) of camera.

        Note:
            No objects in the scene are changed by this function.
        """

        # Setup the renderers' camera. (All renderers use the same camera.)
        self.vtkcamera.SetPosition(pos)
        self.vtkcamera.SetFocalPoint(fp)
        self.vtkcamera.SetViewAngle(vfov)     # vertical view angle of camera

        # Make sure all planar vertical objects are looking at the camera. We
        # need to update these objects' orientations if the scene is dynamic or
        # when the camera's position changes.
        curpos = np.array(pos)
        if self.dynamic_env or np.any(np.abs(curpos - self.last_camera_pos) > 0.1):
            self.look_at_camera()
        self.last_camera_pos[:] = curpos

        # Render the images. We render the color image last. Otherwise, distant
        # parts of the scene sometimes seem to be clipped out. I don't know why
        # this happens. (2020-10-16)
        if self.idx_labels != None:
            self.renderers[self.idx_labels].ResetCameraClippingRange()
            self.renwindow[self.idx_labels].Render()
        if self.idx_objid != None:
            self.renderers[self.idx_objid].ResetCameraClippingRange()
            self.renwindow[self.idx_objid].Render()
        if self.idx_color != None:
            self.renderers[self.idx_color].ResetCameraClippingRange()
            self.renwindow[self.idx_color].Render()

        # renwin.WaitForCompletion() and renwin.CheckInRenderStatus() don't
        # seem to work! So, just add a short pause.
        time.sleep(0.05)


    def get_images(self, imlist=[], filtersize=1):
        """
        Get images of the world from the VTK renderer.

        Usage:
            imgs = SimWorld.get_images(imlist=[], filtersize=1)

        Arguments:
            imlist: (list) List of names of images to retrieve.  Each image name
            is a string from the allowed images described below.  If this list
            is empty, then all valid images are returned. Default is empty (all
            images returned).

            filtersize: (int) Radius (in pixels) of filter used to remove small
            regions from the non-ground-truth semantic label image. The attempt
            is to simulate real image perception algorithms where very small
            objects are not recognized. Default is 1.

        Returns:
            imgs: A dictionary of images containing zero or more of the
            following:

                'color': RGB color image.

                'label': Single-channel semantic label image. The value of each
                pixel is one of the IDs from the `label2id` dictionary. To
                simulate reality, the labels of very small objects (as
                determined by the `filtersize` argument) can be removed from
                this image.

                'objectid': Image of object IDs, which index into the list of
                objects, self.objs.

                'labelgt': Single-channel ground-truth semantic label image.
                The value of each pixel is one of the IDs from the `label2id`
                dictionary.

                'labelrgb': RGB semantic label image. Objects in this image are
                colored using the RGB values from the `label_colors` list. This
                image is probably useful only for display purposes.

                'depth': Single-channel, float-valued, depth image. Depth values
                are in meters. Currently, the depth and ground-truth depth
                images are identical.

                'depthgt': Single-channel, float-valued, ground-truth depth
                image. Depth values are in meters. Currently, the depth and
                ground-truth depth images are identical.

        Note:
            The position and orientation of the VTK camera used to render
            the world must be set prior to calling this function.  Also, all
            flat objects in the scene should already be looking at the camera.
        """
        if imlist == []:
            # Get all images.
            imset = {'color', 'label', 'objectid', 'labelgt', 'labelrgb',
                     'depth', 'depthgt'}
        else:
            imset = set(imlist)

        retimgs = dict()                            # images to return
        imlabelgt = None

        if self.idx_color != None and 'color' in imset:
            # RGB image of texture-mapped world model.
            retimgs['color'] = vtu.renwin2numpy(self.renwindow[self.idx_color])

        if self.idx_objid != None and 'objectid' in imset:
            # RGB image of object ID world model.
            im = vtu.renwin2numpy(self.renwindow[self.idx_objid])
            retimgs['objectid'] = rgbid2uint(im, filter=True)

        if self.idx_labels != None and imset.intersection(['label','labelgt','labelrgb','depth','depthgt']):
            # RGB image of object-class-colored world model.
            imlabelrgb = vtu.renwin2numpy(self.renwindow[self.idx_labels])

            # Semantic label and ground-truth semantic label images.
            imlabel, imlabelgt = semantic_labels(imlabelrgb, filtersize=filtersize)

            if 'label' in imset: retimgs['label'] = imlabel
            if 'labelgt' in imset: retimgs['labelgt'] = imlabelgt
            if 'labelrgb' in imset: retimgs['labelrgb'] = imlabelrgb

        if imset.intersection(['depth', 'depthgt']):
            if imlabelgt is not None:
                # Fix problem of bad depth values at boundary of sky and other objects.
                imsky = (imlabelgt == label2id['sky']).astype(np.uint8)
                ###imsky = skimage.filters.rank.maximum(imsky, disk(1)) > 0
            else:
                imsky = None

            # The depth image can be generated from any of the world views.
            if self.idx_labels != None:
                imdepthgt = vtu.getdepthmap(self.renwindow[self.idx_labels], self.vtkcamera)
            elif self.idx_objid != None:
                imdepthgt = vtu.getdepthmap(self.renwindow[self.idx_objid], self.vtkcamera)
            elif self.idx_color != None:
                imdepthgt = vtu.getdepthmap(self.renwindow[self.idx_color], self.vtkcamera)
            else:
                imdepthgt = None

            if imdepthgt is not None:
                if imsky is not None:
                    if imdepthgt.shape[0:2] != imsky.shape[0:2]:
                        raise Exception('Image dimension mismatch.\n'+
                            'Camera image size can exceed screen size only when using\n'+
                            'off-screen rendering (via renWin.SetOffScreenRendering(True).')
                    imdepthgt[imsky] = np.Inf
                imdepthgt[imdepthgt > 2*self.env_radius+1] = np.Inf  # fix rendering errors
                imdepth = imdepthgt.copy()

                if 'depth' in imset: retimgs['depth'] = imdepth
                if 'depthgt' in imset: retimgs['depthgt'] = imdepthgt

        return retimgs


    def rand_road_intersection(self, align='road'):
        """
        Generate random position at a road intersection.

        Usage:
            pos, viewdir = SimWorld().rand_road_intersection(align='road')

        Arguments:
            align: (str) How to choose the viewing direction. Default is 'road'.
            This may be any of the following: 'road', 'random'.

        Returns:
            pos: The 2D position (X,Y).

            viewdir: The 3D viewing direction, (dX,dY,dZ).

        Description:
            The position will not be too close to the edge of the city.
        """

        #
        if len(self.nsroads) == 0 and len(self.ewroads) == 0:
            raise Exception('There are no roads to position camera on.')

        # Get the center of a random road intersection that is inside the
        # radius of the city.
        ns = ew = None
        while True:
            if len(self.nsroads) > 0:
                ns = np.random.randint(0, high=len(self.nsroads))  # pick a NS road
            if len(self.ewroads) > 0:
                ew = np.random.randint(0, high=len(self.ewroads))  # pick an EW road
            if ns == None:
                # There are only EW roads. Pick a random position on one of these.
                x = self.ewroads[ew].xctr + (2*np.random.rand()-1)*self.ewroads[ew].xhalflen
                y = self.ewroads[ew].yctr + (2*np.random.rand()-1)*self.ewroads[ew].yhalflen
            elif ew == None:
                # There are only NS roads. Pick a random position on one of these.
                x = self.nsroads[ns].xctr + (2*np.random.rand()-1)*self.nsroads[ns].xhalflen
                y = self.nsroads[ns].yctr + (2*np.random.rand()-1)*self.nsroads[ns].yhalflen
            else:
                # Get the intersection of the NS and EW roads.
                x = self.nsroads[ns].xctr
                y = self.ewroads[ew].yctr
            d = np.sqrt(x**2 + y**2)     # distance from center of city
            if d < 0.9*self.env_radius:
                break

        if align == 'road':
            # Randomly choose a road to look down.
            if np.random.rand() > 0.5:
                # Look east or west (in the X-direction).
                nsdir = 0
                ewdir = 1 - 2*np.random.randint(0,high=2)
            else:
                # Look north or south (in the Y-direction).
                nsdir = 1 - 2*np.random.randint(0,high=2)
                ewdir = 0
            viewdir = (ewdir, nsdir, 0)
        elif align == 'random':
            # Choose a random viewing direction.
            theta = np.random.uniform(0, 2*np.pi)
            viewdir = (np.cos(theta), np.sin(theta), 0)
        else:
            raise ValueError('Unrecognized alignment type: "{:s}"'.format(align))

        return (x,y), viewdir


    def rand_building_egress(self):
        """
        Generate random position and viewing direction at a building egress.

        Usage:
            pos, viewdir = SimWorld().rand_building_egress()

        Returns:
            pos: The 2D position (X,Y).

            viewdir: The 3D viewing direction, (dX,dY,dZ).

        Description:
            The position will not be too close to the edge of the environement.
        """

        # Pick a random building.
        bldg = self.buildings[np.random.randint(0, high=len(self.buildings))]
        road = bldg.closestroad

        # Get a point on the road to look at.
        if road.type == 'road-ns':
            x = road.xctr
            y = bldg.yctr + (2*np.random.rand()-1)*bldg.yhalflen
        else:
            x = bldg.xctr + (2*np.random.rand()-1)*bldg.xhalflen
            y = road.yctr

        # Get the point on the base of the building closest to the point (x,y).
        bldgpt = bldg.closestBBPnt(x, y, dst2edge=0.5)
        viewdir = (x,y) - bldgpt

        # Move away from the building by 1 m.
        viewdir = viewdir/np.linalg.norm(viewdir)
        bldgpt = bldgpt + viewdir

        return bldgpt, np.hstack((viewdir,0))


def TagValue(tags, key):
    """
    Get the value of a key tag.

    Usage:
        value = TagValue(tags, key)

    Arguments:
        tags: (set or list) The object tags. If a string, each tag in the string
        must be seperated by a period.

        key: (str) The key of the tag whose value is to be retrieved (not
        including the equal sign). Keyed tags look like KEY=VALUE where KEY and
        VALUE are any strings not containing "_".

    Returns:
        value: (str) The value of the tag is returned in a string. If the tag
        doesn't exist, then the empty string is returned.
    """

    if type(tags) == str:
        tags = set(tags.split('_'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    value = ""
    key = key.lower() + "="
    for t in tags:
        if t.startswith(key):
            value = t[len(key):]
            break

    return value


def TagTypeID(tags):
    """
    Get the type and ID from a set of tags. The type and ID tags should identify
    a unique object.

    Usage:
        otype, oid = TagTypeID(tags)

    Arguments:
        tags: (set or list) The object tags. If a string, each tag in the string
        must be seperated by a period.

    Returns:
        otype: (str) The type tag.

        oid: (str) The ID tag.
    """

    if type(tags) == str:
        tags = set(tags.split('_'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    otype = list(tags & base_tags)
    otype = otype[0] if len(otype) > 0 else 'unknown'

    oid = ""
    for t in tags:
        if t.isdigit():
            oid = t
            break

    return otype, oid


def basetype(objname):
    """
    Strip off from the end of a string the first dash ('-') and all characters after that.
    """
    p = objname.find('-')
    return objname if p == -1 else objname[0:p]


def objboxes(semlabel, depth, otype, minarea=16):
    """
    Extract, from semantic label and depth images, a list of object bounding
    boxes for a specific class of object.

    Usage:
        oboxes = objboxes(semlabel, depth, otype, minarea=16)

    Arguments:
        semlabel: (numpy.ndarray) Single-channel, int-valued image of semantic
        labels. The integer values must correspond to those in "label2id".

        depth: (numpy.ndarray) Single-channel, float-valued image of depth
        values.

        otype: (str) Name of object class (from "label2id") to get bounding
        boxes for.

        minarea:(float) Minimum area (pixels) of an object to return its
        properties.

    Returns:
        oboxes: (list) List of Blob objects for each spatially seperate object
        in the image. Each Blob will have the following properties:
            x, y: the minimum x and y coordinates (pixels) of the box
            width, height: the width and height (pixels) of the box
            depth: the approximate depth (meters) of the object
    """

    # Replace non-object pixels with zero.
    im = semlabel.copy()
    im[im != label2id[otype]] = 0

    # Perform opening (erosion then dilation) to remove thin regions.
    # im = skimage.morphology.binary_opening(im, disk(1))

    # Label spatially seperated objects in the semantic label image.
    labels = skm.label(im, connectivity=2)

    # Get properties of all objects.
    props1 = skm.regionprops(labels)

    # Create a filtered list of objects to return.
    oboxes = []
    for p in props1:
        if p.area >= minarea:
            # p.bbox is [rowmin, colmin, rowmax+1, colmax+1]
            d = depth[p.bbox[2]-1, int((p.bbox[1]+p.bbox[3])/2)]
            oboxes.append(Blob(yxs=p.slice, pixels=True,
                               props={'depth':d, 'label':otype,
                                      'confidence':1.0, 'gt':True}))

    return oboxes


def semantic_labels(imlabelrgb, filtersize=1):
    """
    Transform an RGB image of the object-class-colored world model into a
    single-channel semantic label image where all pixel values come from the IDs
    in the `label2id` dictionary.

    Usage:
        imlabel, imlabelgt = semantic_labels(imlabelrgb, filtersize=3)

    Arguments:
        imlabelrgb: RGB image of object-class-colored world model. This is the
        RGB "flat" world-view image of the scene (pixel values in [0,255]x3)
        where objects have fixed color (no texture) and are uniformly lit only
        by ambient lighting.

        filtersize: (int) Radius (in pixels) of filter used to remove small
        regions from the non-ground-truth semantic label image. We attempt to
        simulate real image perception algorithms where very small objects are
        not recognized. Default is 3.

    Returns:
        imlabel: Single-channel semantic label image. The value of each pixel is
        one of the IDs from the `label2id` dictionary. To simulate reality, the
        labels of very small objects (as determined by the `filtersize`
        argument) will often not appear in this image.

        imlabelgt: Single-channel ground-truth semantic label image. The value
        of each pixel is one of the IDs from the `label2id` dictionary.

    Description:
        Objects classes are identified by their RGB colors, as defined by
        label2id and id2color1.
    """

    # Map image colors (3-channel) to the class ID (1-channel) of the
    # corresponding object types.
    imlabelgt = np.zeros(imlabelrgb.shape[0:2], dtype=np.uint8)
    for key in label2id:
        class_color = otype2color255[key]
        class_loc = np.logical_and(imlabelrgb[:,:,0] == class_color[0],
                                   np.logical_and(imlabelrgb[:,:,1] == class_color[1],
                                                  imlabelrgb[:,:,2] == class_color[2]))
        # if key == trglabel:
            # class_target = class_loc.copy()            # binary target image
        imlabelgt[class_loc] = label2id[key]

    if filtersize > 0:
        # Apply a modal filter to remove the labels of small regions in the
        # image. 'filtersize' is the radius of the filter neighborhood.
        imlabel = skimage.filters.rank.modal(imlabelgt, disk(filtersize))
    else:
        imlabel = imlabelgt.copy()

    # Get the target groundtruth, a list of target bounding boxes.
    # imtargets, numtargets = ndi.label(class_target, structure=np.ones((3,3),dtype=int))
    # trgslices = ndi.find_objects(imtargets)
    # trglist = []
    # for k, st in enumerate(trgslices, start=1):
        # # st is a tuple of two slices: (yslice, xslice).
        # trglist = trglist + [Blob(yxs=st, pixels=True, label=trglabel,
                                  # props={'confidence':1.0, 'gt':trglabel, 'tnum':k})]

    # Replace all unlabeled pixels with the nearest valid label.
    invalid = np.zeros_like(imlabel, dtype=bool)
    invalid[imlabel == label2id['unknown']] = True      # non-labeled pixels
    # invalid[imlabel == label2id['person']] = True     # pixels labeled as target
    imlabel = dtfill(imlabel, invalid)

    # Overwrite semantic IDs with the target label where large enough targets
    # are found.
    #
    # mintargetwidth: In the returned groundtruth image 'imall', show all
    # targets whos width in the current image is greater or equal to
    # 'mintargetwidth'. When 'mintargetwidth' is None, no targets will appear
    # in this groundtruth image. When 'mintargetwidth' is is 0, all targets
    # will appear in the groundtruth image. Default is 0. Regardless of the
    # value of 'mintargetwidth', all targets are always returned in 'trglist'.
    # imgtall = imgtnt.copy()
    # if mintargetwidth is not None:
        # bigtargets = np.zeros_like(imtargets, dtype=bool)
        # for t in trglist:
            # if t.width >= mintargetwidth:
                # bigtargets[imtargets == t.tnum] = True
        # imgtall[bigtargets] = label2id[trglabel]

    if False:
        # Display the images.
        with Fig(figsize=(15,5), axpos=[131,132,133], link=[0,1,2]) as f:
            f.set(image=imlabelrgb, axisnum=0, axistitle='RGB Labels')
            f.set(image=imlabelgt, axisnum=1, axistitle='GT Labels')
            f.set(image=imlabel, axisnum=2, axistitle='Labels')
            # for t in trglist:
                # print('Target {}: [{},{},{},{}]'.format(t.id, t.xmin,t.ymin,t.width,t.height))
                # f.draw(axisnum=0, rect=t, edgecolor='w', linestyle='--')
            f.wait(event='key_press_event')

    return imlabel, imlabelgt





def rgbid2uint(imrgb, filter=False):
    """
    Transform an RGB image of the object-ID-colored world model into a
    single-channel, unsigned int, object ID image.

    Usage:
        imuint = rgbid2uint(imrgb, filter=False)

    Arguments:
        imrgb: RGB image of object-ID-colored world model. This is the RGB
        world-view image of the scene (pixel values in [0,255]x3) where objects
        are uniformly colored with their object IDs and are uniformly lit only
        by ambient lighting.

        filter: (bool) If True, then perform a morphological dilation on the
        object ID image to remove blurred IDs at the edges of objects. The
        non-object regions are dilated, which shirnks (erodes) the object
        regions. Default is False.

    Returns:
        imuint: Single-channel, unisgend-int-valued, object ID image.

    Description:
        When two nonzero objects (with different IDs) overlap in the image, the
        boundary at the intersection will usually be blured and therefore
        assigned an incorrect object ID. These incorrect IDs are ussually only
        one or two pixels in size and can be filtered out later based on their
        small sizes.
    """

    # Convert the 3-channel RGB values into single-channel unsigned ints.
    imuint = ((imrgb[:,:,0]).astype(np.uint) << 16) + \
             ((imrgb[:,:,1]).astype(np.uint) << 8) + imrgb[:,:,2].astype(np.uint)

    # The IDs at the boundaries of objects are blured by the VTK rendering
    # process. To remove most of these incorrect object IDs, we can expand (via
    # a morphological dilation) the region of zero values by one pixel. The
    # remainder of incorrect IDs (at boundaries between two non-zero regions)
    # can be filtered out based on their very small areas and whether or not the
    # IDs correspond to groundtruthed objects. A disk radius of 1.76 gives a
    # fully-filled 3x3 structuring element.
    if filter:
        zeros = imuint == 0
        zeros = skimage.morphology.binary_dilation(zeros, disk(1.76))
        imuint[zeros] = 0

    return imuint


def semlabel2objid(im, objrgb, objid, nonobjid):
    """
    Create a VTK texture image of object IDs from a semantic label image.

    Usage:
        vtktxt = semlabel2objid(im, objrgb, objid, nonobjid)

    Arguments:
        im: (numpy.ndarray) Semantic label image to transform into a VTK
        texture. This image must be a Numpy.ndarray with 3 color channels and
        optionally one alpha (transparency) channel (so, RGB or RGBA).

        objrgb: (3-array like) A 3-array like (tuple, list, or Numpy array)
        giving the R, G, and B values (in this order) of the semantic label of
        the object to be identified in the texture image. R, G, and B values are
        ints in [0,255].

        objid: (int) The object ID to write into the texture image at all pixel
        locations matching {objrgb}. {objid} is converted to a 3-tuple of RGB
        values as described below.

        nonobjid: (int) The value to write into the texture image at all nonzero
        pixel locations not matching {objrgb}. {nonobjid} is converted to a
        3-tuple of RGB values as described below.

    Returns:
        vtktxt: (vtkTexture) A VTK texture.

    Description:
        Pixles in the semantic label image with RGB color {objrgb} are recolored
        to {objid} (converted to an RGB value), and pixels with any other
        nonzero RGB values are recolored to {nonobjid} (converted to an RGB
        value). The image is then saved to the local drive, and then read in as
        a VTK texture, which is returned to the calling function. Conversion of
        an object ID to an RGB tuple is done according to the following:
            R = (ID & 0xFF0000) >> 16,
            G = (ID & 0x00FF00) >> 8,
            B = (ID & 0x0000FF).
        So, conversion from R,G,B to the original object ID is
            ID = (R << 16) + (G << 8) + B.
    """
    assert len(objrgb) == 3, 'Argument "objrgb" must be an RGB triple'

    # im = np.array(imageio.imread(imfile))        # read the semantic label image
    assert type(im) == numpy.ndarray
    numchan = im.shape[2]

    # Get pixel locations of the matching object ('this_obj') and the pixel
    # locations of all nonmatching objects ('other_obj').
    if numchan == 4:
        this_obj = (im[:,:,0] == objrgb[0]) & (im[:,:,1] == objrgb[1]) & \
                   (im[:,:,2] == objrgb[2]) & (im[:,:,3] == 255)
        other_objs = (im[:,:,3] > 0) & (this_obj == False)
    elif numchan == 3:
        this_obj = (im[:,:,0] == objrgb[0]) & (im[:,:,1] == objrgb[1]) & \
                   (im[:,:,2] == objrgb[2])
        other_objs = this_obj == False
    else:
        raise Exception('Image must be 3 or 4 channel (RGB or RGBA):{:s}'.format(
                        imfile))

    # Convert the object IDs to RGB colors.
    this_id = np.array(((objid & 0xff0000) >> 16, (objid & 0x00ff00) >> 8,
                        (objid & 0x0000ff)), dtype=int)
    other_id = np.array(((nonobjid & 0xff0000) >> 16, (nonobjid & 0x00ff00) >> 8,
                        (nonobjid & 0x0000ff)), dtype=int)

    # Color a new image with the two different IDs.
    imnew = np.zeros_like(im)
    for k in range(3):
        imnew[this_obj, k] = this_id[k]
        imnew[other_objs, k] = other_id[k]
    if numchan == 4:
        imnew[this_obj, 3] = 255

    # plt.figure(); plt.imshow(imnew); plt.pause(0.1)

    if True:
        # Direct conversion of Numpy array to VTK texture.
        texture = vtu.numpy2texture(imnew)
    else:
        # Caution: very slow for large number of objects!
        # Save the recolored image and then create a VTK texture from it.
        fname = './simworld_objid_txts/obj_{:d}.png'.format(objid)
        imageio.imwrite(fname, imnew)
        texture = vtk.vtkTexture()
        texture.SetInterpolate(False)
        pngreader = vtk.vtkPNGReader()
        pngreader.SetFileName(fname)
        texture.SetInputConnection(pngreader.GetOutputPort())

    return texture


def plotsignal(f, sig=None, anum=0, tmax=None, amax=None, title="",
               xmarks=[], psd=None, freq=None):
    """
    Plot multiple 1D signals in a single figure.

    Arguments:
        f: (tuple) The Matplotlib figure and axis objects, stored in a tuple
        (fig, axs). If None, then a new figure is created.

        sig: (1D array-like) The signal to plot. If None, `psd` will be plotted.

        anum: (int) The number of the axis (0,1,...) to plot the current
        signal.

        tmax: (float) The maximum x-axis (sample number) value.

        amax: (float) The maximum y-axis (amplitude) value.

        title: (str) The title of the current subplot.

        xmarks: (list of float) List of x-axis positions to draw vertical lines.

        psd: (1D array-like) The power spectral density of the signal.

        freq: (1D array-like) The frequency (Hz) at each sample in PSD.
    """
    if f is None:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6,10))
        f = (fig, axs)
    elif type(f) is list:
        fig, axs = plt.subplots(nrows=f[0], ncols=f[1], figsize=(6,10))
        f = (fig, axs)
    elif type(f) is not tuple:
        raise Exception('Arg f must be None, a list [nrows, ncols], or a tuple of figure data')
    if title == "":
        title = "Signal"
    axs = f[1][anum]
    if sig is not None:
        if tmax is None:
            tmax = len(sig)
        if amax is None:
            amax = np.abs(sig).max()
        if xmarks != []:
            tmax = max(tmax, max(xmarks))
        axs.cla()
        axs.set_title(title)
        axs.plot(sig, 'b-')
        axs.set_xlim(0, tmax)
        axs.set_ylim(-amax, amax)
        for x in xmarks:
            axs.plot([x,x], [-amax,amax], 'r-', linewidth=1.5)
    elif psd is not None and freq is not None:
        axs.semilogy(freq, psd, 'b')
        axs.set_ylim([1e-12, 1e12])
        axs.set_xlabel('frequency [Hz]')
        axs.set_ylabel('PSD [V^2/Hz]')
        axs.set_title(title)
    else:
        raise Exception('Nothing to plot: signal and PSD are both empty')
    plt.draw()
    plt.pause(0.1)
    return f


if __name__ == '__main__':
    from camera import *

    # Create the PTZ camera.
    camera = PTZCamera(imsize=(1280,720), rnghfov=(3,54),
                       rngpan=(-np.Inf,np.Inf), rngtilt=(-45,60),
                       pos=(0,0,0.5), pan=0, tilt=0, zoom=0)

    # Create an urban model.
    sim = SimWorld(timeofday=[500,1900], env_radius=200, bldg_density=1,
                   road_density=1, plant_density=1, people_density=1,
                   animal_density=1, clutter_density=1, vehicle_density=0.1,
                   bldg_plant_density=1, barrier_density=1, gndfeat_density=1,
                   p_over_building={'person':0.5, 'clutter':0.1, 'animal':1.0, 'vehicle':0},
                   p_over_road={'person':0.1, 'clutter':0.05, 'animal':0.2, 'vehicle':0.5},
                   textures='textures', rand_seed=None, imsize=camera.imsize,
                   verbosity='low', envtype='urban', show_obj_ids=False,
                   views={'color','label'})

    # Display the 2D map.
    mymap = Map2D(maps=sim.map3d, ptzcam=camera, label_colors=label_colors)

    # Manually move the camera.
    sim.you_drive(camera, map2d=mymap)

    if False:
        # Display some images.
        imgs = camera.get_images(sim)
        with Fig(axpos=[131,132,133], figtitle='My World', figsize=(10,4), link=[0,1,2]) as f:
            f.set(axisnum=0, image=imgs['color'], axistitle='Color')
            f.set(axisnum=1, image=imgs['label'], axistitle='Semantic labels')
            f.set(axisnum=2, image=imgs['depth'], axistitle='Depth')
            print('Press any key to continue...')
            f.wait(event='key_press_event')


