"""
VTK utilities.

Author:
    Phil David, Army Research Laboratory, December 2017.

Notes:
    Display (screen) coordinates have origin (0,0) at the lower left corner of the
    screen.

    Viewport coordinates go from -1 to 1 with (-1,-1) at the lower left corner of
    the image.

    Z-buffer values range from 0 to 1.
"""

import scipy
import vtk
import vtk.util.colors
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from phutils import *
import pyvista
import numpy as np
import matplotlib.pyplot as plt


def close_renwin(iren):
    """
    After calling this function, please execute the following:
        del renwin, iren
    """
    renwin = iren.GetRenderWindow()
    renwin.Finalize()
    iren.TerminateApp()


def getdepthmap(renwin, camera):
    """
    Get a depth map of the currently rendered scene.

    Usage:
        depth = getdepthmap(renwin, camera)

    Notes:
        Objects that are texture-mappped with PNG images seem to be transparent in
        the Z-buffer, and therefore are not given a depth value.
    """
    assert camera is not None
    assert renwin is not None

    xsize, ysize = renwin.GetSize()
    depth = np.zeros((ysize, xsize))
    numpts = xsize*ysize

    zbuf = zbuffer2numpy(renwin)      # zbuf[0,0] is at upper left corner of screen.

    # Get the viewport coordinates (-1 to 1) of every pixel in the window.
    xc1 = np.linspace(-1, 1, xsize, endpoint=True)
    yc1 = np.linspace(1, -1, ysize, endpoint=True)
    xc2, yc2 = np.meshgrid(xc1, yc1, indexing='xy')
    viewpts = np.zeros((ysize, xsize, 4))
    viewpts[:,:,0] = xc2
    viewpts[:,:,1] = yc2
    viewpts[:,:,2] = zbuf
    viewpts[:,:,3] = 1.0
    viewpts = np.reshape(viewpts.T, (4,numpts), order='f')  # New shape: 4 x Numpts

    # The transformation matrix 'hmat1' will convert homogeneous world coordinates
    # to viewport coordinates. 'Aspect' is the width/height for the viewport,
    # and the 'nearz' and 'farz' are the Z-buffer values that map to the near
    # and far clipping planes. The viewport coordinates of a point located
    # inside the frustum are in the range ([-1,+1],[-1,+1],[nearz,farz]).
    hmat1 = camera.GetCompositeProjectionTransformMatrix(xsize/ysize, 0, 1)

    # The inverse transformation converts viewport coordinates to world coordinates.
    hmat1.Invert()
    hmat2 = np.zeros((4,4))
    for r in range(4):
        for c in range(4):
            hmat2[r,c] = hmat1.GetElement(r,c)

    hwp = mymatmul(hmat2, viewpts)       # Homogeneous world points (4xNumpts).

    # Convert world coordinates to camera coordinates.
    # z = hwp[2,:]/hwp[3,:]
    # y = hwp[1,:]/hwp[3,:]
    wp = hwp[0:3,:]/hwp[3,:]              # Nonhomogeneous world points (3xNumpts)

    # Get distance of each point from camera center.
    cpos = np.array(camera.GetPosition())
    cp = wp - cpos[:,None]                # Points in camera coordinates (with rotation)
    depth = np.linalg.norm(cp, axis=0)    # Distance from camera center.
    depth = depth.reshape((ysize, xsize)) # Form back into original image shape.

    return depth


def matprint(m, nrows, ncols):
    """
    Print a VTK matrix.
    """
    for r in range(nrows):
        for c in range(ncols):
            print('{:7.3f} '.format(m.GetElement(r, c)), end='')
        if r < nrows-1:
            print('')


def filter2numpy(filter):
    """
    Convert output of a VTK filter to a Numpy array.

    Usage:
        im = filter2numpy(filter)

    Arguments:
        filter: The VTK filter to convert.

    Returns:
        The Numpy array containing the filter output.
    """

    filter.Update()
    im1 = filter.GetOutput()
    cols, rows, depth = im1.GetDimensions()
    assert depth == 1, 'depth of image != 1: {}'.format(depth)
    data = im1.GetPointData().GetScalars()
    im2 = vtk_to_numpy(data)
    if depth == 1:
        im2 = im2.reshape(rows, cols)
    else:
        im2 = im2.reshape(rows, cols, depth)
    return im2


def renwin2numpy(renwin):
    """
    Convert the current VTK rendered scene to a Numpy array.

    Usage:
        im = renwin2numpy(renwin)

    Arguments:
        renwin: The VTK render window to get the image of.

    Returns:
        im: The Numpy array containing the rendered image.
    """

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renwin)

    w2if.Update()
    im = w2if.GetOutput()
    cols, rows, depth = im.GetDimensions()

    if cols == 0 or rows == 0 or depth == 0:
        raise Exception('Unable to get VTK rendered image. Make sure window is active.')

    assert depth == 1, 'depth of image != 1: {}'.format(depth)

    data = im.GetPointData().GetScalars()
    im2 = vtk_to_numpy(data)
    # im2 = im2.reshape(rows, cols, -1)
    datatype = data.GetDataType()
    if datatype == 3:
        depth = 3
    if depth == 1:
        im2 = im2.reshape(rows, cols)
    else:
        im2 = im2.reshape(rows, cols, depth)
    im2 = np.flipud(im2)    # Index (0,0) maps to upper left corner of window.
    return im2


def zbuffer2numpy(renwin):
    """
    Convert the z-buffer of the render window to a Numpy array.

    Usage:
        im = zbuffer2numpy(renwin)

    Arguments:
        renwin: the render window to convert the z-buffer of.

    Returns:
        im: the z-buffer as a Numpy array. Index (0,0) (e.g., im[0,0])
        corresponds to the upper left corner of the window.

    Description:
        A z-buffer value of 1.0 (maybe any value > 0.999999?) indicates that the
        screen point has not been rendered into.
    """

    filter = vtk.vtkWindowToImageFilter()
    filter.SetInput(renwin)
    filter.SetScale(1) # SetMagnification(1)    # Resolution of output relative to input resolution.
    filter.SetInputBufferTypeToZBuffer()

    # scale = vtk.vtkImageShiftScale()
    # scale.SetInput(filter.GetOutput())
    # scale.SetOutputScalarTypeToDouble()
    # scale.SetShift(0)
    # scale.SetScale(-255)

    #im = filter2numpy(scale)
    im = filter2numpy(filter)
    im = np.flipud(im)    # Make index (0,0) access z value for upper left corner of window.
    return im


def set_keypress_callback(renderer, interactor, callbackfun):
    """
    This function assigns a callback function to process renderer keyboard events.

    Usage:
        set_keypress_callback(renderer, interactor, callbackfun)

    Example:
        def my_keypress_callback(renderer, interactor, key):
            key = key.lower()
            print("Pressed key {}".format(key))
            ...
            renderer.ResetCameraClippingRange()
            renderer.GetRenderWindow().Render()
            return
        set_keypress_callback(renderer, interactor, my_keypress_callback)
        interactor.Initialize()
        renwindow.Render()
        interactor.Start()
    """

    class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

        def keyPressEvent(self, obj, event):
            key = self.interactor.GetKeySym()
            self.callback(self.renderer, self.interactor, key)
            return

        def __init__(self, renderer, interactor, callbackfun):
            self.renderer = renderer
            self.interactor = interactor
            self.callback = callbackfun
            interactor.AddObserver("KeyPressEvent", self.keyPressEvent)

    interactor.SetInteractorStyle(MyInteractorStyle(renderer, interactor, callbackfun))
    interactor.GetInteractorStyle().EnabledOn()


def make_ellipsoid(renderer, param, color=None, texture=None, lightcoef=(1,1,1),
                   opacity=1.0, res=30, tags=''):
    """
    Make an upright ellipsoid.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.
        param: Parameters of the ellipsoid: [xctr, yctr, zctr, xyhalfwidth, zhalfwidth]
        texture: (vtkTexture, r_scale, s_scale)
        color: the color (R,G,B) of the object. Used if texture is not used.
        lightcoef: the object's reflectance coefficients: (ambient, diffuse, specular).
        opacity: how opaque (in [0,1]) the ellipsoid is (0=transparent, 1=fully opaque)
        res: resolution of facets on ellipsoid.

    Notes:
        The VTK texture object may be created from an image file as follows:
            reader = vtk.vtkPNGReader()
            reader.SetFileName(pngfilename)
            texture = vtk.vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
    """

    if color is None and texture is None:
        raise Exception('Must provide either color or texture for ellipsoid')

    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(param[3])
    ellipsoid.SetYRadius(param[3])
    ellipsoid.SetZRadius(param[4])
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(ellipsoid)
    source.SetUResolution(res)
    source.SetVResolution(res)
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()
    actor.GetProperty().SetOpacity(opacity)

    if texture is None:
        # Ellipsoid is single color.
        mapper.SetInputConnection(source.GetOutputPort())
        mapper.SetScalarRange(-0.5, 0.5)
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOff()
        # actor.GetProperty().SetEdgeColor(.2, .2, .5)
        actor.GetProperty().SetColor(vtk.util.colors.green)
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)
    else:
        # Texture map ellipsoid.
        source.GenerateTextureCoordinatesOn()
        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(source.GetOutputPort())
        xform.SetScale(texture[1], texture[2], 1)
        xform.SetFlipT(1)
        xform.SetFlipS(0)
        mapper.SetInputConnection(xform.GetOutputPort())
        mapper.SetScalarRange(-.5, 0.5)
        actor.SetMapper(mapper)
        actor.SetTexture(texture[0])

    actor.SetPosition(param[0:3])
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    renderer.AddActor(actor)
    return actor


def make_cuboid(renderer, param, color=None, texture=None, lightcoef=(1,1,1),
                alr_variation=0.3, tags=''):
    """
    Make an upright cuboid.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.
        param: Parameters of the cuboid: [xctr, yctr, zctr, xhalfwidth, yhalfwidth, zhalfwidth]
        texture: (vtkTexture, r_scale, s_scale)
        color: the color (R,G,B) of the object. Used if texture is not used.
        lightcoef: the object's reflectance coefficients: (ambient, diffuse, specular).

    Notes:
        The VTK texture object may be created from an image file as follows:
            reader = vtk.vtkPNGReader()
            reader.SetFileName(pngfilename)
            texture = vtk.vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
    """

    if color is None and texture is None:
        raise Exception('Must provide either color or texture for cuboid')

    cube = vtk.vtkCubeSource()
    cube.SetXLength(2*param[3])
    cube.SetYLength(2*param[5])
    cube.SetZLength(2*param[4])
    cube.SetCenter(0, 0, 0)
    mapper = vtk.vtkPolyDataMapper()
    actor = vtk.vtkActor()

    if texture is None:
        # Cuboid is single color.
        mapper.SetInputConnection(cube.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOff()
        # actor.GetProperty().SetEdgeColor(.2, .2, .5)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)
    else:
        # Texture map cuboid.
        if True:
            xform = vtk.vtkTransformTextureCoords()
            xform.SetInputConnection(cube.GetOutputPort())
            xform.SetScale(texture[1], texture[2], 1)
            # xform.SetOrigin(0,0,0)   #  point about which texture map is flipped (e.g., rotated)
            xform.SetFlipT(1)
            xform.SetFlipS(0)
            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputConnection(xform.GetOutputPort())
        else:
            mapper.SetInputConnection(cube.GetOutputPort())
        actor.SetMapper(mapper)
        actor.SetTexture(texture[0])

    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    # alr = 0.6 + alr_variation*(np.random.rand(1)[0] - 0.5)
    # actor.GetProperty().SetAmbient(alr)
    actor.SetPosition(param[0:3])        # Center of cubeoid.
    actor.RotateX(-90.0)
    actor.RotateY(0.0)
    renderer.AddActor(actor)
    return actor


def make_sphere(renderer, param, color=None, texture=None, lightcoef=(1,1,1),
                rscale=1, sscale=1, res=50, alr_variation=0.3, tags=''):
    """
    Make a sphere.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.
        param: Parameters of the sphere: [xctr, yctr, zctr, radius]
        texture: vtkTexture
        color: the color (R,G,B) of the object. Used if texture is not used.
        lightcoef: the object's reflectance coefficients: (ambient, diffuse, specular).
            Default is (1, 1, 1).
    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for sphere')

    # Generate sphere polydata
    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(res)
    sphere.SetPhiResolution(res)
    sphere.SetCenter(param[0], param[1], param[2])
    sphere.SetRadius(param[3])
    actor = vtk.vtkActor()

    if texture is None:
        # Sphere is single color.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)  # Set one color for ambient, diffuse, and specular
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)
    else:
        # Texture map sphere.
        maptosphere = vtk.vtkTextureMapToSphere()
        maptosphere.SetInputConnection(sphere.GetOutputPort())
        maptosphere.PreventSeamOn()
        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(maptosphere.GetOutputPort())
        xform.SetScale(rscale, sscale, 1)
        xform.SetFlipT(1)
        xform.SetFlipS(0)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(xform.GetOutputPort())
        actor.SetMapper(mapper)
        actor.SetTexture(texture)

    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    actor.ForceOpaqueOn()

    renderer.AddActor(actor)
    return actor



def make_cyl_old(renderer, origin=(0,0,0), radius=1, width=1, height=1, numfacets=8,
              colors=None, textures=None, reps=(1,1,1), lightcoef=(1,1,1),
              opacity=1.0, tags=''):
    """
    Make a cylinder where each facet is texture mapped by a seperate image.

    Usage: actors = make_cyl(renderer, origin=(0,0,0), radius=1, width=1,
                             height=1, numfacets=8, colors=None, textures=None,
                             reps=(1,1,1), lightcoef=(1,1,1), opacity=1.0, tags=''):

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        origin: (3-array-like) The origin (x,y,z) of the cylinder. This is the
        taken to be the bottom of the cylinder along the main axis.
        Default is (0,0,0).

        radius: (float) Radius of the cylinder. Default is 1.

        width: (float) Width of each texture image in world coordinates.
        Default is 1.

        height: (float) Height of each texture image in world coordinates.
        Default is 1.

        numfacets: (int) Number of facets used to represent the cylinder. This
        must be 3 or more. Default is 8.

        reps: (3-array-like) The triple, (rx, ry,rz), of repetitions of the texture
        image on the cylinder's facets in the X, Y, and Z directions, respectively.
        Default is (1,1,1)

        textures: List of vtkTexture texture objects. The textures are cycled
        through (possibly more than once) to texture map the cylinder facets.
        Default is None.

        colors: List of (R,G,B) colors for each surface facet. Used if texture
        is not provided.  R, G, and B are floats in [0,1].

        lightcoef: (3-array-like) The object's reflectance coefficients:
        (ambient, diffuse, specular). Default is (1,1,1).

        tags: Any data that is to be saved to the actor's 'tag' property. This
        data may be retrieved using the following: actor.GetProperty().tags.
        Default value is ''.

    Returns:
        actors: (list) List of actors associated with the created cylinder.

    Description:
        Vtk texture data passed in via `textures` are mapped onto planar facets
        of the cylinder in a counter-clockwise order (when viewed from above),
        starting with the facet centered on the positive X axis. The N texture
        images in `textures` list, when viewed as a panoramic image, should be
        ordered from right ("textures[0]") to left ("textures[N-1]").
    """

    if colors is None and textures is None:
        raise Exception('Must provide either `colors` or `textures`')

    if type(tags) == str:
        tags = set(tags.split('_'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    actors = []
    origin = np.array(origin)
    dtheta = 360/numfacets
    halfwidth = radius*np.tan(np.deg2rad(dtheta/2))  # half the width of one facet
    pos = origin + np.array([radius,0,0]) + np.array([0,halfwidth,0])

    if textures is None:
        texture = None
        numcolors = len(colors)
    else:
        color = None
        numtextures = len(textures)
        tags = tags | set(["noflip"])

    s = 2*halfwidth/width
    v1 = -np.array([0, s*width, 0])      # world vector along texture image x axis
    v2 = np.array([0, 0, s*height])      # world vector along texture image y axis

    theta = 0
    for k in range(numfacets):
        if textures is None:
            color = colors[k % numcolors]
        else:
            texture = textures[k % numtextures]
        actors.append(make_quad(renderer, origin=origin, pos=pos, v1=v1, v2=v2,
                                orient=(0,0,theta), reps=reps, color=color,
                                texture=texture, lightcoef=lightcoef,
                                opacity=opacity, tags=tags))
        theta += dtheta

    return actors


def make_cyl(renderer, origin=(0,0,0), radius=1, width=1, height=1, seamless=False,
             colors=None, textures=None, reps=(1,1,1), lightcoef=(1,1,1),
             opacity=1.0, tags=''):
    """
    Make an upright cylinder where each facet is texture mapped by a seperate image.

    Usage: actors = make_cyl(renderer, origin=(0,0,0), radius=1, width=1,
                             height=1, colors=None, textures=None,
                             reps=(1,1,1), lightcoef=(1,1,1), opacity=1.0, tags=''):

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        origin: (3-array-like) The origin (x,y,z) of the cylinder. This is the
        taken to be the bottom of the cylinder along the main axis.
        Default is (0,0,0).

        radius: (float) Radius of the cylinder. Default is 1.

        width: (float) Width of each texture image in world coordinates.
        Default is 1.

        height: (float) Height of each texture image in world coordinates.
        Default is 1.

        seamless: (bool) If true, then the number of cylinder facets will be
        increased if necessary in order to have an integer multiple of TEXTURES
        factes. This is done assuming the right edge of the 1st texture image
        matches up with the left edge of the last texture image. Default is
        False.

        reps: (3-array-like) The triple, (rx, ry,rz), of repetitions of the
        texture image on the cylinder's facets in the X, Y, and Z directions,
        respectively. Default is (1,1,1)

        textures: List of vtkTexture texture objects. The textures are cycled
        through (possibly more than once) to texture map the cylinder facets.
        Default is None.

        colors: List of (R,G,B) colors for each surface facet. Used if texture
        is not provided.  R, G, and B are floats in [0,1].

        lightcoef: (3-array-like) The object's reflectance coefficients:
        (ambient, diffuse, specular). Default is (1,1,1).

        tags: Any data that is to be saved to the actor's 'tag' property. This
        data may be retrieved using the following: actor.GetProperty().tags.
        Default value is ''.

    Returns:
        actors: (list) List of actors associated with the created cylinder.

    Description:
        Vtk texture data passed in via `textures` are mapped onto planar facets
        of the cylinder in a counter-clockwise order (when viewed from above),
        starting with the facet centered on the positive X axis. The N texture
        images in `textures` list, when viewed as a panoramic image, should be
        ordered from right ("textures[0]") to left ("textures[N-1]").

        Texture images are resized in the horizontal direction only in order to
        make them perfectly span the width of each cylinder facet. The vertical
        direction is never resized.
    """

    if colors is None and textures is None:
        raise Exception('Must provide either `colors` or `textures`')

    if type(tags) == str:
        tags = set(tags.split('_'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    actors = []
    origin = np.array(origin)
    dtheta = 2*np.rad2deg(np.arctan(width/(2*radius))) # approx. dtheta based on txtr image width
    numfacets = int(360/dtheta)
    if seamless:
        numfacets += len(textures) - (numfacets % len(textures))
    dtheta = 360/numfacets           # actual dtheta required to make a closed cylinder
    halfwidth = radius*np.tan(np.deg2rad(dtheta/2))       # half the width of one facet
    pos = np.array([radius,0,0]) + np.array([0,halfwidth,0])      # corner of 1st facet

    if textures is None:
        texture = None
        numcolors = len(colors)
    else:
        color = None
        numtextures = len(textures)
        tags = tags | set(["noflip"])

    # Scale the width of the texture images to make them fit the facets.
    s = 2*halfwidth/width
    v1 = -np.array([0, s*width, 0])    # world vector along texture image x axis
    v2 = np.array([0, 0, height])      # world vector along texture image y axis

    theta = 0
    for k in range(numfacets):
        if textures is None:
            color = colors[k % numcolors]
        else:
            texture = textures[k % numtextures]
        actors.append(make_quad(renderer, origin=origin, pos=pos, v1=v1, v2=v2,
                                orient=(0,0,theta), reps=reps, color=color,
                                texture=texture, lightcoef=lightcoef,
                                opacity=opacity, tags=tags))
        theta += dtheta

    return actors


def make_cylinder(renderer, param, color=None, texture=None, scale=(1,1,1),
                  lightcoef=(1,1,1), cap=False, res=20, camera=None, tags=''):
    """
    Make an upright cylinder that is textured mapped with a single image.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        param: Parameters of the cylinder: [xctr, yctr, zctr, radius, halfheight]

        texture: The VTK texture data.

        scale: (tuple) Texture scale factors: (# repetitions of texture around
        cylinder's circumference, # repititions of texture in direction of
        cylinder's axis, ???).

        res: (int) Number of planar facets to approximate curved cylinder wall.

        cap: (bool) Should the cylinder have caps at each end? Default is False.

        color: the color (R,G,B) of the object. Used if texture is not used.

        lightcoef: the object's reflectance coefficients: (ambient, diffuse, specular).

        camera: the current camera. If provided, the cylinder will rotate to follow
            the camera.
    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for cylinder')

    cylinder = vtk.vtkCylinderSource()
    cylinder.SetResolution(res)
    cylinder.SetHeight(2*param[4])
    cylinder.SetRadius(param[3])
    cylinder.SetCapping(cap)

    if texture is None:
        # Cylinder is a single color.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(color)  # Set one color for ambient, diffuse, and specular
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)
    else:
        # Texture map cylinder.
        maptocylinder = vtk.vtkTextureMapToCylinder()
        maptocylinder.SetInputConnection(cylinder.GetOutputPort())
        # maptocylinder.PreventSeamOn()
        maptocylinder.SetPreventSeam(False)
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(maptocylinder.GetOutputPort())
        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(maptocylinder.GetOutputPort())
        xform.SetScale(scale)
        xform.SetFlipT(1)
        xform.SetFlipS(0)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(xform.GetOutputPort())
        if camera is None:
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(texture)
        else:
            # Make the cylinder follow the camera.
            actor = vtk.vtkFollower()
            actor.SetMapper(mapper)
            actor.SetCamera(camera)
            actor.SetTexture(texture)

    actor.SetPosition(param[0:3])
    actor.RotateX(-90.0)
    actor.RotateY(360*np.random.rand())      # cylinder has random "front side"

    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    renderer.AddActor(actor)
    return actor


def make_cone(renderer, param, color=None, texture=None, lightcoef=(1,1,1),
              res=20, camera=None, tags=''):
    """
    Make a cone.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.
        param: Parameters of the cone: [xctr, yctr, zctr, radius, halfheight]
        texture: (vtkTexture, r_scale, s_scale)
        color: the color (R,G,B) of the object. Used if texture is not used.
        lightcoef: the object's reflectance coefficients: (ambient, diffuse, specular).
        camera: the current camera. If provided, the cylinder will rotate to follow
            the camera.

    Notes:
        The VTK texture object may be created from an image file as follows:
            reader = vtk.vtkPNGReader()
            reader.SetFileName(pngfilename)
            texture = vtk.vtkTexture()
            texture.SetInputConnection(reader.GetOutputPort())
    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for cylinder')

    cylinder = vtk.vtkCylinderSource()
    cylinder.SetResolution(res)
    cylinder.SetHeight(2*param[4])
    cylinder.SetRadius(param[3])

    if texture is None:
        # Cylinder is a single color.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # actor.GetProperty().SetColor(color)  # Set one color for ambient, diffuse, and specular
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)
    else:
        # Texture map cylinder.
        maptocylinder = vtk.vtkTextureMapToCylinder()
        maptocylinder.SetInputConnection(cylinder.GetOutputPort())
        maptocylinder.PreventSeamOn()
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(maptocylinder.GetOutputPort())
        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(maptocylinder.GetOutputPort())
        xform.SetScale(texture[1], texture[2], 1)
        xform.SetFlipT(1)
        xform.SetFlipS(0)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(xform.GetOutputPort())
        if camera is None:
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetTexture(texture[0])
        else:
            # Make the cylinder follow the camera.
            actor = vtk.vtkFollower()
            actor.SetMapper(mapper)
            actor.SetCamera(camera)
            actor.SetTexture(texture[0])

    actor.SetPosition(param[0:3])
    actor.RotateX(-90.0)
    actor.RotateY(360*np.random.rand())      # cylinder has random "front side"

    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    renderer.AddActor(actor)
    return actor


def make_vrect(renderer, param, color=None, texture=None, lightcoef=(1,1,1),
               tags=''):
    """
    Make a vertical rectangle, one whose edges are parallel to the XY plane and
    Z axis.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        param: Parameters of the rectangle: [xstart, ystart, zstart, xend, yend,
        zend]

        texture: (texture, hscale, vscale, topcolor, repetition, aspectwdh)

        color: the color (R,G,B) of the object. Used if texture is not used.

        lightcoef: the object's reflectance coefficients: (ambient, diffuse,
        specular).

    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for vert. rect')

    # Define the rectangle verticies [x, y, z].
    xs = param[0]; ys = param[1]; zs = param[2];
    xe = param[3]; ye = param[4]; ze = param[5]
    pts = [[xs, ys, zs], [xe, ye, zs], [xe, ye, ze], [xs, ys, ze]]
    poly_height = abs(zs - ze)
    poly_width = np.sqrt((xs - xe)**2 + (ys - ye)**2)
    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(*pt)

    # Create the polygon
    numpts = len(pts)
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(numpts)
    for k in range(numpts):
        polygon.GetPointIds().SetId(k,k)

    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)

    # Create a PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if texture:           # Texture map the polygon.
        # Assign texture image to polygon.
        actor.SetTexture(texture[0])

        # Assign texture coordinates.
        textureCoordinates = vtk.vtkFloatArray()
        textureCoordinates.SetNumberOfComponents(2);
        textureCoordinates.SetName("TextureCoordinates")
        xrep = poly_width/texture[1]
        yrep = poly_height/texture[2]
        textureCoordinates.InsertNextTuple([0, 0])
        textureCoordinates.InsertNextTuple([xrep, 0])
        textureCoordinates.InsertNextTuple([xrep, yrep])
        textureCoordinates.InsertNextTuple([0, yrep])
        polydata.GetPointData().SetTCoords(textureCoordinates)
    else:               # Polygon is single color.
        actor.GetProperty().EdgeVisibilityOff()
        # actor.GetProperty().SetEdgeColor(.2, .2, .5)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    actor.ForceOpaqueOn()            # this is needed for scene depth estimation

    renderer.AddActor(actor)


def make_rect(renderer, params, color=None, texture=None, hsize=1, vsize=1,
              lightcoef=(1,1,1), opacity=1.0, camera=None, tags=''):
    """
    Make a 3D rectangle.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        params: [(x0,y0,z0), (x1,y1,z1), (x2,y2,z2)]. (x0,y0,z0) is one corner
        of the rectangle. (x1,y1,z1) is the vector from (x0,y0,z0) to the
        adjacent corner in the direction corresponding to the X axis of the
        texture image. (x2,y2,z2) is the vector from (x0,y0,z0) to the adjacent
        corner in the direction corresponding to the Y axis of the texture
        image.

        texture: VTK texture data.

        hsize: (float) Size of texture image horizontal axis in world coordiantes.

        vsize: (float) Size of texture image vertical axis in world coordiantes.

        color: the color (R,G,B) of the object. Used if texture is not used.

        lightcoef: the object's reflectance coefficients: (ambient, diffuse,
        specular).

    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for vert. rect')

    p0 = np.array(params[0])  # origin (one corner) of the rectange
    dx = np.array(params[1])  # direction vector corresponding to x-axis of texture image
    dy = np.array(params[2])  # direction vector corresponding to y-axis of texture image

    # Get lengths of X and Y sides of rectangle.
    poly_xsize = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2])
    poly_ysize = np.sqrt(dy[0]*dy[0] + dy[1]*dy[1] + dy[2]*dy[2])

    pts = [p0, p0+dx, p0+dx+dy, p0+dy]

    points = vtk.vtkPoints()
    for pt in pts:
        points.InsertNextPoint(pt)

    # Create the polygon
    numpts = len(pts)
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(numpts)
    for k in range(numpts):
        polygon.GetPointIds().SetId(k,k)

    # Add the polygon to a list of polygons
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)

    # Create a PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if texture:           # Texture map the polygon.
        # Assign texture image to polygon.
        actor.SetTexture(texture)

        # Assign texture coordinates.
        textureCoordinates = vtk.vtkFloatArray()
        textureCoordinates.SetNumberOfComponents(2);
        textureCoordinates.SetName("TextureCoordinates")
        xrep = poly_xsize/hsize
        yrep = poly_ysize/vsize
        textureCoordinates.InsertNextTuple([0, 0])
        textureCoordinates.InsertNextTuple([xrep, 0])
        textureCoordinates.InsertNextTuple([xrep, yrep])
        textureCoordinates.InsertNextTuple([0, yrep])
        polydata.GetPointData().SetTCoords(textureCoordinates)
    else:               # Polygon is single color.
        actor.GetProperty().EdgeVisibilityOff()
        # actor.GetProperty().SetEdgeColor(.2, .2, .5)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetAmbientColor(color)
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetSpecularColor(color)

    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    renderer.AddActor(actor)




def make_rect2(renderer, pos=(0,0,0), rot=(0,0,0), scale=(1,1,1),
               color=None, texture=None, lightcoef=(1,1,1),
               opacity=1.0, camera=None, tags=''):
    """
    Make a 3D rectangle.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        pos: (array-like) The position of the center of the rectangle.

        rot: (array-like of floats) The rotation of the plane about the X, Y,
        and Z axes.

        texture: vtkTexture (texture, hscale, vscale)

        color: the color (R,G,B) of the object. Used if texture is not used.

        lightcoef: the object's reflectance coefficients: (ambient, diffuse,
        specular).

        camera: (vtkCamera()) The camera whose viewpoint the rectangle will
        follow.  If None, then the rectangle does not follow any camera.

    Description:
        By default, the rectangle lies in the X-Z plane with edges parallel to
        the X and Z axes.
        a normal parallel to the Y axis.
        The bottom of the rectangle will be on the plane Z = 0 and parallel to
        the Y axis.  The normal to the plane is parallel to the X axis.

    """
    if color is None and texture is None:
        raise Exception('Must provide either color or texture for vert. rect')
    # camera = renderer.GetActiveCamera()

    # transform = vtk.vtkTransform()
    # transform.Identity()
    # # transform.Translate(x,y,z)
    # transform.RotateWXYZ(90, 0, 1, 0)   # rotate about an axis: (angle in degrees, x, y, z)
    # texture.SetTransform(transform)
    texture[0].SetInterpolate(True)              # smooth texture images when viewed close up

    plane = vtk.vtkPlaneSource()
    # plane.SetOrigin(0,0,0)
    plane.SetCenter(0,0,0)
    plane.SetNormal(0,0,1)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())
    actor = vtk.vtkFollower()  # vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture[0])
    actor.SetScale(texture[1], texture[2], 1)
    actor.RotateY(180.0)
    actor.SetPosition(pos)
    # actor.SetOrientation(1,1,1)

    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    if camera is None:
        actor.RotateX(-90.0)
        if rot != (0,0,0):
            actor.RotateX(rot[0])
            actor.RotateY(rot[2])
            actor.RotateZ(rot[1])
    else:
        actor.SetCamera(camera)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    renderer.AddActor(actor)


def make_box(renderer, origin=(0,0,0), pos=(0,0,0), v1=(1,0,0), v2=(0,1,0),
             v3=(0,0,1), reps=(1,1,1), color=(0.3,0.3,0.3), texture=None,
             toptexture=None, topreps=(1,1), lightcoef=(1,1,1), opacity=1.0,
             tags=''):
    """
    Make a 3D box.

    Usage:
        actors = make_box(renderer, origin=(0,0,0), pos=(0,0,0), v1=(1,0,0),
                          v2=(0,1,0), v3=(0,0,1), reps=(1,1,1), color=(.3,.3,.3),
                          texture=None, toptexture=None, lightcoef=(1,1,1),
                          opacity=1.0):

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        origin: (3-array-like) The origin (x,y,z) of the actor's coordinate
        system relative to the world coordinate system. Rotation and translation
        of the actor is relative to this coordinate system. Default is (0,0,0).

        pos: (3-array-like) The position (x,y,z) of one corner (refered to as
        the 1st corner, below) of the box relative to the actor's origin.
        Default is (0,0,0).

        v1: (3-array-like) The 3-vector (dx,dy,dz) from the box's 1st corner
        (defined by "pos") that defines the 1st edge of the box (both its
        direction and length). If the box is being texture-mapped, then this
        edge will correspond to the X axis in texture image. Default is (1,0,0).

        v2: (3-array-like) The 3-vector (dx,dy,dz) from the box's 1st corner
        (defined by "pos") that defines the 2nd edge of the box (both its
        direction and length). If the box is being texture-mapped, then this
        edge will correspond to the X axis in texture image. Default is (0,1,0).

        v3: (3-array-like) The 3-vector (dx,dy,dz) from the box's 1st corner
        (defined by "pos") that defines the height of the box (both its
        direction and length). If the box is being texture-mapped, then this
        edge will correspond to the Y axis in texture image. Default is (0,0,1).

        reps: (3-array-like) The triple, (r1,r2,r3), of repetitions of the texture
        image on the box's surfaces in the V1, V2, and V3 directions, respectively.
        Default is (1,1,1).

        topreps: (2-array-like) The triple, (r1,r2), of repetitions of the
        texture image on the box's top surface in the V1, V2 directions,
        respectively. Default is (1,1).

        texture: The vtkTexture texture data for the sides of the box. Default
        is None.

        toptexture: The vtkTexture texture data for the top of the box. Default
        is None.

        color: The color (R,G,B) of the surface. Used if "texture" is None. Used
        for the top of the box if "toptexture" is None. R, G, and B are all
        floats in [0,1]. Default is None.

        lightcoef: (3-array-like) The object's reflectance coefficients:
        (ambient, diffuse, specular). Default is (1,1,1).

    Returns:
        actors: (list) List of actors associated with the created box.
    """

    if color is None and texture is None:
        raise Exception('Must provide either color or texture')

    actors = []
    pos = np.array(pos)
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)

    color1 = None if texture else color

    # Side 1: Corner at "pos" and parallel to "v1".
    actors.append(make_quad(renderer, origin=origin, pos=pos, v1=v1, v2=v3,
                            reps=(reps[0],reps[2]), color=color1, texture=texture,
                            lightcoef=lightcoef, opacity=opacity, tags=tags))

    # Side 2: Corner at `pos+v1` and parallel to "v2".
    actors.append(make_quad(renderer, origin=origin, pos=pos+v1, v1=v2, v2=v3,
                            reps=(reps[1],reps[2]), color=color1, texture=texture,
                            lightcoef=lightcoef, opacity=opacity, tags=tags))

    # Side 3: Corner at `pos+v1+v2` and parallel to "-v1".
    actors.append(make_quad(renderer, origin=origin, pos=pos+v1+v2, v1=-v1, v2=v3,
                            reps=(reps[0],reps[2]), color=color1, texture=texture,
                            lightcoef=lightcoef, opacity=opacity, tags=tags))

    # Side 4: Corner at `pos+v2` and parallel to "-v2".
    actors.append(make_quad(renderer, origin=origin, pos=pos+v2, v1=-v2, v2=v3,
                            reps=(reps[1],reps[2]), color=color1, texture=texture,
                            lightcoef=lightcoef, opacity=opacity, tags=tags))

    # Side 5: Top of box.
    if toptexture is not None:
        actors.append(make_quad(renderer, origin=origin, pos=pos+v3, v1=v1, v2=v2,
                                reps=topreps, color=None, texture=toptexture,
                                lightcoef=lightcoef, opacity=opacity, tags=tags))
    elif color is not None:
        actors.append(make_quad(renderer, origin=origin, pos=pos+v3, v1=v1, v2=v2,
                                reps=(1,1), color=color, texture=None,
                                lightcoef=lightcoef, opacity=opacity, tags=tags))

    return actors


def make_poly(renderer, origin=(0,0,0), pts=None, orient=(0,0,0), color=None,
              texture=None, lightcoef=(1,1,1), opacity=1.0, tags=''):
    """
    Make a planar polygon.

    Usage:
        actor = make_poly(renderer, origin=(0,0,0), pts=None, orient=(0,0,0),
                          color=None, texture=None, lightcoef=(1,1,1),
                          opacity=1.0, tags='')

    Arguments:
        renderer: The renderer that the object's actor is to be added to.

        origin: (3-array-like) The origin (x,y,z) of the actor's coordinate
        system relative to the world coordinate system. Rotation and translation
        of the actor is relative to this coordinate system. Default is (0,0,0).

        pts: (list) A list or array of three or more 3D ponts lying in a plane.
        These points are the corners of the polygon. No points should be
        repeated. The polygon may be nonconvex, but may not have internal loops.
        Default is None.

        orient: (3-array-like) The 3-vector (rx, ry, rz) gives the rotation
        angles about the X, Y, and Z axes, respectively, of the surface around
        the actor's origin. All angles are in degrees.

        texture: The VTK texture data. Default is None.

        color: The color (R,G,B) of the surface. Used only if texture is not
        used. R, G, and B are all floats in [0,1]. Default is None.

        lightcoef: (3-array-like) The object's reflectance coefficients:
        (ambient, diffuse, specular). Default is (1,1,1).

        tags: Any data that is to be saved to the actor's 'tag' property. This
        data may be retrieved using the following: actor.GetProperty().tags.
        Default value is ''.

    Returns:
        actor: (vtkActor()) The actor associated with the created quadrilateral.
    """

    numpts = len(pts)
    if numpts < 3:
        raise ValueError('Argument PTS must be 3 or more 3D points')

    if color is None and texture is None:
        raise Exception('Must provide either color or texture')
    elif color is not None and texture is not None:
        raise Exception('Only one of "color" or "texture" argument should be given.')

    if type(tags) == str:
        tags = set(tags.split('.'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    # Create a polygon in the actor's coordinate system.
    points = vtk.vtkPoints()
    for k in range(numpts):
        points.InsertNextPoint(pts[k])
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(numpts)
    for k in range(numpts):
        polygon.GetPointIds().SetId(k,k)
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if texture == None:
        # Quadrilateral is a solid color.
        actor.GetProperty().SetColor(color)
    else:
        # Texture-map the quadrilateral.
        textcoords = vtk.vtkFloatArray()
        textcoords.SetNumberOfComponents(2)
        textcoords.SetName("TextureCoordinates")

        # Use the original texture.
        textcoords.InsertNextTuple((0,0))
        textcoords.InsertNextTuple((reps[0],0))
        textcoords.InsertNextTuple((reps[0],reps[1]))
        textcoords.InsertNextTuple((0,reps[1]))

        poly.GetPointData().SetTCoords(textcoords)
        actor.SetTexture(texture)

    # Set the lighting properties of the surface.
    actor.GetProperty().SetOpacity(opacity)        # Opacity: 0 = invisible, 1 = opaic
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Setup the actor relative to the world coordinate system.
    actor.SetOrientation(orient)
    actor.SetPosition(origin)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    actor.ForceOpaqueOn()            # this is needed for scene depth estimation

    renderer.AddActor(actor)
    return actor


def make_quad(renderer, origin=(0,0,0), pos=(0,0,0), v1=(1,0,0), v2=(0,1,0),
              orient=(0,0,0), reps=(1,1), color=None, texture=None, hflip=False,
              lightcoef=(1,1,1), opacity=1.0, tags=''):
    """
    Make a quadrilateral.  (Currently, just a parallelogram.)

    Usage:
        actor = make_quad(renderer, origin=(0,0,0), pos=(0,0,0), v1=(1,0,0),
                          v2=(0,1,0), orient=(0,0,0), reps=(1,1), color=None,
                          texture=None, hflip=False, lightcoef=(1,1,1),
                          opacity=1.0, tags='')

    Arguments:
        renderer: The renderer that the object's actor is to be added to.

        origin: (3-array-like) The origin (x,y,z) of the actor's coordinate
        system relative to the world coordinate system. Rotation and translation
        of the actor is relative to this coordinate system. Default is (0,0,0).

        pos: (3-array-like) The position (x,y,z) of one corner of the
        quadrilateral relative to the actor's origin.  Default
        is (0,0,0).

        v1: (3-array-like) The 3-vector (dx,dy,dz) from the quadrilateral's 1st
        corner (defined by "pos") that defines the 1st edge of the quadrilateral
        (both its direction and length). If the quadrilateral is being
        texture-mapped, then this edge will correspond to the X axis in texture
        image. Default is (1,0,0).

        v2: (3-array-like) The 3-vector (dx,dy,dz) from the quadrilateral's 1st
        corner (defined by "pos") that defines the 2nd edge of the quadrilateral
        (both its direction and length). If the quadrilateral is being
        texture-mapped, then this edge will correspond to the Y axis in texture
        image. Default is (0,1,0).

        orient: (3-array-like) The 3-vector (rx, ry, rz) gives the rotation
        angles of the surface about the X, Y, and Z axes, respectively, passing
        through the actor's origin. All angles are in degrees. Positive rotation
        angles give counterclockwise rotations about the respective axis
        (following the right-hand rule).

        reps: (2-array-like) The pair, (rx, ry), of repetitions of the texture
        image on the quadrilateral in the X and Y directions, respectively.
        Default value is (1,1)

        texture: The VTK texture data. Default is None.

        hflip: (bool) Flip the texture image horizontally? Default is False.

        color: The color (R,G,B) of the surface. Used only if texture is not used.
        Default is None.  R, G, and B are all floats in [0,1].

        lightcoef: (3-array-like) The object's reflectance coefficients:
        (ambient, diffuse, specular). Default is (1,1,1).

        tags: Any data that is to be saved to the actor's 'tag' property. This
        data may be retrieved using the following: actor.GetProperty().tags.
        Default value is ''.

    Returns:
        actor: (vtkActor()) The actor associated with the created quadrilateral.
    """

    if color is None and texture is None:
        raise Exception('Must provide either color or texture')
    elif color is not None and texture is not None:
        raise Exception('Only one of "color" or "texture" argument should be given.')

    if type(tags) == str:
        tags = set(tags.split('.'))        # convert string to set
    elif type(tags) != set:
        raise ValueError('Tags argument must be a string or set of strings')

    pos = np.array(pos)
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Create a quadrilateral in the actor's coordinate system.
    points = vtk.vtkPoints()
    points.InsertNextPoint(pos)         # (0, 0, 0)
    points.InsertNextPoint(pos+v1)      # (1, 0, 0)
    points.InsertNextPoint(pos+v1+v2)   # (1, 1, 0)
    points.InsertNextPoint(pos+v2)      # (0, 1, 0)
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)
    polygon.GetPointIds().SetId(0, 0)
    polygon.GetPointIds().SetId(1, 1)
    polygon.GetPointIds().SetId(2, 2)
    polygon.GetPointIds().SetId(3, 3)
    polygons = vtk.vtkCellArray()
    polygons.InsertNextCell(polygon)
    quad = vtk.vtkPolyData()
    quad.SetPoints(points)
    quad.SetPolys(polygons)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(quad)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if texture == None:
        # Quadrilateral is a solid color.
        actor.GetProperty().SetColor(color)
    else:
        # Texture-map the quadrilateral.
        textcoords = vtk.vtkFloatArray()
        textcoords.SetNumberOfComponents(2)
        textcoords.SetName("TextureCoordinates")

        if hflip:
            # Flip the texture horizontally.
            textcoords.InsertNextTuple((reps[0],0))
            textcoords.InsertNextTuple((0,0))
            textcoords.InsertNextTuple((0,reps[1]))
            textcoords.InsertNextTuple((reps[0],reps[1]))
        else:
            # Use the original texture.
            textcoords.InsertNextTuple((0,0))
            textcoords.InsertNextTuple((reps[0],0))
            textcoords.InsertNextTuple((reps[0],reps[1]))
            textcoords.InsertNextTuple((0,reps[1]))

        quad.GetPointData().SetTCoords(textcoords)
        actor.SetTexture(texture)

    # Set the lighting properties of the surface.
    actor.GetProperty().SetOpacity(opacity)        # Opacity: 0 = invisible, 1 = opaic
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    # Setup the actor relative to the world coordinate system.
    actor.SetOrientation(orient)
    actor.SetPosition(origin)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    actor.ForceOpaqueOn()            # this is needed for scene depth estimation

    renderer.AddActor(actor)
    return actor


def make_text(renderer, text=None, camera=None, pos=(0,0,0), textscale=0.2,
              opacity=1.0, lightcoef=(1,1,1), tags=''):
    """
    Make text that follows the camera.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.
        text: The text (a string) to insert.
        pos: The position (X,Y,Z) to place the text.
        camera: The camera to follow. Uses the active camera if none is given.
        textscale: The scale factor of the text.
    """

    if camera is None:
        camera = renderer.GetActiveCamera()
    vtext = vtk.vtkVectorText()
    vtext.SetText(text)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vtext.GetOutputPort())
    actor = vtk.vtkFollower()
    actor.SetMapper(mapper)
    actor.SetScale(textscale, textscale, textscale)
    actor.AddPosition(pos)
    # textprop = actor.GetProperty()
    renderer.AddActor(actor)
    actor.SetCamera(camera)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    actor.GetProperty().SetOpacity(opacity)        # Opacity: 0 = invisible, 1 = opaic
    actor.GetProperty().SetAmbient(lightcoef[0])   # Ambient (nondirectional) lighting coefficient
    actor.GetProperty().SetDiffuse(lightcoef[1])   # Diffuse (direct) lighting coefficient
    actor.GetProperty().SetSpecular(lightcoef[2])  # Specular (highlight) lighting coefficient

    return actor


def make_axes(renderer, camera=None, pos=(0,0,0), axscale=5, linewidth=5,
              labelscale=0.5, tags=''):
    """
    Make a labeled axes.

    Arguments:
        renderer: The renderer that the obect's actor is to be added to.

        pos: The position, (X,Y,Z), to place the axes.

        camera: The camera that axes labels will follow. Uses the active camera if
            none is given.

        axscale: The size of the axes.

        linewidth: The thickness of the axes lines.

        labelscale: The scale factor of the axes labels.
    """

    # Create the 3D axes.
    axes = vtk.vtkAxes()
    axes.SetSymmetric(False)
    axesMapper = vtk.vtkPolyDataMapper()
    axesMapper.SetInputConnection(axes.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(axesMapper)
    actor.SetScale(axscale, axscale, axscale)
    actor.SetPosition(pos)
    axprop = actor.GetProperty()
    axprop.SetLineWidth(linewidth)
    renderer.AddActor(actor)

    if camera is None:
        camera = renderer.GetActiveCamera()

    # Label the coordinate axes.
    axpos = np.array([axscale+2*labelscale, 0, 0])   # labelscale])
    pos = np.array(pos)
    for k in range(0,3):
        vtext = vtk.vtkVectorText()
        vtext.SetText("XYZ"[k])
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputConnection(vtext.GetOutputPort())
        textActor = vtk.vtkFollower()
        textActor.SetMapper(textMapper)
        textActor.SetScale(labelscale, labelscale, labelscale)
        textActor.AddPosition(pos + np.roll(axpos, k))
        # textprop = textActor.GetProperty()
        renderer.AddActor(textActor)
        textActor.SetCamera(camera)

    # Save user-defined data in the actor's "tags" property.
    p = actor.GetProperty()
    p.tags = tags
    actor.SetProperty(p)

    return actor



def label_points(renderer, pts3d, labels, camera=None, markers=True, markersize=0.1,
                 markercolor=(0,0,0), labelscale=0.5, tags=''):
    """
    Label some points in the work model.

    Usage:
        label_points(renderer, pts3d, labels, camera=None, markers=True,
                     markersize=0.1, markercolor=(0,0,0), labelscale=0.5)

    Arguments:
        renderer: the VTK renderer.
        pts3d: a list of 3D points.
        labels: a list of text labels, one for each point in 'pts3d'.
        camera: the camera that the labels should follow. Default: current active
            camera.
        labelscale: scale factor for labels. Default: 0.5.
        markers: True or False. Draw a marker (sphere) at each point? Default: True.
        markersize: size of the marker. Default: 0.1.
        markercolor: color of all markers (an 3-tuple). Default: black.

    Notes:
        The apparent colors of markers is affected by the lighting, so they
        don't always appear as the requested color.
    """

    if len(pts3d) != len(labels):
        raise Exception('Error: arguments "pts3d" and "labels" must be the same length')

    if camera is None:
        camera = renderer.GetActiveCamera()

    for k in range(0, len(pts3d)):
        atext = vtk.vtkVectorText()
        atext.SetText(labels[k])
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputConnection(atext.GetOutputPort())
        textActor = vtk.vtkFollower()
        textActor.SetMapper(textMapper)
        textActor.SetScale(labelscale, labelscale, labelscale)
        pos = np.array(pts3d[k])
        textActor.AddPosition(pos)
        # textprop = textActor.GetProperty()
        renderer.AddActor(textActor)
        textActor.SetCamera(camera)

        # Save user-defined data in the actor's "tags" property.
        p = textActor.GetProperty()
        p.tags = tags
        textActor.SetProperty(p)

        if markers:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pos)
            sphere.SetRadius(markersize/2)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(markercolor)
            renderer.AddActor(actor)

            # Save user-defined data in the actor's "tags" property.
            p = actor.GetProperty()
            p.tags = tags
            actor.SetProperty(p)


def update_renderers(myworld, camera, renderers, pos=None, viewdir=None, vfov=None):
    """
    Upate the camera and rendered scenes for the given camera parameters.

    Arguments:
        myworld: (SimWorld) The simulated world.

        camera: The vtkCamera().

        renderers: A list of vtkRenderer() objects.

        pos: The new camera position, (X,Y,Z).  Default is None.

        viewdir: The new camera viewing direction, (dX,dY,dZ). Default is None.
        The VTK focal point (FP) does not move with the camera's position; it
        enables a camera to look at a finite fixed point from different points
        of view. Viewdir does move with (it is independent of) camera position;
        it is a viewing *direction* not a *fixed point*: VIEWDIR = FP - POS.

        vfov: The new camera vertical viewing angle (vertical FOV, in degrees).
    """
    if pos is not None:
        camera.SetPosition(pos)

    if viewdir is not None:
        if pos is None:
            pos = camera.GetPosition()
        camera.SetFocalPoint(pos[0]+viewdir[0], pos[1]+viewdir[1], pos[2]+viewdir[2])

    if vfov is not None:
        camera.SetViewAngle(vfov)

    for ren in renderers:
        ren.ResetCameraClippingRange()
        ren.GetRenderWindow().Render()


def get_camera_pose(camera):
    """
    Get the position, orientation, and field of view, of the renderer's camera.

    Usage:
        pos, viewdir, vfov = get_camera_pose(camera)

    Arguments:
        camera: A vtkCamera object.

    Returns:
        pos: The camera position (center of projection), a tuple (X,Y,Z).
        viewdir: The camera viewing direction, a tuple (dX,dY,dZ).
        vfov: The camera's vertical field-of-view, in degrees, a float.

    Description:
        The camera orientation is returned in 'viewdir.' This is a vector pointing
        along the camera's optical axis. This "viewing direction" of the camera
        changes when the camera pans or tilts, but not when the camera translates.
    """
    pos = camera.GetPosition()   # (X,Y,Z) camera absolute position
    fp = camera.GetFocalPoint()  # 3D focal point, changes with camera translation
    vfov = camera.GetViewAngle() # vertical field-of-view (degrees)
    viewdir = tuple(np.array(fp) - np.array(pos))
    return pos, viewdir, vfov


def numpy2texture(im):
    """
    Convert a numpy array to a VTK texture.

    Usage:
        vtktex = numpy2texture(im)

    Arguments:
        im: (numpy.ndarray) This is the NumRows X NumCols X NumChans Numpy array
        that is to be converted to a VTK texture. NumChans must be 3 or 4.

    Returns:
        vtktex: (vtkTexture) The VTK texture.
    """

    if im.ndim != 3:
        raise ValueError('Argument {im} must be a nrows x ncols x nchans Numpy array')
    elif im.shape[2] > 4:
        raise ValueError('Argument {im} must have 3 or 4 channels')
    nchan = im.shape[2]

    if True:
        vtktex = pyvista.numpy_to_texture(im)
    else:
        grid = vtk.vtkImageData()
        grid.SetDimensions(im.shape[1], im.shape[0], 1)
        vtkarr = numpy_to_vtk(np.flip(im.swapaxes(0,1), axis=1).reshape((-1,nchan), order='F'),
                              deep=True, array_type=vtk.VTK_INT)
        vtkarr.SetName('Image')
        grid.GetPointData().AddArray(vtkarr)
        grid.GetPointData().SetActiveScalars('Image')

        vtktex = vtk.vtkTexture()
        vtktex.SetInputDataObject(grid)
        vtktex.Update()

    return vtktex
