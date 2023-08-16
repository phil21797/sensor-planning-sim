wavio
=====

``wavio`` is a Python module that defines two functions:

* ``wavio.read`` reads a WAV file and returns an object that holds the
  sampling rate, sample width (in bytes), and a numpy array containing the
  data.
* ``wavio.write`` writes a numpy array to a WAV file, optionally using a
  specified sample width.

The functions can read and write 8-, 16-, 24- and 32-bit integer WAV files.

The module uses the ``wave`` module in Python's standard library, so it has
the same limitations as that module.  In particular, the ``wave`` module
does not support compressed WAV files, and it does not handle floating
point WAV files.  When floating point data is passed to ``wavio.write`` it
is converted to integers before being written to the WAV file.

``wavio`` requires Python 3.7 or later.

``wavio`` depends on numpy (http://www.numpy.org).  NumPy version 1.19.0 or
later is required.    The unit tests in ``wavio`` require ``pytest``.

The API of the functions in ``wavio`` should not be considered stable.  There
may be backwards-incompatible API changes between releases.

*Important notice*

In version 0.0.5, the data handling in ``wavio.write`` has been changed in
a backwards-incompatible way.  The API for scaling the input in 0.0.4 was
a flexible interface that only its creator could love.  The new API is
simpler, and it is hoped that it does the right thing by default in
most cases.  In particular:

* When the input data is an integer type, the values are not scaled or
  shifted.  The only change that might happen is the data will be clipped
  if the values do not fit in the output integer type.
* If the input data is a floating point type, ``sampwidth`` must be given.
  The default behavior is to scale input values in the range [-1.0, 1.0]
  to the output range [min_int+1, max_int], where min_int and max_int are
  the minimum and maximum values of the output data type determined by
  ``sampwidth``.  See the description of ``scale`` in the docstring of
  ``wavio.write`` for more options.  Regardless of the value of ``scale``,
  the float input 0.0 is always mapped to the midpoint of the output type;
  ``wavio.write`` will not translate the values up or down.
* A warning is now generated if any data values are clipped.  A parameter
  allows the generation of the warning to be disabled or converted to an
  exception.

Example
~~~~~~~

The following code (also found in the docstring of ``wavio.write``) writes
a three second 440 Hz sine wave to a 24-bit WAV file::

    import numpy as np
    import wavio

    rate = 22050           # samples per second
    T = 3                  # sample duration (seconds)
    n = int(rate*T)        # number of samples
    t = np.arange(n)/rate  # grid of time values

    f = 440.0              # sound frequency (Hz)
    x = np.sin(2*np.pi * f * t)

    wavio.write("sine24.wav", x, rate, sampwidth=3)


-----

:Author:     Warren Weckesser
:Repository: https://github.com/WarrenWeckesser/wavio
:License:    BSD 2-clause (http://opensource.org/licenses/BSD-2-Clause)
