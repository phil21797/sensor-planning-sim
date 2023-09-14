
# External Environment Definitions

The environment definitions file is used to define the position and properties of moving objects and agents in the environment. 

The general file format is:

    OBJECT <Object_1_tags>      # this is a comment
    START <x> <y> <z>
    <move_command_1>
    <move_command_2>
    ...
    <move_command_N>
    END
    ...
    AGENT <x> <y> <xfwd> <yfwd> <hascam> <hasmic> <initzoom> <panrange>
    ...

Any number of objects and agents my be defined. "Objects" are the things that move around the environment that the agents are expected to detect and track. Each object definition starts with the `OBJECT <object_tags>` command. Then, the trajectory of each object is defined by a sequence of object movement commands:

    START <x> <y> <z>           -- The starting position of the object (at
                                   time 0). This is required for all objects.

    TIME <time> <x> <y> <z>     -- Move to position (x,y,z) at absolute time
                                   <time> sec.

    DTIME <dtime> <x> <y> <z>   -- Move for <dtime> sec. to position (x,y,z).

    SPEED <speed> <x> <y> <z>   -- Move at speed <speed> to position (x,y,z)
                                   where <speed> > 0.

    ARC <dir> <rad> <deg> <speed>
                                -- Move through an arc in direction <dir>,
                                   either LEFT or RIGHT, radius <rad>, for
                                   <deg> degrees, and at speed <speed>.

    STOP <dtime>                -- Stop for <dtime> sec.

    END                         -- End the definition of the object. This is
                                   required for all objects.

Times for an object must be listed sequentially and be increasing. After an object reaches its final defined position, the object jumps back to its starting position on the next frame of the simulation.

Any number of agents may be defined. Agents, which are stationary except when moved by control algorithms external to the environment simulation, possess sensors for detecting objects. Each agent definition starts with the `AGENT` keyword and is followed by eight parameters:

     <x> <y>        -- The 2D position (z is 0) of the agent in the environment.

     <xfwd> <yfwd>  -- The 2D forward-facing direction of the agent.

     <hascam>       -- 1 if the agent has a camera; otherwise, 0.

     <hasmic>       -- 1 if the agent has a microphone; otherwise, 0

     <initzoom>     -- Initial zoom (in [0,1]) of the agent's camera.

     <panrange>     -- Range of pan angles (in degrees) of the agent's camera
                       (e.g., 180 => pan from -90° to 90°). Note: A pan angle
                       of 0° will point the agent's camera in the direction
                       (<xfwd>, <yfwd>).

For reference, vehicles typically move at speeds between 22 and 36 m/s on highways, and move at speeds up to 22 m/s on non-highways; and humans typically walk at around 3 m/s, and jog at around 5-6 m/s.

