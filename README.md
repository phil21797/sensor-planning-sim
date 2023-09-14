

# Sensor Planning Simulation Environment
<br>
 <p align="center">  
 <img src="README.images/camera_image.png">
 </p><br>
 
## Overview

**The goal of this project is to provide a simulation environment to support the research and development of multi-agent, multi-sensor, surveillance algorithms in large, outdoor environments.**

This project supports the Army Research Laboratory's Internet of Battlefield Things (IoBT) program (https://www.arl.army.mil/cras/iobt-cra/). 

This simulator procedurally generates random, quasi-photorealistic, 3D urban environments that may be used to train and evaluate mutl-agent perception and planning algorithms. The size and compexity of the generated environments, and the number of sensor-equiped agents, are limited only by the available computing resources. See the [sim_world/README](sim_world/README.md) file for more information about the simulator's capabilities.

<br>

## Multiagent, Multisensor Surveillance


<table>
<tr>
<td><img src="README.images/map_2d.png"><br>
<p align="left"> 
The groundtruth map shows static and dynamic objects, agents,  and their sensors. Each agent may possess a camera and/or a  microphone. A camera is represented by a wedge showing the  camera's horizontal field of view. A microphone is represented by a double ring. A camera wedge or microphone ring is colored red when the camera or microphone, respectively, is turned on and  sensing the environment. In this figure, agent 1 has a camera and a microphone, but the microphone is turned off; agent 4 has just a microphone, which is turned on; agent 5 has just a camera; and agent 6 has a camera and a microphone with the camera turned off. For microphones that are powered on, we identify sound-producing objects that are in audible range of a microphone by connecting the object to the microphone with a colored dotted line, where the color indicates the strength of the audio signal at the microphone (red is a strong signal, blue is a weak signal). 
 </td></p>
</tr>
</table><br>

## Multiagent Display

<table>
<tr>
<td><img src="README.images/montage.png"><br>
<p align="Left"> The above shows the display at one instant in a multiagent simulation.  Left: A 2D "map" representation of the environment.  Red and transparent wedges represent PTZ camera fields-of-view. Red and black rings represent microphones. Colored dotted lines represent possible acoustic detections. Right: Camera images and overlayed microphone sensor data as observed by the six agents shown in the 2D map. </td></p>
</tr>
</table>

## Updates

**14 Aug 2023** -- The method of creating objects to follow fixed trajectories has been updated. Now, these objects, as well as sensing agents, are described in a text file created by the user. Various object movement commands are available to describe the objects' trajectories.  See the external environment definitions [README](README_envdefs.md) for more information.

**11 Aug 2023** -- Here's a [video](./sim_demo_01.mp4) showing a short run of the simulator.

**7 Sept. 2023** -- The YOLO object detector has been installed in the code and is being used in`multiagent.py`to process all camera images.

**9 Sept. 2023** -- The multiagent code (`multiagent.py`) and a lot of the supporting code has been updated. A simple multi-sensor control algorithm is implemented where an agent's microphone is turned on for a short time (e.g., 1 sec.) to check for sounds with amplitudes above a given threshold (i.e., audio detections). Whenever there are audio detections, the agent's camera will scan the scene looking for the sources. When nothing is detected by the microphone for a short time, the agent will shut off the camera and microphone for a few seconds (e.g., 5).  After this "sleep" period, the agent starts this process over by turning the microphone back on (with the camera still off) to check for new sounds. Here's a [video](./sim_demo_02.mp4) demonstrating this multiagent algorithm.  This is not meant to be a useful algorithm (there is no inter-agent coordination), but is only a simple example of how the agents can control their sensors in this simulation environment.

<br>

## Running the code

* This code was developed and tested on a Linux system, but should run on a Windows system with minor modifications. The following command may be used to recreate the environment in which this software was developed:
~~~bash
$ conda create --name <env> --file requirements.txt
~~~

* Add `.../sim_world` to your `PYTHONPATH` environment variable.

* If desired, edit `multiagent.py` to change various program parameters.

* If you want to run the YOLO object detector, download the YOLO weights into the folder `obj_detector/weights`.  See [obj_detector/weights/README](obj_detector/weights/README.md) for details.

* See [sim_world/README](sim_world/README.md)  for additional setup information.

* Run it:
~~~bash
$ python multiagent.py
~~~
 
 <br>
 
## Author

Phil David

&nbsp;&nbsp;&nbsp;&nbsp; U.S. Army DEVCOM, Army Research Laboratory, 2020-04-07.

&nbsp;&nbsp;&nbsp;&nbsp; Parsons Corp., 2023-06-01.

<br>

## Acknowledgment

Mr. Jeff Goldman, U.S. Army DEVCOM, Army Research Laboratory, helped with the design of the acoustic sensor model.



