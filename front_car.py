from __future__ import print_function

import glob
import os
import sys
import time
from time import sleep

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
from carla import ColorConverter as cc
from carla import Transform 
from carla import Location
from carla import Rotation
from agents.navigation.global_route_planner import GlobalRoutePlanner

from PIL import Image
from tqdm import tqdm
import random


SHOW_PREVIEW = True
IM_WIDTH = 640
IM_HEIGHT = 480
final_destination = [190, 306.886, 5]
final_destination = [194.01885986328125,262.87078857421875, 5]

actor_list = []


epsilon = 0.1 


FPS = 60

def trajectory(world,vehicle, draw = False):
    amap = world.get_map()
    sampling_resolution = 2
    # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
    grp = GlobalRoutePlanner(amap, sampling_resolution)
    # grp.setup()
    
    start_location = vehicle.get_transform().location
    end_location = carla.Location(x=final_destination[0], y=final_destination[1], z=0)
    a = amap.get_waypoint(start_location, project_to_road=True)
    b = amap.get_waypoint(end_location, project_to_road=True)
    spawn_points = world.get_map().get_spawn_points()
    #print(spawn_points)
    a = a.transform.location
    b = b.transform.location
    w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
    i = 0
    if draw:
        for w in w1:
            if i % 10 == 0:
                world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                persistent_lines=True)
            else:
                world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                persistent_lines=True)
            i += 1
    return w1


# Connect to the Carla simulator
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
world = client.get_world()

# Spawn a vehicle in the world
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = carla.Transform(Location(x=90, y=306.886, z=5), Rotation(yaw=0))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
actor_list.append(vehicle)

# to wait for the agent to spawn and start moving
sleep(5)


flag = True


try:


    # Get the location of the vehicle at every timestep
    while flag == True:

        # Get the current location of the vehicle
        location = vehicle.get_location()

        
        # to stop the car at these location for 2 seconds
        if 0xFF == ord('q'):
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake = 1.0))

        elif 194 < location.x < 195 and 260 < location.y < 265:
            print("Final Destination Reached....")
            print("Destroying all actors....")

            flag = False

            for actor in actor_list:
                actor.destroy()
        
        else:      
            # Do some simulation step (e.g. apply control to the vehicle)
            way = trajectory(world, vehicle)[0][0]
            waypoint_rot = way.transform.rotation
            rot = vehicle.get_transform().rotation
            orientation_diff = waypoint_rot.yaw - rot.yaw
            phi = orientation_diff%360 -360*(orientation_diff%360>180)
            if abs(phi)<1:
                control = carla.VehicleControl(throttle=0.5, steer=0.0) 
            elif abs(phi)<2:
                if phi<0:
                    control = carla.VehicleControl(throttle=0.5, steer=-0.1) 
                elif phi>=0:
                    control = carla.VehicleControl(throttle=0.5, steer=0.1) 
            else:
                if phi<0:
                    control = carla.VehicleControl(throttle=0.5, steer=-0.4) 
                elif phi>=0:
                    control = carla.VehicleControl(throttle=0.5, steer=0.4) 
            
            vehicle.apply_control(control)
            world.tick()
            print(f"Vehicle location: x={location.x}, y={location.y}, z={location.z}")
        
        time.sleep(1/FPS)



except KeyboardInterrupt:
    for actor in actor_list:
        actor.destroy()