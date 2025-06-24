#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time
import threading
import argparse, sys

import numpy as np
import pandas as pd

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
import random
import pdb
import math

# con, top
tk = 0.4
# behav
# tk = 0.2
# large
# tk = 3.0
# manhattan
# tk = 4.0
# z_default=70
# cathedral, berlin tv tower
# tk = 3.0
z_default = 1.5

color_min_max = ((150, 200), (0, 50), (0, 20))
walker_color_code = [(234, 56,41 ), (246, 133, 17), (248, 217, 4), (170, 216, 22), (39, 187, 54), (0, 143, 93), (15, 181, 174), (51, 197, 232), (56, 146, 243), (104, 109, 244), (137, 61, 231), (224, 85, 226), (222, 61, 130)]
#walker_color_code = [(234, 56,41 )]
vehicle_color_code = [(234, 56,41 ), (246, 133, 17), (248, 217, 4), (170, 216, 22), (39, 187, 54), (0, 143, 93), (15, 181, 174), (51, 197, 232), (56, 146, 243), (104, 109, 244), (137, 61, 231), (224, 85, 226), (222, 61, 130)]
#vehicle_color_code = [(56, 146, 243)]

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def load_csv(path, origin=np.array([[0, 0]]), frame_every=5, multi=1.8, swap_xy=False, flip_x=False, flip_y=False):
    actors_dict = {}
    dat = pd.read_csv(path)
    
    agent_ids = dat['agent_id'].unique()
    
    for a_id in agent_ids:
        agent_dat = dat.loc[dat['agent_id']==a_id]
        start_frame = agent_dat['frame'].min()
        
        if not start_frame in actors_dict:
            actors_dict[start_frame] = ({}, {})
            
        position_array = np.array(agent_dat[["x", "y"]])
        if swap_xy:
            position_array = position_array[:, [1, 0]]
        frame_array = np.array(agent_dat["frame"])
        
        sorted_inds = frame_array.argsort()
        sorted_inds = sorted_inds[::frame_every]
        position_array = position_array[sorted_inds] * multi
        if flip_x:
            position_array[:, 0] = -position_array[:, 0]
        if flip_y:
            position_array[:, 1] = -position_array[:, 1]
        position_array = position_array + origin
        frame_array = (frame_array[sorted_inds] - start_frame)
        
        assert position_array.shape[0] == frame_array.shape[0]
        
        if agent_dat['agent_type'].iloc()[0] == 0:
            # human
            actors_dict[start_frame][0][a_id] = (frame_array, position_array)
        else:
            # vehicle
            if not 'vehicle' in actors_dict[start_frame]:
                actors_dict[start_frame][1][(agent_dat['agent_type'].iloc()[0], a_id)] = (frame_array, position_array)
            
            actors_dict[start_frame][1][(agent_dat['agent_type'].iloc()[0], a_id)] = (frame_array, position_array)
        
    return actors_dict


def thread_actor_handler(client, world, actors_dict, update_interval=0.05, frame_every=5, bound_box=((0,0),(1,1)), draw_every=1, draw_len=10):
    actor_list = list(actors_dict.keys())
    # walkers_ai_list = []
    
    # TODO: add blueprint options
    blueprintsWalkers = get_actor_blueprints(world, "walker.pedestrian.*", '2')
    walker_blacklist = ["walker.pedestrian.0011",
        "walker.pedestrian.0010",
        "walker.pedestrian.0009",
        "walker.pedestrian.0014",
        "walker.pedestrian.0013",
        "walker.pedestrian.0012",
        "walker.pedestrian.0048",
        "walker.pedestrian.0049",
        "walker.pedestrian.0030",
        "walker.pedestrian.0032"]
            
    blueprintsWalkers = [i for i in blueprintsWalkers if i.id not in walker_blacklist]
    if not blueprintsWalkers:
        raise ValueError("Couldn't find any walkers with the specified filters")

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
        
    # 1. we spawn the walker object
    batch = []
    for agent_id in actor_list:
        walker_bp = np.random.choice(blueprintsWalkers)
        spawn_point = carla.Transform()
        
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'true')
        
        spawn_point.location.x = actors_dict[agent_id][1][0, 0]
        spawn_point.location.y = actors_dict[agent_id][1][0, 1]
        spawn_point.location.z = z_default
        
        batch.append(SpawnActor(walker_bp, spawn_point))
        
    results = client.apply_batch_sync(batch, True)
    
    zip_ids = list(zip(actor_list, results))
    c_actor_list = [x[1].actor_id for x in zip_ids if x[1].actor_id != 0]
    zip_ids = list(zip(actor_list, c_actor_list))
    
    time_count = time.time()
        
    # 4. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    for idx, (agent_id, client_id) in enumerate(zip_ids):
        # set walk to random point
        control = carla.WalkerControl()
        
        # set transformation state
        first_pos = actors_dict[agent_id][1][0]
        second_pos = actors_dict[agent_id][1][1]
        transform = world.get_actor(client_id).get_transform()
        
        fv = carla.Vector3D(second_pos[0] - first_pos[0], second_pos[1] - first_pos[1], 0)
        angle = math.degrees(fv.get_vector_angle(carla.Vector3D(1, 0, 0)))
        
        transform.location.x = first_pos[0]
        transform.location.y = first_pos[1]
        transform.rotation.yaw = angle * math.copysign(1, fv.y)
        world.get_actor(client_id).set_transform(transform)
        
        # set destination
        target_point = carla.Vector3D()
        target_point.x = actors_dict[agent_id][1][1, 0]
        target_point.y = actors_dict[agent_id][1][1, 1]
        target_point.z = z_default
                
        cur_loc = world.get_actor(client_id).get_location()
        direction = target_point - cur_loc
        direction.z = 0.0
        
        target_dist = cur_loc.distance_2d(target_point)
       
        control.direction = direction
        control.speed = target_dist * 0.98
        
        if draw_len < 0:
            for i in range(1, actors_dict[agent_id][1].shape[0]+1, draw_every):
                frame_no = i
                if frame_no >= actors_dict[agent_id][1].shape[0]:
                    break
                
                random.seed(client_id)
                rgb = random.choice(walker_color_code)
                world.debug.draw_line(carla.Location(x=actors_dict[agent_id][1][frame_no-1, 0], 
                                                     y=actors_dict[agent_id][1][frame_no-1, 1], 
                                                     z=z_default+0.2),
                                      carla.Location(x=actors_dict[agent_id][1][frame_no, 0], 
                                                     y=actors_dict[agent_id][1][frame_no, 1], 
                                                     z=z_default+0.2), 
                                     color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                     life_time=(actors_dict[agent_id][0][-1]+5*frame_every)*update_interval,
                                     thickness=tk)
        else:
            for i in range(1, draw_len+1):
                frame_no = i * draw_every
                if frame_no >= actors_dict[agent_id][1].shape[0]:
                    break
                
                random.seed(client_id)
                rgb = random.choice(walker_color_code)
                world.debug.draw_line(carla.Location(x=actors_dict[agent_id][1][frame_no-1, 0], 
                                                     y=actors_dict[agent_id][1][frame_no-1, 1], 
                                                     z=z_default+0.2),
                                      carla.Location(x=actors_dict[agent_id][1][frame_no, 0], 
                                                     y=actors_dict[agent_id][1][frame_no, 1], 
                                                     z=z_default+0.2), 
                                     color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                     life_time=(actors_dict[agent_id][0][frame_no]+5*frame_every)*update_interval,
                                     thickness=tk)
            
        world.get_actor(client_id).apply_control(control)

    print('spawned %d walkers.' % (len(zip_ids)))
        
    try:
        # spawn actors
        # attach Ai walker
        
        frame_count = 0
        
        # update every given frames
        while(len(zip_ids) != 0):
            sleep_time = update_interval * frame_every - (time.time() - time_count)
            if sleep_time> 0:
                time.sleep(sleep_time)
            time_count = time.time()
            frame_count += frame_every

            # update actor destination
            
            del_idxs = []
            
            for idx, (agent_id, client_id) in enumerate(zip_ids):
                frame_idx = np.searchsorted(actors_dict[agent_id][0], frame_count)
                cur_loc = world.get_actor(client_id).get_location()
        
                if frame_count >= actors_dict[agent_id][0][-1]:
                    
                    dest_loc = carla.Vector3D()
                    dest_loc.x = actors_dict[agent_id][1][-1, 0]
                    dest_loc.y = actors_dict[agent_id][1][-1, 1]
                
                    if target_dist < 10000:
                        print('Delete walker. id: {}'.format(agent_id))
                        client.apply_batch([carla.command.DestroyActor(client_id)])
                        del_idxs.append(idx)
                else:
                    dest_loc = carla.Vector3D()
                    dest_loc.x = actors_dict[agent_id][1][frame_idx, 0]
                    dest_loc.y = actors_dict[agent_id][1][frame_idx, 1]
            
                target_dist = cur_loc.distance_2d(dest_loc)
                
                control = carla.WalkerControl()
                direction = dest_loc - cur_loc
                direction.z = 0.0
                control.direction = direction
                control.speed = target_dist * 0.98
                    
                world.get_actor(client_id).apply_control(control)
            
                if not draw_len < 0:
                    if frame_count % (draw_every * frame_every) == 0:
                        frame_no = frame_idx + draw_len
                        if frame_no < actors_dict[agent_id][1].shape[0]:
                            random.seed(client_id)
                            rgb = random.choice(walker_color_code)
                            
                            world.debug.draw_line(carla.Location(x=actors_dict[agent_id][1][frame_no-1, 0], 
                                                                y=actors_dict[agent_id][1][frame_no-1, 1], 
                                                                z=z_default+0.2),
                                                carla.Location(x=actors_dict[agent_id][1][frame_no, 0], 
                                                                y=actors_dict[agent_id][1][frame_no, 1], 
                                                                z=z_default+0.2), 
                                                color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                                life_time=(actors_dict[agent_id][0][frame_no]-frame_count+5* frame_every)*update_interval,
                                                thickness=tk)
                                            
            for x in sorted(del_idxs, reverse=True):
                del zip_ids[x]
    finally:
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for (agent_id, client_id) in zip_ids:
            print('delete walker. id: {}'.format(agent_id))
            client.apply_batch([carla.command.DestroyActor(client_id)])
            del_idxs.append(idx)
   

def carculate_steer(cur_transform, dest_vec, vehicle_len, time_diff):
    forward_vec = cur_transform.get_forward_vector()
    theta = forward_vec.make_unit_vector().get_vector_angle(dest_vec.make_unit_vector())
    
    if theta < 1e-4:
        # go straight
        return 0.0, dest_vec.length()
    
    r = dest_vec.length() / 2 / abs(math.cos(math.pi/2 - theta))
    r = max(r, vehicle_len*2)
    
    c = forward_vec.cross(dest_vec)
    # left turn
    if c.z > 0:
        return math.asin(vehicle_len / 2 / r)*2, r*theta
    else:
        return -math.asin(vehicle_len / 2 / r)*2, r*theta
        

def thread_vehicle_handler(client, world, vehicles_dict, update_interval=0.05, frame_every=5, bound_box=((0,0),(1,1)), draw_every=1, draw_len=10):
    vehicles_list = list(vehicles_dict.keys())
    
    #cars_model = ["microlino", "micra"]
    cars_model = ["microlino", "micra", "A2", "Gran Tourer", "C3", "Wrangler Rubicon", "Cooper S", "Leon", "Prius", "CarlaCola", "Sprinter", ]
    bicycles_model = ["crossbike", "century", "omafiets"]
    
    car_blueprints = []
    bicycle_blueprints = []
    
    # TODO: add blueprint options
    for vehicle in world.get_blueprint_library().filter('*vehicle*'):
        if any(model in vehicle.tags for model in cars_model):
            car_blueprints.append(vehicle)
        elif any(model in vehicle.tags for model in bicycles_model):
            bicycle_blueprints.append(vehicle)

    if not car_blueprints or not bicycle_blueprints:
        raise ValueError("Couldn't find any vehicles with the specified filters")

    def set_desired_speed_torque(vehicle_id, dest_loc, next_dest_loc, time_diff, max_speed=10):
        target_point = carla.Vector3D()
        target_point.x = dest_loc[0]
        target_point.y = dest_loc[1]
        target_point.z = z_default
        
        next_target_point = carla.Vector3D()
        next_target_point.x = next_dest_loc[0]
        next_target_point.y = next_dest_loc[1]
        next_target_point.z = z_default

        cur_loc = world.get_actor(vehicle_id).get_location()
        # cur_velocity = world.get_actor(vehicle_id).get_velocity()
        
        desire_state = next_target_point - target_point
        desire_state.z = 0.0
        
        target_displacement = target_point - cur_loc
        target_displacement.z = 0.0
        
        if target_displacement.length() < 1e-4:
            world.get_actor(vehicle_id).apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            return
            
        physics_control = world.get_actor(vehicle_id).get_physics_control()
        vehicle_len = physics_control.wheels[1].position.distance(physics_control.wheels[0].position)* 0.0254
        max_steer = math.radians(physics_control.wheels[1].max_steer_angle)
        steer, path_len = carculate_steer(world.get_actor(vehicle_id).get_transform(), target_displacement, vehicle_len, time_diff)
        
        steer = max(min(steer / max_steer, 1), -1)
        
        target_velocity = path_len / time_diff
        
        world.get_actor(vehicle_id).apply_control(carla.VehicleControl(throttle=max(min(target_velocity/max_speed, 1), 0), steer=steer))

    def teleport_here(vehicle_id, dest_loc, next_dest_loc=None, next_next_dest_loc=None, time_interval=0.05):
        transform = world.get_actor(vehicle_id).get_transform()
        
        fv = carla.Vector3D(next_dest_loc[0] - dest_loc[0], next_dest_loc[1] - dest_loc[1], 0)
        if next_dest_loc is None:
            angle = 0
        else:
            angle = math.degrees(fv.get_vector_angle(carla.Vector3D(1, 0, 0)))
        
        transform.location.x = dest_loc[0]
        transform.location.y = dest_loc[1]
        transform.rotation.yaw = angle * math.copysign(1, fv.y)
        world.get_actor(vehicle_id).set_transform(transform)
        
        world.get_actor(vehicle_id).set_target_velocity(fv/time_interval)
    
    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    
    # 1. we spawn the vehicle object
    batch = []
    for agent_key in vehicles_list:
        (agent_type, agent_id) = agent_key
        if agent_type == 1:
            # bicycle
            vehicle_bp = np.random.choice(bicycle_blueprints)
        elif agent_type == 2:
            # car
            vehicle_bp = np.random.choice(car_blueprints)
        else:
            raise ValueError("Invalid agent type")
            
        spawn_point = carla.Transform()
        
        if vehicle_bp.has_attribute('is_invincible'):
            vehicle_bp.set_attribute('is_invincible', 'true')
        
        spawn_point.location.x = vehicles_dict[agent_key][1][0, 0]
        spawn_point.location.y = vehicles_dict[agent_key][1][0, 1]
        spawn_point.location.z = z_default
        
        target_displacement = carla.Vector3D()
        target_displacement.x = vehicles_dict[agent_key][1][1, 0] - vehicles_dict[agent_key][1][0, 0]
        target_displacement.y = vehicles_dict[agent_key][1][1, 1] - vehicles_dict[agent_key][1][0, 1]
        target_displacement.z = 0.0
        
        if target_displacement.length() > 1e-4:
            spawn_point.rotation.yaw = spawn_point.get_forward_vector().make_unit_vector().get_vector_angle(target_displacement.make_unit_vector())
        
        batch.append(SpawnActor(vehicle_bp, spawn_point))
        
    results = client.apply_batch_sync(batch, True)
    
    zip_ids = list(zip(vehicles_list, results))
    c_actor_list = [x[1].actor_id for x in zip_ids if x[1].actor_id != 0 and x[1].error=='']
    zip_ids = list(zip(vehicles_list, c_actor_list))
    
    time_count = time.time()
        
    # 4. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    for idx, (agent_key, client_id) in enumerate(zip_ids):
        # set transformation state        
        world.get_actor(client_id).set_collisions(False)
        
        first_pos = vehicles_dict[agent_key][1][0]
        second_pos = vehicles_dict[agent_key][1][1]
        third_pos = vehicles_dict[agent_key][1][1]
        transform = world.get_actor(client_id).get_transform()
        
        fv = carla.Vector3D(second_pos[0] - first_pos[0], second_pos[1] - first_pos[1], 0)
        angle = math.degrees(fv.get_vector_angle(carla.Vector3D(1, 0, 0)))
        
        transform.location.x = first_pos[0]
        transform.location.y = first_pos[1]
        transform.rotation.yaw = angle * math.copysign(1, fv.y)
        world.get_actor(client_id).set_transform(transform)
    
        if draw_len < 0:
            for i in range(1, vehicles_dict[agent_key][0].shape[0], draw_every):
                frame_no = i
                if frame_no >= vehicles_dict[agent_key][0].shape[0]:
                    break
                
                random.seed(client_id)
                rgb = random.choice(vehicle_color_code)
                
                world.debug.draw_line(carla.Location(x=vehicles_dict[agent_key][1][frame_no-1, 0], 
                                                     y=vehicles_dict[agent_key][1][frame_no-1, 1], 
                                                     z=z_default+0.2),
                                      carla.Location(x=vehicles_dict[agent_key][1][frame_no, 0], 
                                                     y=vehicles_dict[agent_key][1][frame_no, 1], 
                                                     z=z_default+0.2), 
                                       color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                     life_time=vehicles_dict[agent_key][0][-1]*update_interval,
                                     thickness=tk)

        else:
            for i in range(1, draw_len+1):
                frame_no = i * draw_every
                
                if frame_no >= vehicles_dict[agent_key][0].shape[0]:
                    break
                random.seed(client_id)
                rgb = random.choice(vehicle_color_code)
                
                world.debug.draw_line(carla.Location(x=vehicles_dict[agent_key][1][frame_no-1, 0], 
                                                     y=vehicles_dict[agent_key][1][frame_no-1, 1], 
                                                     z=z_default+0.2),
                                      carla.Location(x=vehicles_dict[agent_key][1][frame_no, 0], 
                                                     y=vehicles_dict[agent_key][1][frame_no, 1], 
                                                     z=z_default+0.2), 
                                       color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                     life_time=vehicles_dict[agent_key][0][frame_no]*update_interval,
                                     thickness=tk)
            
    print('spawned %d vehicle.' % (len(zip_ids)))
        
    try:
        frame_count = 0
        # update every given frames
        while(len(zip_ids) != 0):
            sleep_time = update_interval * frame_every - (time.time() - time_count)
            if sleep_time> 0:
                time.sleep(sleep_time)
            time_count = time.time()
            frame_count += frame_every

            # update actor destination
            del_idxs = []
            
            for idx, (agent_key, client_id) in enumerate(zip_ids):
                frame_idx = np.searchsorted(vehicles_dict[agent_key][0], frame_count)
                if frame_count >= vehicles_dict[agent_key][0][-1]:
                    print('Delete vehicle. id: {}, cid: {}'.format(agent_key[0], agent_key[1]))
                    client.apply_batch([carla.command.DestroyActor(client_id)])
                    frame_diff = frame_every * update_interval
                    del_idxs.append(idx)
                    continue
                elif frame_count >= vehicles_dict[agent_key][0][-3]:
                    dest_loc = vehicles_dict[agent_key][1][-1, :]
                    next_dest_loc = 2*dest_loc - vehicles_dict[agent_key][1][-2, :] # extrapolate next location
                    frame_diff = frame_every * update_interval
                    next_next_dest_loc=None
                else:
                    dest_loc = vehicles_dict[agent_key][1][frame_idx, :]
                    next_dest_loc = vehicles_dict[agent_key][1][frame_idx+1, :]
                    frame_diff = (vehicles_dict[agent_key][0][frame_idx+1] - frame_count) * update_interval
                    next_next_dest_loc = vehicles_dict[agent_key][1][frame_idx+2, :]
            
                # set_desired_speed_torque(client_id, dest_loc, next_dest_loc, frame_diff, max_speed=(10 if agent_key == 1 else 20))
                teleport_here(client_id, dest_loc, next_dest_loc, next_next_dest_loc, time_interval=frame_diff)
                
                if not  draw_len < 0:    
                    if frame_count % (draw_every * frame_every) == 0:
                        frame_no = frame_idx + draw_len
                        if frame_no < vehicles_dict[agent_key][1].shape[0]:
                            random.seed(client_id)
                            rgb = random.choice(vehicle_color_code)
                    
                            world.debug.draw_line(carla.Location(x=vehicles_dict[agent_key][1][frame_no-1, 0], 
                                                                y=vehicles_dict[agent_key][1][frame_no-1, 1], 
                                                                z=z_default+0.2),
                                                carla.Location(x=vehicles_dict[agent_key][1][frame_no, 0], 
                                                                y=vehicles_dict[agent_key][1][frame_no, 1], 
                                                                z=z_default+0.2), 
                                                color=carla.Color(rgb[0],rgb[1],rgb[2]), 
                                                life_time=(vehicles_dict[agent_key][0][frame_no]-frame_count)*update_interval,
                                                thickness=tk)
            
            for x in sorted(del_idxs, reverse=True):
                del zip_ids[x]

    finally:
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for (agent_key, client_id) in zip_ids:
            print('delete vehicle. id: {}, cid: {}'.format(agent_key[0], agent_key[1]))
            client.apply_batch([carla.command.DestroyActor(client_id)])
            del_idxs.append(idx)
            

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--path',
        help='path to data')
    argparser.add_argument(
        '--scene',
        choices=['hyang', 'zara'],
        default='hyang',
        help='name of the scene')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--draw_len',
        default=-1,
        type=int,
        help='length of future trajctory')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    np.random.seed(args.seedw if args.seedw is not None else int(time.time()))
    
    args.asynch = True
    update_interval = 1/30
    frame_every = 3
    
    if args.scene == 'hyang':
        # hyang
        # origin, multi, swap_xy, flip_x, flip_y =  np.array([[119.2, 224.7]]), 1 / 29.17, False, False, False
        origin, multi, swap_xy, flip_x, flip_y =  np.array([[120.2, 223.3]]), 1 / 29.17, False, False, False  #shifted
        
        client.load_world('CrowdES_hyang')
        z_default=1.5
    elif args.scene == 'zara':
        origin, multi, swap_xy, flip_x, flip_y = np.array([[315 - 15.6535488*0.5*1.3, 215 - 12.52283904*0.5*1.3 + 0.5]]), 0.02174104*1.3, False, False, False     # zara              15.6535488  12.52283904
        
        client.load_world('CrowdES_zara')
        z_default=2.5
        
        # set camera
        world = client.get_world()
        transform = world.get_spectator().get_transform()
        transform.location = carla.Location(x=315.328552, y=220.958313, z=7.827725)
        transform.rotation = carla.Rotation(pitch=-51.071339, yaw=-90.952713, roll=0.000014)
        world.get_spectator().set_transform(transform)
    else:
        raise NotImplementedError
    path = args.path
    
    # path = "test.csv"
    bound_box = (0,0)
    draw_every = 1
    draw_len = args.draw_len
    
    csv_data = load_csv(path, origin=origin, frame_every=frame_every, multi=multi, 
                                swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)
    
    def get_actor_from_data(csv_data):
        frame_dat = sorted(list(csv_data.keys()))
        
        for frame in frame_dat:
            yield frame, csv_data[frame]

    try:
        world = client.get_world()

        settings = world.get_settings()
        if not args.asynch:
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = update_interval
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        settings.no_rendering_mode = False
        world.apply_settings(settings)
        
        host_time_count = time.time()
        prev_frame = 0
        
        for frame, agents in get_actor_from_data(csv_data):
            sleep_time = (frame - prev_frame) * update_interval - (time.time() - host_time_count)
            if sleep_time> 0:
                time.sleep(sleep_time)
            prev_frame = frame
            host_time_count = time.time()
                
            actors, vehicles = agents
            if bool(actors):
                thread = threading.Thread(target = thread_actor_handler, args = (client, world, actors, update_interval, frame_every, bound_box, draw_every, draw_len))
                thread.daemon = False
                thread.start()
                
            # if True:
            if bool(vehicles):
                thread = threading.Thread(target = thread_vehicle_handler, args = (client, world, vehicles, update_interval, frame_every, bound_box, draw_every, draw_len))
                thread.daemon = False    
                thread.start()
            
    finally:
        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
