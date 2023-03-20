from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from agents.navigation.global_route_planner import GlobalRoutePlanner


import carla

from carla import ColorConverter as cc
from carla import Transform 
from carla import Location
from carla import Rotation

from PIL import Image

import keras
import tensorflow as tf

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import numpy as np
import cv2
from collections import deque
from keras.applications.xception import Xception 
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

from keras.models import Sequential, Model, load_model
from keras.layers import AveragePooling2D, Conv2D, Activation, Flatten, GlobalAveragePooling2D, Dense, Concatenate, Input


#from tensorboard import *

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tensorflow.keras import regularizers

from tqdm import tqdm


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 20
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 2
MODEL_NAME = "Driving"

MEMORY_FRACTION = 0.8
MIN_REWARD = 0

EPISODES = 40
DISCOUNT = 0.99
epsilon = 0.5
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01

AGGREGATE_STATS_EVERY = 1


# path for training
town2 = {1: [193.75,269.2610168457031, 5, 270], 2:[135.25,206]} #left turn

# #Trajectory 1
# town2 = {1: [80, 306.6, 5, 0], 2:[135.25,206]} 

# #Trajectory 2
# town2 = {1: [-7.498, 284.716, 5, 90], 2:[81.98,241.954]}

# #Trajectory 3
# town2 = {1: [-7.498, 165.809, 5, 90], 2:[81.98,241.954]}

# #Trajectory 4
# town2 = {1: [106.411, 191.63, 5, 0], 2:[170.551,240.054]}


curves = [0, town2]

#MODEL_PATH = "models/Driving__3120.00max_3120.00avg_3120.00min__1679091203.model"



'''
Custom Tensorboard class. Updates logs after every episode
'''
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, model, log_dir):
        super().__init__(model, log_dir)
        self.log_dir = log_dir
        self.model = model
        self.step = 1
        print("SELF: LOG DIR:      ", self.log_dir)
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with tf.compat.v1.Session() as sess:
            for name, value in logs.items():
                summary = tf.compat.v1.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.writer.add_summary(summary, index)
                if self.model is not None:
                    self.writer.add_graph(sess.graph, global_step=index)
            self.writer.flush()
                


'''
Defining the Carla Environment Class. 
'''
class CarEnv:

    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0   # actions that the agent can take [-1, 0, 1] --> [turn left, go straight, turn right]
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None


    def __init__(self):
        # to initialize
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.find("vehicle.tesla.model3")
        self.via = 2
        self.crossing = 0
        self.curves = 1
        self.reached = 0
        self.start = town2[1]
        self.phi = []
        self.dc = []
        self.vel = []
        self.time = []

        
    def reset(self):

        # store any collision detected
        self.collision_history = []
        # to store all the actors that are present in the environment
        self.actor_list = []
        # store the number of times the vehicles crosses the lane marking
        self.lanecrossing_history = []
        
        
        '''
        To spawn the Vehicle (agent)
        '''
        initial_pos = self.start
        self.transform = Transform(Location(x=initial_pos[0], y=initial_pos[1], z=initial_pos[2]), Rotation(yaw=initial_pos[3]))
        # to spawn the actor; the veichle
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)


        # '''
        # To spawn the SEGMENTATION camera
        # '''
        # self.rgb_camera = self.blueprint_library.find("sensor.camera.rgb")
        # self.rgb_camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        # self.rgb_camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        # self.rgb_camera.set_attribute("fov", f"110")

        # self.camera_spawn_point = carla.Transform(carla.Location(x=2, y=0, z=1.4))

        # # to spawn the camera
        # self.camera_sensor = self.world.spawn_actor(self.rgb_camera, self.camera_spawn_point, attach_to = self.vehicle)
        # self.actor_list.append(self.camera_sensor)

        # # to record the data from the camera sensor
        # self.camera_sensor.listen(lambda data: self.process_image(data))


        # to initialize the car quickly and get it going
        self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.0))
        time.sleep(4)

        '''
        To spawn the collision sensor
        '''
        # to introduce the collision sensor to detect what type of collision is happening
        col_sensor = self.blueprint_library.find("sensor.other.collision")
        
        # keeping the location of the sensor to be same as that of the RGB camera
        self.collision_sensor = self.world.spawn_actor(col_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # to record the data from the collision sensor
        self.collision_sensor.listen(lambda event: self.collision_data(event))


        # to introduce the lanecrossing sensor to identify vehicles trajectory
        lane_crossing_sensor = self.blueprint_library.find("sensor.other.lane_invasion")

        # keeping the location of the sensor to be same as that of RGM Camera
        self.lanecrossing_sensor = self.world.spawn_actor(lane_crossing_sensor, self.camera_spawn_point, attach_to = self.vehicle)
        self.actor_list.append(self.lanecrossing_sensor)

        # to record the data from the lanecrossing_sensor
        self.lanecrossing_sensor.listen(lambda event: self.lanecrossing_data(event))
        
        traj = self.trajectory()
        self.path = []
        for el in traj:
            self.path.append(el[0])

        # while self.front_camera is None:
        #     time.sleep(0.01)


        # going to keep an episode length of 10 seconds otherwise the car learns to go around a circle and keeps doing the same thing
        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(throttle = 1.0, brake = 0.0))
        
        return [0,0] #return [self.front_camera, 0,0, initial_pos[0], initial_pos[1]]



    def collision_data(self, event):
        self.collision_history.append(event)
    
    def lanecrossing_data(self, event):
        self.lanecrossing_history.append(event)
        print("Lane crossing history: ", event)



    # def process_image(self, image):
    #     #image.save_to_disk('tutorial/new_sem_output/%.6d.jpg' % image.frame,carla.ColorConverter.CityScapesPalette)
    #     # Define color codes for each class
    #     colors = {
    #         0: [0, 0, 0],         # None
    #         1: [70, 70, 70],      # Buildings
    #         2: [190, 153, 153],   # Fences
    #         3: [72, 0, 90],       # Other
    #         4: [220, 20, 60],     # Pedestrians
    #         5: [153, 153, 153],   # Poles
    #         6: [157, 234, 50],    # RoadLines
    #         7: [128, 64, 128],    # Roads
    #         8: [244, 35, 232],    # Sidewalks
    #         9: [107, 142, 35],    # Vegetation
    #         10: [0, 0, 255],      # Vehicles
    #         11: [102, 102, 156],  # Walls
    #         12: [220, 220, 0],    # TrafficSigns
    #     }

    #     # Convert segmentation image to numpy array
    #     i = np.array(image.raw_data)
    #     #self.front_camera = image_np

    #     # convert the flattened array into an image and we have 4 channels in the order "BGRA"
    #     i2 = i.reshape((self.im_height, self.im_width, 4))
    #     i3 = i2[:, :, :3]

    #     if self.SHOW_CAM:
    #         cv2.imshow("Car Camera", self.front_camera)
    #         cv2.waitKey(1)

    #     self.front_camera = i3
    #     #return self.front_camera/255.0    # to normalize the values of image between 0 to 255


    def step(self, action, current_state):
        '''
        Take 5 actions; go straight, turn left, turn right, turn slightly left, turn slightly right
        '''

        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0*self.STEER_AMT))

        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=-0.6*self.STEER_AMT))

        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.6*self.STEER_AMT))

        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=-0.1*self.STEER_AMT))

        if action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.1*self.STEER_AMT))

        
        # initialize a reward for a single action 
        reward = 0
        # to calculate the kmh of the vehicle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # to get the position and orientation of the car
        pos = self.vehicle.get_transform().location
        rot = self.vehicle.get_transform().rotation
        
        # to get the closest waypoint to the car
        waypoint = self.client.get_world().get_map().get_waypoint(pos, project_to_road=True)
        #path = self.trajectory()
        #waypoint = path[0][0]
        waypoint_ind = self.get_closest_waypoint(self.path, waypoint) + 1
        print(waypoint_ind)
        waypoint = self.path[waypoint_ind]
        if len(self.path) != 1:
            next_waypoint = self.path[waypoint_ind+1]
        else:
            next_waypoint = waypoint
        waypoint_loc = waypoint.transform.location
        waypoint_rot = waypoint.transform.rotation
        next_waypoint_loc = next_waypoint.transform.location
        next_waypoint_rot = next_waypoint.transform.rotation
        
        final = [curves[self.curves][2][0], curves[self.curves][2][1]]
        final_destination = [curves[self.curves][self.via][0], curves[self.curves][self.via][1]]
        dist_from_goal = np.sqrt((pos.x - final_destination[0])**2 + (pos.y-final_destination[1])**2)

        done = False


        '''
        TO DEFINE THE REWARDS
        '''
        # to get the orientation difference between the car and the road "phi"
        orientation_diff = waypoint_rot.yaw - rot.yaw
        phi = orientation_diff%360 -360*(orientation_diff%360>180)
        
        #current_state[0] = current_state[0]*25
        current_state[1] = current_state[1]/15
        
        u = [waypoint_loc.x-next_waypoint_loc.x, waypoint_loc.y-next_waypoint_loc.y]
        v = [pos.x-next_waypoint_loc.x, pos.y-next_waypoint_loc.y]
        if np.linalg.norm(u) > 0.1 and np.linalg.norm(v) > 0.1:
            signed_dis = np.linalg.norm(v)*np.sin(np.sign(np.cross(u,v))*np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))
        else:
            signed_dis = 0
        

        print(current_state[0]) 
        print(current_state[1]) 
        

        # Defining the Reward function by comparing the action taken to a suboptimal policy
        if abs(current_state[0])<5:
            if action == 0:
                reward += 2
            else:
                reward -= 1
        elif abs(current_state[0])<10:
            if current_state[0]<0:
                if action == 3:
                    reward += 2
                elif action == 1:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 4:
                    reward += 2
                elif action == 2:
                    reward += 1
                else:
                    reward -= 1
        else:
            if current_state[0]<0:
                if action == 1:
                    reward += 2
                elif action == 3:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 2:
                    reward += 2
                elif action == 4:
                    reward += 1
                else:
                    reward -= 1
                    
        if abs(current_state[1])<0.1:
            if action == 0:
                reward += 4
            else:
                reward -= 2
        elif abs(current_state[1])<0.5:
            if current_state[1]<0:
                if action == 3:
                    reward += 2
                elif action == 1:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 4:
                    reward += 2
                elif action == 2:
                    reward += 1
                else:
                    reward -= 1
        else:
            if current_state[1]<0:
                if action == 1:
                    reward += 2
                elif action == 3:
                    reward += 1
                else:
                    reward -= 1
            else:
                if action == 2:
                    reward += 2
                elif action == 4:
                    reward += 1
                else:
                    reward -= 1
                    
        
        if abs(signed_dis)>2:
            reward -= 10
        
        # to avoid collisions
        if len(self.collision_history) != 0:
            done = True
            reward = - 200

        # to end the episode if phi value goes high
        if abs(phi)>100:
            done = True
            reward = -200
            
        # Ending the episode if the distance to the centerline of the road is greater than 3
        if abs(signed_dis)>3:
            done = True
            reward = -200
            
        # to end the episode if the car reaches close to the final destination
        if dist_from_goal < 5:
            self.reached = 1
            done = True

        # to run each episode for just 30 secodns
        if self.episode_start + 200 < time.time():
            done = True

        print(reward)
        
        self.phi.append(phi)
        self.dc.append(signed_dis)
        self.vel.append(kmh)
        self.time.append(time.time())

        return [phi, signed_dis*15], reward, done, waypoint


    def trajectory(self, draw = False):
        amap = self.world.get_map()
        sampling_resolution = 0.5
        # dao = GlobalRoutePlannerDAO(amap, sampling_resolution)
        grp = GlobalRoutePlanner(amap, sampling_resolution)
        # grp.setup()
        
        #start_location = self.vehicle.get_transform().location
        start_location = carla.Location(x=self.start[0], y=self.start[1], z=0)
        end_location = carla.Location(x=town2[2][0], y=town2[2][1], z=0)
        a = amap.get_waypoint(start_location, project_to_road=True)
        b = amap.get_waypoint(end_location, project_to_road=True)
        spawn_points = self.world.get_map().get_spawn_points()
        #print(spawn_points)
        a = a.transform.location
        b = b.transform.location
        w1 = grp.trace_route(a, b) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
        i = 0
        if draw:
            for w in w1:
                if i % 10 == 0:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                    persistent_lines=True)
                else:
                    self.world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
                    color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
                    persistent_lines=True)
                i += 1
        return w1
    

    def get_closest_waypoint(self, waypoint_list, target_waypoint):
        closest_waypoint = None
        closest_distance = float('inf')
        for i, waypoint in enumerate(waypoint_list):
            distance = math.sqrt((waypoint.transform.location.x - target_waypoint.transform.location.x)**2 +
                                 (waypoint.transform.location.y - target_waypoint.transform.location.y)**2)
            if distance < closest_distance:
                closest_waypoint = i
                closest_distance = distance
        return closest_waypoint



'''
TO define the Deep Q Network agent class
'''
class DQNAgent:
    def __init__(self):
        #self.model = load_model(MODEL_PATH)
        #self.target_model = load_model(MODEL_PATH)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.graph = tf.get_default_graph()


        self.tensorboard = ModifiedTensorBoard(self.model, log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model3 = Sequential()
        model3.add(Dense(8, input_shape=(2,), activation='relu', name='dense1'))
        model3.add(Dense(5, activation='linear', name='output'))
        combined_model = Model(inputs=model3.input, outputs=model3.output)
        
        # compile the model
        combined_model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        
        return combined_model

    
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)


    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # to sample a minibatch
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # to normalize the image
        current_data = np.array([[transition[0][i] for i in range(2)] for transition in minibatch])
        # predicting all the datapoints present in the mini-batch
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_data, PREDICTION_BATCH_SIZE)

        new_current_data = np.array([[transition[3][i] for i in range(2)] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_data, PREDICTION_BATCH_SIZE)

        X_img = []
        X_data = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X_data.append([current_state[i] for i in range(2)])
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # to contnuously train the base model 
        with self.graph.as_default():
            self.model.fit(np.array(X_data), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        # to assign the weights of the base model to the target model 
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape))[0]

    def train_in_loop(self):
        X2 = np.random.uniform(size=(1, 2)).astype(np.float32)
        y = np.random.uniform(size=(1, 5)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X2,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

           
     
if __name__ == '__main__':

    FPS = 400
    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    # Memory fraction, used mostly when trai8ning multiple agents
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    path = r"/home/tejas/Documents/Stanford/CS 238/Final Project/Stanford-CS-238/Stanford-CS-238/Stanford-CS-238/models"
    
    # Create models folder
    if not os.path.isdir(path):
        os.makedirs(path)

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()


    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    agent.get_qs([0,0])

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        #try:
        #for direction in range(2):
            #print("direction: ", direction)
            if episode%2 == 0:
                town2 = {1: [182.65191650390625,236.9, 5, 180], 2:[135.25,206]}
                env.start = town2[1]
            else:
                town2 = {1: [193.75,269.2610168457031, 5, 270], 2:[135.25,206]}
                env.start = town2[1]
            env.reached = 0
            env.collision_hist = []
            # Update tensorboard step every episode
            agent.tensorboard.step = episode
    
            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
    
            # Reset environment and get initial state
            current_state = env.reset()
            
            
            # Reset flag and start iterating until episode ends
            done = False
            episode_start = time.time()

    
            # Play for given number of seconds only
            while True:
    
                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    qs = agent.get_qs(current_state)
                    print(qs)
                    action = np.argmax(qs)
                    #time.sleep(1/FPS)
                else:
                    # Get random action
                    action = np.random.randint(0, 5)
                        
                    # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                    time.sleep(1/FPS)
                new_state, reward, done, waypoint = env.step(action, current_state)
                
  
                # Transform new continous state to new discrete state and count reward
                episode_reward += reward
    
                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))

                current_state = new_state
                step += 1
   
                if done:
                    break
        
            print("EPISODE {} REWARD IS: {}".format(episode, episode_reward))
            
            # End of episode - destroy agents
            for actor in env.actor_list:
                actor.destroy()
            
    
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    
                # Save model, but only when min reward is greater or equal a set value
                if min_reward >= MIN_REWARD:
                    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    
            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        


    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')