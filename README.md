# Safe Navigation: Training Autonomous Vehicles using Deep Reinforcement Learning in CARLA

## Introduction
The development of safe and reliable autonomous driving technology remains a critical research area, with numerous studies exploring various approaches to enhancing the safety of self-driving vehicles. When it comes to beyond level 4 in autonomous driving technology, the vehicle should be able to deal with various kinds of situations around the self-driving vehicle in order to arrive at the targeted destination successfully. Different sensor and algorithm-based approaches have been tried and developed to navigate safely to the target destination in the CARLA simulator. Conventional control systems are based on mathematical models, but they only control the vehicle in a limited range of situations. Therefore, Machine Learning (ML) algorithms have been applied in autonomous systems to better control vehicles in varied situations.

## Problem Statement
The deployment of autonomous vehicles on road to safely navigate in uncertain environments to the final destination require fast computation, low latency, and high accuracy. Therefore, there is a need to develop efficient techniques to process high-dimensional input data from sensors and images to reduce the state space and improve the training and testing efficiency. This paper proposes a Deep RL method using DQN, which includes sensor data pre-processing steps to reduce the state space and demonstrates its effectiveness in successfully navigating through 4 traffic scenarios with high levels of safety and accuracy.

<img src="/images/trajs.png" alt="Trajectories" width=50% style="display: block; margin: 0 auto;">

## Installtion
1. Clone this repo: `git clone https://github.com/Tejas-Deo/Safe-Navigation-Training-Autonomous-Vehicles-using-Deep-Reinforcement-Learning-in-CARLA.git`
2. Download Carla 0.9.13 from [source](https://github.com/carla-simulator/carla/releases). 
3. Install required packages: `pip install -r requirements.txt`

## Run
1. Run Carla Server using: `./CarlaUE4.sh`
2. Run `config.py` file to load Town02
3. Generate traffic either using `generate_traffic.py` or spawn pedestrians at random location along Trajectory 1 and 2 using the `pedestrians_1.py` and `pedestrians_2.py` script. Change the number of vehicles and pedestrians to spawn using `generate_traffic.py` by passing the corresponding arguments.
4. Select an existing trajectory (Trajectory 1, Trajectory 2, Trajectory 3, Trajectory 4) or set custom trajectory using the format given in `test_everything.py` arguments.
5. To find the initial and final locations of the custom trajectory, make use of `get_location.py` file and navigate the map using W,A,S,D, E, and Q keys. 
6. Enter locations or select existing trajectories and run the `test_everything.py` file.

## Methodology
Model Architecture Encapsulating the workflow:

<center><img src="/images/model.jpg" alt="Model Architecture"></center>
