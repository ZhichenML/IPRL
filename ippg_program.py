from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import logging
import sys
import datetime
import time
import functools
import copy
from bayes_opt import BayesianOptimization
from scipy import spatial
from neural_update import NeuralAgent
from controllers import Controller
from utils import *

class ParameterFinder():
    def __init__(self, inputs, actions, steer_prog, accel_prog, brake_prog):
        self.inputs = inputs
        self.actions = actions
        self.steer = steer_prog
        self.accel = accel_prog
        self.brake = brake_prog

    def find_distance_paras(self, sp0, sp1, sp2, spt, ap0, ap1, ap2, apt, api, apc, bp0, bp1, bp2, bpt):
        self.steer.update_parameters([sp0, sp1, sp2], spt)
        self.accel.update_parameters([ap0, ap1, ap2], apt, api, apc)
        self.brake.update_parameters([bp0, bp1, bp2], bpt)
        steer_acts = []
        accel_acts = []
        brake_acts = []
        for window_list in self.inputs:
            steer_acts.append(clip_to_range(self.steer.pid_execute(window_list), -1, 1))
            accel_acts.append(clip_to_range(self.accel.pid_execute(window_list), 0, 1))
            brake_acts.append(clip_to_range(self.brake.pid_execute(window_list), 0, 1))
        steer_diff = spatial.distance.euclidean(steer_acts, np.array(self.actions)[:, 0])
        accel_diff = spatial.distance.euclidean(accel_acts, np.array(self.actions)[:, 1])
        brake_diff = spatial.distance.euclidean(brake_acts, np.array(self.actions)[:, 2])
        diff_total = -(steer_diff + accel_diff + brake_diff)/float(len(self.actions))
        return diff_total

    def pid_parameters(self, info_list):

        gp_params = {"alpha": 1e-4, "n_restarts_optimizer": 2}  # Optimizer configuration
        logging.info('Optimizing Controller')
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                        {'sp0': info_list[0][0], 'sp1': info_list[0][1], 'sp2': info_list[0][2], 'spt': info_list[0][3],
                                         'ap0': info_list[1][0], 'ap1': info_list[1][1], 'ap2': info_list[1][2], 'apt': info_list[1][3], 'api': info_list[1][4], 'apc': info_list[1][5],
                                         'bp0': info_list[2][0], 'bp1': info_list[2][1], 'bp2': info_list[2][2], 'bpt': info_list[2][3]}, verbose=0)

        bo_pid.maximize(init_points=50, n_iter=10, kappa=5, **gp_params)
        logging.info(bo_pid.max['params'])

        return bo_pid.max['params']


def programmatic_game(steer, accel, brake, track_name='practgt2.xml'):
    episode_count = 5
    max_steps = 10000
    window = 5

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False, track_name=track_name)

    logging.info("TORCS Experiment Start with Priors on " + track_name)
    for i_episode in range(episode_count):
        ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                   list(ob.wheelSpinVel / 100.0), list(ob.track), [0, 0, 0]]
        window_list = [tempObs[:] for _ in range(window)]

        for j in range(max_steps):
            steer_action = clip_to_range(steer.pid_execute(window_list), -1, 1)
            accel_action = clip_to_range(accel.pid_execute(window_list), 0, 1)
            brake_action = clip_to_range(brake.pid_execute(window_list), 0, 1)
            action_prior = [steer_action, accel_action, brake_action]

            tempObs = [[ob.speedX], [ob.angle], [ob.trackPos], [ob.speedY], [ob.speedZ], [ob.rpm],
                       list(ob.wheelSpinVel / 100.0), list(ob.track), action_prior]
            window_list.pop(0)
            window_list.append(tempObs[:])

            ob, r_t, done, info = env.step(action_prior)
            #if np.mod(j, 1000) == 0:

            if done:
                print('Done. Steps: ', j)
                break
                
        logging.info("Episode: " + str(i_episode) + " step: " + str(j+1) + " Distance: " + str(ob.distRaced) + ' ' + str(ob.distFromStart) + " Lap Times: " + str(ob.lastLapTime))


        env.end()  # This is for shutting down TORCS
        logging.info("Finish.")



def learn_policy(track_name, test_program):

    # Define Pi_0
    # def __init__(self, pid_constants=(0, 0, 0), pid_target=0.0, pid_sensor=0, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.0, condition='False')
    #steer_prog = Controller(pid_constants=[0.97, 0.05, 49.98], pid_target=0, pid_sensor=2, pid_sub_sensor=0)
    #accel_prog = Controller(pid_constants=[3.97, 0.01, 48.79], pid_target=0.30, pid_sensor=5, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.01, condition='obs[-1][2][0] > -self.para_condition and obs[-1][2][0] < self.para_condition')
    #brake_prog = Controller(pid_constants=[0, 0, 0], pid_target=0, pid_sensor=2, pid_sub_sensor=0)

    steer_prog = Controller(pid_constants=[0.97, 0.05, 49.98], pid_target=0, pid_sensor=2, pid_sub_sensor=0)
    accel_prog = Controller(pid_constants=[3.97, 0.01, 48.79], pid_target=0.30, pid_sensor=5, pid_sub_sensor=0, pid_increment=0.0, para_condition=0.01, condition='obs[-1][2][0] > -self.para_condition and obs[-1][2][0] < self.para_condition')
    brake_prog = Controller(pid_constants=[0, 0, 0], pid_target=0, pid_sensor=2, pid_sub_sensor=0)

    if test_program == True:
        programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)
        return None

    nn_agent = NeuralAgent(track_name=track_name)
    all_observations = []
    all_actions = []
    for i_iter in range(1): # optimize controller parameters
        logging.info("Iteration {}".format(i_iter))
        # Learn/Update Neural Policy
        if i_iter == 0:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=1500)
        else:
            nn_agent.update_neural([steer_prog, accel_prog, brake_prog], episode_count=100)

        # Collect Trajectories

        observation_list, action_list = nn_agent.collect_data([steer_prog, accel_prog, brake_prog])
        #print('observation_list', observation_list[0])
        #print('\n action_list', action_list[0])

        all_observations += observation_list
        # Relabel Observations
        _, _, all_actions = nn_agent.label_data([steer_prog, accel_prog, brake_prog], all_observations)
        #print('\n all_actions', all_actions[0])

        # Learn new programmatic policy
        #print('observations: ', np.array(all_observations)[0])
        #print('actions', np.array(all_actions).shape)
        param_finder = ParameterFinder(all_observations, all_actions, steer_prog, accel_prog, brake_prog)
        #print('observations: ', np.array(all_observations).shape())
        #print('actions', np.array(all_actions).shape())

        #steer_ranges = [[create_interval(steer_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(steer_prog.pid_info()[1], 0.01)]
        #accel_ranges = [[create_interval(accel_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(accel_prog.pid_info()[1], 0.5), create_interval(accel_prog.pid_info()[2], 0.1), create_interval(accel_prog.pid_info()[3], 0.01)]
        #brake_ranges = [[create_interval(brake_prog.pid_info()[0][const], 0.05) for const in range(3)], create_interval(brake_prog.pid_info()[1], 0.001)]
        steer_ranges = [create_interval(steer_prog.pid_info()[0][const], 0.05) for const in range(3)]
        steer_ranges.append(create_interval(steer_prog.pid_info()[1], 0.01))

        accel_ranges = [create_interval(accel_prog.pid_info()[0][const], 0.05) for const in range(3)]
        accel_ranges.append(create_interval(accel_prog.pid_info()[1], 0.5))
        accel_ranges.append(create_interval(accel_prog.pid_info()[2], 0.1))
        accel_ranges.append(create_interval(accel_prog.pid_info()[3], 0.01))

        brake_ranges = [create_interval(brake_prog.pid_info()[0][const], 0.05) for const in range(3)]
        brake_ranges.append(create_interval(brake_prog.pid_info()[1], 0.001))

        pid_ranges = [steer_ranges, accel_ranges, brake_ranges]
        new_paras = param_finder.pid_parameters(pid_ranges)

        steer_prog.update_parameters([new_paras[i] for i in ['sp0', 'sp1', 'sp2']], new_paras['spt'])
        accel_prog.update_parameters([new_paras[i] for i in ['ap0', 'ap1', 'ap2']], new_paras['apt'], new_paras['api'], new_paras['apc'])
        brake_prog.update_parameters([new_paras[i] for i in ['bp0', 'bp1', 'bp2']], new_paras['bpt'])

        #programmatic_game(steer_prog, accel_prog, brake_prog, track_name=track_name)

    logging.info("Steering Controller" + str(steer_prog.pid_info()))
    logging.info("Acceleration Controller" + str(accel_prog.pid_info()))
    logging.info("Brake Controller" + str(brake_prog.pid_info()))

    return None

#[
#    [(0.9199999999999999, 1.02), (0.0, 0.1), (49.93, 50.029999999999994), (-0.01, 0.01)],
#    [(3.9200000000000004, 4.0200000000000005), (-0.04, 0.060000000000000005), (48.74, 48.839999999999996), (-0.2, 0.8), (-0.1, 0.1), (0.0, 0.02)],
#    [(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05), (-0.001, 0.001)]
# ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trackfile', default='practice.xml') #
    parser.add_argument('--seed', default=1337)
    parser.add_argument('--logname', default='AdaptiveProgramIPPG_')
    parser.add_argument('--test_program', default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    logPath = 'logs'
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    logFileName = args.logname + args.trackfile[:-4] + now
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] %(module)s  %(funcName)s %(lineno)d [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, logFileName)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    learn_policy(track_name=args.trackfile, test_program=args.test_program)
