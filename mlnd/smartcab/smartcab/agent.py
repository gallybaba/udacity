import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import math

class stats:
    """ hold stats on run and help make decisions"""
    def __init__(self):
        self.iteration = 0
        self.bad_moves = 0
        self.good_moves = 0
        self.random_picks = 0
        self.q_ppicks = 0
        self.poor_choices = 0

    def __str__(self):
        return "iteration: " + str(self.iteration)  + ", random picks: "  + str(self.random_picks) + ", poor choices: " + str(self.poor_choices)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Light, Oncoming, NextWaypoint, Left
        self.state = (None, None, None, None)
        self.q = {}
        self.r = {}
        self.gamma = 1.0
        self.alpha = 1.0
        self.epsilon = 0.0
        self.stats = stats()
        self.next_state = (None, None, None, None)
        self.old_waypoint = None
        self.successfull_runs = 0


    def is_q_converged(self):
        noneeqs = []
        forwardqs = []
        leftqs = []
        rightqs = []
        for state, qs in self.q.iteritems():
            noneeqs.append(qs[0])
            forwardqs.append(qs[1])
            leftqs.append(qs[2])
            rightqs.append(qs[3])
        
        if len(forwardqs) < 16:
            return False
        if np.mean(forwardqs) == 0:
            return False
        if np.count_nonzero(forwardqs == 0) / len(forwardqs) * 100 >= 70:
            return False
        else:
            for i in range(0, len(forwardqs) -1):
                if forwardqs[i] != 0 and forwards[i+1] != 0:
                    ratio = float(forwardqs[i]) / forwardqs[i+1]
                    if ratio > 1.05 or ratio < 0.95:
                        return False 


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
  


    def read_sense(self):
        ### TODO: refactor reading sense
        pass
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.old_waypoint = self.next_waypoint
        inputs = self.env.sense(self)
        print "inputs: ", inputs
        print "waypoint: ", self.next_waypoint
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        ### Light, Oncoming, NextWaypoint, Left
        self.state = (inputs['light'], inputs['oncoming'], self.next_waypoint, inputs['left'])
        self.stats.iteration = self.stats.iteration + 1
        # Control how much you want to explore using epsilon
        shouldExploreCount = math.ceil(self.epsilon * self.stats.iteration)
        
        max_index = 0 
        if self.state in self.q:
            qs = self.q[self.state]
            maxq = max(qs)
            max_index = qs.index(maxq)
            # if more than one mazqs, chose random
            indices = []
            for i in range(0, 4):
                if qs[i] == maxq:
                    indices.append(i)
            if len(indices) > 0:
                max_index = random.choice(indices)
            #valid_actions = [None, 'forward', 'left', 'right']
        else:
            self.q[self.state] = [0,0,0,0]
            max_index = random.choice([0,1,2,3])
    
        # TODO: Select action according to your policy
        # choose randomnly until  either q is converged or based on epsilon
        # epsilon controls how many actions to chose randomnly out of total 
        # iterations.  
        if random.random() < self.epsilon:
        #if shouldExploreCount > self.stats.random_picks and not self.is_q_converged():         
            action = random.choice(self.env.valid_actions)
            print "random pick"
            self.stats.random_picks += 1
        else:
            action = self.env.valid_actions[max_index]
            self.stats.q_ppicks += 1
            print "q pick"

        action_index = self.env.valid_actions.index(action)
        actualQ = self.q[self.state][action_index]
        # Execute action and get reward
        reward = self.env.act(self, action)
        negative_reward_str = []
        if reward < 0:
            self.stats.poor_choices += 1
            s = ''.join(["Wrong Action: Light: ", str(self.state[0]), ", Oncoming: ", str(self.state[1]), ", NextWaypoint: ", str(self.state[2]), ", action: " , str(action), ", Reward: ", str(reward)])
            negative_reward_str.append(s)

        # TODO: Learn policy based on state, action, reward
        # act will place agent into next location
        # incremental updates to q matrix since we cannot go through all actions until goal state is reached
        # Q(s,a) = alpha * LearnedQ + (1-alpha) * OldQ # How much we want to learn?
        # LeanredQ = IR(s,a) + gamma * E[Max(s',a')]{1,n}
        # lets sense and get next state and set the next state to current
        #self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.next_state = (inputs['light'], inputs['oncoming'], self.planner.next_waypoint(), inputs['left'])
        # we need to choose max q from this state.
        self.stats.iteration = self.stats.iteration + 1
        
        new_max_index = 0
        new_maxq = 0 
        if self.next_state in self.q:
            qs = self.q[self.next_state]
            new_maxq = max(qs)
            new_max_index = qs.index(new_maxq)
        else:
            self.q[self.next_state] = [0,0,0,0]
    
        expectedQ = new_maxq
        learnedQ = reward + self.gamma * expectedQ
        actualQ = self.alpha * learnedQ + (1-self.alpha) * actualQ
        qs = self.q[self.state]
        qs[action_index] = actualQ
        self.q[self.state] = qs
        if self.env.done:
            print "Destination Reached in ", self.stats.iteration

        if self.env.done and self.env.get_deadline(self) > 0:
            self.successfull_runs += 1
            print "Successfull Run: " + str(self.successfull_runs)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, stats = {}, state = {}, oldwaypoint = {}, nextwaypoint = {}, q = {},  actualQ = {}, maxindex = {}, current_negative_reward= {}".format(deadline, inputs, action, reward, self.stats, self.state, self.old_waypoint, self.next_waypoint, self.q, actualQ, max_index, negative_reward_str)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print "SuccessFull Runs: ", str(a.successfull_runs), ", Poor Choices: ", str(a.stats.poor_choices) 

if __name__ == '__main__':
    run()
