# Do the necessary imports
import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time

# Import functions for perception and decision making
from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images
# Initialize socketio server and Flask application 
# (learn more at: https://python-socketio.readthedocs.io/en/latest/)
sio = socketio.Server()
app = Flask(__name__)

# Read in ground truth map and create 3-channel green version for overplotting
# NOTE: images are read in by default with the origin (0, 0) in the upper left
# and y-axis increasing downward.
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
# This next line creates arrays of zeros in the red and blue channels
# and puts the map into the green channel.  This is why the underlying 
# map output looks green in the display image
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Define RoverState() class to retain rover state parameters
class RoverState():
    def __init__(self):
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.img = None # Current camera image
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.steer = 0 # Current steering angle
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels
        self.ground_truth = ground_truth_3d # Ground truth worldmap
        self.mode = 'forward' # Current mode (can be forward or stop)
        self.throttle_set = 0.5 # Throttle setting when accelerating
        self.brake_set = 10 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 150 # Threshold to initiate stopping
        self.go_forward = 2000 # Threshold to go forward again
        self.max_vel = 3 # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        self.samples_pos = None # To store the actual sample positions
        self.samples_to_find = 0 # To store the initial count of samples.samples_found = 0 # To count the number of samples found
        self.near_sample = 0 # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0 # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False # Set to True to trigger rock pickup
        
        ## custom state 
        ## navigational state
        self.mean_nav_angle = 0 # mean angle to navigate
        self.max_nav_angle = 0 # max angle to navigate
        self.min_nav_angle = 0 # min angle to navigate
        self.mean_nav_dist = 0 # mean distance to navigate
        self.min_nav_dist = 0 # min distance to navigate
        self.max_nav_dist = 0 # max distance to navigate
        self.nav_ratio = 1 # length of obstacle angles / length of navigable angles
        self.len_navs = 0 # length of navigational angles
        
        ## obstacle state
        self.mean_obstacle_angle = 0 # mean angle to obstacle
        self.min_obstacle_angle = 0 # min angle to obstacle
        self.max_obstacle_angle = 0 # max angle to obstacle
        self.obstacle_angles = None # obstacle angles list
        self.obstacle_dists = None # obstacle distances list
        self.mean_obstacle_dist = 0 # mean distance to obstacle
        self.len_obstacles = 0 # length of obstacle angles
        
        ## rock state
        self.mean_rock_dist = 0 # mean rock dist from rover
        self.mean_rock_angle = 0 # mean rock angle from rover
        self.rock_angles = None # rock angles dist
        self.rock_dists = None # rock distance dist
        self.len_rock = 0 # length of rock angles
        self.is_sample_in_vision = False #do we see a rock near by?
        
        ## other state
        self.last_pos = None # what was rover's last position
        self.last_move = 0 # how much did we move from last time?
        self.is_init = True # are we just starting?
        self.was_sample_collected = False # did rover just pick up a sample?
        self.nav_ratio_thresh = 3 # length of obstacle angles / length of navigable angles threshold
        self.in_backup_mode = False # tells state that rover is backing off, -ve velocity
        self.nav_vision_thresh = 1000 # how much we want to see and to filter out far off
        self.rock_vision_thresh = 1200 # how much we want to see and to filter out far off for rocks
        self.pitch_thresh = 1.1 # controls whether vision image is updated to manage fidelity
        self.roll_thresh = 1.1 # controls whether vision image is updated to manage fidelity
        self.num_backups = 0 # to store number of times backup is called
        self.num_unstucks = 0 # to store number of times unstuck is called
        self.backup_thresh = 100 # number of times after which backup is claimed to not work
        self.unstuck_thresh = 100 # number of times after which unstuck is claimedto not work
        self.nav_angle_offset = 1.1 # how much nav angle to offset to avoid sudden stops and turns
        self.currently_unstucking = False # is rover currently in unstuck mode?
        self.currently_backing = False # is rover currently in backup mode?
        self.forward_mode = True
        self.terrain = np.zeros_like(self.vision_image[:,:,0])
        self.visited_terrain = np.zeros_like(self.vision_image[:,:,0])
        
        
# Initialize our rover 
Rover = RoverState()

# Variables to track frames per second (FPS)
# Intitialize frame counter
frame_counter = 0
# Initalize second counter
second_counter = time.time()
fps = None


# Define telemetry function for what to do with incoming data
@sio.on('telemetry')
def telemetry(sid, data):

    global frame_counter, second_counter, fps
    frame_counter+=1
    # Do a rough calculation of frames per second (FPS)
    if (time.time() - second_counter) > 1:
        fps = frame_counter
        frame_counter = 0
        second_counter = time.time()
    print("Current FPS: {}".format(fps))

    if data:
        global Rover
        # Initialize / update Rover with current telemetry
        Rover, image = update_rover(Rover, data)

        if np.isfinite(Rover.vel):

            # Execute the perception and decision steps to update the Rover's state
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)

            # Create output images to send to server
            out_image_string1, out_image_string2 = create_output_images(Rover)

            # The action step!  Send commands to the rover!
            commands = (Rover.throttle, Rover.brake, Rover.steer)
            print('commands: throttle, brake, steer: ', str(commands))
            
            send_control(commands, out_image_string1, out_image_string2)
 
            # If in a state where want to pickup a rock send pickup command
            if Rover.send_pickup and not Rover.picking_up:
                send_pickup()
                # Reset Rover flags
                Rover.send_pickup = False
        # In case of invalid telemetry, send null commands
        else:

            # Send zeros for throttle, brake and steer and empty images
            print('sending zeros to rover')
            send_control((0, 0, 0), '', '')

        # If you want to save camera images from autonomous driving specify a path
        # Example: $ python drive_rover.py image_folder_path
        # Conditional to save image frame if folder was specified
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit(
        "get_samples",
        sample_data,
        skip_sid=True)

def send_control(commands, image_string1, image_string2):
    # Define commands to be sent to the rover
    data={
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
        }
    # Send commands via socketIO server
    sio.emit(
        "data",
        data,
        skip_sid=True)
    eventlet.sleep(0)
# Define a function to send the "pickup" command 
def send_pickup():
    print("Picking up")
    pickup = {}
    sio.emit(
        "pickup",
        pickup,
        skip_sid=True)
    eventlet.sleep(0)
    global Rover
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    #os.system('rm -rf IMG_stream/*')
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Recording this run ...")
    else:
        print("NOT recording this run ...")
    
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
