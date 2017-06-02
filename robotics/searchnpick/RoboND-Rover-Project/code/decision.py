import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        Rover = calculate_metrics(Rover)
        ## hangle init
        if Rover.is_init:
            print('hanldle init')
            Rover = handle_init(Rover)
            return Rover
        
        if Rover.vel == 0:
            print('handling zero velocity')
            Rover = handle_zero_velocity(Rover)
            return Rover
        
        print('normal flow')
        ### normal flow
        Rover = handle_normal_flow(Rover)
            
        print('returning from decision')
        return Rover
    
    else:
        print('No Nav')
        Rover.steer = 0
        Rover.throttle = 0
        Rover.brake = 0
        
    return Rover



def handle_sample_in_vision(Rover):
    if Rover.is_sample_in_vision == True:
        Rover.throttle = 0.5
        Rover.steer = np.clip(Rover.steer, -15, 15)
        print('sample in vision steering to sample now ')
        if Rover.near_sample and not Rover.picking_up and Rover.vel != 0:
            Rover.throttle = 0
            Rover.brake = 10
            print('near sample so slowing down')
        if Rover.nav_ratio > Rover.nav_ratio_thresh:
            print('obstacle in the way')
            Rover = adjust_steer(Rover)
    
    return Rover

def handle_zero_velocity(Rover):
    if Rover.vel == 0 and Rover.throttle != 0:
        print('stuck in zero vel')
        Rover = backout(Rover)
        return Rover
    ### zero velocity
    if np.absolute(Rover.vel) == 0.0: 
        ## ready to pick up sample
        if Rover.near_sample and not Rover.picking_up:
            print('rover is stopped for pickup')
            Rover.brake = 1
            Rover.send_pickup = True
            Rover.steer = 0
            Rover.throttle = 0
            return Rover

        ## picking up sample
        if Rover.picking_up and Rover.near_sample:
            print('rover is picking up')
            Rover.was_sample_collected = True
            Rover.brake = 0
            Rover.steer = 0
            Rover.throttle = 0
            return Rover

        ## have enough nav
        if Rover.nav_ratio <= Rover.nav_ratio_thresh:
            if Rover.len_navs > 30:
                print('some nav to go starting')
                Rover = adjust_throttle(Rover)
                Rover = adjust_steer(Rover)
                return Rover
            else:
                print('no nav unstuck')
                Rover = unstuck(Rover)

        ## there is nav but still stuck or there is no nav
        if Rover.nav_ratio > Rover.nav_ratio_thresh:
            print('not enough nav unstuck it')
            Rover = unstuck(Rover)

    return Rover    

    
def unstuck(Rover):
    print('unstuck')
    Rover.brake = 0
    print('circle right')
    Rover.throttle = 0
    Rover.steer = -15

    return Rover


def handle_normal_flow(Rover):
    if Rover.vel <= 0.3 and Rover.throttle != 0 \
    and Rover.len_navs <= 30:
        print('rover seem stuck ')
        Rover = unstuck(Rover)
        return Rover
    
    if Rover.is_sample_in_vision:
        Rover =  handle_sample_in_vision(Rover)
        return Rover
        
    if Rover.nav_ratio <= Rover.nav_ratio_thresh:
        if Rover.len_navs > 30:
            print('normal mean navigation')
            Rover = adjust_throttle(Rover)
            Rover = adjust_steer(Rover)
            Rover.brake = 0
        else:
            print('backing out')
            Rover = backout(Rover)
    else:
        print('nav is less')
        Rover = backout(Rover)
   
    return Rover


def calculate_metrics(Rover):

    Rover.mean_nav_angle = 0
    Rover.max_nav_angle = 0
    Rover.min_nav_angle = 0
    Rover.max_obstacle_angle = 0
    Rover.min_obstacle_angle = 0
    Rover.mean_nav_dist = 0
    Rover.min_nav_dist = 0
    Rover.max_nav_dist = 0
    if not np.isnan(Rover.max_obstacle_angle):
        Rover.max_obstacle_angle = Rover.max_obstacle_angle

    if not np.isnan(Rover.min_obstacle_angle):    
        Rover.min_obstacle_angle = Rover.min_obstacle_angle

    if Rover.nav_angles is not None:
        Rover.len_navs = len(Rover.nav_angles)
    
    if Rover.obstacle_angles is not None:
        Rover.len_obstacles = len(Rover.obstacle_angles)
        Rover.min_obstacle_angle = np.min(Rover.obstacle_angles)
        Rover.max_obstacle_angle = np.max(Rover.obstacle_angles)
    
    if Rover.rock_angles is not None:
        Rover.len_rock = len(Rover.rock_angles)    
    terrain_img = Rover.vision_image[:,:,2]

#    y, x = terrain_img.nonzero()
#    if len(y) > 0:
#        angles = np.arctan2(y, x)
#        mimg = np.mean(angles)
#        minimg = np.min(angles)
#        maximg = np.max(angles)
#        Rover.mean_nav_angle = mimg
#        Rover.min_nav_angle = minimg
#        Rover.max_nav_angle = maximg
#        Rover.len_navs = len(angles)


#    obstacle_img = Rover.vision_image[:,:,0]
#    y1, x1 = obstacle_img.nonzero()
#    if len(y1) > 0:
#        angles1 = np.arctan2(y1, x1)
#        mimg1 = np.mean(angles1)
#        minimg1 = np.min(angles1)
#        maximg1 = np.max(angles1)
#        Rover.obstacle_angle = mimg1
#        Rover.min_obstacle_angle = minimg1
#        Rover.max_obstacle_angle = maximg1
#        Rover.len_obstacles = len(angles1)
    
#    rock_img = Rover.vision_image[:,:,1]
#    y2, x2 = rock_img.nonzero()
#    if len(y2) > 0:
#        mimg2 = np.mean(np.arctan2(y2, x2))
#        minimg2 = np.min(np.arctan2(y2, x2))
#        maximg2 = np.max(np.arctan2(y2, x2))
#        Rover.rock_angle = mimg2

        
    if Rover.nav_angles is not None and len(Rover.nav_angles) >= Rover.stop_forward:
        Rover.mean_nav_angle = np.mean(Rover.nav_angles * 180 / np.pi)
        Rover.max_nav_angle = np.max(Rover.nav_angles * 180 / np.pi)
        Rover.min_nav_angle = np.min(Rover.nav_angles * 180 / np.pi)
        Rover.mean_nav_dist = np.mean(Rover.nav_dists)
        Rover.min_nav_dist = np.min(Rover.nav_dists)
        Rover.max_nav_dist = np.max(Rover.nav_dists)
        Rover.range_nav_angle = Rover.max_nav_angle - Rover.min_nav_angle
        Rover.range_obstacle_angle = Rover.max_obstacle_angle - Rover.min_obstacle_angle
    Rover.nav_ratio = 1.0
    if Rover.len_navs > 0:
        Rover.nav_ratio = Rover.len_obstacles / Rover.len_navs
        if Rover.nav_ratio == 0:
            Rover.nav_ratio = 1

    if Rover.vel < 0:
        Rover.in_backup_mode = True
    else:
        Rover.in_backup_mode = False
        
            
    print('rover velocity: ', Rover.vel)
    print('rover steer: ', Rover.steer)
    print('rover throttle: ', Rover.throttle)
    print('last move: ', Rover.last_move)
    
    print('calculated nav ratio: ', Rover.nav_ratio)
    print('nav len: ', Rover.len_navs)
    print('mean nav angle: ', Rover.mean_nav_angle)
    print('min nav angle: ', Rover.min_nav_angle)
    print('max nav angle: ', Rover.max_nav_angle)    

    print('rock len: ', Rover.len_rock)
    print('rock angle: ', Rover.rock_angle)
    print('rock dist: ', Rover.rock_dist)
    
    print('obstacle len: ', Rover.len_obstacles)
    print('mean obstacle angle: ', Rover.obstacle_angle)
    print('min obstacle angle: ', Rover.min_obstacle_angle)
    print('max obstacle angle: ', Rover.max_obstacle_angle)
    print('obstacle dist: ', Rover.obstacle_dist)
    print('range nav angle: ', Rover.range_nav_angle)
    print('range obstacle angle: ', Rover.range_obstacle_angle)
    
    return Rover



def handle_init(Rover):
    if Rover.is_sample_in_vision:
        Rover =  handle_sample_in_vision(Rover)
        return Rover
    
    if Rover.nav_ratio <= Rover.nav_ratio_thresh:
        print('enough nav to steer')
        Rover = adjust_throttle(Rover)
        Rover = adjust_steer(Rover)

    else:    
        print('backing off')
        Rover = backoff(Rover)

    Rover.brake = 0
    return Rover


def backout(Rover):
    print('in backout')
    Rover.steer = 0
    Rover.brake = 0
    Rover.throttle = -1
    return Rover

def adjust_throttle(Rover):
    ## facing wall
    if np.absolute(np.absolute(Rover.obstacle_angle) - np.absolute(Rover.mean_nav_angle)) \
    <= 8:
        print('adjust throttle mean obstacle and mean nav are same place checking ratio')
        if Rover.nav_ratio > Rover.nav_ratio_thresh:
            print('looks like hitting wall')
            Rover.throttle = 0
            Rover.brake = 10
            return Rover
    
        if Rover.nav_ratio <= Rover.nav_ratio_thresh:
            if Rover.len_navs >= 30:
                print('adjust throttle moving relatively ok normal throttle')
                Rover.throttle = 0.5
                return Rover
            else:
                print('adjust throttle no range to move, need to stop')
                Rover.throttle = 0
                return Rover
    
    else:
        print('adjust throttle far apart nav from obstacle checking for ratio')
        if Rover.nav_ratio > Rover.nav_ratio_thresh:
            print('adjust throttle looks like hitting wall')
            Rover.throttle = 0
            return Rover
    
        if Rover.nav_ratio <= Rover.nav_ratio_thresh:
            if Rover.len_navs >= 30:
                print('adjust throttle far apart moving relatively ok normal throttle')
                if Rover.vel <= 0:
                    Rover.throttle = 0
                else:    
                    Rover.throttle = 0.5
                return Rover
            else:
                print('adjust throttle far apart no range to move, need to stop')
                Rover.throttle = 0
                return Rover

    return Rover


def adjust_steer(Rover):
    if Rover.in_backup_mode:
        print('adjust steer rover in backup mode, reversing steer')
        Rover.steer = np.clip(-1 * Rover.steer, -15, 15)
        return Rover
    
    if np.absolute(np.absolute(Rover.obstacle_angle) - np.absolute(Rover.mean_nav_angle)) \
    <= 2:
        print('adjust steer mean obstacle and mean nav are same place checking for max / min')
        if Rover.len_navs >= 30:
            Rover.steer = np.clip(Rover.mean_nav_angle * Rover.nav_ratio, -15, 15)
            print('there is enough range steering')
        else:
            print('there is not enough range steering')
            Rover.steer = -15
    else:
        print('obstacle and nav are far apart')
        if Rover.vel <= 0 and Rover.throttle :
            print('reversing steer as velocity is -ve')
            Rover.steer = -15
        else:    
            Rover.steer = np.clip(Rover.nav_ratio * Rover.mean_nav_angle, -15, 15)
    
    return Rover
