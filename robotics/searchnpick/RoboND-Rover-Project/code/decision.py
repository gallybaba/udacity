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
        Rover = print_metrics(Rover)
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
        Rover = random_unstuck(Rover)
        
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
    if Rover.vel == 0 and Rover.throttle != 0 \
    and not Rover.in_backup_mode and Rover.len_navs < Rover.go_forward:
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
        if Rover.len_navs >= Rover.go_forward:
            print('zero vel some nav to go starting')
            Rover = adjust_throttle(Rover)
            Rover = adjust_steer(Rover)
            return Rover
        else:
            print('zero vel no nav unstuck')
            Rover = unstuck(Rover)

        ## there is nav but still stuck or there is no nav
        print('zero vel default')
        Rover = unstuck(Rover)

    return Rover    

    
def unstuck(Rover):
    if Rover.num_unstucks <= Rover.unstuck_thresh:
        print('unstuck')
        Rover.brake = 0
        print('circle right')
        Rover.throttle = 0
        Rover.steer = -15
        Rover.num_unstucks += 1
    elif Rover.num_backups <=  Rover.backup_thresh:
        Rover = backout(Rover)
    else:
        Rover = random_unstuck(Rover)

    return Rover


def handle_normal_flow(Rover):
    if np.absolute(Rover.vel) <= 0.3 and Rover.throttle != 0 \
    and Rover.len_navs <= Rover.go_forward:
        print('normal flow rover seem stuck ')
        Rover = unstuck(Rover)
        return Rover
    
    if Rover.is_sample_in_vision:
        print('normal flow sample in vision')
        Rover =  handle_sample_in_vision(Rover)
        return Rover
        
    if Rover.len_navs >= Rover.go_forward:
        print('normal flow mean navigation')
        Rover = adjust_throttle(Rover)
        Rover = adjust_steer(Rover)
        return Rover
    else:
        print('normal flow backing out')
        Rover = backout(Rover)
        return Rover

    print('normal flow default')
    Rover = backout(Rover)
   
    return Rover


def handle_init(Rover):
    if Rover.is_sample_in_vision:
        print('init sample in vision')
        Rover =  handle_sample_in_vision(Rover)
        return Rover
    
    if Rover.len_navs >= Rover.go_forward:
        print('init enough nav to steer')
        Rover = adjust_throttle(Rover)
        Rover = adjust_steer(Rover)

    else:    
        print('init backing off')
        Rover = backoff(Rover)

    return Rover


def backout(Rover):
    if Rover.num_backups <= Rover.backup_thresh:
        print('in backout')
        Rover.steer = 0
        Rover.brake = 0
        Rover.throttle = -1
    elif Rover.num_unstucks <= Rover.unstuck_thresh:
        Rover = unstuck(Rover)
    else:
        Rover = random_unstuck(Rover)
    return Rover

def adjust_throttle(Rover):
    
    if Rover.in_backup_mode:
        print('adjust throttle in backup mode')
        if Rover.len_navs <= Rover.go_forward:
            prinnt('adjust throttle pretty close to wall keep going back')
            Rover.throttle = -1

        elif Rover.len_navs > Rover.go_forward:
            print('adjust throttle enough room to go forward brake')
            Rover.throttle = 0
            Rover.brake = 10
        return Rover
    
    if Rover.len_navs <= Rover.go_forward:
        
        print('adjust throttle looks like hitting wall')
        Rover.throttle = 0
        Rover.brake = 10
        return Rover
    

    if Rover.len_navs >= Rover.go_forward:
        if Rover.vel < Rover.max_vel:
            print('adjust throttle room to go normal throttle')
            Rover.throttle = 0.5
            Rover.brake = 0
        else:
            print('adjust throttle room to go but high vel slowing throttle')
            Rover.throttle = 0
            Rover.brake = 0
        return Rover
    
    print('adjust throttle default case.. not changing throttle')
    return Rover

def adjust_steer(Rover):
    if Rover.in_backup_mode:
        print('adjust steer rover in backup mode')
        if Rover.len_navs <= Rover.stop_forward:
            prinnt('adjust steer pretty close to wall / obstacle spinning')
            Rover.steer = -15
        elif Rover.len_navs > Rover.stop_forward and \
        Rover.len_navs <= Rover.go_forward:
            print('adjust steer some room but not enough keep going back')
            Rover.steer = 0
        elif Rover.len_navs > Rover.go_forward:
            print('adjust steer enough room to go keep steer')
            Rover.steer = Rover.steer
        return Rover
    
    if Rover.len_navs <= Rover.stop_forward:
        print('adjust steer there is not enough range steering')
        Rover.steer = -15
        return Rover

    if Rover.len_navs > Rover.stop_forward and \
    Rover.len_navs <= Rover.go_forward:
        print('adjust steer nearing obstacle not changing steer')
        Rover.steer = Rover.steer
        return Rover
    
    if Rover.len_navs > Rover.go_forward:    
        print('adjust steer going to mean')
        Rover = clip_to_mean(Rover)
        return Rover
    
    print('adjust steer default case not changing steer')
    return Rover


def print_metrics(Rover):
    print('------------rover state---------------------')
    print('rover velocity: ', Rover.vel)
    print('rover steer: ', Rover.steer)
    print('rover throttle: ', Rover.throttle)
    print('last move: ', Rover.last_move)
    print('------------nav state-----------------------')
    print('mean nav angle: ', Rover.mean_nav_angle)
    print('max  nav angle: ', Rover.max_nav_angle )
    print('min  nav angle: ', Rover.min_nav_angle )
    print('mean nav  dist: ', Rover.mean_nav_dist )
    print('min  nav  dist: ', Rover.min_nav_dist  )
    print('max  nav  dist: ', Rover.max_nav_dist  )
    print('nav      ratio: ', Rover.nav_ratio     )
    print('nav     length: ', Rover.len_navs      )
    print('------------rock state-----------------------')
    print('mean rock  dist: ', Rover.mean_rock_dist )
    print('mean rock angle: ', Rover.mean_rock_angle)
    print('len        rock: ', Rover.len_rock       )
    print('------------obstacle state-----------------------')    
    print('mean obstacle angle: ', Rover.mean_obstacle_angle)
    print('min obstacle  angle: ', Rover.min_obstacle_angle )
    print('max obstacle  angle: ', Rover.max_obstacle_angle )
    print('mean obstacle  dist: ', Rover.mean_obstacle_dist )
    print('len       obstacles: ', Rover.len_obstacles      )
    print('------------engine state-----------------------')        
    print('is sample in vision: ', Rover.is_sample_in_vision )
    print('last            pos: ', Rover.last_pos            )
    print('last           move: ', Rover.last_move           )
    print('is             init: ', Rover.is_init             )
    print('nav ratio    thresh: ', Rover.nav_ratio_thresh    )
    print('back up        mode: ', Rover.in_backup_mode      )
    print('------------end state-----------------------')                
    return Rover


def clip_to_mean(Rover):
    print('clipping to mean')
    ## looking straight
    if np.absolute(Rover.mean_nav_angle) <= 2:
        ## if obstacle is right, go left
        if Rover.mean_obstacle_angle < 0:
            Rover.steer = 2
        else:
            Rover.steer = -2
    else:
        Rover.steer = np.clip(Rover.mean_nav_angle, -15, 15)
    
    return Rover


def random_unstuck(Rover):
    print('random unstuck')
    Rover.throttle = np.random.choice([0, 1, -1], 1)[0]
    Rover.steer = np.random.choice([-15, 0, 15], 1)[0]
    Rover.brake = 0
    return Rover
