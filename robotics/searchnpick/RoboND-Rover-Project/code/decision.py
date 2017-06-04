import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    
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


def handle_sample_in_vision(Rover):
    if Rover.is_sample_in_vision == True:
        Rover.throttle = 0.5
        Rover.steer = np.clip(Rover.steer, -15, 15)
        print('sample in vision steering to sample now ')
        if Rover.near_sample and not Rover.picking_up and Rover.vel != 0:
            Rover.throttle = 0
            Rover.brake = 10
            return Rover
            print('near sample so slowing down')
        if Rover.len_navs < Rover.go_forward:
            print('obstacle in the way')
            Rover = adjust_throttle(Rover)
            Rover = adjust_steer(Rover)

    
    return Rover

def handle_zero_velocity(Rover):
    print('zero vel checking sticky')
    Rover = check_sticky(Rover)
    
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
        Rover.currently_unstucking = True
        print('unstuck in unstsuck')
        Rover.brake = 0
        print('circle right')
        Rover.throttle = 0
        Rover = adjust_steer(Rover)
        if Rover.steer == 0:
            Rover.steer = -15
        Rover.num_unstucks += 1
    elif Rover.num_backups <=  Rover.backup_thresh:
        Rover.currently_unstucking = False
        print('unstuck calling backup')
        Rover = backout(Rover)
    else:
        print('unstuck random unstuck')
        Rover = random_unstuck(Rover)

    return Rover


def handle_normal_flow(Rover):
    
    Rover = check_sticky(Rover)
        
    if Rover.forward_mode:
        print('normal flow rover moving forward')
        if Rover.is_sample_in_vision:
            print('normal flow sample in vision')
            Rover =  handle_sample_in_vision(Rover)
            return Rover

        print('normal flow mean navigation')
        Rover = adjust_throttle(Rover)
        Rover = adjust_steer(Rover)
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
        Rover.currently_backing = True
        print('backout in backout')
        Rover.steer = 0
        Rover.brake = 0
        Rover.throttle = -1
        Rover.num_backups += 1
    elif Rover.num_unstucks <= Rover.unstuck_thresh:
        Rover.currently_backing = False
        print('backout number of attemps to backout breached but unstuck is on')
        Rover = unstuck(Rover)
    else:
        Rover.currently_backing = False
        print('backout number of attemps to backout and unstuck breached randomizing')
        Rover = random_unstuck(Rover)
    return Rover

def adjust_throttle(Rover):    
    
    have_nav = Rover.len_navs > 0
    have_obstacles = Rover.len_obstacles > 0
    
    if np.absolute(Rover.vel) <= 0.2:
        print('adjust throttle very low velocity, release breaks')
        Rover.brake = 0
    
    if have_nav and have_obstacles:
        print('adjust throttle have both nav and obstacles')
        if Rover.mean_obstacle_dist < Rover.mean_nav_dist:
            print('adjust throttle close to obstacle')
            if Rover.vel >= 0.2 and Rover.vel <= 2:
                print('adjust throttle vel > 0.2 and less than 2')
                Rover.throttle = 1
            elif Rover.vel > 2:
                print('adjust throttle vel > max')
                Rover.throttle = 0
            else:
                print('adjust throttle obstacle near default to zero')
                Rover.throttle = 0
            return Rover    
        else:
            print('adjust throttle more nav distance ')
            if Rover.vel > Rover.max_vel:
                print('adjust throttle very high vel slowing down')
                Rover.throttle = 0
            else:
                print('adjust throttle room to go normal throttle')
                Rover.throttle = 0.5
            return Rover    
                
    elif have_nav and not have_obstacles:
        print('adjust throttle have nav and no obstacles')
        if Rover.vel > Rover.max_vel:
            print('adjust throttle very high vel slowing down')
            Rover.throttle = 0
        else:
            print('adjust throttle room to go normal throttle')
            Rover.throttle = 0.5
        return Rover    
    elif have_obstacles:
        print('adjust throttle only abstacles')
        Rover.throttle = 0
        return Rover
    else:
        print('adjust throttle no nav and no obstacles magic')
        Rover.throttle = np.random.choice([-1, 0, 1], 1)[0]
        return Rover
    
    
    print('adjust throttle default case not changing steer')
    return Rover


def adjust_steer(Rover):
           
    have_nav = Rover.len_navs > 0
    have_obstacles = Rover.len_obstacles > 0
    
    if have_nav and have_obstacles:
        print('adjust steer have both nav and obstacles')
        if Rover.mean_nav_angle < 0:
            print('adjust steer clipping to right most')
            if Rover.mean_obstacle_dist < Rover.mean_nav_dist:
                Rover.steer = -15
            else:    
                Rover.steer = np.clip(Rover.min_nav_angle, -15, 15)
            return Rover
        elif Rover.mean_nav_angle > 0:
            print('adjust steer clipping to left most')
            if Rover.mean_obstacle_dist < Rover.mean_nav_dist:
                print('adjust steer clipping to full angle')
                Rover.steer = 15
            else:
                print('adjust steer clipping to max angle')
                Rover.steer = np.clip(Rover.max_nav_angle, -15, 15)
            return Rover
        else:
            print('adjust steer mean angle is zero')
            if Rover.mean_obstacle_angle < 0:
                print('adjust steer obstacle is right so trying to go left')
                if Rover.mean_obstacle_dist < Rover.mean_nav_dist:
                    print('adjust steer obstacle is right so trying to go left full angle')
                    Rover.steer = 15
                else:
                    print('adjust steer obstacle is right so trying to go left steer')
                    left_steer = np.clip(Rover.steer + 5, -15, 15)
                    Rover.steer = left_steer
                return Rover
            elif Rover.mean_obstacle_angle > 0:
                print('adjust steer obstacle is left so trying to go right')
                if Rover.mean_obstacle_dist < Rover.mean_nav_dist:
                    print('adjust steer obstacle is left so trying to go right full angle')
                    Rover.steer = -15
                else:
                    print('adjust steer obstacle is left so trying to go right steer')
                    right_steer = np.clip(Rover.steer - 5, -15, 15)
                    Rover.steer = right_steer
                return Rover
            else:
                print('adjust steer both obstacle and nav are zero')
                Rover.steer = 0
                return Rover
    elif have_nav and not have_obstacles:
        print('adjust steer going to mean')
        if Rover.len_navs > Rover.go_forward:
            if np.allclose([Rover.mean_nav_angle],[0.0]):
                print('adjust steer mean angle zero so randomly picking from range')
                angles = np.arange(Rover.min_nav_angle, Rover.max_nav_angle, 0.5)
                Rover.steer = np.clip(np.random.choice(angles, 1)[0] + Rover.steer, -15, 15)
            else:
                print('adjust steer offsetting')
                Rover.steer = np.clip(Rover.mean_nav_angle * Rover.nav_angle_offset, -15, 15)
        else:
            print('adjust steer need more move not enough nav left')
            if Rover.mean_nav_angle < 0:
                print('adjust steer moving hard right')
                Rover.steer = -15
            elif Rover.mean_nav_angle > 0:
                print('adjust steer moving hard left')
                Rover.steer = 15
            else:
                print('adjust steer mean agle is zero randomizing')
                Rover.steer = np.random.choice([-15, 0, 15], 1)[0]
        return Rover
    elif have_obstacles:
        print('adjust steer only abstacles')
        if Rover.mean_obstacle_angle < 0:
            print('adjust steer obstacle is right so trying to go left')
            Rover.steer = 15
            return Rover
        elif Rover.mean_obstacle_angle > 0:
            print('adjust steer obstacle is left so trying to go right')
            Rover.steer = -15
            return Rover
        else:
            print('adjust steer both obstacle and nav have zero angles, randomizing')
            Rover.steer = 0
            return Rover
    else:
        print('adjust steer no nav and no obstacles magic')
        Rover.steer = np.random.choice([-15, 0, 15], 1)[0]
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




def random_unstuck(Rover):
    print('random unstuck')
    Rover.throttle = np.random.choice([0, 1, -1], 1)[0]
    Rover.steer = np.random.choice([-15, 0, 15], 1)[0]
    Rover.brake = 0
    return Rover

def stop(Rover):
    Rover.currently_backing = False
    Rover.currently_unstucking = False
    print('stopping mode')
    Rover.brake = 10
    Rover.steer = 0
    Rover.throttle = 0
    return Rover

def check_sticky(Rover):
    print('check sticky')
    if not Rover.currently_unstucking and not Rover.currently_backing:
        print('check sticky moving forward mode')
        Rover.forward_mode = True
    elif Rover.currently_unstucking:
        print('check sticky currently unstucking checking to see if need to stop')
        if Rover.len_navs >= Rover.go_forward:
            print('check sticky enough room to go disable unstucking')
            Rover.currently_unstucking = False
            Rover.forward_mode = True
        else:
            print('check sticky need to unstuck more')
            Rover.currently_unstuck = True
            Rover = unstuck(Rover)
            return Rover
    elif Rover.currently_backing or Rover.in_backup_mode:
        print('check sticky in backup mode or currently backing checking to see if need to stop')
        if Rover.len_navs <= Rover.go_forward:
            print('check sticky pretty close to wall keep going back')
            Rover.currently_backing = True
            Rover = backout(Rover)
            return Rover
        else:
            print('check sticky enough room to go forward stopping')
            Rover = stop(Rover)
            Rover.currently_backing = False
            Rover.forward_mode = True
        return Rover
    
    return Rover