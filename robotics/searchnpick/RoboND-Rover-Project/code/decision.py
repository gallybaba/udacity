import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    
    print_metrics(Rover)
    
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                mean_calc = np.mean(Rover.nav_angles * 180/np.pi)
                print('mean_calc: ', mean_calc)
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    #Rover.throttle = 0
                    ## Release the brake to allow turning
                    #Rover.brake = 0
                    ## Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    #Rover.steer = -15 # Could be more clever here about which way to turn
                    Rover = unstuck(Rover)
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print('no navs')
        if Rover.max_vel > 0.3:
            print('no navs but vel high enough stopping')
            Rover = stop(Rover)
        else:
            print('no navs but vel low unstucking')            
            Rover = unstuck(Rover)
            Rover.brake = 0

        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
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

def stop(Rover):
    Rover.currently_backing = False
    Rover.currently_unstucking = False
    print('stopping mode')
    Rover.brake = 10
    Rover.steer = 0
    Rover.throttle = 0
    return Rover

def unstuck(Rover):
    Rover.currently_unstucking = True
    print('unstuck in unstsuck')
    Rover.brake = 0
    print('circle right')
    Rover.throttle = 0
    if Rover.mean_obstacle_angle <= 0:
        Rover.steer = 15
    else:
        Rover.steer = -15
            
    return Rover
