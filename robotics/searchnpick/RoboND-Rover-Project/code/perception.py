import numpy as np
import cv2

def filter_pixels_by_distance(x, y, distance):
    filter_mask = np.sqrt(x**2 + y**2) < distance
    new_x = x[filter_mask]
    new_y = y[filter_mask]
    return new_x, new_y

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def color_below_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be below all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify pixels below the threshold
# Threshold of RGB < 160 does a nice job of identifying obstacles pixels only
#def color_within_thresh(img, rgb_thresh_min=(192, 192, 51), rgb_thresh_max= (255, 255, 114)):
def color_within_thresh(img, rgb_thresh_min=(110, 110, 5), rgb_thresh_max= (255, 255, 90)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be below all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] < rgb_thresh_max[0]) \
                & (img[:,:,1] < rgb_thresh_max[1]) \
                & (img[:,:,2] < rgb_thresh_max[2]) \
                & (img[:,:,0] > rgb_thresh_min[0]) \
                & (img[:,:,1] > rgb_thresh_min[1]) \
                & (img[:,:,2] > rgb_thresh_min[2]) 
    # Index the array of zeros with the boolean array and set to 1
    #print(np.any(above_thresh))
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def color_within_thresh_cv(img, rgb_thresh_min=np.array([5, 110, 110]), rgb_thresh_max= np.array([90, 255, 255])):
    # rgb_thresh_min=(110, 110, 5), rgb_thresh_max= (255, 255, 90)
    # convert to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rock = cv2.inRange(img_hsv, rgb_thresh_min, rgb_thresh_max)
    return rock


# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yawrad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yawrad) - ypix * np.sin(yawrad)
    ypix_rotated = xpix * np.sin(yawrad) + ypix * np.cos(yawrad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + xpix_rot / scale)
    ypix_translated = np.int_(ypos + ypix_rot / scale)
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped



# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    terrain_img = color_thresh(Rover.img)
    obstacle_img = color_below_thresh(Rover.img)
    rock_img = color_within_thresh_cv(Rover.img)
       
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacle_img * 255
    Rover.vision_image[:,:,1] = rock_img * 255
    Rover.vision_image[:,:,2] = terrain_img * 255

    # 5) Convert map image pixel values to rover-centric coords
    ### filter near ones and discard far off for world map
    obstacle_rovx, obstacle_rovy = rover_coords(obstacle_img)
    
    obstacle_x_filtered, obstacle_y_filtered = filter_pixels_by_distance\
    (obstacle_rovx, obstacle_rovy, Rover.nav_vision_thresh)
    print("obstacle x len: ", len(obstacle_x_filtered), ", \
    obstacle y len: ", len(obstacle_y_filtered))
    
    rock_rovx, rock_rovy = rover_coords(rock_img)
    rock_x_filtered, rock_y_filtered = filter_pixels_by_distance\
    (rock_rovx, rock_rovy, Rover.rock_vision_thresh)
    
    print("rock rover x len: ", len(rock_x_filtered), ", \
    rock rover y len: ", len(rock_y_filtered))
    
    terrain_rovx, terrain_rovy = rover_coords(terrain_img)
    terrain_x_filtered, terrain_y_filtered = filter_pixels_by_distance\
    (terrain_rovx, terrain_rovy, Rover.nav_vision_thresh)
    print("terrain x len: ", len(terrain_x_filtered), ", terrain y len: ", len(terrain_y_filtered))
    
    # 6) Convert rover-centric pixel values to world coordinates
    ## set init state, last pos and last move
    rover_posx, rover_posy = Rover.pos
    if Rover.last_pos is not None and len(Rover.last_pos) > 0:
        lastx, lasty = Rover.last_pos
        Rover.last_move = np.sqrt((rover_posx - lastx) ** 2 + (rover_posy - lasty) ** 2)      
        Rover.is_init = False
    else:
        Rover.last_move = 0
        Rover.is_init = True
    ## reset last pos
    Rover.last_pos = Rover.pos[:]

    rover_yaw = Rover.yaw
    world_size = 200
    scale = 10
   
    
    obstacle_dists, obstacle_angles = to_polar_coords(obstacle_x_filtered, obstacle_y_filtered)
    print('perception : obstacle angles: ', len(obstacle_angles))
    ## if there is any obstacle within filtered region
    if obstacle_angles is not None and len(obstacle_angles) > 0:
        Rover.mean_obstacle_angle = np.mean(obstacle_angles)
        Rover.min_obstacle_angle = np.min(obstacle_angles)
        Rover.max_obstacle_angle = np.max(obstacle_angles)
        Rover.obstacle_angles = obstacle_angles
        Rover.len_obstacles = len(obstacle_angles)
        Rover.obstacle_dists = obstacle_dists
        Rover.mean_obstacle_dist = np.mean(obstacle_dists)
        Rover.min_obstacle_dist = np.min(obstacle_dists )
        Rover.max_obstacle_dist = np.max(obstacle_dists )
    else:
        Rover.mean_obstacle_angle = 0
        Rover.min_obstacle_angle = 0
        Rover.max_obstacle_angle = 0
        Rover.obstacle_angles = None
        Rover.len_obstacles = 0
        Rover.obstacle_dists = None
        Rover.mean_obstacle_dist = 0
        Rover.min_obstacle_dist = 0
        Rover.max_obstacle_dist = 0
    
    
    
    obstacle_wx, obstacle_wy = pix_to_world(obstacle_x_filtered, obstacle_y_filtered,\
                                           rover_posx, rover_posy, rover_yaw,\
                                           world_size, scale)

    rock_wx, rock_wy = pix_to_world(rock_x_filtered, rock_y_filtered,\
                                           rover_posx, rover_posy, rover_yaw,\
                                           world_size, scale)


    terrain_wx, terrain_wy = pix_to_world(terrain_x_filtered, terrain_y_filtered,\
                                           rover_posx, rover_posy, rover_yaw,\
                                           world_size, scale)
    
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obstacle_wy, obstacle_wx, 0] = 255
    Rover.worldmap[rock_wy, rock_wx, 1] = 255
    Rover.worldmap[terrain_wy, terrain_wx, 2] = 255
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    rover_dists, rover_angles = to_polar_coords(terrain_x_filtered, terrain_y_filtered)
    print('perception : rover angles: ', len(rover_angles))
    len_nav = 0
    if rover_angles is not None and len(rover_angles) > 0:
        len_nav = len(rover_angles)
        Rover.nav_dists = rover_dists
        Rover.nav_angles = rover_angles
        Rover.mean_nav_angle = np.mean(rover_angles)
        Rover.min_nav_angle = np.min(rover_angles)
        Rover.max_nav_angle = np.max(rover_angles)
        Rover.mean_nav_dist = np.mean(rover_dists)
        Rover.min_nav_dist = np.min(rover_dists )
        Rover.max_nav_dist = np.max(rover_dists )
        Rover.len_navs = len_nav
        if Rover.len_navs > 0 and Rover.len_obstacles > 0:
            Rover.nav_ratio = Rover.len_obstacles / Rover.len_navs
    else:
        Rover.nav_dists = None
        Rover.nav_angles = None
        Rover.mean_nav_angle = 0
        Rover.min_nav_angle = 0
        Rover.max_nav_angle = 0
        Rover.mean_nav_dist = 0
        Rover.min_nav_dist = 0
        Rover.max_nav_dist = 0
        Rover.len_navs = 0

            
    if Rover.vel < 0:
        Rover.in_backup_mode = True
    else:
        Rover.in_backup_mode = False
         
        ## set sample in vision if there are at least 5 pixels showing a rock. removes noise
    ## around 
    Rover.is_sample_in_vision = len(rock_x_filtered) > 5
    if Rover.is_sample_in_vision:
        rockx = np.mean(rock_x_filtered)
        rocky = np.mean(rock_y_filtered)
        angle_to_rover = np.arctan2(rocky, rockx) * 180 / np.pi
        Rover.steer = np.clip(angle_to_rover, -15, 15)
        dist = np.sqrt((rockx - rover_posx)**2 + (rocky - rover_posy)**2)
        Rover.mean_rock_angle = angle_to_rover
        Rover.mean_rock_dist = dist
        Rover.len_rock = len(rock_wx)
    else:
        Rover.is_sample_in_vision = False
        Rover.mean_rock_angle = 0
        Rover.mean_rock_dist = 0
        Rover.len_rock = 0
    #print('from perception: nav_angles: ', len(Rover.nav_angles), ', nav_dists: ', str(len(Rover.nav_dists)))
    
    
    return Rover