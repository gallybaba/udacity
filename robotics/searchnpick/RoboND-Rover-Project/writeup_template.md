## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
Here is an example of how to include an image in your writeup.
__Perspective Transform__
**Original** image 
![original][original.png]
**warped** image
![warped][warped.png]
* Above image shows how perpective transformation takes place. The position on origina image gets shown in warped image on a tiny scale this showing a very high level birds eye view. This transformation is done using *cv2* libraries

__Color Thresholding__
* Below image filters out terrain color and puts everything else as 0. This is done by comparing every pixel in image against a threshold. If that pixel is above  the threshold, it gets assigned 1.M0bileM0ney$

![terrain color threshed][terrain_color_thresh.png
* Obstacle color threshold just negative of terrain
![obstacle color thresh][obstacle_color_thresh.png]
* Rock color thresh is done using a range of color. If the pixel compared sits within that range, it will be set to 1. All else will be 0.
![rock color thresh][rock_color_thresh.png]
* Below is a rock threshold just using cv2 library. It is more precise.
![rock color thresh using cv2][rock_color_thresh_cv.png]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
And another! 

* Following steps were taken in process_image to end up in a movie video
   1) perspective transformation was done two reasons. One to put that image on output image and two, to get a choice to put either warped or flat image on rover vision i.e. the image from Rovers perspective.
   2) color thresh. This is done to identify navigable terrain, obstacles and rock samples. this is a very important step as this decision creates the and controls the navigation tree later on. The logic here is simple.
   any pixel with RGB > 160, 160, 160 is terrain
   any pixel with RGB < 160, 160, 160 is obstacle
   any pixel with RGB in range 110, 110, 5 and 255, 255, 90 is rock. This gets cleaner by using cv2.
   3) once image is threshold with three different copies of original image each indicating the three items discussed above, next step is to convert these pixels to rover centric coordinates. Rover will know these coordinates according to its own perspective. Here we do two things. a) reverse Y axis. b) Swap X and Y. Thats how the frame looks from Rover 's camera.
   4) Then we take these pixels and convert them world coordinates. This is for the world map to know what is where. To do this we do two things. We rotate the pixels from these images according to current Rover position and its yaw angle. Yaw dictates how much Rover is away from X axis. This angle is in degrees so first we convert it into radians using pi radians == 180 degrees. Now since rover scale and world scale are different, we add these rotated pixels to current rover position using the given scale. That projects the given pixel into world map.
   5) Thats it! all we do now place certain things on output image including the warped image, world map and the travel map that shows where all are you going. I could add the rover image and the rover vision image as well but couldn't come up with proper size and locations on output image. I did create vision image as in how the world looks like from Rover's perspective. 

![alt text][image2]
### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
**Perception**
* This is the step that senses the world and fills in state in RoverState class.
It does all of the above and some more.
1) It calculates angles including angles w.r.t min and max navigational distances. The angles that we get by converting terrain pixels to a polar coordinate, we also get how far is that pixel from rover. This min / max angle based min/max distance will then drive Rover. Some times then obstacles and navigation are straight ahead, staying on mean won't help so we create an offset to the mean towards the max distance angle so that rover moves more along where it finds more navigation. This prevents rover to keep going in circles on a circular terrain. Keeping an offset also avoids drastic turns and keeps fidelity on higher side.
2) It also calculates rock positions. From its position it calculates polar coordinates and angle / distances. This information could be used to ) identify rocks while rover is moving, and b) steer towards it in order to collect them. Although in current state collection is turned off. I do intend to improve it later on.
3) Additional states are calculated which are not used in current decision tree but will help improve project later.
**Decision**
Simple decision tree based on 1) rover mode, 2) rover velocity 3) how much navigational terrain is left along with distances, 4) how many obstacle pixels we see along with its distances.
1) Basic idea is while rover is moving forward it can be ready to move on, stuck or ready to stop. While in stop mode, it could be ready to stop, move on and stuck.
2) If stuck, it tries to unstuck itself by steering away from obstacle.
3) It also checks pixel lengths from terrain image captured while color thresholding in perception step. Based on certain threshold as defined in is_clear method, it will decide to completely steer away or clip to the mean with some offset.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  
Fastest Mode:
Ubuntu 16 Xenial 64 Bit
Average FPS about 15. (Default setting)
How it worked?
Rover moves around open terrain and turns away from most obstacles in time (is_clear). It also unstucks itself from most situations like wall, big stones. Having said that it is far from ideal. Sometimes it gets stuck on rocks due to either it doesn't find it in time or it is close enough such that camera cannot see it but it gets stuck on the side wheels. Under these situations it will keep trying to spin and free itself.

What would I do Next to Improve:
1) Put in a time based state that tells whether rover is stuck. It will sense how much it moved from last position. I am actually calculating this state but haven't used it yet.
2) Intelligent backing up. Put in some randomization while backing out so it can see what around it. Also put in auto stop when backing out when it sees enough terrain. I have logic to see that but haven't plugged in yet.
3) Do not revisit portion of map. I tried to subtract current position from terrain image so rover may think it is not a terrain anymore but it didn't help much. It needs a complicated logic that will identify regions travelled and will not go there again given a choice of another terrain near by.
4) Improve fidelity. I tried NOT updating vision image when rolls and pitches are high but then it interferes with navigation as it wouldn't know where to steer if it is close to a wall / obstacle.
5) Once a rock is identified, it should be intelligent enough to avoid obstacles and turn around or try again next time.


**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]


