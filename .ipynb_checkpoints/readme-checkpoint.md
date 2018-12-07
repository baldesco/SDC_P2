# Self-Driving Car Engineer Nanodegree

## Project 2: Advanced Lane Lines Finding

Author: Eduardo Escobar

This project is a continuation of project 1. Here, more advanced computer vision techniques are used to improve the process of detecting lane lines in the road.

The code for this project can be found at the **P2.ipyb** notebook.

## 1. Camera calibration

The first thing to do is to get the distortion coefficients and the camera matrix for the camera that is recording the images in the car. Having these values will allow us to correct the distortion in the images.

<img src="media/Calibration.png" width="700">
___________________________________________________________________________________________________________________________

## 2. Undistort and threshold images

In this section, the lane line images are undistorted using the output from section 1, and then a combination of color and gradient thresholds are applied to obtain a binary image detecting the lane lines.

### 2.1 Color thresholding

Here I transform the undistorted image to the HLS color space, since all 3 channels in this space (hue, lightness and saturation) are useful options for thresholding the image, either by color or by gradients.

<img src="media/hls.png" width="700">

### 2.2 Gradient thresholding

Since the lane lines are mostly vertical, a gradient in the x direction is considered. For this, the `cv2.Sobel` function is used.

The gradient is applied over the lightness channel, over the HLS color space. I chose this channel (over, for example the 'red' channel or the grayscale image) because it seems to give better results.

<img src="media/x_gradient.png" width="700">

### 2.3 Combination of color and gradient thresholding

In order to create a binary mask of the lines, the hue, saturation and x gradient thresholds are combined. Since there are 3 different criteria to detect if there is a line or not, I chose to make a vote, so if at least 2 of the thresholds determine that a pixel in the image contains relevant information, then this pixel is activated in the combined binary mask.

<img src="media/binary_mask.png" width="700">
___________________________________________________________________________________________________________________________

## 3. Perspective transform

After a binary image has been created with the help of color and gradient thresholds, it is necessary to perform a perspective transform, in order to obtain a top view of the region of interest.

For doing this, source and destinations image points must be defined. I used the points showed in the file *example_writeup.pdf* as a guide for this, changing the source points location just a little.

<img src="media/warped.png" width="700">
___________________________________________________________________________________________________________________________

## 4. Line detection

This section deals with the problem of finding a 2nd order equation to approximate the location of both the left and right lane lines.

This task can be divided into smaller actions, shown below.

### 4.1 Define a class to receive the characteristics of each line detection

This is a suggestion given by Udacity, and it helps in keeping track of previous parameters of the lane lines. This can be really useful when the algorithm has trouble finding the lines in a specific frame or group of frames in a video/recording. 

The class is provided with several attributes and two methods, to initialize and update the lines’ parameters.

### 4.2 Find the lines "from scratch"

Two functions are defined to detect the lines. The first function implements a histogram/sliding windows approach to detect where in the warped image are the lines located (x and y coordinates). This function receives the binary warped image and returns the list of x and y coordinates for the points that fall into both the left and right lines. 

The second function receives these lists of coordinates and fits a second-degree polynomial for each line.

### 4.3 Find the lines from prior

Since the lines don't necessarily move a lot from frame to frame, the results found in a previous frame can be used as a starting point for the next one. This is more time efficient because the algorithms look for the lines in a smaller area of the image.

Two functions are defined, very similar to the ones defined in section 4.2. The main difference is the function `search_around_poly`, which does not perform a sliding window search, but only search inside a defined margin around the lane line found on a previous frame.

### 4.4 Find the curvature of the lines and the distance to the center of the lane

When the polynomials for both lines have been found, two more values can be calculated: the curvature for each of the lines and the distance from the center of the car to the center of the lane.

These values need to be given in a distance metric, such as meters or miles, in order to be interpretable and useful. Taking this into account, it is necessary to include the conversion from pixels to the defined distance metric in the calculations.

### 4.5 Create and update the line objects

Two objects of the *Line* class are created; one for each line. Then, the polynomial for each line is calculated, a simple sanity check is performed and the *Line* objects are updated accordingly.

### 4.6 Graph the results

Finally, the two lines can be combined to create and plot a polygon in the undistorted image. In order to do this, it is necessary to:

1. Create a new image, *warp_zero*.
2. Create a polygon with the points from the two lines that were found, and plot it in *warp_zero*.
3. Apply the inverse perspective matrix to *warp_zero*, in order to take it to the original point of view.
4. Use the `cv2.addWeighted` function to add the *warp_zero* image to the *undistorted* image.
5. Plot the result

___________________________________________________________________________________________________________________________

## 5. Putting it all together

We now have all the steps necessary to find lane lines in images and videos. The last thing to do before testing it is to combine all the defined functions into a pipeline.

This is the code for the pipeline:

```
def find_lanes(image, mtx, dist, left_line, right_line):
    ## 1. Undistort image #
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    
    ## 2. Convert to HLS color space and create color channels #
    hls_img = cv2.cvtColor(undistorted,cv2.COLOR_RGB2HLS)
    H = hls_img[:,:,0]
    L = hls_img[:,:,1]
    S = hls_img[:,:,2]
    
    ## 3. Create the 3 separate binary images #
    # Saturation
    sat_thresh = (90, 255)
    binary_sat = np.zeros_like(S)
    binary_sat[(S > sat_thresh[0]) & (S <= sat_thresh[1])] = 1
    # Hue
    hue_thresh = (15, 100)
    binary_hue = np.zeros_like(H)
    binary_hue[(H > hue_thresh[0]) & (H <= hue_thresh[1])] = 1
    # X gradient
    x_thresh = (20, 100)
    binary_x = np.zeros_like(scaled_sobel)
    binary_x[(scaled_sobel >= x_thresh[0]) & (scaled_sobel <= x_thresh[1])] = 1
    # Combined
    combined_binary = np.zeros_like(binary_x)
    combined_binary[(binary_x + binary_sat + binary_hue) > 1] = 1
    
    ## 4. Warp the combined binary image #
    # Define the source and destination points, to perform the perspective transform
    img_size = (undistorted.shape[1], undistorted.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 50, img_size[1] / 2 + 91],
        [((img_size[0] / 6) - 25), img_size[1]],
        [(img_size[0] * 5 / 6) + 30, img_size[1]],
        [(img_size[0] / 2 + 50), img_size[1] / 2 +91]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    M, Minv, warped = warp_image(combined_binary, src, dst)
    
    ## 5. Find the lines # 
    if left_line.reset | right_line.reset:
        left_fit, right_fit, left_fitx, right_fitx, ploty = full_fit_polynomial(warped)
    else:
        left_fit, right_fit, left_fitx, right_fitx, ploty = search_around_poly(warped, left_line.best_fit, right_line.best_fit)
    # Calculate values in meters (curvatures and distance to center)
    left_curverad, right_curverad = measure_curvature_real(ploty,left_fit, right_fit)
    dist_center = calculate_offset(undistorted.shape, left_fitx, right_fitx)
    max_std = 50
    last_frames = 10
    reset_frames = 5
    update = True
    if len(left_fitx) < undistorted.shape[0]:
        update = False
    else:
        lines_distance = np.abs(left_fitx - right_fitx)
        if np.std(lines_distance) > max_std:
            update = False
    left_line.update_line(update=update,lane_inds=left_fitx,coefs=left_fit,curvature=left_curverad,
                          dist_center=dist_center,n_last=last_frames,n_reset=reset_frames)
    right_line.update_line(update=update,lane_inds=right_fitx,coefs=right_fit,curvature=right_curverad,
                           dist_center=dist_center,n_last=last_frames,n_reset=reset_frames)
    
    ## 6. Graph the polygon, add it to the undistorted image and return the outcome
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    try:
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    except:
        pass
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    # Add text to the image, to inform the curvature and the distance from center
    try:
        text1 = 'Radius of curvature: {0:.2f} meters.'.format(left_line.radius_of_curvature)
        rel_pos = 'left' if left_line.line_base_pos > 0 else 'right'
        text2 = 'The vehicle is {0:.3f} meters to the {1} from the center.'.format(np.abs(left_line.line_base_pos),rel_pos)
        cv2.putText(result,text1, (60,60),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
        cv2.putText(result,text2, (60,100),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    except:
        pass
    
    return result
```

___________________________________________________________________________________________________________________________

## 6. Testing the pipeline

Everything is ready now to test the pipeline in images and videos.

### 6.1 Test the pipeline in images

First, I will use the pipeline on one image. Then, all the test images will be entered to produce the output images.

<img src="media/output.png" width="700">

In 3 of the 8 images the algorithm does not find the lines successfully. Let’s see if this can be improved in the video, with the tracking of past frames.

### 6.2 Test the pipeline in videos

First, I will use the pipeline on one video. Then, all the test videos will be entered to produce the outputs. 

<img src="media/video.gif" width="700">
___________________________________________________________________________________________________________________________

## 7. Results on the videos

The algorithm works well on the project video, regular on the challenge video and very bad on the harder challenge video. It definitely needs to be adjusted, and there are many possibilities: 

1. In the creation of the binary image other combinations of criteria and values of thresholds can be considered.
2. The *Line* class can be improved, including more or complementing its existing attributes and methods.
3. The process of finding the lines can use other options, like convolution or Hough transformations in order to improve the quality of the lines that are found.

These are only some options that can be studied. These options could not be studied in depth for reasons of time.
