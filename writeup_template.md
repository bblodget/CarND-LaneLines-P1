# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
1. Convert the image to gray scale
2. Run a gaussian_blur of size 5 on the gray scale image
3. Use canny filter to get the edges
4. Select a trapezoidal region of interest that contains the lane lines
5. Get the hough lines

Instead of modifying the original hough_lines() and draw_lines() functions I made copies of them and named them hough_lines2() and draw_lanes().  The function hough_lines2() was modified from the original to call draw_lanes() instead of draw_lines().

The function draw_lanes() has an additional color parameter from the original draw_lines(). This way the left lane and right lane can be draw with different colors.  There is a loop in draw_lanes() that sorts the hough lines into two lists.  Hough lines that have a negative slope are appended to the left_lines list, and positive slopes are appended to the right_lines list. There is also a check to make sure the slope of the hough line is "reasonable" before being added to the final list.  This check is done by making sure the slope is within a defined range.  The left and right lanes are then drawn by calling a helper function called draw_lane().

The draw_lane() functions get passed the filtered hough lines that define one lane.  It then sorts all of these lines by slope.  It then selects the hough line that has the median slope.  The theory is that the median slope will match the actual slope of the lane line pretty well.  Taking a sample point from the median slope line we calculate the the y intercept.  Given the slope (m) and y intercept (b) we can solve for x values using, x = (y-b)/m.  I solve for x at y=bottom of the image, and at y=top of trapezoidal region of interest.  Using these two points the lane line is drawn.

Lastly the image with the lane lines is merged (overlayed) with the original source image.  Here is a sample resulting image:

![output image][./test_images_output/solidYellowCurve.png]


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming is the filtering of the hough lines could result in no lines left to be used to draw the lane.  If this happens no line is drawn.  I thought I could keep a running average of the lanes and if a frame has no hough lines I could use the previous average lane line.  But at this stage I think it is better to see how well the raw algorithm works.  This issue probably could be reduced with better tuning of the pipeline parameters.

My current pipeline also does not work great with the challenge video.  It does OK until shadows appear on the road.  It also does not follow the lanes around the curve of the road.


### 3. Suggest possible improvements to your pipeline

A possible improvement could be to do some averaging between frames to reduce the lane line dancing.  Make the lane lines move more smoothly.  

I think there is room to make improvements with better tuning of the canny and hough line parameters.  

I also think that maybe doing some color filtering in HSV color space, before going to gray scale,  might help with issues like shadows on the road.

One other idea is to draw many conected lines to be able to follow the lane into a curve.  One could select many y values, and select a hough line that contains that y value.  Then you could draw lines between the midpoints of those select hough lines.


