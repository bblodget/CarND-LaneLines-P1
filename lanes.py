#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lane(img, x1, y1, x2, y2, miny, maxy, color, thickness):
    """
    Draw a lane line given a sample line that has the
    median slope of all the lane lines.  Extrapolate
    the full line from bottom to top of roi.
    """

    # Increase miny to keep lines from touching
    miny = miny + 20

    # Calculate the current slope and intercept
    m = ((y2-y1)/(x2-x1))
    b = y1 - (m*x1)

    # Recalculate points to draw left lane from bottom to top of roi
    y1 = maxy
    x1 = int((y1 - b)/ m)
    y2=miny
    x2 = int((y2 - b)/m)

    # Draw left lane
    if y1>=miny and y2>=miny:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def by_slope(line):
    """
    This function is used by the sort() method
    to sort a list of lines by slope.
    """
    x1,y1,x2,y2 = line[0]
    return (y2-y1)/(x2-x1)


def draw_lanes(img, lines, left_color=[255, 255, 0], right_color=[255,0,0], thickness=10):
    """
    This function extrapolates the lane markers and draws them.
    It marks the left and right lane with two different colors.
    By default the left lane is drawn in yellow and the 
    right lane is drawn in red.
    """

    maxy = img.shape[0] # height of image

    # Lane points need to be greater than miny
    miny = 310

    # Create lists to hold left and right lines.
    left_lines = []
    right_lines = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))

            # negative slope is the left lane line
            if slope <0:

                # Check that slope is in valid range
                # and points y coord is greater than miny
                #if slope > -0.8 and slope < -0.6 and y1>miny and y2>miny:
                if slope > -0.9 and slope < -0.5 and y1>miny and y2>miny:
                    left_lines.append(line)


            # positive slope is the right lane line
            else:

                # Check that slope is in valid range
                # and points y coord is greater than miny
                #if slope > 0.56  and slope < 0.75 and y1>miny and y2>miny:
                if slope > 0.5  and slope < 0.8 and y1>miny and y2>miny:
                    right_lines.append(line)

    if len(left_lines) > 0:
        # Sort the left lines by slope
        left_lines.sort(key=by_slope)


        # find the middle element
        middle = int((len(left_lines)/2.0)+0.5)-1
        x1,y1,x2,y2 = left_lines[middle][0]

        # Draw left lane
        draw_lane(img, x1, y1, x2, y2, miny, maxy, left_color, thickness)

    if len(right_lines) > 0:
        # Sort the right lines by slope
        right_lines.sort(key=by_slope)

        # find the middle element
        middle = int((len(right_lines)/2.0)+0.5)-1
        x1,y1,x2,y2 = right_lines[middle][0]

        # Draw right lane
        draw_lane(img, x1, y1, x2, y2, miny, maxy, right_color, thickness)


def hough_lines2(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    Modified to call draw_lanes instead of draw_lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lanes(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def find_lanes_pipeline(src):
    """ 
    This function implements the pipeline
    to draw lane lines on the specified src image.
    """

    imshape = src.shape

    # Convert image to gray scale
    gray = grayscale(src)

    # Run a gaussian on src image
    blur = gaussian_blur(gray,5)

    # get edges
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur, low_threshold, high_threshold)

    # get region of interest
    vertices = np.array([[(0,imshape[0]),(450, 312), (490, 312), (imshape[1],imshape[0])]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # get the hough lines
    rho = 2
    theta =  np.pi/180
    threshold = 15
    min_line_len = 35
    max_line_gap = 12
    lines = hough_lines2(roi, rho, theta, threshold, min_line_len, max_line_gap)

    # Overlay the lines image on top of src image
    overlay = weighted_img(lines, src)

    return overlay

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = find_lanes_pipeline(image)

    return result


#
# Main
#

# Draw lane lines on the test images
# and save them to the test_images_output
# directory.
filenames = os.listdir("test_images/")
print("filenames: ",filenames)

for filename in filenames:
    src = mpimg.imread("test_images/"+filename)
    dst = find_lanes_pipeline(src)
    filename_png = filename.replace("jpg","png")
    mpimg.imsave("test_images_output/"+filename_png,dst)


# Draw lane lines on the test videos
# and save them to the test_videos_output
# directory.
filenames = os.listdir("test_videos/")
print("filenames: ",filenames)

for filename in filenames:
    src_clip = VideoFileClip("test_videos/"+filename)
    dst_clip = src_clip.fl_image(process_image) #NOTE: this function expects color images!!
    dst_clip.write_videofile("test_videos_output/"+filename)


