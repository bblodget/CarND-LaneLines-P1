#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

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


def draw_lines(img, lines, left_color=[255, 255, 0], right_color=[255,0,0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    imshape = img.shape

    left_near = [imshape[1],0]
    left_far = [0,imshape[0]]

    right_near = [0,0]
    right_far = [imshape[1],imshape[0]]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))

            # negative slope is the left lane line
            if slope <0:

                # Track Left line near point
                if x1<left_near[0] and y1>left_near[1]:
                    left_near = [x1,y1]
                if x2<left_near[0] and y2>left_near[1]:
                    left_near = [x2,y2]

                # Track Left line far point
                if x1>left_far[0] and y1<left_far[1]:
                    left_far = [x1,y1]
                if x2>left_far[0] and y2<left_far[1]:
                    left_far = [x2,y2]

            # positive slope is the right lane line
            else:

                # Track Right line near point
                if x1>right_near[0] and y1>right_near[1]:
                    right_near = [x1,y1]
                if x2>right_near[0] and y2>right_near[1]:
                    right_near = [x2,y2]

                # Track Right line far point
                if x1<right_far[0] and y1<right_far[1]:
                    right_far = [x1,y1]
                if x2<right_far[0] and y2<right_far[1]:
                    right_far = [x2,y2]

    # Draw left lane
    cv2.line(img, (left_near[0], left_near[1]), (left_far[0], left_far[1]), left_color, thickness)

    # Draw right lane
    cv2.line(img, (right_near[0], right_near[1]), (right_far[0], right_far[1]), right_color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
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
def find_lanes(src):
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
    vertices = np.array([[(0,imshape[0]),(450, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array( [[0,539],[480,200], [850,539]], dtype=np.int32 )
    roi = region_of_interest(edges, vertices)

    # get the hough lines
    rho = 2
    theta =  np.pi/180
    threshold = 15
    min_line_len = 40
    max_line_gap = 20
    lines = hough_lines(roi, rho, theta, threshold, min_line_len, max_line_gap)

    # Overlay the lines image on top of src image
    overlay = weighted_img(lines, src)

    return overlay


# Main Loop
filenames = os.listdir("test_images/")
print("filenames: ",filenames)

for filename in filenames:
    src = mpimg.imread("test_images/"+filename)
    dst = find_lanes(src)
    filename_png = filename.replace("jpg","png")
    mpimg.imsave("test_images_output/"+filename_png,dst)


