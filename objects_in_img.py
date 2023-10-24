import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
# Define file paths using raw string literals
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Select the Image")
# Ask the user for the template path
template_path = filedialog.askopenfilename(title="Select the Template")
# image_path = r'C:\python\objects_proj\blake.jpg'
# template_path = r'C:\python\objects_proj\blake_glass.jpg'

# Load the image and the template (tattoo image)
image = cv2.imread(image_path)
template = cv2.imread(template_path)

# Check if the images were loaded successfully
if image is None or template is None:
    print("Error: Unable to load one or both images.")
    exit()

# Ensure both images have the same dimensions
image_height, image_width, _ = image.shape
template = cv2.resize(template, (image_width, image_height))

# Convert the template to the same data type as the input image
template = template.astype(image.dtype)

# Match the template using template matching
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Define a threshold to determine if a match is found
threshold = 0.2  # Adjust this value as needed

if max_val >= threshold:
    # Draw a rectangle around the found object
    h, w, _ = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle for the found object
    print("The object (e.g., tattoo) exists in both images!")
    

    # Display the result
    #cv2.imwrite(r'C:\python\objects_proj\result44'+i+'.jpg', image)  # Save the result as 'result.jpg'
else:
    MIN_MATCH_COUNT = 8
    img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # queryImage
    img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # trainImage
    sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
     if m.distance < 0.7*n.distance:
      good.append(m)
    if len(good)>MIN_MATCH_COUNT:
     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
     matchesMask = mask.ravel().tolist()
     h,w = img1.shape
     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
     dst = cv2.perspectiveTransform(pts,M)
     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
     matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    if len(good) > MIN_MATCH_COUNT:
     print("The object (e.g., tattoo) exists in both images!")
     plt.imshow(img3, 'gray')
     plt.show(block=True)
    else:
     print("The object (e.g., tattoo) does NOT exist in both images or the similarity is too low.")
    
    
     #Wprint("Object not found in the image.")

