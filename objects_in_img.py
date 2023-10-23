import cv2
import numpy as np

# Load the image and the template (tattoo image)
#print("OpenCV version:", cv2.__version__)
image = cv2.imread('C:\python\objects_proj\tom.jpg')  # Replace 'image.jpg' with your image file
template = cv2.imread('C:\python\objects_proj\tom_tatt.jpg')  # Replace 'tattoo.jpg' with your tattoo template file
image_height, image_width, _ = image.shape
template = cv2.resize(template, (image_width, image_height))

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

    # Display the result
    cv2.imwrite('result.jpg', image)  # Save the result as 'result.jpg'
else:
    print("Object not found in the image.")
