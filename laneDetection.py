import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2]
    channel_count = 2
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

cap = cv.VideoCapture("labeled/0.hevc")
ret, image = cap.read()

height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 3),
    (width, height),
]


plt.figure()
plt.imshow(image)
plt.show()


gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
cannyed_image = cv.Canny(gray_image, 100, 200)
cropped_image = region_of_interest(
    cannyed_image,
    np.array([region_of_interest_vertices], np.int32)
)
lines = cv.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

line_image = draw_lines(image, lines) # <---- Add this call.
plt.figure()
plt.imshow(line_image)
plt.show()