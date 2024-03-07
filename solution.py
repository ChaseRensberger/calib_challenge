import cv2 as cv
import numpy as np
import math
import os

def write_tuples_to_file(array_of_tuples, subdirectory, file_name):
    # See if there is a better way to do this
    cwd = os.getcwd()
    file_path = os.path.join(subdirectory, file_name)
    with open(cwd + file_path, 'w') as file:
        for tpl in array_of_tuples:
            line = ' '.join(map(str, tpl)) + '\n'
            file.write(line)

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
    if denom == 0:
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return int(x), int(y)

def processVideo(input_path, output_path, output_name, focal_length):
    cap = cv.VideoCapture(input_path)
    output_offsets = []
    mult = 1
    current_frame_count = 0
    line_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        height = frame.shape[0]
        width = frame.shape[1]

        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 3),
            (width, height),
        ]

        gray_image = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        cannyed_image = cv.Canny(gray_image, 100, 200)
        cropped_image = region_of_interest(
            cannyed_image,
            np.array(
                [region_of_interest_vertices],
                np.int32
            ),
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
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                    if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                        continue
                    if slope <= 0: # <-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else: # <-- Otherwise, right group.
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
        min_y = frame.shape[0] * (3 / 5) # <-- Just below the horizon
        max_y = frame.shape[0] # <-- The bottom of the image
        
        if (len(left_line_x) > 0 and len(left_line_y) > 0) and len(right_line_x) > 0 and len(right_line_y) > 0:
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))

            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
            line_image = draw_lines(
                frame,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]],
                thickness=5,
            )

            vanishing_point = find_intersection(
                (left_x_start, max_y, left_x_end, min_y),
                (right_x_start, max_y, right_x_end, min_y)
            )

            if vanishing_point is not None:
                cv.circle(line_image, vanishing_point, 10, (0, 255, 0), thickness=-1)

            center_x = width / 2
            center_y = height / 2

            disp_x = vanishing_point[0] - center_x
            disp_y = vanishing_point[1] - center_y

            yaw_angle = math.atan(disp_x / focal_length)
            pitch_angle = math.atan(disp_y / focal_length)
            output_offsets.append(("{:e}".format(abs(pitch_angle)), "{:e}".format(abs(yaw_angle))))
            print(f"Estimated Yaw: {yaw_angle} degrees")
            print(f"Estimated Pitch: {pitch_angle} degrees")

        if cv.waitKey(50) & 0xFF == ord('q'):
            break
        if line_image is None:
            line_image = np.copy(frame)
        cv.imshow("frame", line_image)

        if len(output_offsets) != current_frame_count + 1:
            try:
                for i in range(mult):
                    output_offsets.append(output_offsets[-1])
                if mult > 1:
                    mult = 1
            except:
                mult = 2

        current_frame_count += 1

    cap.release()
    cv.destroyAllWindows()
    write_tuples_to_file(output_offsets, output_path, output_name)

def main():
    for i in range(5):
        processVideo(input_path=f"labeled/{i}.hevc", output_path="/labeled-predictions", output_name=f"{i}.txt", focal_length=910)
        


if __name__ == "__main__":
    main()