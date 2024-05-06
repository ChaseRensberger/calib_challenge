import cv2

file_path = "labeled/0.hevc"
cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error reading frame")
    exit()

cv2.imshow("First Frame", frame)
cv2.waitKey(0)

output_file = "frame.jpg"
cv2.imwrite(output_file, frame)
print(f"Frame saved as {output_file}")

cap.release()
cv2.destroyAllWindows()