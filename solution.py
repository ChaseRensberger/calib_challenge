import os
import cv2
import torch
from torchvision import transforms
import numpy as np

def write_tuples_to_file(array_of_tuples, subdirectory, file_name):
    # See if there is a better way to do this
    cwd = os.getcwd()
    file_path = os.path.join(subdirectory, file_name)
    with open(cwd + file_path, 'w') as file:
        for tpl in array_of_tuples:
            line = ' '.join(map(str, tpl)) + '\n'
            file.write(line)

def visualize_lane_lines(frame, ll_seg_out, threshold=0.95):
    lane_line_probs, _ = torch.max(ll_seg_out, dim=1)
    lane_line_mask = lane_line_probs.squeeze().cpu().numpy() > threshold
    lane_line_mask_resized = cv2.resize(lane_line_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

    # Convert binary mask to a 3 channel image to create a color overlay
    lane_line_overlay = np.zeros_like(frame)
    lane_line_overlay[:, :, :] = 0  # Reset all channels to zero - No tint
    lane_line_overlay[lane_line_mask_resized == 1, :] = [0, 255, 0]  # Green color for the lane lines

    # Blend the overlay with the original frame
    visualized_frame = cv2.addWeighted(frame, 1, lane_line_overlay, 0.4, 0)

    return visualized_frame

    # lines = cv2.HoughLinesP(lane_line_mask_resized.astype(np.uint8), 1, np.pi/180, threshold=100, minLineLength=1000, maxLineGap=5)

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # else:
    #     print("No lines detected")

    # return frame

def processVideo(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_tensor = transform(frame).unsqueeze(0)

        if torch.cuda.is_available():
            frame_tensor = frame_tensor.to('cuda')
            model.to('cuda')

        
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(frame_tensor)
        
        lane_lines_frame = visualize_lane_lines(frame, ll_seg_out)
        cv2.imshow("frame", lane_lines_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processVideo("labeled/0.hevc", model, transform)

if __name__ == "__main__":
    main()
