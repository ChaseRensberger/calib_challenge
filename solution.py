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
   pass 

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
        
        # lane_line_frame = visualize_lane_lines(frame, ll_seg_out)
        cv2.imshow("frame", frame)
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
