import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

def visualize_lane_lines(frame, ll_seg_out, threshold=0.95):
    # Assuming ll_seg_out is a tensor of shape [1, 2, H, W]
    # We'll take the max of the two channels to get the best prediction for lane lines
    lane_line_probs, _ = torch.max(ll_seg_out, dim=1)  # Reduce across the channel dimension
    lane_line_mask = lane_line_probs.squeeze().cpu().numpy() > threshold
    
    # Resize the binary mask to match the original frame size
    lane_line_mask = cv2.resize(lane_line_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

    # Convert binary mask to a 3 channel image to create a color overlay
    lane_line_overlay = np.zeros_like(frame)
    lane_line_overlay[:, :, :] = 0  # Reset all channels to zero - No tint
    lane_line_overlay[lane_line_mask == 1, :] = [0, 255, 0]  # Green color for the lane lines

    # Blend the overlay with the original frame
    visualized_frame = cv2.addWeighted(frame, 1, lane_line_overlay, 0.4, 0)

    return visualized_frame

def main():
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture("labeled/0.hevc")

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
        
        lane_line_frame = visualize_lane_lines(frame, ll_seg_out)
        
        cv2.imshow("frame", lane_line_frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()