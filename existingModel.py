import cv2
import torch
from torchvision import transforms
import numpy as np

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
        
        cv2.imshow("frame", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()