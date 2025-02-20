import os
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT.parents[0] / 'weights'

print('ROOT', ROOT)
print('weights from', WEIGHTS)
WEIGHTS.mkdir(parents=True, exist_ok=True)

weights = {
    'street': {
        'weights': WEIGHTS / 'pLitterStreet_YOLOv5l.pt',
        'url': 'https://github.com/gicait/pLitter/releases/download/v0.0.0-street/pLitterStreet_YOLOv5l.pt'
    },
    'cctv': {
        'weights': WEIGHTS / 'pLitterFloat_800x752_to_640x640.pt',
        'url': 'https://github.com/gicait/pLitter/releases/download/v0.1/pLitterFloat_800x752_to_640x640.pt'
    }
}

colors = [(0, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)] * 20

@torch.no_grad()
def detector(type='street'):
    if type not in weights:
        raise ValueError(f'Invalid type. Valid types: {", ".join(weights.keys())}')

    model_path = weights[type]['weights']
    
    if not os.path.isfile(model_path):
        if weights[type]["uri"]:
            print(f'Downloading model from {weights[type]["url"]}')
            torch.hub.download_url_to_file(weights[type]['url'], str(model_path), hash_prefix=None, progress=True)

        else:
            raise FileNotFoundError(f"Model file {model_path} not found. Please ensure it exists.")

    return torch.hub.load(str(ROOT / 'Yolov5_StrongSORT_OSNet/yolov5'), 'custom', str(model_path), source='local', force_reload=True)

def draw_boxes_on_image(image, detections, min_score_thresh=0.3):
    """ Draws bounding boxes on an image based on model detections. """
    num = 0
    for index, row in detections.iterrows():
        if row['confidence'] < min_score_thresh:
            continue

        num += 1
        x1, y1, x2, y2, class_id = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), int(row['class'])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = colors[class_id % len(colors)]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def run_inference(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    results = model(img)
    detections = results.pandas().xyxy[0]  # Pandas DataFrame

    img_with_boxes = draw_boxes_on_image(img, detections)
    output_path = image_path.replace(".jpg", "_output.jpg")
    cv2.imwrite(output_path, img_with_boxes)
    print(f"Processed image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Plastic Litter Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--type", type=str, choices=['street', 'cctv'], default="street", help="Model type")
    
    args = parser.parse_args()
    model = detector(type=args.type)
    run_inference(args.image, model)
