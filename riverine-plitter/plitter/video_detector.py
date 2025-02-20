import os
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from detector import detector, draw_boxes_on_image
from preprocess_water import preprocess_water_image  # Our water preprocessing function

def is_same_detection(det1, det2, score_diff_threshold=0.1, distance_threshold=30):
    """
    Determine if two detections are the same based on class, confidence difference, 
    and center distance.
    """
    if int(det1['class']) != int(det2['class']):
        return False
    if abs(det1['confidence'] - det2['confidence']) > score_diff_threshold:
        return False
    cx1 = (det1['xmin'] + det1['xmax']) / 2
    cy1 = (det1['ymin'] + det1['ymax']) / 2
    cx2 = (det2['xmin'] + det2['xmax']) / 2
    cy2 = (det2['ymin'] + det2['ymax']) / 2
    distance = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
    return distance < distance_threshold

def process_video(input_path, output_path, model_type, frame_skip=1, show_live=True, 
                  save_screenshots=False, preprocess=False, save_preprocessed=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = detector(model_type).to(device).half()  # Use GPU & FP16 for faster inference

    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}")
        return

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get original video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define target resolution for faster inference
    target_size = (640, 640)

    # Final output video writer (resized back to original resolution)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

    # If preprocessing is enabled, create a "preprocessed" folder in the output directory,
    # and use it to save the preprocessed video (with "_preprocessed" appended) and screenshots.
    if preprocess:
        preprocessed_dir = output_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_video_path = preprocessed_dir / (Path(output_path).stem + "_preprocessed.mp4")
        pre_out = cv2.VideoWriter(str(preprocessed_video_path), fourcc, fps, target_size)
        print(f"🖥 Preprocessed video will be saved at: {preprocessed_video_path}")
        # If saving screenshots, they will also be saved in a "screenshots" subfolder within preprocessed_dir.
        if save_screenshots:
            screenshots_dir = preprocessed_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
    else:
        pre_out = None
        if save_screenshots:
            screenshots_dir = output_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames} | FPS: {fps} | Original Resolution: {orig_width}x{orig_height}")

    start_time = time.time()
    unique_detections = 0
    frame_count = 0
    tracked_detections = []  # Detections from the previous frame

    if show_live:
        cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)

    with tqdm(total=total_frames // frame_skip, desc="Processing Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip frames for optimization

            # Resize frame for faster inference
            frame_resized = cv2.resize(frame, target_size)

            # If preprocessing is enabled, apply it to the resized frame
            if preprocess:
                frame_resized = preprocess_water_image(frame_resized)
                if pre_out is not None:
                    pre_out.write(frame_resized)

            
            # Run inference on the tensor input
            results = model(frame_resized)
            detections_df = results.pandas().xyxy[0]

            # Convert detections DataFrame to a list of detection dictionaries
            current_detections = [row.to_dict() for _, row in detections_df.iterrows()]
            
            # Deduplicate: count new detections compared to previous frame
            new_detections = 0
            for det in current_detections:
                if not any(is_same_detection(det, prev_det) for prev_det in tracked_detections):
                    new_detections += 1
            unique_detections += new_detections
            tracked_detections = current_detections

            # Draw bounding boxes on the resized frame
            processed_frame = draw_boxes_on_image(frame_resized, detections_df)

            # Resize processed frame back to original resolution for final output video
            processed_frame_full = cv2.resize(processed_frame, (orig_width, orig_height))

            # Dynamically create an overlay for text:
            line1 = f"Unique Detections: {unique_detections}"
            line2 = f"Frame Detections: {len(current_detections)} (New: {new_detections})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text1_width, text1_height), baseline1 = cv2.getTextSize(line1, font, font_scale, thickness)
            (text2_width, text2_height), baseline2 = cv2.getTextSize(line2, font, font_scale, thickness)
            margin = 10
            x0, y0 = 20, 20
            x1 = x0 + max(text1_width, text2_width) + 2 * margin
            y1 = y0 + text1_height + text2_height + baseline1 + baseline2 + 3 * margin

            overlay = processed_frame_full.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, processed_frame_full, 1 - alpha, 0, processed_frame_full)

            text_x = x0 + margin
            text_y = y0 + text1_height + margin
            cv2.putText(processed_frame_full, line1, (text_x, text_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            text_y += text1_height + baseline1 + margin
            cv2.putText(processed_frame_full, line2, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            if save_screenshots and new_detections > 0:
                screenshot_filename = screenshots_dir / f"frame_{frame_count:06d}_new_{new_detections}_preprocessed.jpg"
                cv2.imwrite(str(screenshot_filename), processed_frame_full)

            if show_live:
                cv2.imshow("Live Detection", processed_frame_full)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty("Live Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break

            out.write(processed_frame_full)
            pbar.update(1)

    cap.release()
    out.release()
    if pre_out is not None:
        pre_out.release()
    cv2.destroyAllWindows()

    elapsed_time = time.time() - start_time
    print(f"✅ Final processed video saved at: {output_path}")
    if pre_out is not None:
        print(f"✅ Preprocessed video saved at: {preprocessed_video_path}")
    print(f"📊 Unique Detections: {unique_detections}")
    print(f"⏳ Total Processing Time: {elapsed_time:.2f} seconds")
    print(f"Total Frames Processed: {frame_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Plastic Litter Detection on Video with Optional Preprocessing")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save final output video")
    parser.add_argument("--type", type=str, choices=['street', 'cctv', 'river'], default="cctv", help="Model type")
    parser.add_argument("--frame_skip", type=int, default=1, help="Number of frames to skip (higher = faster)")
    parser.add_argument("--live", action="store_true", help="Show live detection")
    parser.add_argument("--save_screenshots", action="store_true", help="Save screenshots of detection frames")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess frames for water conditions")
    parser.add_argument("--save_preprocessed", action="store_true", help="Save preprocessed video/photo (if preprocessing is enabled)")
    
    args = parser.parse_args()

    if not args.preprocess:
        ans = input("Do you want to preprocess the input for water conditions? (y/n): ").strip().lower()
        preprocess_flag = True if ans == 'y' else False
    else:
        preprocess_flag = True

    if preprocess_flag:
        if not args.save_preprocessed:
            ans = input("Do you want to save the preprocessed video/photo? (y/n): ").strip().lower()
            save_preprocessed_flag = True if ans == 'y' else False
        else:
            save_preprocessed_flag = True
    else:
        save_preprocessed_flag = False

    if not args.live:
        ans = input("Do you want to show live detection? (y/n): ").strip().lower()
        show_live = True if ans == 'y' else False
    else:
        show_live = True

    if not args.save_screenshots:
        ans = input("Do you want to save screenshots of detections? (y/n): ").strip().lower()
        save_screenshots = True if ans == 'y' else False
    else:
        save_screenshots = True

    process_video(args.input, args.output, args.type, args.frame_skip, show_live, 
                  save_screenshots, preprocess_flag, save_preprocessed_flag)
