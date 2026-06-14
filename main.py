import cv2 
from src.detection.detector import VehicleDetector
from src.detection.tracker import Tracker

def main(video_path, output_path): 
    # prisha note: logic for opening video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"Error: cannot open video {video_path}")
        return 

    # prisha note: get video properties -- need to do this so the output matches the input
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # prisha note: setting up the video writer for mp4 output
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # prisha note: initialize the detector and tracker
    detector = VehicleDetector()
    tracker = Tracker()

    frame_index = 0
    while True: 
        ret, frame = cap.read()
        if not ret: 
            break 

        # prisha note: first have to detect vehicles in the current frame
        detections = detector.detect(frame)

        # prisha note: then update the tracker with the this frame's detections
        detections = tracker.update(detections, frame_index)

        # prisha note: drawing boxes + labels on the frame yayyy
        annotated_frame = detector.annotate_frame(frame, detections)

        # prisha note: now writing the track ID + speed on top of the boxes
        for det in detections: 
            if 'track_id' not in det: 
                continue
            x1, y1, x2, y2 = det['bbox']
            track_id = det['track_id']
            speed = tracker.get_track_speed(track_id, fps = fps)
            label = f"ID: {track_id} | {speed: .1f} km/h"
            cv2.putText(annotated_frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        out.write(annotated_frame)
        frame_index += 1

        if frame_index % 30 == 0: 
            print(f"Processed {frame_index} frames")
        
    cap.release()
    out.release()
    print(f"Finished processing. Output saved to {output_path}")

if __name__ == "__main__":
    main("data/raw/kaggle.mp4", "data/annotated/annotated_kaggle.mp4")