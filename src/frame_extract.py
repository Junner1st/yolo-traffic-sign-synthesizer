import cv2
import os

def extract_frames(video_path, output_dir, interval=0.5):
    """
    Extract frames from a video every 'interval' seconds and save them to output_dir.
    
    :param video_path: Path to the video file
    :param output_dir: Directory to save the extracted frames
    :param interval: Time interval in seconds between frames
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get FPS from video.")
        return
    
    frame_interval = int(fps * interval)
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}.")

if __name__ == "__main__":
    output_dir = "../data/not_synthesized/"
    video_path = "../data/videos/McL8WnjMvxM.mp4"
    
    extract_frames(video_path, output_dir, interval=0.3)
