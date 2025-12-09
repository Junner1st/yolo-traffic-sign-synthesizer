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
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frame_interval = int(fps * interval)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            sec = count // int(fps)
            frame_in_sec = count % int(fps)
            frame_filename = os.path.join(output_dir, f"{video_name}_{sec}s_frame_{frame_in_sec}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        count += 1
    
    cap.release()
    print(f"Frames extracted to {output_dir}.")

if __name__ == "__main__":
    output_dir = "../data/not_synthesized/"
    video_dir = "../data/videos/"
    for file in os.listdir(video_dir):
        if file.endswith('.mp4'):
            video_path = os.path.join(video_dir, file)
            extract_frames(video_path, output_dir, interval=0.3)

