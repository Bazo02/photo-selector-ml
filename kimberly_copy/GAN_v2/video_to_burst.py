import cv2
import os

def extract_bursts_from_video(input_dir, output_dir, burst_size=15, frame_stride=3, spacing=10):
    """
    Extracts multiple bursts from each video in input_dir.
    Each burst is saved in its own folder under a parent folder named after the video.

    Args:
        input_dir (str): Folder containing video files.
        output_dir (str): Folder where burst folders will be saved.
        burst_size (int): Number of frames to extract per burst.
        frame_stride (int): Frames to skip between saved frames within a burst.
        spacing (int): Frames to skip between bursts.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    total_videos = len(video_files)

    for video_index, filename in enumerate(video_files, start=1):
        video_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open {video_path}")
            continue

        base_name = os.path.splitext(filename)[0]
        video_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(video_output_dir, exist_ok=True)

        burst_idx = 0

        print(f"📽 Processing video {video_index}/{total_videos}: {filename}")

        while True:
            # 🔍 Look ahead: do we have enough frames for a full burst?
            start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            frames_needed = burst_size * frame_stride
            temp_count = 0
            enough_frames = True

            while temp_count < frames_needed:
                ret = cap.grab()
                if not ret:
                    enough_frames = False
                    break
                temp_count += 1

            if not enough_frames:
                break  # Not enough frames left for a complete burst

            # Reset to burst start position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

            # 🎞️ Extract burst
            burst_folder = os.path.join(video_output_dir, f"{base_name}_burst_{burst_idx:02d}")
            os.makedirs(burst_folder, exist_ok=True)

            frames_saved = 0
            while frames_saved < burst_size:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Incomplete burst at {burst_folder}, removing.")
                    break
                frame_path = os.path.join(burst_folder, f"frame_{frames_saved:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames_saved += 1

                # Skip frames within burst
                for _ in range(frame_stride - 1):
                    cap.grab()

            # 🧹 Clean up if burst was incomplete
            if frames_saved < burst_size:
                for f in os.listdir(burst_folder):
                    os.remove(os.path.join(burst_folder, f))
                os.rmdir(burst_folder)
                break

            print(f"🧾 Burst {burst_idx:02d}: saved {frames_saved} frames")

            burst_idx += 1

            # Skip frames between bursts
            for _ in range(spacing):
                cap.grab()

        cap.release()
        print(f"✅ {burst_idx} complete bursts extracted from {filename}\n")


burst_size = 10
spacing = 30      #frames skipped between bursts
frame_stride = 3


run_burst_gen = extract_bursts_from_video(
    input_dir="input_dir",
    output_dir="output_dir",
    burst_size=burst_size,
    frame_stride=frame_stride,
    spacing=spacing
)
