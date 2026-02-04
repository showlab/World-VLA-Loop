import cv2
import os
import subprocess
import tempfile
from pathlib import Path


def get_video_duration(video_path):
    """Get the duration of a video in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration


def process_video(input_path, output_path, target_duration, flip=True):
    """
    Process video: optionally flip left-right and adjust FPS to match target duration.
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        target_duration: Target duration in seconds
        flip: Whether to flip the video left-right (default: True)
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new FPS to match target duration
    new_fps = frame_count / target_duration if target_duration > 0 else 30
    
    print(f"    Frame count: {frame_count}")
    print(f"    Original dimensions: {width}x{height}")
    print(f"    Calculated FPS: {new_fps:.2f} (to match duration {target_duration:.2f}s)")
    
    # Setup video writer with temporary file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_output_path = temp_output.name
    temp_output.close()
    
    # Use MJPEG codec for intermediate file (most compatible with OpenCV)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_output_path, fourcc, new_fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        os.unlink(temp_output_path)
        raise ValueError(f"Cannot create video writer for: {output_path}")
    
    # Process frames
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optionally flip frame horizontally (left-right)
        if flip:
            processed_frame = cv2.flip(frame, 1)
        else:
            processed_frame = frame
        
        # Write processed frame
        out.write(processed_frame)
        frame_num += 1
        
        if frame_num % 10 == 0:
            print(f"    Processed {frame_num}/{frame_count} frames", end='\r')
    
    print(f"    Processed {frame_num}/{frame_count} frames - Complete!")
    
    # Release resources
    cap.release()
    out.release()
    
    # Re-encode with ffmpeg to standard H.264 MP4 format
    print(f"    Re-encoding to standard H.264 MP4 format...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(output_path)
        ], check=True, capture_output=True, text=True)
        os.unlink(temp_output_path)
    except subprocess.CalledProcessError as e:
        os.unlink(temp_output_path)
        raise ValueError(f"FFmpeg encoding failed: {e.stderr}")
    except FileNotFoundError:
        os.unlink(temp_output_path)
        raise ValueError("FFmpeg not found. Please install ffmpeg: brew install ffmpeg")


def main():
    """Main function to process videos in page1-page4."""
    base_dir = Path("/Users/hiski/Documents/projects/robot_failure_prediction/World-VLA-Loop/figures/worldgen_videos")
    
    # Process page1-page4
    for page_num in range(1, 5):
        page_dir = base_dir / f"page{page_num}"
        
        if not page_dir.exists():
            print(f"Warning: Page directory not found: {page_dir}")
            continue
        
        # Skip flipping for page1, flip for others
        should_flip = (page_num != 1)
        
        print(f"\n{'='*50}")
        print(f"Processing page{page_num}: {'NO FLIP' if not should_flip else 'WITH FLIP'}")
        print('='*50)
        
        # Process videos 1-4 in each page directory
        for video_num in range(1, 5):
            gt_video = page_dir / f"{video_num}-GT.mp4"
            reference_video = page_dir / f"{video_num}.mp4"
            output_video = page_dir / f"{video_num}-GT-p.mp4"
            
            # Check if input files exist
            if not gt_video.exists():
                print(f"  Warning: GT video not found: {gt_video.name}")
                continue
            
            if not reference_video.exists():
                print(f"  Warning: Reference video not found: {reference_video.name}")
                continue
            
            print(f"\n  Video {video_num}:")
            print(f"    Input: {gt_video.name}")
            print(f"    Reference: {reference_video.name}")
            print(f"    Output: {output_video.name}")
            
            try:
                # Get target duration from reference video
                target_duration = get_video_duration(reference_video)
                print(f"    Target duration: {target_duration:.2f}s")
                
                # Process the GT video
                process_video(gt_video, output_video, target_duration, flip=should_flip)
                print(f"    ✓ Successfully saved: {output_video.name}")
                
            except Exception as e:
                print(f"    ✗ Error processing video {video_num}: {e}")
                continue
    
    print("\n" + "="*50)
    print("All video processing complete!")
    print("="*50)


if __name__ == "__main__":
    main()
