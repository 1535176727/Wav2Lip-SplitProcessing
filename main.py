import os
import subprocess
import argparse
from glob import glob
import cv2
from tqdm import tqdm

# 解析输入参数
parser = argparse.ArgumentParser(description='Process videos with Wav2Lip in 5-second chunks.')
parser.add_argument('--video', type=str, required=True, help='Path to the input video')
parser.add_argument('--audio', type=str, required=True, help='Path to the input audio')
parser.add_argument('--output', type=str, required=True, help='Path to save the final output video')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/wav2lip.pth', help='Path to Wav2Lip checkpoint')
parser.add_argument('--chunk_duration', type=int, default=5, help='Duration of each video chunk in seconds')
args = parser.parse_args()

# 分割视频
def split_video(video_path, chunk_duration):
    print("Splitting video into 5-second chunks...")
    video_dir = 'temp_video_chunks'
    os.makedirs(video_dir, exist_ok=True)

    # 获取视频信息
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    chunk_size = int(chunk_duration * fps)

    # 分割视频
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_name = os.path.join(video_dir, f'chunk_{start // chunk_size}.mp4')
        cmd = f"ffmpeg -y -i {video_path} -vf 'select=between(n\,{start}\,{end-1})' -vsync vfr {chunk_name}"
        subprocess.call(cmd, shell=True)

    video_capture.release()
    return sorted(glob(os.path.join(video_dir, 'chunk_*.mp4')))

# 处理视频片段
def process_chunks(chunks, audio_path, checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed_chunks = []

    for chunk in tqdm(chunks, desc="Processing chunks"):
        output_chunk_path = os.path.join(output_dir, os.path.basename(chunk))
        cmd = f"python inference.py --checkpoint_path {checkpoint_path} --face {chunk} --audio {audio_path} --outfile {output_chunk_path}"
        subprocess.call(cmd, shell=True)
        processed_chunks.append(output_chunk_path)

    return processed_chunks

# 合并视频片段
def merge_chunks(chunks, output_path):
    with open('temp_filelist.txt', 'w') as filelist:
        for chunk in chunks:
            filelist.write(f"file '{os.path.abspath(chunk)}'\n")

    cmd = f"ffmpeg -y -f concat -safe 0 -i temp_filelist.txt -c copy {output_path}"
    subprocess.call(cmd, shell=True)
    os.remove('temp_filelist.txt')

def main():
    print("Starting video processing...")
    # Step 1: Split the video into 5-second chunks
    chunks = split_video(args.video, args.chunk_duration)

    # Step 2: Process each chunk with Wav2Lip
    processed_chunks = process_chunks(chunks, args.audio, args.checkpoint_path, 'processed_chunks')

    # Step 3: Merge processed chunks back into a single video
    merge_chunks(processed_chunks, args.output)

    # Cleanup
    for chunk in chunks + processed_chunks:
        os.remove(chunk)
    os.rmdir('temp_video_chunks')
    os.rmdir('processed_chunks')

    print("Processing complete. Output video saved to", args.output)

if __name__ == '__main__':
    main()
