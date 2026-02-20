import requests
import time
import json
import base64
import os
import cv2
import re
import logging
from vlm_application.api_client import VLMAPIClient

logger = logging.getLogger(__name__)
from decord import VideoReader, cpu
from pathlib import Path
import subprocess
import math


def send_prompt(vllm_api_endpoint, vllm_model, prompt, video_base64, system_prompt=None):
    """Legacy wrapper for VLMAPIClient.send_prompt_with_video().

    Deprecated: Use VLMAPIClient directly for new code.
    """
    client = VLMAPIClient(vllm_api_endpoint, vllm_model)
    return client.send_prompt_with_video(prompt, video_base64, system_prompt)


def send_prompt_for_image(vllm_api_endpoint, vllm_model, prompt, image_base64, system_prompt=None):
    """Legacy wrapper for VLMAPIClient.send_prompt_with_image().

    Deprecated: Use VLMAPIClient directly for new code.
    """
    client = VLMAPIClient(vllm_api_endpoint, vllm_model)
    return client.send_prompt_with_image(prompt, image_base64, system_prompt)


def send_text_query_prompt(vllm_api_endpoint, vllm_model, prompt, summary_report, system_prompt=None):
    """Legacy wrapper for VLMAPIClient.send_text_query().

    Deprecated: Use VLMAPIClient directly for new code.
    """
    client = VLMAPIClient(vllm_api_endpoint, vllm_model)
    return client.send_text_query(prompt, summary_report, system_prompt)


def encode_file_to_base64(file_path: str) -> str:
    """Encode any file to base64 string.

    Works with video files, images, documents, or any binary file.

    Args:
        file_path: Path to file

    Returns:
        Base64 encoded string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read

    Example:
        >>> b64_video = encode_file_to_base64("video.mp4")
        >>> b64_image = encode_file_to_base64("image.jpg")
    """
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Cannot read file {file_path}: {e}")


# Legacy aliases for backward compatibility
def encode_base64_content_from_file(file_path: str) -> str:
    """Deprecated: Use encode_file_to_base64() instead."""
    return encode_file_to_base64(file_path)


def encode_base64_content_for_imagefile(image_path: str) -> str:
    """Deprecated: Use encode_file_to_base64() instead."""
    return encode_file_to_base64(image_path)


def process_video(
    input_video_file,
    output_video_file,
    total_samples=10,
    fps=1,
    resize=None,
):
    try:
        vr = VideoReader(str(input_video_file), ctx=cpu(0))
        total_frame_num = len(vr)

        frame_height, frame_width, _ = vr[0].asnumpy().shape

        batch_array = []
        for i in range(total_samples):
            batch_array.append(int(total_frame_num / total_samples * i))
        frames = vr.get_batch(batch_array).asnumpy()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
        if resize != None:
            out = cv2.VideoWriter(output_video_file, fourcc, fps, resize)
        else:
            out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        for frame in frames:
            if resize != None:
                resize_frame = cv2.cvtColor(cv2.resize(frame, resize), cv2.COLOR_BGR2RGB)
                out.write(resize_frame)
            else:
                _frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(_frame)

            frame_count += 1

        out.release()

    except cv2.error as e:
        print(f"resize: {resize} {input_video_file} {e}")

    return


def extract_index(filename):
    match = re.search(r"_(\d+)\.mp4$", filename)
    return int(match.group(1)) if match else None


def scan_video_files(video_dir):
    """
    Scans a directory for all subfolders and finds .mp4 video files in each subfolder.
    :param video_dir: The root directory containing subfolders with video files.
    :return: List of tuples (folder, subfolder, video file name)
    """
    video_list = []

    for folder in os.listdir(video_dir):
        folder_path = os.path.join(video_dir, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            video_file = f"{os.path.basename(folder_path)}.mp4"
            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):
                    chunk_file = f"{folder_path}/{file}"
                    index = extract_index(file)
                    video_list.append((video_file, chunk_file, index))

    return video_list


def do_chunking(input_video, out_dir, chunk_duration=10):
    tag = Path(input_video).stem
    output_dir = os.path.join(out_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    # Get total duration of video (in seconds)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_video,
    ]
    duration = float(subprocess.check_output(cmd).decode().strip())
    time.sleep(1)  # Sleep to ensure ffprobe has time to process
    print(f"video duration={duration}")
    num_chunks = math.ceil(duration / chunk_duration)

    for i in range(num_chunks):
        start = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        if end_time > duration:
            end_time = duration
        actual_duration = min(chunk_duration, duration - start)
        print(f"chunk#{i}, {chunk_duration}, {start}, {actual_duration}, {end_time} == {start + actual_duration}")
        if actual_duration <= 0.0:
            break  # Nothing more to extract
        output_file = os.path.join(output_dir, f"{tag}_{i:03d}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-ss",
            str(start),
            "-to",
            str(end_time),
            "-c:v",
            "libx264",  # Re-encode video
            "-preset",
            "veryfast",  # Encoding speed
            "-c:a",
            "aac",  # Re-encode audio
            "-strict",
            "experimental",  # Allow experimental AAC encoder
            output_file,
        ]
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                break  # Command succeeded, exit the loop

            print(f"Error in chunk {i} (attempt {attempts + 1}/{max_attempts}): {result.stderr.decode()}")
            attempts += 1
            time.sleep(1)  # Wait before retrying

        if attempts == max_attempts:
            print(f"Failed to process chunk {i} after {max_attempts} attempts")
        else:
            time.sleep(0.5)  # Only sleep on success
        time.sleep(1)  # Sleep to ensure ffmpeg has time to process

        # Check the duration of the output file
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            output_file,
        ]
        try:
            output_duration = float(subprocess.check_output(cmd).decode().strip())
        except:
            output_duration = -1
        print(f"Output file {output_file} duration: {output_duration} seconds")

        # Break if the output file is empty or has no duration
        if output_duration <= 0:
            print(f"Output file {output_file} is empty or has no duration, stopping.")
            break


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    base64_img = encode_base64_content_for_imagefile(data_config["SAMPLE_IMG"])
    vllm_api_endpoint = data_config["REMOTE_END"]
    resize = None  # (640, 360)
    vllm_model = data_config["MODEL_NAME"]
    prompt = "describe the image in detail"
    output = send_prompt_for_image(vllm_api_endpoint, vllm_model, prompt, base64_img)
    print(output)

    # Sample test for video description
    video_path = data_config["SAMPLE_VIDEO"]
    base64_video = encode_base64_content_from_file(video_path)
    prompt = "describe the video in detail"
    output = send_prompt(vllm_api_endpoint, vllm_model, prompt, base64_video)
    print(output)


# if __name__ == "__main__":
#     # Testinf the functions
#     sample_image = "vlm_application/sample_image.jpg"
#     base64_img = encode_base64_content_for_imagefile(sample_image)
#     vllm_api_endpoint = "http://localhost:8000/v1/chat/completions"
#     resize = (640, 360)
#     vllm_model = "qwen2.5-vl-3b-instruct"
#     prompt = "describe the image in detail"
#     output = send_prompt_for_image(vllm_api_endpoint, vllm_model, prompt, base64_img)
#     print(output)
