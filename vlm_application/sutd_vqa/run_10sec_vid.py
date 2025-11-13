import json
import yaml
import os
from tqdm import tqdm
import re
import sys
from pqdm.processes import pqdm # For process-based parallelism


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils import process_video, encode_base64_content_from_file, send_prompt

with open("../config.yaml") as f:
    config = yaml.safe_load(f)

SUTD_DATA = config.get("SUTD_DATA", {})
VIDEO_DATA = SUTD_DATA.get('videos', "")
vllm_api_endpoint = config.get("REMOTE_END", "")
vllm_model = config.get("MODEL_NAME", "")
data_out_yaml = os.path.join(SUTD_DATA['annotation'], "R2_all_filtered.jsonl")
data_out_yaml_result = os.path.join(SUTD_DATA['annotation'], "R2_all_filtered_result.jsonl")

# Open jsonl file
content = []
with open(data_out_yaml) as f:
    lines = f.readlines()
    content = [json.loads(line.strip()) for line in lines]

processed_video_dir = os.path.join(VIDEO_DATA, "../processed")
os.makedirs(processed_video_dir, exist_ok=True)
system_prompt = """You are a smart traffive video analyzer.  Answer the question in the following format: <think>
your reasoning
</think>

<answer>
your answer
</answer>"""
send_prompt_fallback = "You are a helpful assistant."
output_content = []

processed_videos = []

content = content[:100]  # Limit to first 100 items for testing
    
def process_single_video(video_item):
    """Process a single video with FFmpeg - run sequentially to avoid conflicts"""
    try:
        in_file = os.path.join(VIDEO_DATA, video_item['vid_filename'])
        out_file = os.path.join(processed_video_dir, video_item['vid_filename'])
        total_samples = round(video_item['video_duration'])
        
        # Check if processed video already exists
        if os.path.exists(out_file):
            print(f"✅ Video already processed: {video_item['vid_filename']}")
            return out_file, total_samples
        
        # Process video with error handling
        print(f"🎬 Processing video: {video_item['vid_filename']}")
        process_video(in_file, out_file, total_samples=total_samples, fps=1, resize=(640, 360))
        
        # Verify the output file was created
        if not os.path.exists(out_file):
            raise FileNotFoundError(f"Processed video not created: {out_file}")
            
        return out_file, total_samples
    except Exception as e:
        print(f"❌ Error processing video {video_item['vid_filename']}: {e}")
        return None, None        


for video_item in tqdm(content, desc="Processing videos with FFmpeg"):
    processed_path, total_samples = process_single_video(video_item)
    video_item['processed_path'] = processed_path
    video_item['total_samples'] = total_samples
    processed_videos.append(video_item)


# STEP 2: Run inference in parallel (API calls are sa
#


# for x in tqdm(content, desc="Processing videos", total=len(content)):
def run_for_content(x):
    # in_file = os.path.join(VIDEO_DATA, x['vid_filename'])
    out_file = os.path.join(processed_video_dir,  x['vid_filename'])

    total_samples = round(x['video_duration'])
    # process_video(in_file, out_file, total_samples=total_samples, fps=1, resize=(640, 360))
    
    video_base64 = encode_base64_content_from_file(out_file)
    options_str = "".join(
        f"{n}: {v}\n" if v else "" for n, v in zip(["A", "B", "C", "D"], x.get('options'))
    )
    print(x)
    prompt = f"""You are given {total_samples} video frames for analysis.

    Question: {x['q_body']}

    Respond with:
    {options_str}"""
    def extract_answer_thing(result):
        think, answer = "", result
        if "<think>" in result and "<answer>" in result:
            think = result.split("<think>")[1].split("</think>")[0].strip()
            answer = result.split("<answer>")[1].split("</answer>")[0].strip()
        return think, answer
    
    if x.get("q_type") == "Basic Understanding":
        
        try:
            # Split <think> and <answer> parts if needed
            result = send_prompt(vllm_api_endpoint, vllm_model, prompt, video_base64, system_prompt)
            think, answer = extract_answer_thing(result)
        except Exception as e:
            # fallback system prompt
            try:
                result = send_prompt(vllm_api_endpoint, vllm_model, prompt, video_base64, send_prompt_fallback)
                # Split <think> and <answer> parts if needed
                think, answer = extract_answer_thing(result)
            except Exception as e:
                print(f"Error processing {x['vid_filename']}: {e}")
                print(prompt, result)
                think, answer = "", "Error processing video (N.A.)"
        x['think'] = think
        x['pred_answer'] = answer

    else:
        x['think'] = ""
        x['pred_answer'] = f"N.A. -> different {x['q_type']}"
    return x



    
    # y = run_for_content(x)
    # output_content.append(y)
    
output_content = pqdm(content, run_for_content, n_jobs=3)
    
# Save the output to a new jsonl file
with open(data_out_yaml_result, 'w') as f:
    for item in output_content:
        f.write(json.dumps(item) + "\n")
