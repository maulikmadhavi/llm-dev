import json
import yaml
import os
from tqdm import tqdm
import decord

def get_vid_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    vr = decord.VideoReader(video_path)
    return len(vr) / vr.get_avg_fps()

def main():

    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    SUTD_DATA = config.get("SUTD_DATA", "annotations")
    
    data_in_yaml = os.path.join(SUTD_DATA['path'], "R2_all.jsonl")
    with open(data_in_yaml) as f:
        lines = f.readlines()

    
    _header = lines.pop(0)

    Q_TYPE_MAP = {
        "U": "Basic Understanding",
        "A": "Attribution",
        "F": "Event Forecasting",
        "R": "Reverse Reasoning",
        "C": "Counterfactual Inference",
        "I": "Introspection",
    }

    for line in lines:
        data: list = json.loads(line.strip())

        record_id: int = data[0]
        vid_id: int = data[1]
        vid_filename: str = data[2]
        q_body: str = data[4]
        q_type: str = data[5]
        options: list = data[6:10]
        answer_idx: int = data[10]
        answer_str: str = options[answer_idx]

        q_type = Q_TYPE_MAP.get(q_type, "Unknown")

        print(
            f"record_id: {record_id} | vid_id: {vid_id} | filename: {vid_filename}\nq_type: {q_type}\nQ: {q_body}\nA: {answer_str}\n"
        )

def jsonl_writer(data: list, file_path: str) -> None:
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def prepare_filtered_data() -> None:
    """Filter and prepare SUTD VQA dataset for training."""

    with open("../config.yaml") as f:
        config = yaml.safe_load(f)

    SUTD_DATA = config.get("SUTD_DATA", {})
    
    VIDEO_DATA = SUTD_DATA.get('videos', "")
    data_in_yaml = os.path.join(SUTD_DATA['annotation'], "R2_all.jsonl")
    data_out_yaml = os.path.join(SUTD_DATA['annotation'], "R2_all_filtered.jsonl")
    selected_items = []
    if os.path.exists(data_in_yaml):
        with open(data_in_yaml) as f:
            lines = f.readlines()

        
        _header = lines.pop(0)

        Q_TYPE_MAP = {
            "U": "Basic Understanding",
            "A": "Attribution",
            "F": "Event Forecasting",
            "R": "Reverse Reasoning",
            "C": "Counterfactual Inference",
            "I": "Introspection",
        }

        for line in tqdm(lines, desc="Processing data", unit="record", total=len(lines)):
            data: list = json.loads(line.strip())
        
            record_id: int = data[0]
            vid_id: int = data[1]
            vid_filename: str = data[2]
            q_body: str = data[4]
            q_type: str = data[5]
            options: list = data[6:10]
            answer_idx: int = data[10]
            answer_str: str = options[answer_idx]
        
            q_type = Q_TYPE_MAP.get(q_type, "Unknown")
        
            # print(
            #     f"record_id: {record_id} | vid_id: {vid_id} | filename: {vid_filename}\nq_type: {q_type}\nQ: {q_body}\nA: {answer_str}\n"
            # )
            output_video_path = os.path.join(VIDEO_DATA, vid_filename)
            # print(f"Output video path: {output_video_path}, exists: {os.path.exists(output_video_path)}")
            vid_dur = get_vid_duration(output_video_path)
            
            if vid_dur <= 10.0:
                options = [opt for opt in options if opt] # Remove empty string
                answer_idx = options.index(answer_str) if answer_str in options else -1
                selected_items.append({
                    "record_id": record_id,
                    "vid_id": vid_id,
                    "vid_filename": vid_filename,
                    "q_type": q_type,
                    "q_body": q_body,
                    "options": options,
                    "answer_idx": answer_idx,
                    "answer_str": answer_str,
                    "video_duration": vid_dur
                })

        jsonl_writer(selected_items, data_out_yaml)

        if not selected_items:
            print(f"No data found in {data_out_yaml}. Please run the data extraction first.")
            
            
if __name__ == "__main__":
    # main()
    prepare_filtered_data()
    print("Data preparation completed.")