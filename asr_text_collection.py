import time
import json
import os
import argparse
from huggingface_hub import InferenceClient
from google.generativeai import configure, GenerativeModel

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate sentences using LLM services")
parser.add_argument("--llm", choices=["huggingface", "gemini"], default="gemini", 
                    help="Select which LLM service to use")
args = parser.parse_args()

# Initialize the appropriate LLM client
if args.llm == "huggingface":
    API_TOKEN = os.environ.get("HF_TOKEN")
    if not API_TOKEN:
        raise ValueError("HF_TOKEN environment variable is not set")
    MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    client = InferenceClient(model=MODEL_NAME, token=API_TOKEN)
    print(f"Using Hugging Face model: {MODEL_NAME}")
    
elif args.llm == "gemini":
    # Only run this block for Gemini Developer API
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Configure the API with your API key
    configure(api_key=API_KEY)
    
    # Create a client adapter with compatible interface
    class GeminiClientAdapter:
        def __init__(self):
            self.model = GenerativeModel('gemini-2.0-flash-001')
        
        def text_generation(self, prompt, max_new_tokens=2048, temperature=0.7):
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_new_tokens}
            )
            return response.text
    
    client = GeminiClientAdapter()
    print("Using Gemini model: gemini-2.0-flash-001")

# Define the prompt template
PROMPT_TEMPLATE = """Generate {num_sentences} diverse English sentences (10-20 words each) for ASR fine-tuning. These will be read aloud by a non-native English speaker with moderate proficiency. Focus on:

1. Phonetic diversity: Include all English phonemes across the dataset:
    - Vowel sounds: short/long vowels (bit/bite, cut/cute)
    - Consonant pairs: p/b, t/d, k/g, f/v, s/z, sh/zh, ch/j
    - Challenging sounds: th (thin/then), r/l distinctions, w/v, ng

2. Everyday topics:
    - Daily activities and routines
    - Weather and seasons
    - Common conversations and questions
    - Technology and household objects
    - Food, travel, and basic descriptions

3. Speech patterns:
    - Common contractions (I'm, don't, we'll)
    - Simple questions and answers
    - Brief commands and requests
    - Different sentence structures and intonations

4. Vocabulary guidelines:
    - Use common, everyday words (1000-3000 most frequent English words)
    - Include numbers, dates, and time expressions
    - Avoid complex terminology or rare words


Example sentences:
Could you tell me where the nearest bus stop is?
I need to buy some groceries for dinner tonight.
The weather has been quite cold for this time of year.
What time does the movie start this evening?
My brother and I went fishing at the lake last weekend.

Do not include number in the beginning of the sentence and only generate sentences. Start generating {num_sentences} sentences, one per line:"""

# Output file
OUTPUT_FILE = f"data/generated_sentences_{args.llm}.txt"

# Parameters
TOTAL_SENTENCES = 1200
BATCH_SIZE = 60  # Adjust based on API rate limits
RETRIES = 3
DELAY = 5  # Seconds between requests

def generate_batch(num_sentences):
    prompt = PROMPT_TEMPLATE.format(num_sentences=num_sentences)
    for attempt in range(RETRIES):
        try:
            response = client.text_generation(prompt, max_new_tokens=2048, temperature=0.7)
            return response.strip().split('\n')
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(DELAY)
    return []

def main():
    all_sentences = []
    remaining = TOTAL_SENTENCES

    with open(OUTPUT_FILE, "w") as f:
        while remaining > 0:
            batch_size = min(BATCH_SIZE, remaining)
            print(f"Generating {batch_size} sentences...")
            batch = generate_batch(batch_size)
            if not batch:
                print("Failed to generate valid sentences. Retrying...")
                continue
            for i, sentence in enumerate(batch):
                if sentence.strip():
                    clean_sentence = f"{len(all_sentences) + 1}. {sentence.strip()}"
                    all_sentences.append(clean_sentence)
                    f.write(clean_sentence + "\n")
            remaining -= len(batch)
            print(f"Progress: {len(all_sentences)}/{TOTAL_SENTENCES} sentences saved.")
            time.sleep(DELAY)

    print(f"Generated {len(all_sentences)} sentences in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()