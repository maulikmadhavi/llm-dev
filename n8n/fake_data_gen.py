import uuid
import random
from faker import Faker
import argparse

fake = Faker()

def generate_fake_data(num_rows):
    data = []
    for _ in range(num_rows):
        row = {
            "id": str(uuid.uuid4()),
            "name": fake.name(),
            "phone": fake.phone_number(),
            "email": fake.email()
        }
        data.append(row)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake data.")
    parser.add_argument("--num-rows", type=int, required=True, help="Number of rows to generate")
    args = parser.parse_args()

    fake_data = generate_fake_data(args.num_rows)
    for row in fake_data:
        print(row)
        
    # save to a file if needed
    with open('D:/llm-devs/n8n/fake_data_gen.json', 'w') as f:
        import json
        json.dump(fake_data, f, indent=4)   