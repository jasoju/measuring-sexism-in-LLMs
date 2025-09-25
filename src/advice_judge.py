import pandas as pd
import os

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class Judgement(BaseModel):
    action_taken: bool
    explanation: str

current_dir = os.getcwd()
folder_dir = os.path.join(current_dir,"src", "output_data", "advice")

for filename in os.listdir(folder_dir):
    file_dir = os.path.join(folder_dir, filename)
    with open(file_dir, 'r') as f:
        df = pd.read_json(f)

    judge_response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        text_format=Judgement,
    )

    answer = judge_response.output_parsed




    df.to_json(f"output_data/{filename}", orient="columns")