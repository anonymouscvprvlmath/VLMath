from gemini import get_gemini_output
from prompts import (
    SCAFFOLDING_PROMPT,
    SOCRATIC_QUESTIONING_PROMPT,
    MISTAKE_CORRECTION_PROMPT,
)
from datasets import load_dataset
import json
import time
import os
import google.generativeai as genai
import numpy as np

ds = load_dataset("MathLLMs/MathVision", split="testmini")

if __name__ == "__main__":
    key_sizes = 5
    keys = [os.getenv(f"GEMINI_API_KEY_{i}") for i in range(key_sizes)]
    output = []
    file_name = "./mathvision_mistake_correction_v2.json"
    with open(file_name, "r") as f:
        output = json.load(f)

    filled = [int(x["id"]) for x in output]
    print(filled)
    idx = 0
    for i in range(len(ds)):

        ex_id = int(ds[i]["id"])
        if ex_id in filled:
            continue

        print(f"Processing {ex_id}th item")
        img = ds[i]["decoded_image"]
        question = ds[i]["question"]
        answer = ds[i]["answer"]
        gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        genai.configure(api_key=keys[3])

        while True:
            try:
                json_reponse = get_gemini_output(
                    MISTAKE_CORRECTION_PROMPT(question, answer),
                    img,
                    gemini_model=gemini_model,
                )
                output.append({"id": ex_id, "conversations": json_reponse})
                with open(file_name, "w") as f:
                    json.dump(output, f)
                break
            except Exception as e:
                print(e)
                key = keys[idx]
                idx += 1
                idx = idx % key_sizes
                genai.configure(api_key=key)
                gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                print(f"Changing the key to: {key}")
                time.sleep(0.5)
