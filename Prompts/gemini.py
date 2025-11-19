from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import io
from PIL import Image

ENV_PATH = "./.env"
GEMINI_JSON = "./data/gemini.json"
GEMINI_TEST_JSON = "./data/gemini_test.json"
GEMINI_PROB = "./data/mistral_prob.json"

MIN_VAL = 1e-5
load_dotenv(ENV_PATH)
CURR_GEMINI_API = os.getenv("GEMINI_API_KEY_1")
genai.configure(api_key=CURR_GEMINI_API)

gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")


def get_gemini_output(prompt, image: Image.Image, gemini_model):

    inputs = [prompt]
    if image:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        inputs.append({"mime_type": "image/jpeg", "data": image_bytes.getvalue()})

    response = gemini_model.generate_content(inputs)
    clean_response = remove_formatting(response.text.strip())
    json_output = json.loads(clean_response, strict=False)

    return json_output


def remove_formatting(response):
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    return response
