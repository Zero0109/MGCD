import base64
import mimetypes
import os
from openai import OpenAI
import time
import numpy as np

client = OpenAI(api_key="sk-xxx")

prompt = "For the 3V1 vehicle pursuit task, our team is the red pursuers and the enemy is the blue evader. The objective is for any pursuer to make contact with the evader within the time limit. The positions of the agents within the arena are shown in the image. Describe the current state and potential situations in a single sentence respectively. Only these two sentences are required. No title, colons, or other unnecessary content is needed. Please strictly adhere to these requirements."

scene = 'hunting'
for i in range(50):
    def image_to_data_url(path: str) -> str:
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    data_url = image_to_data_url(f"/data/images/{scene}/{i}.jpg")

    messages = [
        { "role": "user", "content": [
            { "type": "text", "text": prompt },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                }
            }
        ] }
    ]

    response = client.chat.completions.create(
        model="gpt-5",
        messages=messages
    )

    print(response.choices[0].message.content)
    os.makedirs(f'situationDesc/{scene}/', exist_ok=True)
    with open(f'situationDesc/{scene}/{i}.txt', 'w', encoding='utf-8') as f:
        f.write(response.choices[0].message.content)

    time.sleep(3)