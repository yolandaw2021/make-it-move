import os
import time
from PIL import Image
from cffi import model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from chatgpt import query_gpt

device = "cuda" if torch.cuda.is_available() else "cpu"


class BLIP2:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if not os.path.exists("temp/blip.pt"):
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
            )
            self.model.save_pretrained("temp/blip.pt")
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "temp/blip.pt", torch_dtype=torch.float16
            ) 
        self.model.to(device)

    def process_image(self, image_path, prompt):
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            device, torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_length=128)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text


def process_image(image_path, prompts):
    answers = []
    model = BLIP2()
    for prompt in prompts:
        answers.append(model.process_image(image_path, prompt))
    return answers


if __name__ == "__main__":
    prompts = [
        "Question: What is the text inside this image? Answer:",
        "Question: What does the image mean? You must neglect any text in the image. Answer:",
        "Create a very long and detailed caption for this image. You must neglect any text in the image.  Answer:",
        "Question: Why is this image funny? Answer:",
    ]
    all_answers = [process_image(f"test_assets/meme{i}.jpg", prompts) for i in range(6)]
    print(all_answers)
    print("-------------")
    for answer in all_answers:
        caption, ocr = query_gpt(prompts, answer)
        print(caption)
        print(ocr)
        print("-------------")
