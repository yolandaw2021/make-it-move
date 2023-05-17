import os
import time
from PIL import Image
from cffi import model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from chatgpt import query_gpt
import os
from tqdm import tqdm

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

    
def caption_folder(image_folder="frames", output_folder="data"):
    N = len(os.listdir(image_folder))
    prompts = [
        "Question: What is the text inside this image? Answer:",
        "Question: Who is the main character? Describe with complete sentence. Answer:",
        "Question: What is the main character doing? Describe with complete sentence. Answer:",
        "Question: What is the color of the scene? Describe with complete sentence. Answer:",
        "Question: What is in the background? Describe with complete sentence. Answer:",
        "Question: What is the style of drawing? Describe with complete sentence. Answer:",
        "Question: What is the emotion of the meme? Describe with complete sentence. Answer:"
    ]
    model = BLIP2()
    for i in tqdm(range(N), desc="Captioning images"):
        image = f'{image_folder}/meme{i}.jpg'
        answers = []
        for prompt in prompts:
            answers.append(model.process_image(image, prompt))
        caption, ocr = query_gpt(prompts, answers)
        with open(f'{output_folder}/meme{i}.txt', 'w') as f:
            f.write(caption)
        with open(f'{output_folder}/meme{i}_ocr.txt', 'w') as f:
            f.write(ocr)
    return True

if __name__ == "__main__":
    # demo trial
    prompts = [
        "Question: What is the text inside this image? Answer:",
        "Question: Who is the main character? Describe in detail. Answer:",
        "Question: What is the main character doing? Describe with complete sentence. Answer:",
        "Question: What is the color of the scene? Describe with complete sentence. Answer:",
        "Question: What is in the background? Describe with complete sentence. Answer:",
        "Question: What is the style of drawing? Describe with phrases like cartoon style, realistic style, fantasy, and so on. Answer:",
        "Question: What is the emotion of the meme? Describe with phrases like, happy vibes, sad emotion, angry emotion, sarcastic humor, and so on. Answer:"
    ]
    model = BLIP2()
    i = 122
    image = f'frames/meme{i}.jpg'
    answers = []
    for prompt in prompts:
        answers.append(model.process_image(image, prompt))
    print("BLIP2 Answers: ", answers)
    caption, ocr = query_gpt(prompts, answers)
    print("caption: ",caption)
    print("ocr: ", ocr)
    
    # create caption for dataset
    # caption_folder()
