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
        "Question: Exactly describes the scene in this image, including the color of the scene, the position and expression of the character, and more. You must neglect any text in the image. Answer:"
    ]
    model = BLIP2()
    for i in tqdm(range(N), desc="Captioning images"):
        image = f'{image_folder}/meme{i}_detext.jpg'
        answers = []
        for prompt in prompts:
            answers.append(model.process_image(image, prompt))
        caption, ocr = query_gpt(prompts, answers)
        with open(f'{output_folder}/meme{i}_detext.txt', 'w') as f:
            f.write(caption)
        with open(f'{output_folder}/meme{i}_detext_ocr.txt', 'w') as f:
            f.write(ocr)
    return True

if __name__ == "__main__":
    # demo trial
    prompts = [
        "Question: What is the text inside this image? Answer:",
        "Question: Is there a main character in the meme? If yes, describe the character in detail: position, motion, facial expression, and so on. Answer:"
        "Question: What does the background look like? Exactly describe the background including the color of the scene, the components in the background, and the position of the components. You must neglect any text in the image. Answer:",
        "Question: What is the style of drawing? Describe with words like: cartoon, realistic, magical, and so on. Answer:",
        "Question: What is the emotion of the meme? Describe with words like: happy, sad, angry, sarcastic, and so on. Answer:"
    ]
    all_answers = [process_image(f"frames/meme{i}.jpg", prompts) for i in range(6)]
    print(all_answers)
    print("-------------")
    for answer in all_answers:
        caption, ocr = query_gpt(prompts, answer)
        print(caption)
        print(ocr)
        print("-------------")
    
    # create caption for dataset
    caption_folder(image_folder="frames/image2_detext", output_folder="data")
