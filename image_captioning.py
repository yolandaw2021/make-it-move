from PIL import Image
from cffi import model
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Current device: {device}')


class BLIP2:
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        self.model.to(device)

    def caption_image(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def caption_folder(self, image_folder="frames", output_folder="data"):
        N = len(os.listdir(image_folder))
        for i in tqdm(range(N), desc="Captioning images"):
            image = f'{image_folder}/meme{i}.jpg'
            caption = self.caption_image(image)
            with open(f'meme{i}.txt', 'w') as f:
                f.write(caption)
        return True

if __name__ == "__main__":
    model = BLIP2()
    model.caption_folder()
    # image_path = [f'frames/meme{i}.jpg' for i in range(6)]
    # captions = [model.caption_image(image) for image in image_path]
    # print(captions)
