import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from imageio import imread, imwrite
from tqdm import tqdm

if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    for i in tqdm(range(50,219), desc="Generating videos"):
        with open(f'text_output/GPT/meme{i}.txt', 'r') as gpt:
            with open(f'text_output/OCR/meme{i}_ocr.txt', 'r') as ocr:
                prompt = f"{gpt.read()}. Style: no words in the image; high-quality; generate best quality video. "
                print(prompt)
        video_frames = pipe(prompt, num_inference_steps=100, num_frames=16).frames
        video_path = export_to_video(video_frames, output_video_path=f'outputs/meme{i}.mp4')
        print("Generated video to: meme", i)