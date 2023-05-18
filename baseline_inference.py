import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from image_captioning import BLIP2
from imageio import imread, imwrite

def generate(image_path, pipeline, prompt, num_frames=16, i):
    video_frames = pipe(prompt, num_inference_steps=100, num_frames=num_frames).frames
    video_path = export_to_video(video_frames, output_video_path='outputs/meme{i}.mp4')
    print("Generated video to: meme", i)

if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    for i in tqdm(range(N), desc="Generating videos"):
        with open(f'text_output/GPT/meme{i}.txt', 'r') as f:
            prompt = f.read()
        generate('frames/meme{i}.jpg', pipe, prompt, i)