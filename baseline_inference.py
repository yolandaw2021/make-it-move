import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# prompt = "A frog sitting under a pouring water pipe, looking very sad."
# prompt = "An astronaut riding a horse."

video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames, output_video_path='videos/frog.mp4')
print(video_path)