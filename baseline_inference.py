import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from image_captioning import BLIP2
from imageio import imread, imwrite

def generate(image_path, pipeline, prompt, captioning, num_frames=16):
    # image = (torch.from_numpy(imread(image_path))/255.0).to(torch.float16)
    # print(image.shape)
    # w,h,c = image.shape
    # image = image[0:w//8*8, 0:h//8*8, :] #cropping to multiple of 8
    # w,h,c = image.shape
    # if captioning:
    #     prompt = utils.caption_image(image)
    # latents = torch.tile(image, (1, num_frames, 1, 1, 1))
    # latents = latents.permute(0, 4, 1, 2, 3)

    # video_frames = pipe(prompt, num_inference_steps=100, latents=latents, height=h, width=w, num_frames=num_frames).frames
    video_frames = pipe(prompt, num_inference_steps=100, num_frames=num_frames).frames
    video_path = export_to_video(video_frames, output_video_path='outputs/meme0.mp4')
    print(video_path)

if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    prompt = "Stick figure style. A sad and lonely person walking in the middle of a crowd, the crowd avoids the person whereever it moves. There is a radius of empty space around the person at all times."

    generate('frames/meme0.jpg', pipe, prompt, captioning=False)