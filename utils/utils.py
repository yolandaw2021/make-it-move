import subprocess
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
import os

pipeline = keras_ocr.pipeline.Pipeline()


def key_frame(video_path, output_path=None):
    if output_path is None:
        output_path = video_path.replace(".mp4", ".jpg")
    print(f"Extracting a key frame from {video_path}")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "select=eq(n\,0)",
        "-q:v",
        "2",
        "-f",
        "image2",
        output_path,
    ]
    subprocess.run(cmd)

def mass_key_frame(folder_path):
    files = os.listdir(folder_path)
    for i in len(files):
        key_frame(f"{folder_path}/meme{i}.mp4", f"{folder_path}/frames/meme{i}.jpg")

def mass_rename(folder_path, filetype="mp4"):
   
    folder = folder_path
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"temp/meme{count}.{filetype}"
        os.rename(src, dst)

def select_frames(input_video_path, output_video_path, num_frames_to_select, fps=30):
    """
    Selects a specified number of frames evenly from the beginning to the end of a given video, and combines them into a new video.
    """
    video = VideoFileClip(input_video_path)
    total_frames = int(video.reader.nframes)
    if(total_frames < num_frames_to_select):
        video = np.tile(video, num_frames_to_select//total_frames+1)[:num_frames_to_select]
        video.write_videofile(output_video_path)
        return
    indices = np.linspace(0, total_frames-1, num=num_frames_to_select, dtype=int)
    selected_frames = [video.get_frame(idx) for idx in indices]
    output_clip = ImageSequenceClip(selected_frames, fps=fps)
    output_clip.write_videofile(output_video_path)

def caption_image(image):
    pass

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)

    # generate (word, box) tuples
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return img


if __name__ == "__main__":
    img = inpaint_text("/test_asset/meme1.jpg", pipeline)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/test_asset/meme1_without_word.jpg", img_rgb)

    # generate key frame script
    # for i in range(0,219):
    #     key_frame(f"/home/yw583/workspace/make-it-move/data/meme{i}.mp4", f"/home/yw583/workspace/make-it-move/frames/meme{i}.jpg")

    # rename frames script
    # mass_rename("./data")

    # select frames script
    # select_frames("data/meme5.mp4", "outputs/meme5_n.mp4", 180, fps=30)


