import subprocess


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


if __name__ == "__main__":
    for i in range(265):
        key_frame(f"./data/meme{i}.mp4", f"./frames/meme{i}.jpg")
