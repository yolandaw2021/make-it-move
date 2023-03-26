import subprocess


def key_frame(video_path):
    print(f"Extracting a key frame from {video_path}")
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "select=eq(n\,0)",
            "-q:v",
            "2",
            "-f",
            "image2",
            "key_frame.jpg",
        ]
    )


if __name__ == "__main__":
    key_frame("/Users/fengyuli/Desktop/temp.mp4")    
