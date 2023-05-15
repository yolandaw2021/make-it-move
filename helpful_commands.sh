# Command for interactive session:
srun --gres gpu:a6000:1 -n 4 -t 48:00:00 -p kilian --mem 32G --pty /bin/bash
# Command for observing gpu condition
watch -d -n 0.5 nvidia-smi