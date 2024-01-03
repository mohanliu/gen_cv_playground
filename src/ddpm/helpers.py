import os

import cv2
import torch
import torchvision
from PIL import Image


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def setup_log_directory(config):
    """Log and Model checkpoint directory Setup"""

    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [
            int(folder.replace("version_", ""))
            for folder in os.listdir(config.root_log_dir)
        ]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory
    log_dir = os.path.join(config.root_log_dir, version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir


def frames2vid(images, save_path):
    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     fourcc = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    # Appending the images to the video one by one
    for image in images:
        video.write(image)

    # Deallocating memories taken for window creation
    #     cv2.destroyAllWindows()
    video.release()
    return
