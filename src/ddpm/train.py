import gc
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision
from datasets import get_dataloader
from diffusion import SimpleDiffusion, forward_diffusion, reverse_diffusion
from helpers import (
    frames2vid,
    get_default_device,
    inverse_transform,
    setup_log_directory,
)
from torch.cuda import amp
from torch.optim import AdamW
from torchmetrics import MeanMetric
from tqdm import tqdm
from unet import UNet


@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "MNIST"  # "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "local_data", "ddpm", "inference"
    )
    root_checkpoint_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "local_data", "ddpm", "checkpoints"
    )

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)
    NUM_EPOCHS = 30
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 2


def train_one_epoch(
    model,
    sd,
    loader,
    optimizer,
    scaler,
    loss_fn,
    epoch=800,
    base_config=BaseConfig(),
    training_config=TrainingConfig(),
):
    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s, _ in loader:
            tq.update(1)

            ts = torch.randint(
                low=1,
                high=training_config.TIMESTEPS,
                size=(x0s.shape[0],),
                device=base_config.DEVICE,
            )
            xts, gt_noise = forward_diffusion(sd, x0s, ts)

            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss


def main():
    # get data
    dataloader = get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=TrainingConfig.BATCH_SIZE,
        device=BaseConfig.DEVICE,
        pin_memory=True,
        num_workers=TrainingConfig.NUM_WORKERS,
        root_dir=os.path.join(os.path.dirname(__file__), "..", "..", "local_data"),
    )

    # initialize the model
    model = UNet(
        input_channels=TrainingConfig.IMG_SHAPE[0],
        output_channels=TrainingConfig.IMG_SHAPE[0],
        base_channels=64,  # 64, 128, 256, 512
        base_channels_multiples=[1, 2, 4, 8],  # 32, 16, 8, 4
        apply_attention=[False, False, True, False],
        dropout_rate=0.1,
        time_multiple=2,  # 64 -> 128
    )
    model.to(BaseConfig.DEVICE)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # initialize the diffusion
    sd = SimpleDiffusion(
        num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
        img_shape=TrainingConfig.IMG_SHAPE,
        device=BaseConfig.DEVICE,
    )

    # optimizer
    optimizer = AdamW(model.parameters(), lr=TrainingConfig.LR)

    # loss function
    loss_fn = nn.MSELoss()

    # scaler
    scaler = amp.GradScaler()

    # set up loggers
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

    # training loop
    for epoch in range(1, TrainingConfig.NUM_EPOCHS + 1):
        torch.cuda.empty_cache()
        gc.collect()

        # Training loop
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        # Inference sampling
        out, steps = reverse_diffusion(
            model,
            sd,
            timesteps=TrainingConfig.TIMESTEPS,
            num_images=32,
            img_shape=TrainingConfig.IMG_SHAPE,
            device=BaseConfig.DEVICE,
            record_process=True,
        )

        # Save the images and video
        video_frames_arr = []

        for x in steps:
            x_inv = inverse_transform(x).type(torch.uint8)
            grid = torchvision.utils.make_grid(x_inv, nrow=8, pad_value=255.0).to("cpu")
            ndarr = torch.permute(grid, (1, 2, 0)).numpy()[:, :, ::-1]
            video_frames_arr.append(ndarr)

        frames2vid(video_frames_arr, os.path.join(log_dir, f"video_{epoch}.mp4"))

        # clear_output()
        checkpoint_dict = {
            "opt": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model": model.state_dict(),
        }
        torch.save(
            checkpoint_dict, os.path.join(checkpoint_dir, "ckpt_{}.tar".format(epoch))
        )
        del checkpoint_dict


if __name__ == "__main__":
    main()
