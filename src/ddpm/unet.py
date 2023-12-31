"""
Source: https://github.com/spmallick/learnopencv/blob/master/Guide-to-training-DDPMs-from-Scratch/Generating_MNIST_using_DDPMs.ipynb
"""

import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        # output dimension: [batch_size, time_emb_dims_exp]
        return self.time_blocks(time)


class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        self.group_norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=self.channels, num_heads=4, batch_first=True
        )

    def forward(self, x):
        B, _, H, W = x.shape  # `_` == self.channels

        # group norm
        h = self.group_norm(x)

        # reshape for multi-head attention
        # the attention applies to the 'channels' dimension and interacts with the spatial dimensions
        # ie. channel vector -> nlp token and spatial dimensions -> sequence of inputs
        h = h.reshape(B, self.channels, H * W).swapaxes(
            1, 2
        )  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]

        # self-attention
        # note that multi-head attention module will return both the output and the attention weights
        # as a tuple but we only need the output here
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]

        h = h.swapaxes(2, 1).view(
            B, self.channels, H, W
        )  # [B, H*W, C] --> [B, C, H*W] --> [B, C, H, W]
        return x + h  # residual connection


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        dropout_rate=0.1,
        time_emb_dims=512,
        apply_attention=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act_fn = nn.SiLU()
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        # Group 2 time embedding
        self.dense_1 = nn.Linear(
            in_features=time_emb_dims, out_features=self.out_channels
        )

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        # match input and output channels
        # if the number of channels is different, we need to do a 1x1 convolution
        # to match the number of channels
        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.conv_1(h)

        # group 2
        # add timestep embedding (with tensor broadcasting over spatial dimensions)
        # Note that having ``None`` will add new dimensions to the tensor with size 1 (like ``np.newaxis``)
        # Note that here we can either use ``[:, :, None, None]`` or ``[..., None, None]`` where ``...`` means
        # all the previous dimensions.
        # <<old code>>
        # h += self.dense_1(self.act_fn(t))[:, :, None, None]  # [B, C, 1, 1]
        # <<new code>>
        h += self.dense_1(self.act_fn(t))[..., None, None]  # [B, C, 1, 1]

        # group 3
        h = self.act_fn(self.normlize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention (residual first)
        h = h + self.match_input(x)
        h = self.attention(h)

        return h


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # downsample the spatial resolution by a factor of 2 using strided convolution
        self.downsample = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # use `nn.Upsample` to upsample the spatial resolution
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x, *args):
        return self.upsample(x)


class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 2, 4, 8),
        apply_attention=(False, False, True, False),
        dropout_rate=0.1,
        time_multiple=4,
    ):
        """
        ResnetBlock itself won't change the spatial resolution of the input. Only DownSample and UpSample will.
        """
        super().__init__()

        time_emb_dims_exp = base_channels * time_multiple  # stay constant throughout
        self.time_embeddings = SinusoidalPositionEmbeddings(
            time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp
        )

        # == First layer ==
        # increase the number of channels from 3 (RGB) to base_channels
        self.first = nn.Conv2d(
            in_channels=input_channels,  # 3 for RGB
            out_channels=base_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        num_resolutions = len(base_channels_multiples)  # number of levels

        # == Encoder part of the UNet. Dimension reduction. ==
        self.encoder_blocks = nn.ModuleList()
        curr_channels = [
            base_channels
        ]  # a stack to keep track of the current channels (for both resnet blocks and downsample blocks)
        in_channels = base_channels  # input channels for next block (will be updated)

        # Nested for loops
        # First loop goes through each level
        # Second loop goes through each residual block
        for level in range(num_resolutions):
            # get the output channels for the current level, which is also the
            # input channels for the next level
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):
                # first residual block will change (mostly increase) the number of channels
                # the rest will keep the same number of channels
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            if level != (num_resolutions - 1):
                # downsample spatial resolution if not the last level
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # == Bottleneck in between ==
        # maintain same spatial resolution and number of channels
        # no skip connections so no need to update curr_channels
        # just to add a few deep layers to increase model complexity
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )

        # == Decoder part of the UNet. Dimension restoration with skip-connections. ==
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                # the first block will do the concatenation
                # the rest will not
                # additional +1 resnet block for the decoder
                encoder_in_channels = curr_channels.pop()  # pop no. of channels
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,  # concat
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                # upsample spatial resolution if not the first level
                self.decoder_blocks.append(UpSample(in_channels))

        # == Final layer ==
        # reduce the number of channels to 3 (RGB)
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

    def forward(self, x, t):
        time_emb = self.time_embeddings(t)

        h = self.first(x)
        outs = [h]  # a stack to keep track of the outputs of each level

        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)  # save the output of each level

        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)

        for layer in self.decoder_blocks:
            # retrieve and concatenate the output of the corresponding level
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)

        h = self.final(h)

        return h
