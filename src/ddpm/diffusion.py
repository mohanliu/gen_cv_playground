import torch


class SimpleDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        img_shape=(3, 64, 64),
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.img_shape = img_shape
        self.device = device

        self.initialize()

    def initialize(self):
        """
        Original single step forward pass formula:

            x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon_t

        x_t: image at time t
        x_{t-1}: image at time t-1
        beta_t: diffusion rate at time t
        epsilon_t: noise at time t

        Define:
        - alpha_t = 1 - beta_t
        - alpha_bar = alpha_1 * alpha_2 * ... * alpha_t, cumulative product of alpha_t from t=1 to t=T

        Then, the formula becomes:

                x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon_t
        """
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()  # vector of beta values
        self.alpha = 1 - self.beta  # let: alpha = 1 - beta

        self.sqrt_beta = torch.sqrt(self.beta)  # sqrt(beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)  # alpha_bar
        self.sqrt_alpha_cumulative = torch.sqrt(
            self.alpha_cumulative
        )  # sqrt(alpha_bar), prefactor for image, mean of x_t (when times x_0)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)  # 1 / sqrt(alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(
            1 - self.alpha_cumulative
        )  # sqrt(1 - alpha_bar), prefactor for current noise, std of x_t

    def get_betas(self):
        """
        beta values based on linear schedule (beta -> diffusion rate)

        based on the original ddpm paper: beta will vary from 0.0001 to 0.02
        over 1000 steps linearly
        """
        scale = 1000 / self.num_diffusion_timesteps

        beta_start = scale * 1e-4
        beta_end = scale * 0.02

        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )


def get_value_per_time(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.

    Args:
        element: tensor of values (usually diffusion rates schedules,
            e.g. ``sqrt_alpha_cumulative`` or ``sqrt_one_minus_alpha_cumulative``)
        t: tensor of time stamp (int, or tensor of ints)
    """
    # gather value at index t at the last dimension
    # note that `gather` is used to select elements in a tensor based on provided indices
    # source: https://pytorch.org/docs/stable/generated/torch.gather.html
    # crucial argument:
    # - dim: dimension along which to index
    # - index: indices of elements to gather
    ele = element.gather(-1, t)

    # reshape to have same number of dimensions as batch of images (B, 1, 1, 1) for broadcasting
    # note 1: use ``reshape`` instead of ``view`` to make it contiguous
    # note 2: since this is a 1D tensor, we can also use ``unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)``
    # or ``[..., None, None, None]``
    return ele.reshape(-1, 1, 1, 1)


def forward_diffusion(sd: SimpleDiffusion, x0: torch.Tensor, timesteps: torch.Tensor):
    # note: ``torch.randn_like()`` Returns a tensor with the same size as ``input`` that is filled with random numbers
    # from a normal distribution with mean 0 and variance 1
    # source: https://pytorch.org/docs/stable/generated/torch.randn_like.html
    eps = torch.randn_like(x0)  # Noise

    # Image scaled
    # note: ``*`` is element-wise multiplication, so it also works for broadcasting
    # all last 3 dimensions are multiplied by the same ``sqrt(alpha_bar)`` scalar (H, W, C)
    # we only have to make sure that the first dimension (batch size) is matched,
    # either by having the same batch size or by having a batch size of 1 for ``sqrt(alpha_bar)``
    mean = get_value_per_time(sd.sqrt_alpha_cumulative, t=timesteps) * x0

    # Noise scaled
    std_dev = get_value_per_time(sd.sqrt_one_minus_alpha_cumulative, t=timesteps)

    # New image (forward pass)
    sample = mean + std_dev * eps

    return sample, eps  # return ... , gt noise --> model predicts this
