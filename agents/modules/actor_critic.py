import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetStack(nn.Module):
    def __init__(self, c_in, c, device=None):
        super().__init__()
        
        self.conv_in = nn.Conv2d(c_in, c, kernel_size=3, stride=1, padding=1, device=device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = self.conv_in(x)
        x_in = self.maxpool(x)
        # residual block
        x = self.conv1(F.relu(x_in))
        x = self.conv2(F.relu(x)) + x_in
        return x


class ImpalaEncoder(nn.Module):
    """Goal-conditioned image encoder: encodes [B, H, W, C] observation and goal frames (0..255), stacked on the
    channel axis."""

    def __init__(self, cfg, device=None):
        super().__init__()

        channels, out_dim = cfg.arch_mlp['conv_channels'], cfg.arch_mlp['h_dim']
        img_h, img_w, img_c = cfg.obs_shape
        c = [2 * img_c, *channels]

        self.resnet = nn.Sequential(*[ResnetStack(c[i], c[i+1], device) for i in range(len(c)-1)])

        with torch.no_grad():
            dummy_input = torch.randn(1, 2 * img_c, img_h, img_w, device=device)
            conv_out_flat_dim = self.resnet(dummy_input).numel()

        self.linear = nn.Linear(conv_out_flat_dim, out_dim, device=device)

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)             # [B, H, W, 2C]
        x = x.permute(0, 3, 1, 2).float() / 255.0      # [B, 2C, H, W]
        x = F.relu(self.resnet(x))
        x = F.gelu(self.linear(x.reshape(*x.shape[:-3], -1)))
        return x


class Actor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_mu = nn.Linear(h_dim, a_dim, device=device)

    def forward(self, x, g):
        x = self.gc_encoder(x, g) if self.gc_encoder is not None else torch.cat([x, g], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_mu(x).tanh()
        return x


class QNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim + a_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim + a_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, a, g):
        x = self.gc_encoder(x, g) if self.gc_encoder is not None else torch.cat([x, g], dim=-1)
        x = torch.cat([x, a], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x


class VNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, h_dim = cfg.obs_shape[-1], cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, g):
        x = self.gc_encoder(x, g) if self.gc_encoder is not None else torch.cat([x, g], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x


class GoalActor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp['h_dim']

        if cfg.obs == 'rgb':
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, obs_dim, device=device)

    def forward(self, x, g):
        x = self.gc_encoder(x, g) if self.gc_encoder is not None else torch.cat([x, g], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x
