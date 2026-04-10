import torch
import torch.nn as nn

SCALAR_SIZE = 24
GRID_CHANNELS = 3
GRID_SIZE = 9


class DQN(nn.Module):
    def __init__(self, input_size=SCALAR_SIZE + GRID_CHANNELS * GRID_SIZE * GRID_SIZE,
                 output_size=3):
        super().__init__()
        self.scalar_size = SCALAR_SIZE
        self.grid_shape = (GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

        # Path A: scalar features MLP
        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_SIZE, 128),
            nn.ReLU(),
        )

        # Path B: local grid CNN
        self.conv_net = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 16, kernel_size=3),  # 9x9 -> 7x7
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),             # 7x7 -> 5x5
            nn.ReLU(),
        )
        conv_spatial = GRID_SIZE - 4  # two 3x3 convs shrink by 2 each
        conv_out_size = 16 * conv_spatial * conv_spatial

        # Merge and output
        self.head = nn.Sequential(
            nn.Linear(128 + conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 24 + 147) — split into scalar and grid
        scalar = x[:, :SCALAR_SIZE]
        grid = x[:, SCALAR_SIZE:].reshape(-1, *self.grid_shape)

        s = self.scalar_net(scalar)
        g = self.conv_net(grid)
        g = g.reshape(g.size(0), -1)
        combined = torch.cat([s, g], dim=1)
        return self.head(combined)


if __name__ == "__main__":
    model = DQN()
    x = torch.randn(1, SCALAR_SIZE + GRID_CHANNELS * GRID_SIZE * GRID_SIZE)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model OK!")
