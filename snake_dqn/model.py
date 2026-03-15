import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_size=24, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    model = DQN()
    x = torch.randn(1, 24)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")
    assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"
    print("Model OK!")
