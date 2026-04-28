import torch
import torch.nn as nn


class CNNLoss(nn.Module):
    def __init__(self, spectral_alpha: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.spectral_alpha = spectral_alpha
        self.eps = eps

    def forward(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        pearson_loss = self.negative_pearson(predicted_ppg, target_ppg)
        if self.spectral_alpha <= 0:
            return pearson_loss
        return pearson_loss + self.spectral_alpha * self.spectral_loss(predicted_ppg, target_ppg)

    def negative_pearson(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        predicted_ppg = predicted_ppg - predicted_ppg.mean(dim=1, keepdim=True)
        target_ppg = target_ppg - target_ppg.mean(dim=1, keepdim=True)

        numerator = torch.sum(predicted_ppg * target_ppg, dim=1)
        denominator = torch.sqrt(
            torch.sum(predicted_ppg.pow(2), dim=1)
            * torch.sum(target_ppg.pow(2), dim=1)
            + self.eps
        )
        correlation = numerator / denominator
        return torch.mean(1 - correlation)

    def spectral_loss(self, predicted_ppg: torch.Tensor, target_ppg: torch.Tensor) -> torch.Tensor:
        predicted_spectrum = torch.abs(torch.fft.rfft(predicted_ppg, dim=1))
        target_spectrum = torch.abs(torch.fft.rfft(target_ppg, dim=1))

        predicted_spectrum = predicted_spectrum / (predicted_spectrum.sum(dim=1, keepdim=True) + self.eps)
        target_spectrum = target_spectrum / (target_spectrum.sum(dim=1, keepdim=True) + self.eps)

        return torch.mean(torch.abs(predicted_spectrum - target_spectrum))
