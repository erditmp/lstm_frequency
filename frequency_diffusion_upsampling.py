import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

# Configuration
UPSAMPLE_STEPS = 59  # Number of minute-level steps to generate between hourly data points
FEATURE_COLUMNS = [None]  # Set dynamically based on dataframe if None

class WindTurbineFrequencyDataset:
    """Prepare frequency domain sequences and conditions for diffusion model."""
    def __init__(self, dataframe, upsample_steps=UPSAMPLE_STEPS, feature_columns=None):
        self.data = dataframe.copy()
        self.upsample_steps = upsample_steps
        global FEATURE_COLUMNS
        if feature_columns is not None:
            FEATURE_COLUMNS = feature_columns
        elif FEATURE_COLUMNS == [None]:
            FEATURE_COLUMNS = [col for col in self.data.columns if col != "time"]

        self.data["time"] = pd.to_datetime(self.data["time"])
        self.data = self.data.sort_values("time")

        # Scale features
        self.scalers = {}
        self.normalized_features = pd.DataFrame(index=self.data.index)
        for feature in FEATURE_COLUMNS:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.scalers[feature] = scaler
            self.normalized_features[feature] = scaler.fit_transform(self.data[[feature]])

        # Hourly averages
        self.hourly_data = self.data.copy()
        self.hourly_data["hour"] = self.hourly_data["time"].dt.floor("H")
        self.hourly_data = self.hourly_data.groupby("hour")[FEATURE_COLUMNS].mean().reset_index()
        self.hourly_data = self.hourly_data.rename(columns={"hour": "time"})

        self.normalized_hourly = pd.DataFrame(index=self.hourly_data.index)
        for feature in FEATURE_COLUMNS:
            self.normalized_hourly[feature] = self.scalers[feature].transform(self.hourly_data[[feature]])

        self.freq_len = upsample_steps // 2 + 1
        self.freq_dim = self.freq_len * len(FEATURE_COLUMNS) * 2
        self.sequences = []
        self.conditions = []

        for i in tqdm(range(len(self.hourly_data) - 1), desc="Preparing frequency sequences"):
            current_hour_time = self.hourly_data["time"].iloc[i]
            next_hour_time = self.hourly_data["time"].iloc[i + 1]

            minute_slice = self.data[(self.data["time"] > current_hour_time) &
                                     (self.data["time"] <= next_hour_time)].head(self.upsample_steps)
            if len(minute_slice) != self.upsample_steps:
                continue

            seq_norm = pd.DataFrame(index=minute_slice.index)
            for feature in FEATURE_COLUMNS:
                seq_norm[feature] = self.scalers[feature].transform(minute_slice[[feature]])
            seq = seq_norm[FEATURE_COLUMNS].values  # (upsample_steps, num_features)

            freq = np.fft.rfft(seq, axis=0)
            freq_flat = np.concatenate([freq.real, freq.imag], axis=0).reshape(-1)
            self.sequences.append(freq_flat)

            cond_current = self.normalized_hourly.iloc[i][FEATURE_COLUMNS].values
            cond_next = self.normalized_hourly.iloc[i + 1][FEATURE_COLUMNS].values
            condition = np.concatenate([cond_current, cond_next])
            self.conditions.append(condition)

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.conditions = torch.tensor(np.array(self.conditions), dtype=torch.float32)
        print(f"Prepared {len(self.sequences)} frequency sequences of dim {self.freq_dim}.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.conditions[idx]

    def inverse_transform_freq(self, freq_tensor):
        freq_np = freq_tensor.cpu().numpy().reshape(2, self.freq_len, len(FEATURE_COLUMNS))
        real = freq_np[0]
        imag = freq_np[1]
        complex_freq = real + 1j * imag
        seq_norm = np.fft.irfft(complex_freq, n=self.upsample_steps, axis=0)
        seq_original = np.zeros_like(seq_norm)
        for j, feature in enumerate(FEATURE_COLUMNS):
            seq_original[:, j] = self.scalers[feature].inverse_transform(seq_norm[:, j].reshape(-1, 1)).flatten()
        return seq_original

# Diffusion components

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos(((x + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.exp(-torch.arange(half, device=device) * torch.log(torch.tensor(10000.0)) / (half - 1))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class FrequencyConditionalDiffusionModel(nn.Module):
    def __init__(self, freq_dim, cond_dim, hidden_dim=256, n_timesteps=1000):
        super().__init__()
        self.freq_dim = freq_dim
        self.cond_dim = cond_dim
        self.n_timesteps = n_timesteps

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        input_dim = freq_dim + cond_dim + hidden_dim
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, freq_dim)
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1 / alphas_cumprod))

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        inp = torch.cat([x, cond, t_emb], dim=-1)
        return self.denoiser(inp)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_om * noise

    def p_losses(self, x_start, t, cond):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.forward(x_noisy, t, cond)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, device):
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.n_timesteps)), desc="Sampling", total=self.n_timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            beta_t = extract(self.betas, t, img.shape)
            sqrt_om = extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_rec = extract(self.sqrt_recip_alphas_cumprod, t, img.shape)
            pred_noise = self.forward(img, t, cond)
            model_mean = sqrt_rec * (img - beta_t * pred_noise / sqrt_om)
            if i == 0:
                img = model_mean
            else:
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(beta_t) * noise
        return img

    @torch.no_grad()
    def sample(self, num_samples, cond, device):
        shape = (num_samples, self.freq_dim)
        return self.p_sample_loop(shape, cond.to(device), device)

# Training utilities

def train_model(model, dataset, epochs=100, batch_size=32, lr=1e-3, device='cpu', save_path='best_frequency_model.pth'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, cond in dataloader:
            x, cond = x.to(device), cond.to(device)
            t = torch.randint(0, model.n_timesteps, (x.size(0),), device=device).long()
            loss = model.p_losses(x, t, cond)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} (loss {best_loss:.4f})")
    print("Training complete.")

@torch.no_grad()
def upsample_hourly(model, hourly_df, dataset, device='cpu'):
    model.eval()
    model.to(device)
    normalized = pd.DataFrame(index=hourly_df.index)
    for feature in FEATURE_COLUMNS:
        normalized[feature] = dataset.scalers[feature].transform(hourly_df[[feature]])
    cond_list = []
    for i in range(len(hourly_df) - 1):
        cond_curr = normalized.iloc[i][FEATURE_COLUMNS].values
        cond_next = normalized.iloc[i + 1][FEATURE_COLUMNS].values
        cond_list.append(np.concatenate([cond_curr, cond_next]))
    cond_tensor = torch.tensor(np.array(cond_list), dtype=torch.float32).to(device)
    freq_samples = model.sample(len(cond_tensor), cond_tensor, device).cpu()
    sequences = []
    for freq in freq_samples:
        seq = dataset.inverse_transform_freq(freq)
        sequences.append(seq)
    return np.array(sequences)

if __name__ == "__main__":
    # Dummy dataset for demonstration
    num_hours = 50
    start_time = pd.to_datetime('2023-01-01 00:00:00')
    hourly_times = [start_time + timedelta(hours=i) for i in range(num_hours)]
    hourly_values = np.random.rand(num_hours, 1) * 1000
    minute_list = []
    for i in range(num_hours - 1):
        cur = hourly_values[i, 0]
        nxt = hourly_values[i + 1, 0]
        minutes = np.linspace(cur, nxt, UPSAMPLE_STEPS + 1)[1:]
        for j in range(UPSAMPLE_STEPS):
            minute_list.append({'time': hourly_times[i] + timedelta(minutes=j+1),
                                'active_power_avg': minutes[j]})
    df = pd.DataFrame(minute_list)

    dataset = WindTurbineFrequencyDataset(df, upsample_steps=UPSAMPLE_STEPS, feature_columns=['active_power_avg'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FrequencyConditionalDiffusionModel(freq_dim=dataset.freq_dim,
                                               cond_dim=dataset.conditions.shape[1])
    train_model(model, dataset, epochs=50, batch_size=32, lr=1e-3,
                device=device, save_path='best_frequency_diffusion_model.pth')

    # Example upsampling and plot
    example_hourly = dataset.hourly_data.iloc[10:12].copy()
    upsampled = upsample_hourly(model, example_hourly, dataset, device=device)
    if len(upsampled) > 0:
        plt.figure(figsize=(12,6))
        t0 = example_hourly['time'].iloc[0]
        t1 = example_hourly['time'].iloc[1]
        plt.plot([t0, t1], example_hourly['active_power_avg'].values, 'ro', label='Hourly')
        minutes = [t0 + timedelta(minutes=i+1) for i in range(UPSAMPLE_STEPS)]
        plt.plot(minutes, upsampled[0][:,0], 'b-', label='Predicted minutes')
        actual = dataset.data[(dataset.data['time'] > t0) & (dataset.data['time'] <= t1)].head(UPSAMPLE_STEPS)
        if len(actual) == UPSAMPLE_STEPS:
            plt.plot(actual['time'], actual['active_power_avg'], 'g--', label='Actual minutes')
        plt.legend(); plt.xlabel('Time'); plt.ylabel('active_power_avg'); plt.title('Frequency Diffusion Upsampling')
        plt.grid(True); plt.tight_layout(); plt.show()
