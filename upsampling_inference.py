
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

# Configuration (must match training script)
UPSAMPLE_STEPS = 59  # Number of minute-level steps to generate between hourly data points
FEATURE_COLUMNS = [None] # Features to upsample, will be set dynamically

# 1. Data Preprocessing (re-using the class from training script)
class WindTurbineDataset:
    def __init__(self, dataframe, upsample_steps=UPSAMPLE_STEPS, feature_columns=None):
        self.data = dataframe.copy()
        self.upsample_steps = upsample_steps
        global FEATURE_COLUMNS
        if feature_columns is not None:
            FEATURE_COLUMNS = feature_columns
        elif FEATURE_COLUMNS == [None]: # If not set, use all columns except 'time'
            FEATURE_COLUMNS = [col for col in self.data.columns if col != 'time']

        # Convert time column to datetime
        self.data["time"] = pd.to_datetime(self.data["time"])
        self.data = self.data.sort_values("time")

        # Normalize data
        self.scalers = {}
        self.normalized_features = pd.DataFrame(index=self.data.index) # Renamed to avoid confusion
        for feature in FEATURE_COLUMNS:
            self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
            self.normalized_features[feature] = self.scalers[feature].fit_transform(self.data[[feature]])

        # Create hourly averages
        self.hourly_data = self.data.copy()
        self.hourly_data["hour"] = self.hourly_data["time"].dt.floor("H")
        self.hourly_data = self.hourly_data.groupby("hour")[FEATURE_COLUMNS].mean().reset_index()
        self.hourly_data = self.hourly_data.rename(columns={"hour": "time"})

        # Normalize hourly data using the same scalers
        self.normalized_hourly = pd.DataFrame(index=self.hourly_data.index)
        for feature in FEATURE_COLUMNS:
            self.normalized_hourly[feature] = self.scalers[feature].transform(self.hourly_data[[feature]])

        self.sequences = []
        self.conditions = []

        # Prepare sequences for diffusion model
        # Each sequence will be `upsample_steps` minutes of data, conditioned by the preceding hourly average
        for i in tqdm(range(len(self.hourly_data) - 1), desc="Preparing sequences"):
            current_hour_time = self.hourly_data["time"].iloc[i]
            next_hour_time = self.hourly_data["time"].iloc[i+1]

            # Get minute data slice from original data, then normalize features
            minute_data_slice_original = self.data[
                (self.data["time"] > current_hour_time) &
                (self.data["time"] <= next_hour_time)
            ].head(self.upsample_steps)

            if len(minute_data_slice_original) == self.upsample_steps:
                # Normalize the features of this slice
                sequence_normalized = pd.DataFrame(index=minute_data_slice_original.index)
                for feature in FEATURE_COLUMNS:
                    sequence_normalized[feature] = self.scalers[feature].transform(minute_data_slice_original[[feature]])
                
                sequence = sequence_normalized[FEATURE_COLUMNS].values
                self.sequences.append(sequence)

                # Condition: current hour's average data
                condition = self.normalized_hourly.iloc[i][FEATURE_COLUMNS].values
                self.conditions.append(condition)

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.conditions = torch.tensor(np.array(self.conditions), dtype=torch.float32)

        print(f"Prepared {len(self.sequences)} sequences.")
        print(f"Sequences shape: {self.sequences.shape}") # (num_samples, upsample_steps, num_features)
        print(f"Conditions shape: {self.conditions.shape}") # (num_samples, num_features)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.conditions[idx]

    def inverse_transform(self, data, feature_idx):
        feature = FEATURE_COLUMNS[feature_idx]
        data_reshaped = data.reshape(-1, 1)
        return self.scalers[feature].inverse_transform(data_reshaped).flatten()

# 2. Diffusion Model Components (re-using classes from training script)

def cosine_beta_schedule(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos(((x + s) / (1 + s)) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, upsample_steps, num_features, n_timesteps=1000, hidden_dim=128):
        super().__init__()
        self.num_features = num_features
        self.upsample_steps = upsample_steps
        self.n_timesteps = n_timesteps

        # Model to predict noise (epsilon)
        # Input: noisy_sequence (upsample_steps * num_features), timestep_embedding (hidden_dim), condition (num_features)
        # Output: predicted_noise (upsample_steps * num_features)
        input_dim = (upsample_steps * num_features) + hidden_dim + num_features
        output_dim = upsample_steps * num_features
        self.denoising_model = MLP(input_dim, output_dim, hidden_dim)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / alphas_cumprod - 1))

    def forward(self, x, t, condition):
        # x: (batch_size, upsample_steps, num_features)
        # t: (batch_size,)
        # condition: (batch_size, num_features)

        t_embed = self.time_mlp(t)
        
        # Flatten x for MLP input
        x_flat = x.view(x.shape[0], -1)

        # Concatenate noisy input, timestep embedding, and condition
        model_input = torch.cat([x_flat, t_embed, condition], dim=-1)
        
        # Predict noise
        predicted_noise = self.denoising_model(model_input)
        
        return predicted_noise.view(x.shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, condition):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.forward(x_noisy, t, condition)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample_loop(self, shape, condition, device):
        batch_size = shape[0]
        # Start from pure noise (x_T)
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.n_timesteps)), desc='Sampling loop time step', total=self.n_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # 1. predict noise
            predicted_noise = self.forward(img, t, condition)

            # 2. get alphas
            betas_t = extract(self.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas_cumprod, t, img.shape)

            # 3. compute x_{t-1} mean
            model_mean = sqrt_recip_alphas_t * (img - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

            if i == 0:
                img = model_mean
            else:
                posterior_variance_t = extract(self.betas, t, img.shape)
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return img

    @torch.no_grad()
    def sample(self, num_samples, condition, device):
        # condition: (num_samples, num_features)
        sample_shape = (num_samples, self.upsample_steps, self.num_features)
        return self.p_sample_loop(sample_shape, condition, device)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.exp(-torch.arange(half_dim, device=device) * torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def upsample_data(model, hourly_data_df, dataset_obj, upsample_steps=UPSAMPLE_STEPS, device='cpu'):
    model.eval()
    model.to(device)

    all_upsampled_data = []

    # Normalize the input hourly data using the same scalers used for training
    normalized_hourly_input = pd.DataFrame(index=hourly_data_df.index)
    for feature in FEATURE_COLUMNS:
        normalized_hourly_input[feature] = dataset_obj.scalers[feature].transform(hourly_data_df[[feature]])
    
    normalized_hourly_input_tensor = torch.tensor(normalized_hourly_input[FEATURE_COLUMNS].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Generate upsampled sequences for each hourly data point
        upsampled_sequences_normalized = model.sample(len(normalized_hourly_input_tensor), normalized_hourly_input_tensor, device)
    
    # Inverse transform to original scale
    for i in range(upsampled_sequences_normalized.shape[0]):
        single_upsampled_sequence_normalized = upsampled_sequences_normalized[i].cpu().numpy()
        single_upsampled_sequence_original_scale = np.zeros_like(single_upsampled_sequence_normalized)
        for j, feature in enumerate(FEATURE_COLUMNS):
            single_upsampled_sequence_original_scale[:, j] = dataset_obj.inverse_transform(single_upsampled_sequence_normalized[:, j], j)
        all_upsampled_data.append(single_upsampled_sequence_original_scale)

    return np.array(all_upsampled_data)


if __name__ == "__main__":
    # --- Configuration for Inference ---
    MODEL_PATH = 'best_upsampling_model.pth'
    # Set this to your actual data file or keep dummy data generation for testing
    DATA_PATH = None # 'path/to/your/scada_dataset.csv'
    
    # If using your own data, make sure FEATURE_COLUMNS matches your dataset
    # FEATURE_COLUMNS = ['active_power_avg', 'another_feature'] 
    # For dummy data, it's set dynamically in WindTurbineDataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Data ---
    if DATA_PATH:
        # Load your actual dataset
        df = pd.read_csv(DATA_PATH)
        df["time"] = pd.to_datetime(df["time"])
        # You might need to preprocess df to match the expected minute-level format
        # For example, if your CSV is already minute-level:
        dummy_df = df.copy() # Rename for consistency with the dataset class
        # If FEATURE_COLUMNS is not set globally, it will be inferred from df
        # You can also explicitly set it here: FEATURE_COLUMNS = ['your_feature_name']
    else:
        # Creating dummy dataset (must match the structure used during training)
        print("Creating dummy dataset for inference...")
        num_hours = 100
        start_time = pd.to_datetime('2023-01-01 00:00:00')
        hourly_times = [start_time + timedelta(hours=i) for i in range(num_hours)]
        hourly_values = np.random.rand(num_hours, 1) * 1000 # Example hourly values for one feature
        
        minute_data_list = []
        for i in range(num_hours - 1):
            current_hour_val = hourly_values[i, 0]
            next_hour_val = hourly_values[i+1, 0]
            minute_vals = np.linspace(current_hour_val, next_hour_val, UPSAMPLE_STEPS + 1)[1:]
            for j in range(UPSAMPLE_STEPS):
                minute_data_list.append({
                    'time': hourly_times[i] + timedelta(minutes=j+1),
                    'active_power_avg': minute_vals[j]
                })
        dummy_df = pd.DataFrame(minute_data_list)
        print(f"Dummy dataset shape: {dummy_df.shape}")

    # Initialize dataset object to get scalers and hourly data structure
    # Ensure FEATURE_COLUMNS is correctly set before initializing dataset
    if FEATURE_COLUMNS == [None]: # If still default, infer from dummy_df
        FEATURE_COLUMNS = [col for col in dummy_df.columns if col != 'time']
    dataset = WindTurbineDataset(dummy_df, upsample_steps=UPSAMPLE_STEPS, feature_columns=FEATURE_COLUMNS)

    # --- Load Model ---
    num_features = len(FEATURE_COLUMNS)
    model = ConditionalDiffusionModel(upsample_steps=UPSAMPLE_STEPS, num_features=num_features).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Please train the model first using upsampling_diffusion_model.py.")
        print("Exiting inference script.")
        exit()

    # --- Perform Upsampling ---
    print("Performing upsampling on example hourly data...")
    # Take a few hourly data points to upsample
    # For actual data, select a relevant slice of your hourly data
    example_hourly_df = dataset.hourly_data.iloc[10:15].copy() # Upsample for 5 hours
    
    upsampled_output = upsample_data(model, example_hourly_df, dataset, upsample_steps=UPSAMPLE_STEPS, device=device)

    print(f"Upsampled output shape: {upsampled_output.shape}") # (num_hours_to_upsample, upsample_steps, num_features)

    # --- Plotting Original vs. Predicted ---
    if len(upsampled_output) > 0:
        num_hours_to_plot = upsampled_output.shape[0]
        num_features_to_plot = upsampled_output.shape[2]

        for feature_idx in range(num_features_to_plot):
            feature_name = FEATURE_COLUMNS[feature_idx]
            plt.figure(figsize=(15, 7))
            
            for i in range(num_hours_to_plot):
                current_hour_time = example_hourly_df['time'].iloc[i]
                current_hour_value = example_hourly_df[feature_name].iloc[i]
                
                # Get the actual minute data for this hour from the original dataset
                # This assumes your dummy_df (or loaded actual df) contains the minute-level data
                actual_minute_data_slice = dataset.data[
                    (dataset.data['time'] > current_hour_time) &
                    (dataset.data['time'] <= current_hour_time + timedelta(hours=1))
                ].head(UPSAMPLE_STEPS)

                # Time points for plotting
                plot_start_time = current_hour_time
                plot_end_time = current_hour_time + timedelta(minutes=UPSAMPLE_STEPS)
                time_range = pd.date_range(start=plot_start_time, end=plot_end_time, periods=UPSAMPLE_STEPS + 1)[1:]

                # Plot original hourly point
                plt.plot(current_hour_time, current_hour_value, 'go', markersize=8, label='Original Hourly Data' if i == 0 else "")

                # Plot predicted minute data
                plt.plot(time_range, upsampled_output[i, :, feature_idx], 'b-', alpha=0.7, label='Predicted Minute Data' if i == 0 else "")

                # Plot actual minute data if available
                if not actual_minute_data_slice.empty and len(actual_minute_data_slice) == UPSAMPLE_STEPS:
                    plt.plot(actual_minute_data_slice['time'], actual_minute_data_slice[feature_name], 'r--', alpha=0.7, label='Actual Minute Data' if i == 0 else "")
                elif not actual_minute_data_slice.empty:
                    print(f"Warning: Incomplete actual minute data for hour starting {current_hour_time}. Expected {UPSAMPLE_STEPS}, got {len(actual_minute_data_slice)}.")

            plt.xlabel('Time')
            plt.ylabel(feature_name)
            plt.title(f'Upsampling Comparison for {feature_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    print("Inference script finished. Check the generated plots.")



