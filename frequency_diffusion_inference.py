import torch
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import os

from frequency_diffusion_upsampling import (
    WindTurbineFrequencyDataset,
    FrequencyConditionalDiffusionModel,
    upsample_hourly,
    UPSAMPLE_STEPS,
    FEATURE_COLUMNS
)

if __name__ == "__main__":
    MODEL_PATH = 'best_frequency_diffusion_model.pth'
    DATA_PATH = None  # path to minute-level CSV; if None, use dummy data as in training

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if DATA_PATH and os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['time'] = pd.to_datetime(df['time'])
    else:
        # Create dummy data identical to training script
        print("Creating dummy dataset for inference...")
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
                minute_list.append({
                    'time': hourly_times[i] + timedelta(minutes=j+1),
                    'active_power_avg': minutes[j]
                })
        df = pd.DataFrame(minute_list)
        print(f"Dummy dataset shape: {df.shape}")

    if FEATURE_COLUMNS == [None]:
        cols = [col for col in df.columns if col != 'time']
    else:
        cols = FEATURE_COLUMNS

    dataset = WindTurbineFrequencyDataset(df, upsample_steps=UPSAMPLE_STEPS, feature_columns=cols)

    model = FrequencyConditionalDiffusionModel(freq_dim=dataset.freq_dim,
                                               cond_dim=dataset.conditions.shape[1])
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train the model first.")

    example_hourly = dataset.hourly_data.iloc[10:15].copy()
    upsampled = upsample_hourly(model, example_hourly, dataset, device=device)

    if len(upsampled) > 0:
        num_hours_to_plot = upsampled.shape[0]
        for feature_idx, feature_name in enumerate(cols):
            plt.figure(figsize=(15,6))
            for i in range(num_hours_to_plot):
                t0 = example_hourly['time'].iloc[i]
                t1 = example_hourly['time'].iloc[i+1]
                plt.plot([t0, t1], example_hourly[feature_name].iloc[i:i+2], 'ro')
                minutes = [t0 + timedelta(minutes=j+1) for j in range(UPSAMPLE_STEPS)]
                plt.plot(minutes, upsampled[i][:,feature_idx], 'b-', label='Predicted' if i==0 else '')
                actual = dataset.data[(dataset.data['time']>t0) & (dataset.data['time']<=t1)].head(UPSAMPLE_STEPS)
                if len(actual) == UPSAMPLE_STEPS:
                    plt.plot(actual['time'], actual[feature_name], 'g--', label='Actual' if i==0 else '')
            plt.title(f'Upsampling Comparison for {feature_name}')
            plt.xlabel('Time')
            plt.ylabel(feature_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    print("Inference complete.")
