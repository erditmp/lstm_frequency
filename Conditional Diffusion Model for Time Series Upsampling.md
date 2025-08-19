# Conditional Diffusion Model for Time Series Upsampling

This project implements a conditional diffusion model to upsample hourly wind turbine SCADA data to minute-level resolution. The model generates 59 intermediate timesteps between consecutive hourly data points, aiming to capture minute-level changes, sudden shifts, and volatility.

## Project Structure

-   `upsampling_diffusion_model.py`: The main Python script containing the implementation of the conditional diffusion model, data preprocessing, training (with best model saving).
-   `upsampling_inference.py`: A separate script for loading a trained model and performing upsampling inference with enhanced plotting.
-   `best_upsampling_model.pth`: The saved weights of the best-performing diffusion model after training.
-   `RF_FREQUENCYBASED_LONGDATASET.py`: (Provided by user) An example of an LSTM-based frequency domain upsampling approach.
-   `diffusion.py`: (Provided by user) A reference implementation of a diffusion model.

## Model Overview

The core of this project is a conditional diffusion model adapted for time series upsampling. The model learns to generate the minute-level data distribution conditioned on the preceding hourly average. This allows it to fill in the 59 minute-level data points between two hourly measurements.

### Key Components:

1.  **Data Preprocessing (`WindTurbineDataset` class):**
    -   Loads and normalizes the minute-level SCADA data.
    -   Calculates hourly averages from the minute-level data.
    -   Prepares sequences of 59 minute-level data points as targets, with the corresponding hourly average as the conditioning input.

2.  **Conditional Diffusion Model (`ConditionalDiffusionModel` class):**
    -   **Denoising Network (MLP):** A Multi-Layer Perceptron (MLP) that predicts the noise added to a noisy time series sample at a given timestep, conditioned on the hourly data.
    -   **Diffusion Process:** Uses a cosine beta schedule for noise addition, as proposed in recent diffusion model literature.
    -   **Sampling Process:** Iteratively denoises a pure noise sample, guided by the hourly condition, to generate the upsampled minute-level sequence.

3.  **Training and Inference Functions:**
    -   `train_diffusion_model`: Handles the training loop, optimizing the denoising network to predict the noise. It now saves the model with the lowest validation loss to `best_upsampling_model.pth`.
    -   `upsample_data`: Performs the upsampling inference, taking hourly data as input and generating minute-level predictions.

## Getting Started

### Prerequisites

-   Python 3.x
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `torch` (PyTorch)
-   `tqdm`
-   `matplotlib`

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu tqdm matplotlib
```

### Data

The scripts currently use a dummy dataset for demonstration purposes. To use your actual SCADA data, you need to:

1.  **Replace the dummy data loading** in the `if __name__ == "__main__":` block of `upsampling_diffusion_model.py` and `upsampling_inference.py` with your actual data loading mechanism. Ensure your DataFrame has a `time` column (datetime objects) and the feature columns you want to upsample (e.g., `active_power_avg`).

    ```python
    # Example for loading your data:
    # df = pd.read_csv("path/to/your/scada_dataset.csv")
    # df["time"] = pd.to_datetime(df["time"])
    # dummy_df = df # Or preprocess df to match the expected minute-level format
    ```

2.  **Update `FEATURE_COLUMNS`**: Modify the `FEATURE_COLUMNS` list at the top of `upsampling_diffusion_model.py` and `upsampling_inference.py` to match the names of the features in your dataset that you intend to upsample.

    ```python
    FEATURE_COLUMNS = ["your_feature_1", "your_feature_2"] # e.g., ["active_power_avg"]
    ```

### Running the Training Script

To train the model and save the best weights:

```bash
python upsampling_diffusion_model.py
```

The script will:

1.  Prepare the dataset.
2.  Train the conditional diffusion model and save the best model to `best_upsampling_model.pth`.
3.  Perform a small example upsampling and generate a plot.

### Running the Inference Script

To perform upsampling using a trained model and visualize the results:

```bash
python upsampling_inference.py
```

The script will:

1.  Load the trained model from `best_upsampling_model.pth`.
2.  Perform upsampling on a small example of hourly data.
3.  Generate plots comparing the original hourly data, the predicted minute-level data, and the actual minute-level data (if available in your dataset).

## Configuration

You can adjust the following parameters in `upsampling_diffusion_model.py` and `upsampling_inference.py`:

-   `UPSAMPLE_STEPS`: The number of minute-level steps to generate between hourly data points. Default is 59.
-   `FEATURE_COLUMNS`: A list of strings, specifying the names of the columns in your dataset that contain the features to be upsampled.
-   `epochs`: (Training script only) Number of training epochs for the diffusion model.
-   `batch_size`: (Training script only) Batch size for training.
-   `learning_rate`: (Training script only) Learning rate for the Adam optimizer.
-   `n_timesteps`: Number of diffusion timesteps (influences the quality and training time of the diffusion model).
-   `hidden_dim`: Hidden dimension for the MLP in the denoising network.

## Conditioning Mechanism

The model uses the hourly average of the features as the condition for generating the minute-level data. This means that for each hour, the model is provided with the average value of the features during that hour, and it learns to generate the 59 minute-level data points that are consistent with this average and the overall data distribution.

## Performance and Limitations

### Performance:

-   **Capturing Volatility:** Diffusion models are known for their ability to generate diverse and high-fidelity samples, which makes them suitable for capturing the complex, non-linear patterns, sudden shifts, and volatility often present in SCADA data.
-   **Flexibility:** The conditional nature allows the model to adapt its upsampling based on the specific hourly conditions, providing more realistic interpolations than simpler methods.
-   **Generative Capability:** Unlike interpolation methods, diffusion models can generate multiple plausible upsampled sequences for the same hourly input, reflecting the inherent uncertainty in real-world data.

### Limitations:

-   **Computational Cost:** Training diffusion models can be computationally intensive and time-consuming, especially for large datasets and many diffusion timesteps.
-   **Data Requirements:** Requires a sufficiently large and representative minute-level dataset for training to learn the underlying data distribution effectively.
-   **Hyperparameter Tuning:** Performance is sensitive to hyperparameters such as `n_timesteps`, `hidden_dim`, learning rate, and the choice of beta schedule.
-   **Boundary Conditions:** While the model is conditioned on the preceding hourly average, ensuring perfect continuity and consistency with the *next* hourly average (if available) might require additional mechanisms or post-processing.

## Future Work

-   **Evaluation Metrics:** Implement more sophisticated evaluation metrics specifically designed for time series generation and upsampling (e.g., statistical similarity, frequency domain analysis).
-   **Advanced Architectures:** Explore more complex neural network architectures for the denoising model, such as U-Nets or Transformer-based models, which are commonly used in state-of-the-art diffusion models for sequence generation.
-   **More Complex Conditioning:** Investigate incorporating additional conditioning information, such as the previous minute's data, or the start and end hourly values for the upsampling interval.
-   **Real-world Data Testing:** Test the model extensively on diverse real-world SCADA datasets to assess its robustness and generalization capabilities.



