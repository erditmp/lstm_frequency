#################################################################################################################################
# Random Forest - Frequency Based

################################################################################################################################
# in this script I created a random forest model for upsampling in frequency based:

#################################################################################################################################
# Step 1: First Data Preprocessing Step:
#########################################################

##### Importing Libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import CubicSpline
from scipy.fft import fft, fftfreq

# set random seeds for reproducibility:
np.random.seed(42)  # it sets numpys random number generator to 42. this is important for reproducibility in experiments.

# Load the dataset:
print("Loading dataset...")
df = pd.read_csv("nkg_randomforest/reduced_dataset_first_1percent.csv")
print(f"Original shape: {df.shape}")


# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# Display basic info about the dataset
print(f"Time range: {df['time'].min()} to {df['time'].max()}")

# Select features for our model 
feature_columns = ['active_power_avg']
target_columns = feature_columns  # We'll predict the same features we use as input

# Check for missing values in selected features
print("Missing values in selected features:")
print(df[feature_columns].isna().sum())

# Clean data - remove NaN values
df_clean = df.dropna(subset=feature_columns)
print(f"Shape after cleaning: {df_clean.shape}")

# Data preprocessing
class WindTurbineDataset:
    def __init__(self, dataframe, prediction_length=59, use_fft_features=False, n_fft_features=40):
        """
        Dataset for wind turbine time series for scikit-learn with FFT-based targets.
        
        Args:
            dataframe: DataFrame with time series data
            prediction_length: Length of output sequence to predict (minutes per hour)
            use_fft_features: Whether to include FFT features from cubic spline analysis (deprecated - now always False)
            n_fft_features: Number of FFT coefficients to predict
        """
        self.data = dataframe.copy()
        self.prediction_length = prediction_length
        self.use_fft_features = False  # Always False for new approach
        self.n_fft_features = n_fft_features
        
        # Normalize data using MinMaxScaler
        self.scalers = {}
        self.normalized_data = pd.DataFrame(index=self.data.index)
        
        for feature in feature_columns:
            self.scalers[feature] = MinMaxScaler(feature_range=(0, 1))
            self.normalized_data[feature] = self.scalers[feature].fit_transform(self.data[[feature]])
        
        # Create hourly averages from minute data
        self.hourly_data = self.data.copy()
        self.hourly_data['hour'] = self.hourly_data['time'].dt.floor('H')
        self.hourly_data = self.hourly_data.groupby('hour')[feature_columns].mean().reset_index()
        self.hourly_data = self.hourly_data.rename(columns={'hour': 'time'})
        
        # Normalize hourly data
        self.normalized_hourly = pd.DataFrame(index=self.hourly_data.index)
        for feature in feature_columns:
            self.normalized_hourly[feature] = self.scalers[feature].transform(self.hourly_data[[feature]])
        
        # Compute cubic spline interpolation and FFT targets
        print("Computing cubic spline interpolation and FFT coefficients as targets...")
        self._compute_cubic_spline_and_fft_targets()
        
        # For each hour, create input features and FFT coefficient targets
        X_list = []  # [hourly_features, minute_position]
        y_list = []  # FFT coefficients (magnitude and phase)
        
        for i in range(len(self.hourly_data) - 1):
            hour_time = self.hourly_data['time'].iloc[i]
            # Find corresponding minute data for this hour
            minute_data_indices = (self.data['time'] >= hour_time) & (self.data['time'] < hour_time + timedelta(hours=1))
            minute_data = self.normalized_data[minute_data_indices]
            
            if len(minute_data) >= prediction_length:
                hourly_features = self.normalized_hourly.iloc[i].values
                
                # Get FFT coefficients for this hour
                if i < len(self.fft_coefficients_per_hour):
                    fft_coeffs = self.fft_coefficients_per_hour[i]
                    
                    for j in range(prediction_length):
                        # Input: [hourly_features, minute_position]
                        minute_pos = j / prediction_length
                        pos_encoding = [
                            np.sin(minute_pos * np.pi * 2),
                            np.cos(minute_pos * np.pi * 2),
                            np.sin(minute_pos * np.pi * 4),
                            np.cos(minute_pos * np.pi * 4)
                        ]
                        
                        # Combine features (no FFT features as input anymore)
                        input_features = np.concatenate([hourly_features, pos_encoding])
                        X_list.append(input_features)
                        
                        # Target: FFT coefficients (same for all minutes in this hour)
                        y_list.append(fft_coeffs)
        
        # Convert lists to numpy arrays for sklearn
        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)
        
        print(f"Created {len(self.X)} training samples")
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        print(f"Each target contains {self.y.shape[1]} FFT coefficients per feature")
        
        # Create feature names for interpretability
        self.feature_names = self._create_feature_names()
        print(f"Feature names: {self.feature_names}")
    
    def _compute_cubic_spline_and_fft_targets(self):
        """
        Compute cubic spline interpolation and FFT coefficients as targets.
        """
        # Convert time to hours since start
        start_time = min(self.data['time'].min(), self.hourly_data['time'].min())
        
        # Create time arrays in hours since start
        x_minute = ((self.data['time'] - start_time).dt.total_seconds() / 3600.0).values
        x_hour = ((self.hourly_data['time'] - start_time).dt.total_seconds() / 3600.0).values
        
        # Initialize storage for FFT coefficients organized by hour
        num_hours = len(self.hourly_data) - 1
        fft_coefficients_by_hour = [[] for _ in range(num_hours)]
        
        # Process each feature
        for feature in feature_columns:
            y_minute = self.data[feature].values
            y_hour = self.hourly_data[feature].values
            
            # Fit cubic splines
            cs_minute = CubicSpline(x_minute, y_minute, extrapolate=True)
            cs_hour = CubicSpline(x_hour, y_hour, extrapolate=True)
            
            # Evaluate hourly spline at minute-level time points
            interp_hour_at_minute = cs_hour(x_minute)
            
            # Compute differences (fluctuations)
            fluctuations = y_minute - interp_hour_at_minute
            
            # Split fluctuations into hourly segments and compute FFT
            for i in range(num_hours):
                hour_time = self.hourly_data['time'].iloc[i]
                minute_data_indices = (self.data['time'] >= hour_time) & (self.data['time'] < hour_time + timedelta(hours=1))
                hour_fluctuations = fluctuations[minute_data_indices]
                
                if len(hour_fluctuations) >= self.prediction_length:
                    # Take first prediction_length minutes
                    hour_fluctuations = hour_fluctuations[:self.prediction_length]
                    
                    # Compute FFT
                    fft_values = fft(hour_fluctuations)
                    
                    # Get frequency values using fftfreq (in cycles per hour)
                    freq = fftfreq(len(hour_fluctuations), d=1/60)  # d=1/60 since we sample every minute
                    
                    # Get all magnitudes to find top k components
                    all_magnitudes = np.abs(fft_values)
                    
                    # Find indices of top n_fft_features by magnitude (INCLUDING DC component)
                    # DC component (index 0) represents systematic offset and is important for reconstruction
                    top_indices = np.argsort(all_magnitudes)[-self.n_fft_features:][::-1]
                    
                    # Extract top magnitude, phase, and actual frequency values
                    magnitudes = all_magnitudes[top_indices]
                    phases = np.angle(fft_values[top_indices])
                    top_frequencies = freq[top_indices]  # Actual frequency values in cycles/hour
                    
                    # Combine magnitude, phase, and actual frequencies for reconstruction
                    fft_coeffs = np.concatenate([magnitudes, phases, top_frequencies])
                    
                    # Store coefficients for this hour and feature
                    fft_coefficients_by_hour[i].append(fft_coeffs)
        
        # Convert to final format: each hour has concatenated coefficients for all features
        self.fft_coefficients_per_hour = []
        for i in range(num_hours):
            if len(fft_coefficients_by_hour[i]) == len(feature_columns):
                # Concatenate coefficients for all features
                hour_coeffs = np.concatenate(fft_coefficients_by_hour[i])
                self.fft_coefficients_per_hour.append(hour_coeffs)
        
        print(f"Computed FFT coefficients for {len(self.fft_coefficients_per_hour)} hours")
        if self.fft_coefficients_per_hour:
            print(f"Each hour has {len(self.fft_coefficients_per_hour[0])} total coefficients")
            print(f"  Per feature: {self.n_fft_features} magnitude + {self.n_fft_features} phase + {self.n_fft_features} frequencies = {self.n_fft_features * 3} coefficients")
            print(f"  Total features: {len(feature_columns)}")
            print(f"  Expected total: {self.n_fft_features * 3 * len(feature_columns)}")
        else:
            print("No coefficients computed")
    
    def _create_feature_names(self):
        """Create descriptive names for all input features."""
        names = []
        
        # Hourly features
        for feature in feature_columns:
            names.append(f"hourly_{feature}")
        
        # Positional encoding features
        names.extend(['sin_pos_2pi', 'cos_pos_2pi', 'sin_pos_4pi', 'cos_pos_4pi'])
        
        return names
    
    def get_data(self):
        """Return X and y arrays for use with scikit-learn."""
        return self.X, self.y
    
    def inverse_transform(self, data, feature_idx):
        """Convert normalized values back to original scale for a specific feature."""
        feature = feature_columns[feature_idx]
        data_reshaped = data.reshape(-1, 1)
        return self.scalers[feature].inverse_transform(data_reshaped).flatten()
    

    def reconstruct_from_fft(self, fft_coefficients, feature_idx, hour_idx):
        """
        Reconstruct time series from FFT coefficients.
        
        Args:
            fft_coefficients: Predicted FFT coefficients for all features
            feature_idx: Index of the feature to reconstruct
            hour_idx: Hour index for getting baseline cubic spline
            
        Returns:
            Reconstructed time series for the specified feature
        """
        # Extract coefficients for the specified feature
        coeffs_per_feature = self.n_fft_features * 3  # magnitude + phase + frequency_value
        start_idx = feature_idx * coeffs_per_feature
        end_idx = start_idx + coeffs_per_feature
        
        feature_coeffs = fft_coefficients[start_idx:end_idx]
        
        # Split into magnitude, phase, and frequency values
        magnitudes = feature_coeffs[:self.n_fft_features]
        phases = feature_coeffs[self.n_fft_features:2*self.n_fft_features]
        frequencies = feature_coeffs[2*self.n_fft_features:]
        
        # Create frequency array for the prediction length using fftfreq
        freq_array = fftfreq(self.prediction_length, d=1/60)  # d=1/60 for minute sampling
        
        # Create zero-padded coefficient array for full FFT length
        padded_coeffs = np.zeros(self.prediction_length, dtype=complex)
        
        # Place the top coefficients at their closest frequency positions
        for i, target_freq in enumerate(frequencies):
            # Find closest frequency bin to the predicted frequency
            freq_idx = np.argmin(np.abs(freq_array - target_freq))
            
            if freq_idx < self.prediction_length:
                # Reconstruct complex coefficient from magnitude and phase
                complex_coeff = magnitudes[i] * np.exp(1j * phases[i])
                padded_coeffs[freq_idx] = complex_coeff
        
        # Apply inverse FFT to get fluctuations
        reconstructed_fluctuations = np.fft.ifft(padded_coeffs).real
        
        # Get the cubic spline baseline for this hour
        hour_time = self.hourly_data['time'].iloc[hour_idx]
        start_time = self.data['time'].min()
        x_hour = ((self.hourly_data['time'] - start_time).dt.total_seconds() / 3600.0).values
        y_hour = self.hourly_data[feature_columns[feature_idx]].values
        
        # Create minute-level time points for this hour
        hour_offset = (hour_time - start_time).total_seconds() / 3600.0
        x_minute_hour = np.linspace(hour_offset, hour_offset + 1, self.prediction_length)
        
        # Interpolate baseline
        cs_hour = CubicSpline(x_hour, y_hour, extrapolate=True)
        baseline = cs_hour(x_minute_hour)
        
        # Add fluctuations to baseline
        reconstructed = baseline + reconstructed_fluctuations
        
        return reconstructed

    
class WindTurbineMLPipeline:
    def __init__(self, model_params=None):
        """
        Complete ML pipeline for wind turbine data prediction using a single Random Forest model.
        
        Args:
            model_params: Dictionary of parameters for RandomForestRegressor
        """
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'random_state': 42,
                'n_jobs': -1,
                'max_depth': 12,
                'min_samples_leaf': 5
            }
        
        self.model_params = model_params
        self.model = None  # Single model instead of multiple models
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_score = None
        self.test_score = None
        self.feature_r2_scores = {}  # R2 scores for individual features
        self.train_hourly_data = None  # Hourly data from training set only
        self.test_hourly_data = None   # Hourly data from test set only
        self.train_df = None           # Training minute-level data
        self.test_df = None            # Test minute-level data
    
    def prepare_data(self, dataframe, prediction_length=59, test_size=0.3, random_state=42, 
                     use_fft_features=False, n_fft_features=40):
        """
        Prepare data using WindTurbineDataset and split into train/test.
        
        Args:
            dataframe: Input DataFrame
            prediction_length: Length of prediction sequence
            test_size: Fraction of data for testing
            random_state: Random state for reproducibility
            use_fft_features: Whether to include FFT features from cubic spline analysis
            n_fft_features: Number of top FFT features to include
        """
        print("Preparing dataset...")
        self.use_fft_features = use_fft_features
        self.n_fft_features = n_fft_features
        
        self.dataset = WindTurbineDataset(dataframe, prediction_length, use_fft_features, n_fft_features)
        X, y = self.dataset.get_data()
        
        print("Splitting data into train/test sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False  # Don't shuffle to maintain temporal order
        )
        
        print(f"Training set: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Test set: X={self.X_test.shape}, y={self.y_test.shape}")
        
        # Create separate hourly data for train and test to avoid data leakage
        print("Creating separate train/test hourly datasets...")
        self._create_train_test_hourly_data(dataframe, test_size)
    
    def _create_train_test_hourly_data(self, dataframe, test_size):
        """
        Create separate hourly datasets for training and testing to avoid data leakage.
        """
        # Sort dataframe by time
        df_sorted = dataframe.sort_values('time').reset_index(drop=True)
        
        # Split the original dataframe by time (not random) to maintain temporal integrity
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        print(f"Train data time range: {train_df['time'].min()} to {train_df['time'].max()}")
        print(f"Test data time range: {test_df['time'].min()} to {test_df['time'].max()}")
        
        # Create hourly data for training set
        train_hourly = train_df.copy()
        train_hourly['hour'] = train_hourly['time'].dt.floor('H')
        self.train_hourly_data = train_hourly.groupby('hour')[feature_columns].mean().reset_index()
        self.train_hourly_data = self.train_hourly_data.rename(columns={'hour': 'time'})
        
        # Create hourly data for test set  
        test_hourly = test_df.copy()
        test_hourly['hour'] = test_hourly['time'].dt.floor('H')
        self.test_hourly_data = test_hourly.groupby('hour')[feature_columns].mean().reset_index()
        self.test_hourly_data = self.test_hourly_data.rename(columns={'hour': 'time'})

        
        print(f"Training hourly data shape: {self.train_hourly_data.shape}")
        print(f"Test hourly data shape: {self.test_hourly_data.shape}")
        print("Data leakage avoided: Train and test hourly data are temporally separated")
        
        # Store the train/test DataFrames for accessing actual minute data later
        self.train_df = train_df
        self.test_df = test_df



        # Perform cubic spline analysis
        self._perform_cubic_spline_analysis()
        
    def _perform_cubic_spline_analysis(self):
        """
        Perform cubic spline interpolation analysis and FFT on the differences.
        """
        print("Performing cubic spline analysis.")
        
        # Convert time to hours since start for both datasets
        start_time = min(self.train_df['time'].min(), self.train_hourly_data['time'].min())
        end_time = max(self.train_df['time'].max(), self.train_hourly_data['time'].max())
        
        # Create time arrays in hours since start
        x_minute = ((self.train_df['time'] - start_time).dt.total_seconds() / 3600.0).values
        y_minute = self.train_df['active_power_avg'].values
        x_hour = ((self.train_hourly_data['time'] - start_time).dt.total_seconds() / 3600.0).values
        y_hour = self.train_hourly_data['active_power_avg'].values

        # Fit cubic splines (assumes data is sorted by time)
        print("Fitting cubic splines...")
        cs_minute = CubicSpline(x_minute, y_minute, extrapolate=True)
        cs_hour = CubicSpline(x_hour, y_hour, extrapolate=True)
        
        # Evaluate both splines at minute-level time points to get differences
        interp_minute_at_minute = cs_minute(x_minute)  # Should be identical to y_minute
        interp_hour_at_minute = cs_hour(x_minute)      # Hourly spline evaluated at minute points
        difference = interp_minute_at_minute - interp_hour_at_minute  # x_t = P_t - C_t
        
        # Store results for further analysis
        self.spline_results = {
            'x_minute': x_minute,
            'y_minute': y_minute,
            'x_hour': x_hour,
            'y_hour': y_hour,
            'interp_minute_at_minute': interp_minute_at_minute,
            'interp_hour_at_minute': interp_hour_at_minute,
            'difference': difference,
            'cs_minute': cs_minute,
            'cs_hour': cs_hour
        }

        # Create DataFrame for differences
        df_difference = pd.DataFrame({
            'time': self.train_df['time'],
            'difference': difference
        })

        # Compute FFT on the differences
        print("Computing FFT...")
        fft_values = fft(difference)
        freq = fftfreq(len(difference), d=1/60)  # Frequencies in cycles per hour (sampling interval 1/60 hours)

        # Get positive frequencies and magnitudes (excluding DC component)
        N = len(difference)
        positive_freq_idx = np.arange(1, N//2)
        frequencies = freq[positive_freq_idx]
        magnitudes = 2.0 / N * np.abs(fft_values[positive_freq_idx])
        
        # Store FFT results
        self.fft_results = {
            'fft_values': fft_values,
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'freq_full': freq
        }

        print(f"FFT analysis complete. Found {len(frequencies)} frequency components.")
        print(f"Frequency range: {frequencies.min():.4f} to {frequencies.max():.4f} cycles/hour")
        print(f"Max magnitude: {magnitudes.max():.4f} at frequency {frequencies[np.argmax(magnitudes)]:.4f} cycles/hour")

        # Plot the results
        # self._plot_spline_analysis(df_difference, interp_hour_at_minute)
        # self._plot_fft_analysis()
    
    def _plot_spline_analysis(self, df_difference, interp_hour_at_minute):
        """Plot the cubic spline analysis results."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time series comparison
        plt.subplot(2, 1, 1)
        plt.plot(self.train_df['time'], self.train_df['active_power_avg'], 
                 marker='o', linestyle='-', color='blue', alpha=0.7, markersize=2, label='P_t (Minute data)')
        plt.plot(self.train_hourly_data['time'], self.train_hourly_data['active_power_avg'], 
                 marker='s', linestyle='-', color='black', markersize=4, label='P_h (Hourly data)')
        plt.plot(self.train_df['time'], interp_hour_at_minute, 
                 marker='*', linestyle='--', color='orange', alpha=0.8, markersize=3, label='C_t (Hourly spline at minute points)')
        plt.xlabel('Time')
        plt.ylabel('Active Power')
        plt.title('Original Data and Cubic Spline Interpolations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 2: Differences
        plt.subplot(2, 1, 2)
        plt.plot(df_difference['time'], df_difference['difference'], 
                 marker='o', linestyle='-', color='pink', alpha=0.7, markersize=2, label='x_t = P_t - C_t')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Active Power Difference')
        plt.title('Difference Between Minute Data and Hourly Spline (x_t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_fft_analysis(self):
        """Plot the FFT analysis results."""
        frequencies = self.fft_results['frequencies']
        magnitudes = self.fft_results['magnitudes']
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Frequency spectrum (linear scale)
        plt.subplot(2, 2, 1)
        plt.plot(frequencies, magnitudes, 'b-', linewidth=1)
        plt.xlabel('Frequency (cycles/hour)')
        plt.ylabel('Magnitude')
        plt.title('FFT Magnitude Spectrum (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Frequency spectrum (log scale)
        plt.subplot(2, 2, 2)
        plt.semilogy(frequencies, magnitudes, 'b-', linewidth=1)
        plt.xlabel('Frequency (cycles/hour)')
        plt.ylabel('Magnitude (log scale)')
        plt.title('FFT Magnitude Spectrum (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Top frequency components
        plt.subplot(2, 2, 3)
        top_n = 20
        top_indices = np.argsort(magnitudes)[-top_n:][::-1]
        top_freqs = frequencies[top_indices]
        top_mags = magnitudes[top_indices]
        
        plt.bar(range(len(top_freqs)), top_mags)
        plt.xlabel('Rank')
        plt.ylabel('Magnitude')
        plt.title(f'Top {top_n} Frequency Components')
        plt.xticks(range(len(top_freqs)), [f'{f:.2f}' for f in top_freqs], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Power spectral density
        plt.subplot(2, 2, 4)
        power = magnitudes ** 2
        plt.plot(frequencies, power, 'r-', linewidth=1)
        plt.xlabel('Frequency (cycles/hour)')
        plt.ylabel('Power')
        plt.title('Power Spectral Density')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print dominant frequencies
        print("\nTop 10 dominant frequencies:")
        top_indices = np.argsort(magnitudes)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            freq_val = frequencies[idx]
            mag_val = magnitudes[idx]
            period_hours = 1/freq_val if freq_val > 0 else np.inf
            print(f"  {i+1:2d}. Frequency: {freq_val:8.4f} cycles/hour, "
                  f"Period: {period_hours:8.2f} hours, Magnitude: {mag_val:8.4f}")
    
    def get_spline_results(self):
        """Return the cubic spline analysis results."""
        if not hasattr(self, 'spline_results'):
            print("Spline analysis not performed yet. Call _perform_cubic_spline_analysis() first.")
            return None
        return self.spline_results
    
    def get_fft_results(self):
        """Return the FFT analysis results."""
        if not hasattr(self, 'fft_results'):
            print("FFT analysis not performed yet. Call _perform_cubic_spline_analysis() first.")
            return None
        return self.fft_results

    def _get_actual_minute_data(self, hourly_subset, feature_name, data_type, prediction_length=59):
        """
        Get actual minute-level data corresponding to the hourly predictions.
        
        Args:
            hourly_subset: DataFrame with subset of hourly data being predicted
            feature_name: Name of the feature to extract
            data_type: 'train' or 'test' - which data to use
            prediction_length: Number of minutes per hour
            
        Returns:
            List of actual values corresponding to the predicted time periods
        """
        if data_type == 'train':
            minute_df = self.train_df
        elif data_type == 'test':
            minute_df = self.test_df
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        
        actual_values = []
        
        for _, hour_row in hourly_subset.iterrows():
            hour_time = hour_row['time']
            
            # Find minute-level data for this hour
            hour_start = hour_time
            hour_end = hour_time + timedelta(hours=1)
            
            # Get minute data for this hour

            
            # This is a minute data from the test frame. But sometimes our test frame might have missing data in minute based.
            # In order to compare our real values with the predicted values, we need to have the same number of minutes.
            # So we are repeating the last value of the minute data for the missing minutes in order to have the same number of minutes.
            # these are might be the following shape
            """
            hour_minute_data.shape
            (4, 288)
            hour_minute_data.shape
            (49, 288)
            In this case, our original data do not have these values and these values are missing in original data so in the test_df daata
            Therefore we repeat the last value of the minute data for the missing minutes in order to have the same number of minutes.
            """
            hour_minute_data = minute_df[
                (minute_df['time'] >= hour_start) & 
                (minute_df['time'] < hour_end)
            ].sort_values('time')
            
            if len(hour_minute_data) > 0:
                # Take up to prediction_length values
                values_to_take = min(prediction_length, len(hour_minute_data))
                hour_values = hour_minute_data[feature_name].iloc[:values_to_take].tolist()
                actual_values.extend(hour_values)
                
                # If we have fewer minutes than prediction_length, pad with the last available value
                if len(hour_values) < prediction_length:
                    last_value = hour_values[-1] if hour_values else 0
                    actual_values.extend([last_value] * (prediction_length - len(hour_values)))
        
        return actual_values
    
    def train_models(self):
        """
        Train a single Random Forest model for all target features.
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("Training single Random Forest model for all features...")
        
        # Create and train single model for all features
        self.model = RandomForestRegressor(**self.model_params)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate overall training and test scores
        self.training_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)
        
        print(f"Overall Training R² score: {self.training_score:.4f}")
        print(f"Overall Test R² score: {self.test_score:.4f}")
        
        # Calculate individual feature R² scores
        y_pred = self.model.predict(self.X_test)
        for i, feature_name in enumerate(feature_columns):
            r2 = r2_score(self.y_test[:, i], y_pred[:, i])
            self.feature_r2_scores[feature_name] = r2
            print(f"  {feature_name} R² score: {r2:.4f}")
    
    def predict(self, X=None):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features. If None, uses test set.
            
        Returns:
            Predictions for all features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if X is None:
            X = self.X_test
        
        return self.model.predict(X)
    
    def predict_from_hourly(self, hourly_data=None, data_type='train', prediction_length=59):
        """
        Make predictions from hourly data, generating minute-level predictions using FFT reconstruction.
        
        Args:
            hourly_data: DataFrame with hourly data. If None, uses appropriate train/test hourly data.
            data_type: 'train' or 'test' - which hourly data to use when hourly_data is None
            prediction_length: Number of minutes to predict per hour (default 60)
            
        Returns:
            Predictions for all features across all minutes (reconstructed from FFT)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if hourly_data is None:
            # Use the appropriate hourly data (train or test) to avoid data leakage
            if data_type == 'train':
                if self.train_hourly_data is None:
                    raise ValueError("Training hourly data not available. Call prepare_data() first.")
                hourly_data = self.train_hourly_data
            elif data_type == 'test':
                if self.test_hourly_data is None:
                    raise ValueError("Test hourly data not available. Call prepare_data() first.")
                hourly_data = self.test_hourly_data
            else:
                raise ValueError("data_type must be 'train' or 'test'")
        
        # Normalize the hourly data using scalers fitted on training data only
        normalized_hourly = pd.DataFrame(index=hourly_data.index)
        for feature in feature_columns:
            normalized_hourly[feature] = self.dataset.scalers[feature].transform(hourly_data[[feature]])
        
        all_predictions = []
        
        for i in range(len(hourly_data)):
            hourly_features = normalized_hourly.iloc[i].values
            
            # Generate input features for a representative minute (we only need one since FFT coeffs are same for the hour)
            minute_pos = 0.5  # Use middle of hour as representative
            pos_encoding = [
                np.sin(minute_pos * np.pi * 2),
                np.cos(minute_pos * np.pi * 2),
                np.sin(minute_pos * np.pi * 4),
                np.cos(minute_pos * np.pi * 4)
            ]
            
            # Combine features
            input_features = np.concatenate([hourly_features, pos_encoding])
            
            # Predict FFT coefficients for this hour
            fft_coeffs_pred = self.model.predict(input_features.reshape(1, -1))[0]
            
            # Reconstruct time series from FFT coefficients for each feature
            hour_predictions = []
            for feature_idx in range(len(feature_columns)):
                reconstructed = self.dataset.reconstruct_from_fft(fft_coeffs_pred, feature_idx, i)
                hour_predictions.append(reconstructed)
            
            # Transpose to get (time_steps, features) format
            hour_predictions = np.array(hour_predictions).T  # Shape: (prediction_length, n_features)
            all_predictions.extend(hour_predictions)
        
        return np.array(all_predictions)
    
    def predict_from_hourly_and_inverse_transform(self, hourly_data=None, data_type='train', prediction_length=59):
        """
        Make predictions from hourly data and convert back to original scale using FFT reconstruction.
        
        Args:
            hourly_data: DataFrame with hourly data. If None, uses appropriate train/test hourly data.
            data_type: 'train' or 'test' - which hourly data to use when hourly_data is None
            prediction_length: Number of minutes to predict per hour
            
        Returns:
            Predictions in original scale (reconstructed from FFT)
        """
        predictions_reconstructed = self.predict_from_hourly(hourly_data, data_type, prediction_length)
        
        # The reconstruction already returns values in original scale (not normalized)
        # because reconstruct_from_fft works with original scale cubic splines
        return predictions_reconstructed
    
    def predict_and_inverse_transform(self, X=None):
        """
        Make predictions and convert back to original scale.
        This method is not directly applicable for FFT-based approach.
        Use predict_from_hourly_and_inverse_transform instead.
        
        Args:
            X: Input features. If None, uses test set.
            
        Returns:
            Predictions in original scale (reconstructed using FFT)
        """
        print("Warning: predict_and_inverse_transform is not optimal for FFT-based approach.")
        print("Consider using predict_from_hourly_and_inverse_transform for better results.")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if X is None:
            X = self.X_test
        
        # Predict FFT coefficients
        fft_predictions = self.model.predict(X)
        
        # For test set reconstruction, we need to map predictions back to hours
        # This is complex since we don't have a direct hour mapping for test set samples
        # For now, return a simplified reconstruction using hourly means
        
        predictions_reconstructed = []
        samples_per_hour = self.dataset.prediction_length
        
        for i in range(0, len(fft_predictions), samples_per_hour):
            if i + samples_per_hour <= len(fft_predictions):
                # Take the first prediction for this hour (they should all be the same FFT coeffs)
                fft_coeffs = fft_predictions[i]
                
                # Reconstruct for each feature
                hour_predictions = []
                for feature_idx in range(len(feature_columns)):
                    # Extract coefficients for this feature
                    coeffs_per_feature = self.dataset.n_fft_features * 3
                    start_idx = feature_idx * coeffs_per_feature
                    end_idx = start_idx + coeffs_per_feature
                    
                    feature_coeffs = fft_coeffs[start_idx:end_idx]
                    
                    # Split into magnitude, phase, and frequency values
                    magnitudes = feature_coeffs[:self.dataset.n_fft_features]
                    phases = feature_coeffs[self.dataset.n_fft_features:2*self.dataset.n_fft_features]
                    frequencies = feature_coeffs[2*self.dataset.n_fft_features:]
                    
                    # Create frequency array and zero-padded coefficient array
                    freq_array = fftfreq(self.dataset.prediction_length, d=1/60)
                    padded_coeffs = np.zeros(self.dataset.prediction_length, dtype=complex)
                    
                    # Place coefficients at their closest frequency positions
                    for j, target_freq in enumerate(frequencies):
                        freq_idx = np.argmin(np.abs(freq_array - target_freq))
                        if freq_idx < self.dataset.prediction_length:
                            complex_coeff = magnitudes[j] * np.exp(1j * phases[j])
                            padded_coeffs[freq_idx] = complex_coeff
                    
                    # Apply inverse FFT to get fluctuations
                    fluctuations = np.fft.ifft(padded_coeffs).real
                    
                    # Use a constant baseline (this is approximate)
                    baseline = np.zeros(self.dataset.prediction_length)  # Will need actual hourly values
                    reconstructed = baseline + fluctuations
                    
                    hour_predictions.append(reconstructed)
                
                # Transpose and add to results
                hour_predictions = np.array(hour_predictions).T
                predictions_reconstructed.extend(hour_predictions)
        
        return np.array(predictions_reconstructed)


    # calculating VSS:

    def volatility_similarity_score(self, actual, predicted):
        """ calculate volatility similarity score (vss) to measure similarity in volatility phase"""
        # calculate minute to minute differences:
        actual_diff=np.diff(actual)
        predicted_diff=np.diff(predicted)

        #ensumre same length
        min_len = min(len(actual_diff), len(predicted_diff))
        actual_diff = actual_diff[:min_len]
        predicted_diff = predicted_diff[:min_len]

        #calculate correlation of differences:

        if np.std(actual_diff) == 0 or np.std(predicted_diff) == 0:
            correlation = 0

        else:
            correlation = np.corrcoef(actual_diff, predicted_diff)[0, 1]
            if np.isnan(correlation):
                correlation = 0

        #calculate the variance ratio:

        actual_variance = np.var(actual_diff)
        predicted_variance = np.var(predicted_diff)
        if actual_variance == 0 or predicted_variance == 0:
            variance_ratio = 1 #neutral if no variance
        else:
            variance_ratio = min(predicted_variance/ actual_variance, actual_variance / predicted_variance)

        #volatility similarity score (0 to 1)

        vss = correlation * variance_ratio
        return max(0, min(1, vss))  # Ensure VSS is between 0 and 1
    
    def evaluate_models(self):
        """
        Evaluate the trained FFT-based model and print summary.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        print("\n" + "="*50)
        print("FFT-BASED MODEL EVALUATION SUMMARY")
        print("="*50)
        
        # For FFT-based models, traditional R² on FFT coefficients may not be meaningful
        # Instead, evaluate reconstruction quality
        print(f"\nFFT Coefficient Prediction Performance:")
        print(f"  Training R² score (FFT coeffs): {self.training_score:.4f}")
        print(f"  Test R² score (FFT coeffs): {self.test_score:.4f}")
        
        # Try to evaluate reconstruction quality on test data
        try:
            test_reconstructed = self.predict_from_hourly_and_inverse_transform(data_type='test')
            test_actual = self._get_test_actual_values()
            
            if test_actual is not None and len(test_actual) == len(test_reconstructed):
                print(f"\nReconstruction Quality on Test Data:")
                for feature_idx, feature_name in enumerate(feature_columns):
                    actual_feature = test_actual[:, feature_idx]
                    pred_feature = test_reconstructed[:, feature_idx]
                    
                    mse = np.mean((actual_feature - pred_feature) ** 2)
                    mae = np.mean(np.abs(actual_feature - pred_feature))
                    
                    # Calculate R² for reconstruction
                    ss_res = np.sum((actual_feature - pred_feature) ** 2)
                    ss_tot = np.sum((actual_feature - np.mean(actual_feature)) ** 2)
                    r2_reconstruction = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # calculate vss for reconstruction

                    vss = self.volatility_similarity_score(actual_feature, pred_feature)
                    
                    print(f"  {feature_name}:")
                    print(f"    Reconstruction R²: {r2_reconstruction:.4f}")
                    print(f"    MSE: {mse:.4f}")
                    print(f"    VSS: {vss:.4f}")
                    print(f"    MAE: {mae:.4f}")
                    
                    # Store for later use
                    self.feature_r2_scores[feature_name] = r2_reconstruction
            else:
                print(f"\nNote: Could not evaluate reconstruction quality (data size mismatch)")
                # Use FFT coefficient R² as fallback
                for feature_name in feature_columns:
                    self.feature_r2_scores[feature_name] = self.test_score
                    
        except Exception as e:
            print(f"\nNote: Could not evaluate reconstruction quality: {str(e)}")
            # Use FFT coefficient R² as fallback
            for feature_name in feature_columns:
                self.feature_r2_scores[feature_name] = self.test_score
        
        # Feature importance for FFT coefficient prediction
        print(f"\nFeature Importance for FFT Coefficient Prediction (Top 15):")
        importance = self.model.feature_importances_
        
        # Use feature names from dataset
        if hasattr(self.dataset, 'feature_names') and self.dataset.feature_names is not None:
            input_feature_names = self.dataset.feature_names
        else:
            # Fallback to basic names
            input_feature_names = [f"hourly_{f}" for f in feature_columns] + ['sin_pos_2pi', 'cos_pos_2pi', 'sin_pos_4pi', 'cos_pos_4pi']
        
        top_indices = np.argsort(importance)[-15:][::-1]
        for i, idx in enumerate(top_indices):
            feature_name = input_feature_names[idx] if idx < len(input_feature_names) else f"Feature_{idx}"
            print(f"    {i+1:2d}. {feature_name:<20}: {importance[idx]:.4f}")
        
        print(f"\nFeature Categories Summary:")
        n_hourly = len(feature_columns)
        n_pos = 4
        hourly_importance = np.sum(importance[:n_hourly])
        pos_importance = np.sum(importance[n_hourly:n_hourly+n_pos])
        
        print(f"  Hourly features importance: {hourly_importance:.4f}")
        print(f"  Positional features importance: {pos_importance:.4f}")
        
        # Calculate relative importance
        total_importance = hourly_importance + pos_importance
        print(f"\nRelative Importance:")
        print(f"  Hourly features: {hourly_importance/total_importance*100:.1f}%")
        print(f"  Positional features: {pos_importance/total_importance*100:.1f}%")
        
        print(f"\nModel Configuration:")
        print(f"  FFT coefficients per feature: {self.dataset.n_fft_features} (mag + phase + frequencies)")
        print(f"  Total FFT coefficients: {self.dataset.n_fft_features * 3 * len(feature_columns)}")
        print(f"  Prediction length: {self.dataset.prediction_length} minutes")
    
    def _get_test_actual_values(self):
        """
        Get actual minute-level values for test set for evaluation.
        Returns None if not available.
        """
        try:
            if hasattr(self, 'test_hourly_data') and self.test_hourly_data is not None:
                actual_values = []
                for _, hour_row in self.test_hourly_data.iterrows():
                    hour_time = hour_row['time']
                    hour_start = hour_time
                    hour_end = hour_time + timedelta(hours=1)
                    
                    # Get minute data for this hour from test data
                    if hasattr(self, 'test_df') and self.test_df is not None:
                        hour_minute_data = self.test_df[
                            (self.test_df['time'] >= hour_start) & 
                            (self.test_df['time'] < hour_end)
                        ].sort_values('time')
                        
                        if len(hour_minute_data) > 0:
                            values_to_take = min(self.dataset.prediction_length, len(hour_minute_data))
                            
                            hour_values = []
                            for feature in feature_columns:
                                feature_values = hour_minute_data[feature].iloc[:values_to_take].tolist()
                                # Pad if necessary
                                if len(feature_values) < self.dataset.prediction_length:
                                    last_value = feature_values[-1] if feature_values else 0
                                    feature_values.extend([last_value] * (self.dataset.prediction_length - len(feature_values)))
                                hour_values.append(feature_values)
                            
                            # Transpose to get (time_steps, features) format
                            hour_values = np.array(hour_values).T
                            actual_values.extend(hour_values)
                
                return np.array(actual_values) if actual_values else None
            return None
        except Exception as e:
            print(f"Warning: Could not extract test actual values: {str(e)}")
            return None
    
    def plot_predictions_vs_actual(self, feature_name, n_samples=1000):
        """
        Plot predictions vs actual values for a specific feature.
        
        Args:
            feature_name: Name of the feature to plot
            n_samples: Number of samples to plot
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if feature_name not in feature_columns:
            raise ValueError(f"Feature {feature_name} not found in {feature_columns}")
        
        # Get predictions
        predictions = self.predict_and_inverse_transform()
        feature_idx = feature_columns.index(feature_name)
        
        # Convert actual values back to original scale
        actual_original = self.dataset.inverse_transform(self.y_test[:, feature_idx], feature_idx)
        
        # Sample data for plotting
        if len(predictions) > n_samples:
            indices = np.random.choice(len(predictions), n_samples, replace=False)
            pred_sample = predictions[indices, feature_idx]
            actual_sample = actual_original[indices]
        else:
            pred_sample = predictions[:, feature_idx]
            actual_sample = actual_original
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_sample, pred_sample, alpha=0.5)
        plt.plot([actual_sample.min(), actual_sample.max()], 
                [actual_sample.min(), actual_sample.max()], 'r--', lw=2)
        plt.xlabel(f'Actual {feature_name}')
        plt.ylabel(f'Predicted {feature_name}')
        plt.title(f'Predictions vs Actual: {feature_name}')
        plt.grid(True, alpha=0.3)
        
        # Calculate and display R² score
        r2 = self.feature_r2_scores[feature_name]
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_comparison(self, feature_name, n_samples=500, start_idx=0):
        """
        Plot actual vs predicted values as time series (line graph over horizon).
        
        Args:
            feature_name: Name of the feature to plot
            n_samples: Number of consecutive samples to plot
            start_idx: Starting index for the time series plot
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if feature_name not in feature_columns:
            raise ValueError(f"Feature {feature_name} not found in {feature_columns}")
        
        # Get predictions
        predictions = self.predict_and_inverse_transform()
        feature_idx = feature_columns.index(feature_name)
        
        # Convert actual values back to original scale
        actual_original = self.dataset.inverse_transform(self.y_test[:, feature_idx], feature_idx)
        
        # Select consecutive samples for time series
        end_idx = min(start_idx + n_samples, len(predictions))
        actual_series = actual_original[start_idx:end_idx]
        pred_series = predictions[start_idx:end_idx, feature_idx]
        
        # Create time index (assuming samples are in temporal order)
        time_index = np.arange(start_idx, end_idx)
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual and predicted lines
        plt.plot(time_index, actual_series, 'b-', linewidth=2, label='Actual', alpha=0.8)
        plt.plot(time_index, pred_series, 'r-', linewidth=2, label='Predicted', alpha=0.8)
        
        # Fill between for better visualization
        plt.fill_between(time_index, actual_series, pred_series, alpha=0.2, color='gray', label='Difference')
        
        # Customize the plot
        plt.xlabel('Time Steps')
        plt.ylabel(f'{feature_name}')
        plt.title(f'Time Series Comparison: Actual vs Predicted {feature_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics to the plot
        mse = np.mean((actual_series - pred_series) ** 2)
        mae = np.mean(np.abs(actual_series - pred_series))
        r2 = self.feature_r2_scores[feature_name]
        
        
        stats_text = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nMAE = {mae:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\nTime Series Statistics for {feature_name}:")
        print(f"Samples plotted: {len(actual_series)}")
        print(f"Mean Actual: {np.mean(actual_series):.4f}")
        print(f"Mean Predicted: {np.mean(pred_series):.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
    
    def plot_multiple_features_time_series(self, n_samples=500, start_idx=0):
        """
        Plot time series comparison for all features in subplots.
        
        Args:
            n_samples: Number of consecutive samples to plot
            start_idx: Starting index for the time series plot
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Get predictions
        predictions = self.predict_and_inverse_transform()
        
        # Create subplots
        n_features = len(feature_columns)
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        # Select consecutive samples
        end_idx = min(start_idx + n_samples, len(predictions))
        time_index = np.arange(start_idx, end_idx)
        
        for i, feature_name in enumerate(feature_columns):
            # Convert actual values back to original scale
            actual_original = self.dataset.inverse_transform(self.y_test[:, i], i)
            actual_series = actual_original[start_idx:end_idx]
            pred_series = predictions[start_idx:end_idx, i]
            
            # Plot on subplot
            axes[i].plot(time_index, actual_series, 'b-', linewidth=2, label='Actual', alpha=0.8)
            axes[i].plot(time_index, pred_series, 'r-', linewidth=2, label='Predicted', alpha=0.8)
            axes[i].fill_between(time_index, actual_series, pred_series, alpha=0.2, color='gray')
            
            # Customize subplot
            axes[i].set_ylabel(feature_name)
            axes[i].set_title(f'{feature_name} - R² = {self.feature_r2_scores[feature_name]:.4f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps')
        plt.suptitle('Time Series Comparison: All Features', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.show()
    
    '''def plot_hourly_predictions(self, feature_name, n_hours=10, start_hour=233, data_type='test', prediction_length=None):
        """
        Plot predictions generated from hourly data showing minute horizons with actual values.
        
        Args:
            feature_name: Name of the feature to plot
            n_hours: Number of hours to plot
            start_hour: Starting hour index
            data_type: 'train' or 'test' - which hourly data to use
            prediction_length: Number of minutes per hour (if None, uses dataset's prediction_length)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if feature_name not in feature_columns:
            raise ValueError(f"Feature {feature_name} not found in {feature_columns}")
        
        # Use dataset's prediction_length if not specified
        if prediction_length is None:
            prediction_length = self.dataset.prediction_length
        
        # Get subset of appropriate hourly data
        if data_type == 'train':
            hourly_data = self.train_hourly_data
        elif data_type == 'test':
            hourly_data = self.test_hourly_data
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        
        hourly_subset = hourly_data.iloc[start_hour:start_hour + n_hours]
        
        # Get predictions for this subset
        predictions = self.predict_from_hourly_and_inverse_transform(hourly_subset, data_type, prediction_length)
        feature_idx = feature_columns.index(feature_name)
        
        print(f"Debug: predictions shape = {predictions.shape}, feature_idx = {feature_idx}")
        print(f"Debug: n_hours = {n_hours}, prediction_length = {prediction_length}")
        print(f"Debug: hourly_subset shape = {hourly_subset.shape}")
        
        # Get actual minute-level data for the same time periods
        actual_values = self._get_actual_minute_data(hourly_subset, feature_name, data_type, prediction_length)
        
        # Adjust n_hours based on available predictions
        total_available_predictions = len(predictions)
        total_needed_predictions = len(hourly_subset) * prediction_length
        
        if total_needed_predictions > total_available_predictions:
            print(f"Warning: Expected {total_needed_predictions} predictions but only {total_available_predictions} available")
            n_hours_available = total_available_predictions // prediction_length
            n_hours = min(n_hours, n_hours_available)
            hourly_subset = hourly_subset.iloc[:n_hours]  # Adjust subset too
            print(f"Adjusting to {n_hours} hours")
        
        # Create time index for plotting
        time_minutes = []
        pred_values = []
        hour_boundaries = []
        
        for hour_idx in range(n_hours):
            hour_start_minute = hour_idx * prediction_length
            for minute in range(prediction_length):
                time_minutes.append(hour_start_minute + minute)
                pred_idx = hour_idx * prediction_length + minute
                if pred_idx < len(predictions):
                    pred_values.append(predictions[pred_idx, feature_idx])
                else:
                    print(f"Warning: pred_idx {pred_idx} out of bounds for predictions length {len(predictions)}")
                    break
            hour_boundaries.append(hour_start_minute + prediction_length)
        
        # Adjust actual_values to match the number of hours we're actually plotting
        if len(actual_values) > len(pred_values):
            actual_values = actual_values[:len(pred_values)]
        
        print(f"Debug: Final lengths - time_minutes: {len(time_minutes)}, pred_values: {len(pred_values)}, actual_values: {len(actual_values)}")
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual and predicted values
        if len(actual_values) > 0:
            plt.plot(time_minutes[:len(actual_values)], actual_values, 'g-', linewidth=2, label='Actual', alpha=0.8)
            
        plt.plot(time_minutes[:len(pred_values)], pred_values, 'b-', linewidth=2, label='Predicted', alpha=0.8)
        
        # Fill between actual and predicted if both available
        if len(actual_values) > 0 and len(actual_values) == len(pred_values):
            plt.fill_between(time_minutes, actual_values, pred_values, alpha=0.3, color='gray', label='Difference')
        
        # Add vertical lines to separate hours
        for boundary in hour_boundaries[:-1]:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, label='Hour Boundary' if boundary == hour_boundaries[0] else "")
        
        plt.xlabel('Time (Minutes)')
        plt.ylabel(f'{feature_name}')
        plt.title(f'Hourly Predictions vs Actual: {feature_name} ({data_type.upper()} data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add hour labels
        for hour_idx in range(n_hours):
            hour_time = hourly_subset.iloc[hour_idx]['time']
            hour_center = hour_idx * prediction_length + prediction_length // 2
            plt.text(hour_center, plt.ylim()[1] * 0.95, f'Hour {hour_idx}\n{hour_time.strftime("%H:%M")}', 
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add statistics to the plot if actual values available
        if len(actual_values) > 0 and len(actual_values) == len(pred_values):
            mse = np.mean((np.array(actual_values) - np.array(pred_values)) ** 2)
            mae = np.mean(np.abs(np.array(actual_values) - np.array(pred_values)))
            r2 = 1 - (np.sum((np.array(actual_values) - np.array(pred_values)) ** 2) / 
                     np.sum((np.array(actual_values) - np.mean(actual_values)) ** 2))
            
            stats_text = f'MSE = {mse:.4f}\nMAE = {mae:.4f}\nR² = {r2:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nHourly Prediction Summary for {feature_name}:")
        print(f"Hours plotted: {n_hours}")
        print(f"Total minutes predicted: {len(pred_values)}")
        print(f"Mean predicted value: {np.mean(pred_values):.4f}")
        print(f"Std predicted value: {np.std(pred_values):.4f}")
        
        if len(actual_values) > 0:
            print(f"Actual data points available: {len(actual_values)}")
            if len(actual_values) == len(pred_values):
                print(f"Mean actual value: {np.mean(actual_values):.4f}")
                print(f"Mean Squared Error: {mse:.4f}")
                print(f"Mean Absolute Error: {mae:.4f}")
                print(f"R² Score: {r2:.4f}")
        else:
            print("No actual minute-level data found for comparison")'''



    def plot_hourly_predictions(self, feature_name, n_hours=10, start_hour=233, data_type='test', prediction_length=None):
        """
        Plot predictions generated from hourly data showing minute horizons with actual values.
        
        Args:
            feature_name: Name of the feature to plot
            n_hours: Number of hours to plot
            start_hour: Starting hour index
            data_type: 'train' or 'test' - which hourly data to use
            prediction_length: Number of minutes per hour (if None, uses dataset's prediction_length)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        if feature_name not in feature_columns:
            raise ValueError(f"Feature {feature_name} not found in {feature_columns}")
        
        # Use dataset's prediction_length if not specified
        if prediction_length is None:
            prediction_length = self.dataset.prediction_length
        
        # Get subset of appropriate hourly data
        if data_type == 'train':
            hourly_data = self.train_hourly_data
        elif data_type == 'test':
            hourly_data = self.test_hourly_data
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        
        hourly_subset = hourly_data.iloc[start_hour:start_hour + n_hours]
        
        # Get predictions for this subset
        predictions = self.predict_from_hourly_and_inverse_transform(hourly_subset, data_type, prediction_length)
        feature_idx = feature_columns.index(feature_name)
        
        print(f"Debug: predictions shape = {predictions.shape}, feature_idx = {feature_idx}")
        print(f"Debug: n_hours = {n_hours}, prediction_length = {prediction_length}")
        print(f"Debug: hourly_subset shape = {hourly_subset.shape}")
        
        # Get actual minute-level data for the same time periods
        actual_values = self._get_actual_minute_data(hourly_subset, feature_name, data_type, prediction_length)
        
        # Adjust n_hours based on available predictions
        total_available_predictions = len(predictions)
        total_needed_predictions = len(hourly_subset) * prediction_length
        
        if total_needed_predictions > total_available_predictions:
            print(f"Warning: Expected {total_needed_predictions} predictions but only {total_available_predictions} available")
            n_hours_available = total_available_predictions // prediction_length
            n_hours = min(n_hours, n_hours_available)
            hourly_subset = hourly_subset.iloc[:n_hours]  # Adjust subset too
            print(f"Adjusting to {n_hours} hours")
        
        # Create time index for plotting
        time_minutes = []
        pred_values = []
        hour_boundaries = []
        
        for hour_idx in range(n_hours):
            hour_start_minute = hour_idx * prediction_length
            for minute in range(prediction_length):
                time_minutes.append(hour_start_minute + minute)
                pred_idx = hour_idx * prediction_length + minute
                if pred_idx < len(predictions):
                    pred_values.append(predictions[pred_idx, feature_idx])
                else:
                    print(f"Warning: pred_idx {pred_idx} out of bounds for predictions length {len(predictions)}")
                    break
            hour_boundaries.append(hour_start_minute + prediction_length)
        
        # Adjust actual_values to match the number of hours we're actually plotting
        if len(actual_values) > len(pred_values):
            actual_values = actual_values[:len(pred_values)]
        
        print(f"Debug: Final lengths - time_minutes: {len(time_minutes)}, pred_values: {len(pred_values)}, actual_values: {len(actual_values)}")
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual and predicted values
        if len(actual_values) > 0:
            plt.plot(time_minutes[:len(actual_values)], actual_values, 'g-', linewidth=2, label='Actual', alpha=0.8)
            
        plt.plot(time_minutes[:len(pred_values)], pred_values, 'b-', linewidth=2, label='Predicted', alpha=0.8)
        
        # Fill between actual and predicted if both available
        if len(actual_values) > 0 and len(actual_values) == len(pred_values):
            plt.fill_between(time_minutes, actual_values, pred_values, alpha=0.3, color='gray', label='Difference')
        
        # Add vertical lines to separate hours
        for boundary in hour_boundaries[:-1]:
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, label='Hour Boundary' if boundary == hour_boundaries[0] else "")
        
        # Add hourly average lines (for actual, predicted, and input hourly avg)
        for hour_idx in range(n_hours):
            hour_start_minute = hour_idx * prediction_length
            hour_end_minute = min(hour_start_minute + prediction_length, len(time_minutes))
            
            # Actual average (dashed green) - Alpha 1.0 for visibility
            if len(actual_values) > hour_start_minute:
                hour_actual = actual_values[hour_start_minute:hour_end_minute]
                avg_actual = np.mean(hour_actual)
                plt.hlines(avg_actual, hour_start_minute, hour_end_minute, colors='darkgreen', linestyles='dashed', label='Avg Actual' if hour_idx==0 else None, linewidth=1.5, alpha=1.0)  # Darkgreen and thicker for visibility
                print(f"Hour {hour_idx} Actual Mean: {avg_actual}")
            # Predicted average (dashed blue) - Alpha 1.0
            if len(pred_values) > hour_start_minute:
                hour_pred = pred_values[hour_start_minute:hour_end_minute]
                avg_pred = np.mean(hour_pred)
                plt.hlines(avg_pred, hour_start_minute, hour_end_minute, colors='darkblue', linestyles='dashed', label='Avg Predicted' if hour_idx==0 else None, linewidth=1.5, alpha=1.0)  # Darkblue for contrast
                print(f"Hour {hour_idx} Predicted Mean: {avg_pred}")
            # Input hourly average (dotted black, model's input) - Alpha 1.0
            hourly_avg = hourly_subset.iloc[hour_idx][feature_name]  # Hourly mean from input
            plt.hlines(hourly_avg, hour_start_minute, hour_end_minute, colors='orange', linestyles='dotted', label='Input Hourly Avg' if hour_idx==0 else None, linewidth=2.0, alpha=1.0)
            print(f"Hour {hour_idx} Input Hourly Avg: {hourly_avg}") 
        
        plt.xlabel('Time (Minutes)')
        plt.ylabel(f'{feature_name}')
        plt.title(f'Hourly Predictions vs Actual: {feature_name} ({data_type.upper()} data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add hour labels
        for hour_idx in range(n_hours):
            hour_time = hourly_subset.iloc[hour_idx]['time']
            hour_center = hour_idx * prediction_length + prediction_length // 2
            plt.text(hour_center, plt.ylim()[1] * 0.95, f'Hour {hour_idx}\n{hour_time.strftime("%H:%M")}', 
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add statistics to the plot if actual values available
        if len(actual_values) > 0 and len(actual_values) == len(pred_values):
            mse = np.mean((np.array(actual_values) - np.array(pred_values)) ** 2)
            mae = np.mean(np.abs(np.array(actual_values) - np.array(pred_values)))
            r2 = 1 - (np.sum((np.array(actual_values) - np.array(pred_values)) ** 2) / 
                     np.sum((np.array(actual_values) - np.mean(actual_values)) ** 2))
            vss = self.volatility_similarity_score(np.array(actual_values), np.array(pred_values))  # VSS added
            
            stats_text = f'MAE = {mae:.4f}\nR² = {r2:.4f}\nVSS = {vss:.4f}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nHourly Prediction Summary for {feature_name}:")
        print(f"Hours plotted: {n_hours}")
        print(f"Total minutes predicted: {len(pred_values)}")
        print(f"Mean predicted value: {np.mean(pred_values):.4f}")
        print(f"Std predicted value: {np.std(pred_values):.4f}")
        
        if len(actual_values) > 0:
            print(f"Actual data points available: {len(actual_values)}")
            if len(actual_values) == len(pred_values):
                print(f"Mean actual value: {np.mean(actual_values):.4f}")
                print(f"Mean Squared Error: {mse:.4f}")
                print(f"Mean Absolute Error: {mae:.4f}")
                print(f"R² Score: {r2:.4f}")
                print(f"VSS Score: {vss:.4f}")
        else:
            print("No actual minute-level data found for comparison")
    
    def plot_hourly_predictions_all_features(self, n_hours=10, start_hour=233, data_type='test', prediction_length=None):
        """
        Plot hourly predictions vs actual values for all features in subplots.
        
        Args:
            n_hours: Number of hours to plot
            start_hour: Starting hour index
            data_type: 'train' or 'test' - which hourly data to use
            prediction_length: Number of minutes per hour (if None, uses dataset's prediction_length)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Use dataset's prediction_length if not specified
        if prediction_length is None:
            prediction_length = self.dataset.prediction_length
        
        # Get subset of appropriate hourly data
        if data_type == 'train':
            hourly_data = self.train_hourly_data
        elif data_type == 'test':
            hourly_data = self.test_hourly_data
        else:
            raise ValueError("data_type must be 'train' or 'test'")
        
        hourly_subset = hourly_data.iloc[start_hour:start_hour + n_hours]
        
        # Get predictions for this subset
        predictions = self.predict_from_hourly_and_inverse_transform(hourly_subset, data_type, prediction_length)
        
        # Adjust n_hours based on available predictions
        total_available_predictions = len(predictions)
        total_needed_predictions = len(hourly_subset) * prediction_length
        
        if total_needed_predictions > total_available_predictions:
            print(f"Warning: Expected {total_needed_predictions} predictions but only {total_available_predictions} available")
            n_hours_available = total_available_predictions // prediction_length
            n_hours = min(n_hours, n_hours_available)
            hourly_subset = hourly_subset.iloc[:n_hours]
            print(f"Adjusting to {n_hours} hours")
        
        # Create time index
        time_minutes = []
        for hour_idx in range(n_hours):
            hour_start_minute = hour_idx * prediction_length
            for minute in range(prediction_length):
                time_minutes.append(hour_start_minute + minute)
        
        # Limit time_minutes to available predictions
        time_minutes = time_minutes[:len(predictions)]
        
        # Create subplots
        n_features = len(feature_columns)
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        # Hour boundaries for vertical lines
        hour_boundaries = [i * prediction_length for i in range(1, n_hours)]
        
        for i, feature_name in enumerate(feature_columns):
            pred_values = predictions[:, i]
            
            # Get actual values for this feature
            actual_values = self._get_actual_minute_data(hourly_subset, feature_name, data_type, prediction_length)
            
            # Adjust actual_values to match pred_values length
            if len(actual_values) > len(pred_values):
                actual_values = actual_values[:len(pred_values)]
            
            # Plot actual values first (if available)
            if len(actual_values) > 0:
                axes[i].plot(time_minutes[:len(actual_values)], actual_values, 'g-', 
                           linewidth=2, label='Actual', alpha=0.8)
                
                # Fill between actual and predicted if both available
                if len(actual_values) == len(pred_values):
                    axes[i].fill_between(time_minutes[:len(pred_values)], actual_values, pred_values, 
                                       alpha=0.3, color='gray', label='Difference')
            
            # Plot predictions
            axes[i].plot(time_minutes[:len(pred_values)], pred_values, 'b-', linewidth=2, label='Predicted', alpha=0.8)
            
            # Add hour boundaries
            for boundary in hour_boundaries:
                axes[i].axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
            
            # Calculate and display metrics if actual values available
            if len(actual_values) > 0 and len(actual_values) == len(pred_values):
                mse = np.mean((np.array(actual_values) - np.array(pred_values)) ** 2)
                mae = np.mean(np.abs(np.array(actual_values) - np.array(pred_values)))
                r2 = 1 - (np.sum((np.array(actual_values) - np.array(pred_values)) ** 2) / 
                         np.sum((np.array(actual_values) - np.mean(actual_values)) ** 2))
                
                # Customize subplot with metrics
                axes[i].set_ylabel(feature_name)
                axes[i].set_title(f'{feature_name} - R²={r2:.3f}, MAE={mae:.2f}')
            else:
                # Customize subplot without metrics
                axes[i].set_ylabel(feature_name)
                axes[i].set_title(f'{feature_name} - Predictions Only')
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add hour labels
            for hour_idx in range(n_hours):
                hour_time = hourly_subset.iloc[hour_idx]['time']
                hour_center = hour_idx * prediction_length + prediction_length // 2
                axes[i].text(hour_center, axes[i].get_ylim()[1] * 0.95, 
                           f'H{hour_idx}', ha='center', va='top', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), fontsize=8)
        
        axes[-1].set_xlabel('Time (Minutes)')
        plt.suptitle(f'Hourly Predictions vs Actual: All Features ({n_hours} hours, {data_type.upper()} data)', fontsize=16, y=0.995)
        plt.tight_layout()
        plt.show()
    
    def predict_single_hour(self, hourly_features, prediction_length=59):
        """
        Predict minute-level data for a single hour given hourly features using FFT reconstruction.
        
        Args:
            hourly_features: Dictionary or array of hourly feature values
            prediction_length: Number of minutes to predict
            
        Returns:
            Array of predictions for each minute in original scale
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_models() first.")
        
        # Convert to array if dictionary
        if isinstance(hourly_features, dict):
            hourly_array = np.array([hourly_features[feature] for feature in feature_columns])
        else:
            hourly_array = np.array(hourly_features)
        
        # Normalize the hourly features
        normalized_hourly = np.zeros_like(hourly_array)
        for i, feature in enumerate(feature_columns):
            normalized_hourly[i] = self.dataset.scalers[feature].transform([[hourly_array[i]]])[0, 0]
        
        # Generate input features (using middle of hour as representative)
        minute_pos = 0.5
        pos_encoding = [
            np.sin(minute_pos * np.pi * 2),
            np.cos(minute_pos * np.pi * 2),
            np.sin(minute_pos * np.pi * 4),
            np.cos(minute_pos * np.pi * 4)
        ]
        
        # Combine features
        input_features = np.concatenate([normalized_hourly, pos_encoding])
        
        # Predict FFT coefficients
        fft_coeffs_pred = self.model.predict(input_features.reshape(1, -1))[0]
        
        # For single hour prediction, we need to create a simple reconstruction
        hour_predictions = []
        for feature_idx in range(len(feature_columns)):
            # Extract coefficients for this feature
            coeffs_per_feature = self.dataset.n_fft_features * 3
            start_idx = feature_idx * coeffs_per_feature
            end_idx = start_idx + coeffs_per_feature
            
            feature_coeffs = fft_coeffs_pred[start_idx:end_idx]
            
            # Split into magnitude, phase, and frequency values
            magnitudes = feature_coeffs[:self.dataset.n_fft_features]
            phases = feature_coeffs[self.dataset.n_fft_features:2*self.dataset.n_fft_features]
            frequencies = feature_coeffs[2*self.dataset.n_fft_features:]
            
            # Create frequency array and zero-padded coefficient array
            freq_array = fftfreq(prediction_length, d=1/60)
            padded_coeffs = np.zeros(prediction_length, dtype=complex)
            
            # Place coefficients at their closest frequency positions
            for i, target_freq in enumerate(frequencies):
                freq_idx = np.argmin(np.abs(freq_array - target_freq))
                if freq_idx < prediction_length:
                    complex_coeff = magnitudes[i] * np.exp(1j * phases[i])
                    padded_coeffs[freq_idx] = complex_coeff
            
            # Apply inverse FFT to get fluctuations
            reconstructed_fluctuations = np.fft.ifft(padded_coeffs).real
            
            # Add to the hourly mean (baseline) - use the provided hourly value as constant baseline
            baseline = np.full(prediction_length, hourly_array[feature_idx])
            reconstructed = baseline + reconstructed_fluctuations
            
            hour_predictions.append(reconstructed)
        
        # Transpose to get (time_steps, features) format
        return np.array(hour_predictions).T
    


# Example usage and main execution
if __name__ == "__main__":
    # ===== FFT-BASED APPROACH: Model predicts FFT coefficients =====
    
    print("="*80)
    print("TRAINING FFT-BASED MODEL (Predicts FFT Coefficients)")
    print("="*80)
    
    # Create FFT-based pipeline
    pipeline_fft = WindTurbineMLPipeline()
    
    # Prepare data with FFT coefficient targets
    pipeline_fft.prepare_data(df_clean, prediction_length=59, test_size=0.3, 
                              use_fft_features=False, n_fft_features=40)
    
    # Train FFT-based model
    pipeline_fft.train_models()
    
    # Evaluate FFT-based model
    pipeline_fft.evaluate_models()
    
    # ===== VISUALIZATIONS =====
    print(f"\n" + "="*80)
    print("GENERATING RECONSTRUCTED PREDICTIONS (FFT-Based Model)")
    print("="*80)
    
    # Generate reconstructed predictions using the FFT-based model
    pipeline_fft.plot_hourly_predictions(feature_columns[0], n_hours=5, start_hour=10, data_type='test')
    
    # Show reconstructed predictions shape
    predictions_fft = pipeline_fft.predict_from_hourly_and_inverse_transform(data_type='test')
    print(f"\nFFT-based reconstructed predictions shape: {predictions_fft.shape}")
    
    # ===== ADDITIONAL ANALYSIS =====
    print(f"\n" + "="*80)
    print("ADDITIONAL ANALYSIS WITH FFT-BASED MODEL")
    print("="*80)
    
    # Use the FFT-based model for further analysis
    pipeline = pipeline_fft
    
    print(f"Selected FFT-based model test R² score: {pipeline.test_score:.4f}")
    
    # Additional analysis with the selected model
    # Plot hourly predictions for all features (using TEST data)
    print(f"\nGenerating TEST hourly predictions for all features...")
    pipeline.plot_hourly_predictions_all_features(n_hours=5, start_hour=10, data_type='test')
    
    # Also show training predictions for comparison
    print(f"\nGenerating TRAINING hourly predictions plot for {feature_columns[0]}...")
    pipeline.plot_hourly_predictions(feature_columns[0], n_hours=5, start_hour=10, data_type='train')
    
    # Example: Predict for a single hour with custom hourly features
    print(f"\nExample: Predicting single hour with custom features...")
    sample_hourly_features = {
        'active_power_avg': 1500.0,
        'wind_speed_avg': 8.5,  
    }
    
    single_hour_pred = pipeline.predict_single_hour(sample_hourly_features)
    print(f"Single hour prediction shape: {single_hour_pred.shape}")
    
    '''29_RANDOMFOREST_TIMEDOMAIN_PARAMETERCHANGES.ipynb# Plot the single hour prediction
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(feature_columns):
        plt.subplot(2, 1, i+1)
        plt.plot(range(59), single_hour_pred[:, i], 'b-', linewidth=2, label='Predicted (FFT-based)')
        plt.xlabel('Minute')
        plt.ylabel(feature)
        plt.title(f'{feature}\n(Hourly avg: {sample_hourly_features[feature]})')
        plt.grid(True, alpha=0.3)
        plt.legend()'''
    
    # Plot the single hour prediction
    plt.figure(figsize=(12, 8 * len(feature_columns) // 2))  # Figür boyutunu feature sayısına göre ayarla (daha uzun olsun)
    for i, feature in enumerate(feature_columns):
        plt.subplot(len(feature_columns), 1, i+1)  # Row sayısını feature uzunluğuna göre ayarla (örneğin 3 feature için 3 row)
        plt.plot(range(59), single_hour_pred[:, i], 'b-', linewidth=2, label='Predicted (FFT-based)')
        plt.xlabel('Minute')
        plt.ylabel(feature)
        plt.title(f'{feature}\n(Hourly avg: {sample_hourly_features[feature]})')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.suptitle('Single Hour Prediction (60 minutes) - FFT-based Reconstruction', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # ===== FFT-BASED APPROACH INSIGHTS =====
    print(f"\n" + "="*80)
    print("FFT-BASED APPROACH INSIGHTS")
    print("="*80)
    
    print(f"Model Architecture:")
    print(f"  Input features: {pipeline.X_train.shape[1]} (hourly features + positional encoding)")
    print(f"  Output targets: {pipeline.y_train.shape[1]} (FFT coefficients)")
    print(f"  FFT coefficients per feature: {pipeline.dataset.n_fft_features}")
    print(f"  Total features processed: {len(feature_columns)}")
    
    print(f"\nApproach Benefits:")
    print(f"  ✓ Frequency domain representation captures periodic patterns efficiently")
    print(f"  ✓ Compact representation: {pipeline.dataset.n_fft_features} coeffs vs {pipeline.dataset.prediction_length} time points")
    print(f"  ✓ Model learns to predict dominant frequency components")
    print(f"  ✓ Reconstruction preserves temporal structure through inverse FFT")
    
    print(f"\nData Flow:")
    print(f"  1. Input: Hourly mean + positional encoding")
    print(f"  2. Model predicts: FFT magnitude and phase coefficients")
    print(f"  3. Reconstruction: Inverse FFT + cubic spline baseline")
    print(f"  4. Output: High-resolution minute-level time series")
    
    # ===== COMPARISON PLOTS =====
    print(f"\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    # FFT-based approach plots
    print(f"FFT-based approach - hourly predictions for {feature_columns[0]}...")
    pipeline.plot_hourly_predictions(feature_columns[0], n_hours=3, start_hour=10, data_type='test')
    
    print(f"FFT-based approach - time series comparison using traditional predict method...")
    try:
        pipeline.plot_time_series_comparison(feature_columns[0], n_samples=300)
    except Exception as e:
        print(f"Note: Traditional time series plot not available for FFT-based approach: {str(e)}")
        print("Use hourly prediction plots instead for FFT-based visualization.")

    

    
   