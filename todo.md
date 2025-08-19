## To-do List

### Phase 1: Analyze existing LSTM and diffusion model implementations
- [x] Read RF_FREQUENCYBASED_LONGDATASET.py to understand data handling, feature engineering, and upsampling logic.
- [x] Read diffusion.py to understand the core components of a Gaussian Diffusion model.

### Phase 2: Design conditional diffusion model architecture for time series upsampling
- [x] Define the input and output dimensions for the diffusion model.
- [x] Determine the conditioning mechanism (e.g., concatenation of hourly data).
- [x] Outline the neural network architecture for the diffusion model (e.g., U-Net or Transformer-based).

### Phase 3: Implement conditional diffusion model with configurable upsampling steps
- [x] Adapt the diffusion.py script to create a time-series specific diffusion model.
- [x] Implement data loading and preprocessing similar to RF_FREQUENCYBASED_LONGDATASET.py.
- [x] Integrate the conditional diffusion model for upsampling, making the number of timesteps a variable.
- [x] Implement training loop for the diffusion model.

### Phase 4: Test and validate the implementation
- [x] Train the model on a sample dataset.
- [x] Evaluate the upsampling performance using appropriate metrics.
- [x] Visualize the upsampled data against actual minute-level data.

### Phase 5: Deliver final script and documentation to user
- [x] Provide the complete Python script.
- [x] Explain how to use the script and configure the upsampling steps.
- [x] Summarize the model's performance and limitations.



### Phase 6: Modify training script to save best model
- [x] Modify `upsampling_diffusion_model.py` to save the model with the lowest validation loss.

### Phase 7: Create inference script with enhanced plotting
- [x] Create a new script `upsampling_inference.py` for loading the trained model and performing inference.
- [x] Implement enhanced plotting in `upsampling_inference.py` to compare original and predicted data.

### Phase 8: Test and validate new scripts
- [x] Run `upsampling_diffusion_model.py` to train and save the model.
- [x] Run `upsampling_inference.py` to perform inference and generate plots.

### Phase 9: Deliver updated scripts and documentation
- [x] Provide the updated `upsampling_diffusion_model.py` and new `upsampling_inference.py`.
- [x] Update `README.md` with instructions for the new scripts.

