import subprocess
import joblib
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression 


def run_train_script():
    """
    Runs the train.py script to generate the scikit-learn model.
    This ensures that the model file exists before quantization.
    """
    print("Running training script (src/train.py)...")
    # Use capture_output=True to get stdout/stderr for better debugging
    result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Training script executed successfully.")
        # print(result.stdout) # Uncomment to see stdout from train.py
    else:
        print(f"Training script failed with error: {result.stderr}")
        raise RuntimeError(f"Training script failed: {result.stderr}")


def quantize_affine_int8(param, q_min_int=-128, q_max_int=127):
    """
    Quantizes a given parameter (numpy array or scalar) to a signed 8-bit integer (int8)
    using affine quantization (scale and zero-point).

    Parameters:
    param (np.ndarray or float): The parameter to quantize (float32).
    q_min_int (int): The minimum integer value for the quantized range (default -128).
    q_max_int (int): The maximum integer value for the quantized range (default 127).

    Returns:
    tuple: A tuple containing:
           - np.ndarray: The quantized parameter as an int8 NumPy array.
           - float: The calculated scale factor.
           - int: The calculated zero-point.
    """
    param = np.asarray(param, dtype=np.float32)

    if param.size == 0:
        return np.array([], dtype=np.int8), 1.0, 0 # Default scale and zero_point for empty array

    min_val_float = param.min()
    max_val_float = param.max()

    # Calculate scale
    # Scale = (float_range) / (quant_range)
    if max_val_float == min_val_float:
        scale = 1.0 # Avoid division by zero, treat as identity if range is a single value
    else:
        scale = (max_val_float - min_val_float) / (q_max_int - q_min_int)

    # Calculate zero-point
    # zero_point_float = q_min_int - (min_val_float / scale)
    # The zero-point is an integer that maps the float 0.0 to a specific quantized integer.
    # It must be clipped to be within the target integer range.
    zero_point = q_min_int - np.round(min_val_float / scale).astype(int)
    zero_point = np.clip(zero_point, q_min_int, q_max_int)

    # Quantize: round(float_val / scale + zero_point)
    quantized = np.round(param / scale + zero_point).astype(np.int8)
    
    # Clip quantized values to the target integer range to handle potential overflows
    quantized = np.clip(quantized, q_min_int, q_max_int)

    return quantized, scale, zero_point

def dequantize_affine_int8(quantized_param, scale, zero_point):
    """
    Dequantizes a signed 8-bit integer parameter back to its original float32 range
    using affine dequantization.

    Parameters:
    quantized_param (np.ndarray or int): The quantized parameter (int8).
    scale (float): The scale factor used during quantization.
    zero_point (int): The zero-point used during quantization.

    Returns:
    np.ndarray: The dequantized parameter as a float32 NumPy array.
    """
    quantized_param = np.asarray(quantized_param, dtype=np.float32) # Cast to float32 for calculations

    # Dequantize: (quantized_val - zero_point) * scale
    dequantized = (quantized_param - zero_point) * scale
    return dequantized


def main():
    """
    Main function to perform model loading, quantization, and inference.
    It demonstrates saving unquantized and quantized parameters, then performing
    inference with dequantized weights and reporting R2 score and model size.
    """

    # Create 'model' directory if it doesn't exist to store artifacts
    if not os.path.exists('model'):
        os.makedirs('model')

    # Ensure the scikit-learn model is trained and saved
    try:
        run_train_script()
    except RuntimeError:
        print("Skipping further steps as training script failed.")
        return

    # Define path for the scikit-learn model
    model_path = "model/california_housing_linear_regression_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Scikit-learn model not found at {model_path}. "
              "Please ensure src/train.py creates it successfully.")
        return
    
    # Load the trained scikit-learn model
    model_sklearn = joblib.load(model_path)
    print(f"Loaded scikit-learn model from {model_path}")

    # Extract coefficients and intercept. Crucially, ensure they are 1D NumPy arrays
    # using np.atleast_1d() to avoid TypeError with torch.from_numpy if they are scalars.
    sklearn_coef = np.atleast_1d(model_sklearn.coef_).astype(np.float32)
    sklearn_intercept = np.atleast_1d(model_sklearn.intercept_).astype(np.float32)

    # Store unquantized parameters in a dictionary
    unquant_params = {
        "coef": sklearn_coef,
        "intercept": sklearn_intercept
    }
    # Save unquantized parameters using joblib with high compression
    unquant_params_path = "model/unquant_params.joblib"
    joblib.dump(unquant_params, unquant_params_path, compress=9) # compress=9 for maximum compression
    print(f"Unquantized parameters saved as {unquant_params_path}")

    # Perform manual 8-bit signed integer quantization using affine scheme
    quantized_coef, coef_scale, coef_zero_point = quantize_affine_int8(sklearn_coef)
    quantized_intercept, intercept_scale, intercept_zero_point = quantize_affine_int8(sklearn_intercept)

    # Store quantized parameters along with their scale and zero-point.
    # These values are CRUCIAL for dequantization later.
    quant_params = {
        "coef": quantized_coef,
        "intercept": quantized_intercept,
        "coef_scale": coef_scale,
        "coef_zero_point": coef_zero_point,
        "intercept_scale": intercept_scale,
        "intercept_zero_point": intercept_zero_point
    }
    # Save quantized parameters using joblib with high compression
    quant_params_path = "model/quant_params.joblib"
    joblib.dump(quant_params, quant_params_path, compress=9) # compress=9 for maximum compression
    print(f"Quantized parameters saved as {quant_params_path}")

    # Dequantize parameters back to float32 for use in the PyTorch model
    # Ensure torch.from_numpy receives a NumPy array, not a scalar.
    dequantized_coef_torch = torch.from_numpy(dequantize_affine_int8(quantized_coef, coef_scale, coef_zero_point)).float().unsqueeze(0)
    dequantized_intercept_torch = torch.from_numpy(dequantize_affine_int8(quantized_intercept, intercept_scale, intercept_zero_point)).float()


    # Define a simple PyTorch Linear Regression model
    class SimpleRegressor(nn.Module):
        def __init__(self, in_features):
            super(SimpleRegressor, self).__init__()
            # Initialize a linear layer
            self.linear = nn.Linear(in_features, 1)

        def forward(self, x):
            return self.linear(x)

    # Instantiate the PyTorch model using the number of input features
    input_features = sklearn_coef.shape[0]
    model_torch = SimpleRegressor(in_features=input_features)

    # Manually set the weights and biases of the PyTorch model using the dequantized parameters
    # This simulates the "quantized" model's behavior for inference
    model_torch.linear.weight.data = dequantized_coef_torch
    model_torch.linear.bias.data = dequantized_intercept_torch

    # Save the state_dict of the dequantized PyTorch model
    quantized_torch_model_path = "model/quantized_pytorch_model.pth"
    torch.save(model_torch.state_dict(), quantized_torch_model_path)
    print(f"Dequantized PyTorch model (with quantized weights) saved as {quantized_torch_model_path}")

    
    print("\n--- Performing Inference and Calculating Metrics ---")
    
    # Fetch the California Housing dataset for inference
    # Note: Using as_frame=True requires pandas, which should be in requirements.txt
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Convert features to PyTorch tensor for inference with the PyTorch model
    X_tensor = torch.from_numpy(X.values.astype(np.float32))

    # Inference with the original Scikit-learn model
    y_pred_sklearn = model_sklearn.predict(X)
    r2_sklearn = r2_score(y, y_pred_sklearn)
    print(f"R² Score (Original Scikit-learn Model): {r2_sklearn:.4f}")

    # Inference with the dequantized PyTorch model
    model_torch.eval() # Set model to evaluation mode (e.g., disables dropout if present)
    with torch.no_grad(): # Disable gradient calculations for inference, saves memory
        y_pred_torch = model_torch(X_tensor).squeeze().numpy() # Squeeze to remove single-dimension if present
    r2_torch = r2_score(y, y_pred_torch)
    print(f"R² Score (Dequantized PyTorch Model): {r2_torch:.4f}")

    # --- Analysis of R2 Score ---
    # With affine quantization to signed 8-bit integers, the R2 score for the quantized
    # model should ideally be non-negative and closer to the original, though some
    # accuracy drop due to precision loss is still expected. A negative R2 means the
    # model performs worse than simply predicting the mean of the target variable.
    # While this improved quantization scheme is more robust than a simple min-max to uint8,
    # for sensitive models, further techniques like Quantization-Aware Training (QAT)
    # or higher bit-depth might be needed if higher accuracy is critical.

    # Calculate file sizes for comparison
    unquant_size_kb = os.path.getsize(unquant_params_path) / 1024
    quant_size_kb = os.path.getsize(quant_params_path) / 1024

    formatted_unquant_size = f"{unquant_size_kb:.2f} KB"
    formatted_quant_size = f"{quant_size_kb:.2f} KB"

    print("\n--- Summary ---")
    print(f"{'Metric':<20} {'Original Sklearn Model':<30} {'Quantized Model':<30}")
    print(f"{'R² Score':<20} {r2_sklearn:<30.4f} {r2_torch:<30.4f}")
    print(f"{'Model Size':<20} {formatted_unquant_size:<30} {formatted_quant_size:<30}")

    # --- Analysis of Model Size ---
    # For very small models (like this linear regression with few parameters),
    # the fixed overhead of storing the scale and zero-point (which are floats/ints,
    # and crucial for dequantization) along with the joblib serialization overhead
    # can sometimes outweigh the byte savings from quantizing the few actual weights.
    # This can lead to the 'quantized' model file being slightly larger than the original.
    # The size benefits of quantization are typically more evident in larger models
    # with millions of parameters.

if __name__ == "__main__":
    main()