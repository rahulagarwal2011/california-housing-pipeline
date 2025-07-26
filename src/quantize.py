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
    """Runs the train.py script to generate the scikit-learn model."""
    print("Running training script (src/train.py)...")
    result = subprocess.run(["python", "src/train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Training script executed successfully.")
    else:
        print(f"Training script failed with error: {result.stderr}")
        raise RuntimeError(f"Training script failed: {result.stderr}")


def quantize_to_uint8(param, min_val=None, max_val=None):
    """
    Quantizes a given parameter (numpy array or scalar) to an 8-bit unsigned integer.
    If min_val and max_val are not provided, they are calculated from the parameter.
    """
    param = np.asarray(param, dtype=np.float32) 
    if param.size == 0:
        return np.array([], dtype=np.uint8)

    if min_val is None:
        min_val = param.min()
    if max_val is None:
        max_val = param.max()

    if max_val == min_val:
        
        return np.zeros_like(param, dtype=np.uint8)


    quantized = np.round(((param - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    return quantized

def dequantize_from_uint8(quantized_param, min_val, max_val):
    """
    Dequantizes a given 8-bit unsigned integer parameter back to its original range.
    """
    quantized_param = np.asarray(quantized_param, dtype=np.float32)
    
    if max_val == min_val:
        
        return np.full_like(quantized_param, min_val, dtype=np.float32)

   
    dequantized = ((quantized_param / 255.0) * (max_val - min_val)) + min_val
    return dequantized



def main():
    """
    Main function to perform model loading, quantization, and inference.
    """

    if not os.path.exists('model'):
        os.makedirs('model')


    try:
        run_train_script()
    except RuntimeError:
        print("Skipping further steps as training script failed.")
        return


    model_path = "model/california_housing_linear_regression_model.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Scikit-learn model not found at {model_path}. Please ensure src/train.py creates it.")
        return
    model_sklearn = joblib.load(model_path)
    print(f"Loaded scikit-learn model from {model_path}")


    sklearn_coef = model_sklearn.coef_
    sklearn_intercept = model_sklearn.intercept_


    unquant_params = {
        "coef": sklearn_coef,
        "intercept": sklearn_intercept
    }
    unquant_params_path = "model/unquant_params.joblib"
    joblib.dump(unquant_params, unquant_params_path)
    print(f"Unquantized parameters saved as {unquant_params_path}")

 
    coef_min, coef_max = sklearn_coef.min(), sklearn_coef.max()
    intercept_min, intercept_max = sklearn_intercept.min(), sklearn_intercept.max()


    quantized_coef = quantize_to_uint8(sklearn_coef, coef_min, coef_max)
    quantized_intercept = quantize_to_uint8(sklearn_intercept, intercept_min, intercept_max)


    quant_params = {
        "coef": quantized_coef,
        "intercept": quantized_intercept,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "intercept_min": intercept_min,
        "intercept_max": intercept_max
    }
    quant_params_path = "model/quant_params.joblib"
    joblib.dump(quant_params, quant_params_path)
    print(f"Quantized parameters saved as {quant_params_path}")


    dequantized_coef_torch = torch.from_numpy(dequantize_from_uint8(quantized_coef, coef_min, coef_max).astype(np.float32)).unsqueeze(0)
    dequantized_intercept_torch = torch.from_numpy(dequantize_from_uint8(quantized_intercept, intercept_min, intercept_max).astype(np.float32))


    class SimpleRegressor(nn.Module):
        def __init__(self, in_features):
            super(SimpleRegressor, self).__init__()
            self.linear = nn.Linear(in_features, 1)

        def forward(self, x):
            return self.linear(x)

  
    input_features = sklearn_coef.shape[0]
    model_torch = SimpleRegressor(in_features=input_features)


    model_torch.linear.weight.data = dequantized_coef_torch
    model_torch.linear.bias.data = dequantized_intercept_torch

    quantized_torch_model_path = "model/quantized_pytorch_model.pth"
    torch.save(model_torch.state_dict(), quantized_torch_model_path)
    print(f"Dequantized PyTorch model (with quantized weights) saved as {quantized_torch_model_path}")

    
    print("\n--- Performing Inference and Calculating Metrics ---")
    

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target


    X_tensor = torch.from_numpy(X.values.astype(np.float32))


    y_pred_sklearn = model_sklearn.predict(X)
    r2_sklearn = r2_score(y, y_pred_sklearn)
    print(f"R² Score (Original Scikit-learn Model): {r2_sklearn:.4f}")


    model_torch.eval() 
    with torch.no_grad():
        y_pred_torch = model_torch(X_tensor).squeeze().numpy()
    r2_torch = r2_score(y, y_pred_torch)
    print(f"R² Score (Dequantized PyTorch Model): {r2_torch:.4f}")


    unquant_size_kb = os.path.getsize(unquant_params_path) / 1024
    quant_size_kb = os.path.getsize(quant_params_path) / 1024


    formatted_unquant_size = f"{unquant_size_kb:.2f} KB"
    formatted_quant_size = f"{quant_size_kb:.2f} KB"
    print("\n--- Summary ---")
    print(f"{'Metric':<20} {'Original Sklearn Model':<30} {'Quantized Model':<30}")
    print(f"{'R² Score':<20} {r2_sklearn:<30.4f} {r2_torch:<30.4f}")
    print(f"{'Model Size':<20} {formatted_unquant_size:<30} {formatted_quant_size:<30}")

if __name__ == "__main__":
    main()