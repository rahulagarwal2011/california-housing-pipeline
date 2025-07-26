import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import sys  # Added missing import

def main():
    try:
        # 1. Load the saved scikit-learn model
        model_sklearn = joblib.load("model/california_housing_linear_regression_model.joblib")

        # 2. Load quantized parameters
        quant_params = joblib.load("model/quant_params.joblib")
        quantized_coef = quant_params["coef"]
        quantized_intercept = quant_params["intercept"]

        # Ensure shapes are correct
        if quantized_coef.shape != (8,):
            raise ValueError(f"Expected quantized_coef shape (8,), got {quantized_coef.shape}")
        if quantized_intercept.shape != (1,):
            quantized_intercept = np.array([quantized_intercept])  # Convert to 1D array if needed

        # 3. Define the PyTorch model class
        class SimpleRegressor(nn.Module):
            def __init__(self, in_features):
                super(SimpleRegressor, self).__init__()
                self.linear = nn.Linear(in_features, 1)

            def forward(self, x):
                return self.linear(x)

        # Initialize the PyTorch model
        model_torch = SimpleRegressor(in_features=8)  # 8 features for the California housing dataset

        # Set the quantized parameters to the PyTorch model
        model_torch.linear.weight.data = torch.from_numpy(quantized_coef.astype(np.float32)).reshape(1, 8)
        model_torch.linear.bias.data = torch.from_numpy(quantized_intercept.astype(np.float32))

        # 4. Load the California Housing dataset for prediction
        data = fetch_california_housing()
        X, y = data.data, data.target

        # 5. Make predictions using the original scikit-learn model
        y_pred_sklearn = model_sklearn.predict(X)
        r2_sklearn = r2_score(y, y_pred_sklearn)

        # 6. Make predictions using the quantized PyTorch model
        batch_size = 512
        y_preds = []

        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                # Ensure batch is 2D: [batch_size, 8]
                if batch.ndim == 1:  # Single sample case
                    batch = batch.reshape(1, -1)
                preds = model_torch(batch).numpy().flatten()
                y_preds.append(preds)

        y_pred_torch = np.concatenate(y_preds)
        r2_torch = r2_score(y, y_pred_torch)

        # 7. Print the R² scores
        print(f"Original Sklearn Model R² Score: {r2_sklearn:.4f}")
        print(f"Quantized PyTorch Model R² Score: {r2_torch:.4f}")
        print(f"{r2_sklearn:.4f},{r2_torch:.4f}")  # Output for quantize.py to parse

    except Exception as e:
        print(f"Error in predict.py: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()