import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import sys  

def main():
    try:
       
        model_sklearn = joblib.load("model/california_housing_linear_regression_model.joblib")

   
        quant_params = joblib.load("model/quant_params.joblib")
        quantized_coef = quant_params["coef"]
        quantized_intercept = quant_params["intercept"]


        if quantized_coef.shape != (8,):
            raise ValueError(f"Expected quantized_coef shape (8,), got {quantized_coef.shape}")
        if quantized_intercept.shape != (1,):
            quantized_intercept = np.array([quantized_intercept])  


        class SimpleRegressor(nn.Module):
            def __init__(self, in_features):
                super(SimpleRegressor, self).__init__()
                self.linear = nn.Linear(in_features, 1)

            def forward(self, x):
                return self.linear(x)

    
        model_torch = SimpleRegressor(in_features=8)  

        model_torch.linear.weight.data = torch.from_numpy(quantized_coef.astype(np.float32)).reshape(1, 8)
        model_torch.linear.bias.data = torch.from_numpy(quantized_intercept.astype(np.float32))

        data = fetch_california_housing()
        X, y = data.data, data.target


        y_pred_sklearn = model_sklearn.predict(X)
        r2_sklearn = r2_score(y, y_pred_sklearn)


        batch_size = 512
        y_preds = []

        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
     
                if batch.ndim == 1:  
                    batch = batch.reshape(1, -1)
                preds = model_torch(batch).numpy().flatten()
                y_preds.append(preds)

        y_pred_torch = np.concatenate(y_preds)
        r2_torch = r2_score(y, y_pred_torch)

   
        print(f"Original Sklearn Model R² Score: {r2_sklearn:.4f}")
        print(f"Quantized PyTorch Model R² Score: {r2_torch:.4f}")
        print(f"{r2_sklearn:.4f},{r2_torch:.4f}")  

    except Exception as e:
        print(f"Error in predict.py: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()