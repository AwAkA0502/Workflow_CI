import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings

os.environ["QT_QPA_PLATFORM"] = "offscreen"
warnings.filterwarnings('ignore')

def train_model(n_estimators, max_depth, learning_rate, data_path):
    df = pd.read_csv(data_path)
    
    feature_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize',
                       'model_encoded', 'transmission_encoded', 'fuelType_encoded']
    X = df[feature_columns]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # mlflow.set_experiment("BMW_Price_Prediction_Workflow")
    
    with mlflow.start_run():
        mlflow.log_params({
            "iterations": n_estimators,
            "depth": max_depth,
            "learning_rate": learning_rate
        })

        model = CatBoostRegressor(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            random_seed=42,
            verbose=False,
            loss_function='RMSE'
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2_score": r2})

        importances = model.get_feature_importance()
        importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
        importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')

        summary = {
            "model": "CatBoost_CI",
            "metrics": {"r2": r2, "mae": mae},
            "timestamp": datetime.now().isoformat()
        }
        with open('model_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        mlflow.log_artifact('model_summary.json')

        mlflow.catboost.log_model(model, "model")
        
        print(f"DONE! R2: {r2:.4f}")

if __name__ == "__main__":
    n_est = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    m_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    l_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    d_path = sys.argv[4] if len(sys.argv) > 4 else "bmw_preprocessing.csv"
    
    train_model(n_est, m_depth, l_rate, d_path)