import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.express as px
import plotly.graph_objects as go


def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    # Add 'mae' to metrics to track it during training
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def evaluate_final_performance(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Final Performance: {model_name} ---")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R²: {r2:.4f}")

    # Plotting Actual vs. Predicted
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted', marker=dict(opacity=0.6)))
    fig_compare.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode='lines', name='Ideal', line=dict(dash='dash')))
    fig_compare.update_layout(title=f'Actual vs. Predicted - {model_name}', xaxis_title='Actual Calories', yaxis_title='Predicted Calories', showlegend=True)
    fig_compare.show()

    # Plotting Residuals
    residuals = y_true - y_pred
    fig_residuals = px.scatter(x=y_pred, y=residuals, title=f'Residual Plot - {model_name}', labels={'x': 'Predicted Calories', 'y': 'Residuals'})
    fig_residuals.add_hline(y=0, line_dash="dash")
    fig_residuals.show()


def plot_epoch_history(history, model_name):
    epochs = list(range(1, len(history.history['loss']) + 1))

    # Plot Loss (MSE)
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history.history['loss'], mode='lines', name='Training Loss'))
    if 'val_loss' in history.history:
        fig_loss.add_trace(go.Scatter(x=epochs, y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig_loss.update_layout(title=f'Training & Validation Loss (MSE) per Epoch - {model_name}',
                         xaxis_title='Epoch', yaxis_title='Loss (MSE)', showlegend=True)
    fig_loss.show()

    # Plot MAE
    if 'mae' in history.history:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(x=epochs, y=history.history['mae'], mode='lines', name='Training MAE'))
        if 'val_mae' in history.history:
            fig_mae.add_trace(go.Scatter(x=epochs, y=history.history['val_mae'], mode='lines', name='Validation MAE'))
        fig_mae.update_layout(title=f'Training & Validation MAE per Epoch - {model_name}',
                             xaxis_title='Epoch', yaxis_title='Mean Absolute Error', showlegend=True)
        fig_mae.show()



#Load and process
print("Loading and preprocessing data...")
full_train_df = pd.read_csv('Data/train.csv')

full_train_df.replace({'male': 1, 'female': 0}, inplace=True)

if 'Id' in full_train_df.columns:
    full_train_df.drop('Id', axis=1, inplace=True)
if 'id' in full_train_df.columns:
    full_train_df.drop('id', axis=1, inplace=True)



# Define features and target
X_cols = [col for col in full_train_df.columns if col != 'Calories']
y_col = 'Calories'

X_full = full_train_df[X_cols]
y_full = full_train_df[y_col]



# Split training data
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)


epochs_count = 100

model_name_s1 = "Scaled Base Model"

X_train_s1 = X_train_orig.copy()
y_train_s1 = y_train_orig.copy()
X_val_s1 = X_val_orig.copy()
y_val_s1 = y_val_orig.copy()

scaler_s1 = StandardScaler()
X_train_s1_scaled = scaler_s1.fit_transform(X_train_s1)
X_val_s1_scaled = scaler_s1.transform(X_val_s1)

model_s1 = create_nn_model(X_train_s1_scaled.shape[1])
print(f"Training {model_name_s1} for {epochs_count} epochs...")
history_s1 = model_s1.fit(X_train_s1_scaled, y_train_s1, 
                            epochs=epochs_count, 
                            validation_data=(X_val_s1_scaled, y_val_s1),
                            verbose=1)

# Plot training history by epoch
plot_epoch_history(history_s1, model_name_s1)

# Save the trained model
model_save_path = 'trained_calorie_predictor_model_100epochs.keras'
model_s1.save(model_save_path)

# Make predictions on validation set
predictions_s1 = model_s1.predict(X_val_s1_scaled).flatten()

# Evaluate performance on validation set
evaluate_final_performance(model_name_s1, y_val_s1, predictions_s1)