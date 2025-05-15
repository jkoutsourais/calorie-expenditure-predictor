import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

all_metrics = []
def evaluate_model_performance(model_name, model, X_test_scaled, y_true, y_pred_inverse_transformed_if_needed):
    mse = mean_squared_error(y_true, y_pred_inverse_transformed_if_needed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_inverse_transformed_if_needed)
    r2 = r2_score(y_true, y_pred_inverse_transformed_if_needed)

    print(f"\n--- Performance: {model_name} ---")
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R²: {r2:.4f}")
    all_metrics.append({'Scenario': model_name, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2})

    # Plotting Actual vs. Predicted
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=y_true, y=y_pred_inverse_transformed_if_needed, mode='markers', name='Predicted', marker=dict(opacity=0.6)))
    fig_compare.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()], mode='lines', name='Ideal', line=dict(dash='dash')))
    fig_compare.update_layout(title=f'Actual vs. Predicted - {model_name}', xaxis_title='Actual Calories', yaxis_title='Predicted Calories', showlegend=True)
    fig_compare.show()

    # Plotting Residuals
    residuals = y_true - y_pred_inverse_transformed_if_needed
    fig_residuals = px.scatter(x=y_pred_inverse_transformed_if_needed, y=residuals, title=f'Residual Plot - {model_name}', labels={'x': 'Predicted Calories', 'y': 'Residuals'})
    fig_residuals.add_hline(y=0, line_dash="dash")
    fig_residuals.show()


#Load and preprocess data
print("Loading and preprocessing data...")
full_train_df = pd.read_csv('Data/train.csv')

full_train_df.replace({'male': 1, 'female': 0}, inplace=True)

full_train_df.drop('id', axis=1, inplace=True)


# Define features (X) and target (y) from the full training dataset
X_cols = [col for col in full_train_df.columns if col != 'Calories']
y_col = 'Calories'

X_full = full_train_df[X_cols]
y_full = full_train_df[y_col]

# Split training data
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)


results = []
epochs_count = 10

#----------Base Model (Original Data, Scaled Features)----------#
X_train_s1 = X_train_orig.copy()
y_train_s1 = y_train_orig.copy()
X_val_s1 = X_val_orig.copy()
y_val_s1 = y_val_orig.copy() # This is our y_true for evaluation

scaler_s1 = StandardScaler()
X_train_s1_scaled = scaler_s1.fit_transform(X_train_s1)
X_val_s1_scaled = scaler_s1.transform(X_val_s1)

model_s1 = create_nn_model(X_train_s1_scaled.shape[1])
model_s1.fit(X_train_s1_scaled, y_train_s1, epochs=epochs_count, verbose=0)
predictions_s1 = model_s1.predict(X_val_s1_scaled).flatten()
evaluate_model_performance("Base Model", model_s1, X_val_s1_scaled, y_val_s1, predictions_s1)


#----------Feature Skew Addressed (Body_Temp)----------#
X_train_s2 = X_train_orig.copy()
y_train_s2 = y_train_orig.copy() # Target is not transformed
X_val_s2 = X_val_orig.copy()
y_val_s2 = y_val_orig.copy()

feature_to_transform_s2 = 'Body_Temp'
print(f"Skew of {feature_to_transform_s2} before transformation: {X_train_s2[feature_to_transform_s2].skew():.4f}")
pt_feature_s2 = PowerTransformer(method='yeo-johnson')
X_train_s2[feature_to_transform_s2] = pt_feature_s2.fit_transform(X_train_s2[[feature_to_transform_s2]])
X_val_s2[feature_to_transform_s2] = pt_feature_s2.transform(X_val_s2[[feature_to_transform_s2]]) # Use fitted transformer
print(f"Skew of {feature_to_transform_s2} after transformation: {X_train_s2[feature_to_transform_s2].skew():.4f}")

scaler_s2 = StandardScaler()
X_train_s2_scaled = scaler_s2.fit_transform(X_train_s2)
X_val_s2_scaled = scaler_s2.transform(X_val_s2)

model_s2 = create_nn_model(X_train_s2_scaled.shape[1])
model_s2.fit(X_train_s2_scaled, y_train_s2, epochs=epochs_count, verbose=0)
predictions_s2 = model_s2.predict(X_val_s2_scaled).flatten()
evaluate_model_performance("Feature Skew Addressed (Body_Temp)", model_s2, X_val_s2_scaled, y_val_s2, predictions_s2)


#----------Target Skew Addressed (Calories)----------#
X_train_s3 = X_train_orig.copy() # Features are not specifically transformed for skew here, only scaled
y_train_s3_orig = y_train_orig.copy()
X_val_s3 = X_val_orig.copy()
y_val_s3_true = y_val_orig.copy() # True values in original scale for final evaluation

y_train_s3_transformed = y_train_s3_orig.copy() # Initialize for transformation

print(f"Skew of target 'Calories' before transformation: {y_train_s3_transformed.skew():.4f}")
y_train_s3_transformed = np.log1p(y_train_s3_transformed)
print(f"Skew of target 'Calories' after log1p transformation: {y_train_s3_transformed.skew():.4f}")

scaler_s3 = StandardScaler()
X_train_s3_scaled = scaler_s3.fit_transform(X_train_s3)
X_val_s3_scaled = scaler_s3.transform(X_val_s3)

model_s3 = create_nn_model(X_train_s3_scaled.shape[1])
model_s3.fit(X_train_s3_scaled, y_train_s3_transformed, epochs=epochs_count, verbose=0)
predictions_s3_transformed = model_s3.predict(X_val_s3_scaled).flatten()


predictions_s3_inverse = np.expm1(predictions_s3_transformed) # Inverse of np.log1p
evaluate_model_performance("Target Skew Addressed (Calories)", model_s3, X_val_s3_scaled, y_val_s3_true, predictions_s3_inverse)


#----------Both Feature and Target Skew Addressed----------#
X_train_s4 = X_train_orig.copy()
y_train_s4_orig = y_train_orig.copy()
X_val_s4 = X_val_orig.copy()
y_val_s4_true = y_val_orig.copy() # True values in original scale for final evaluation

y_train_s4_transformed = y_train_s4_orig.copy() # Initialize for transformation

feature_to_transform_s4 = 'Body_Temp'
print(f"Skew of {feature_to_transform_s4} before transformation: {X_train_s4[feature_to_transform_s4].skew():.4f}")
pt_feature_s4 = PowerTransformer(method='yeo-johnson')
X_train_s4[feature_to_transform_s4] = pt_feature_s4.fit_transform(X_train_s4[[feature_to_transform_s4]])
X_val_s4[feature_to_transform_s4] = pt_feature_s4.transform(X_val_s4[[feature_to_transform_s4]])
print(f"Skew of {feature_to_transform_s4} after transformation: {X_train_s4[feature_to_transform_s4].skew():.4f}")

print(f"Skew of target 'Calories' before transformation: {y_train_s4_transformed.skew():.4f}")
y_train_s4_transformed = np.log1p(y_train_s4_transformed)
print(f"Skew of target 'Calories' after log1p transformation: {y_train_s4_transformed.skew():.4f}")

scaler_s4 = StandardScaler()
X_train_s4_scaled = scaler_s4.fit_transform(X_train_s4)
X_val_s4_scaled = scaler_s4.transform(X_val_s4)

model_s4 = create_nn_model(X_train_s4_scaled.shape[1])
model_s4.fit(X_train_s4_scaled, y_train_s4_transformed, epochs=epochs_count, verbose=0)
predictions_s4_transformed = model_s4.predict(X_val_s4_scaled).flatten()

predictions_s4_inverse = np.expm1(predictions_s4_transformed) # Inverse of np.log1p
evaluate_model_performance("Both Feature & Target Skew Addressed", model_s4, X_val_s4_scaled, y_val_s4_true, predictions_s4_inverse)


#----------Results----------#
print("\n--- Overall Performance Summary ---")
results_df = pd.DataFrame(all_metrics)
results_df = results_df.set_index('Scenario')
print(results_df)

fig_table = go.Figure(data=[go.Table(
    header=dict(values=list(results_df.reset_index().columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[results_df.reset_index()[col] for col in results_df.reset_index().columns],
               fill_color='lavender',
               align='left'))
])
fig_table.update_layout(title_text="Model Performance Comparison")
fig_table.show()