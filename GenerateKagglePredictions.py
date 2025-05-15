import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


MODEL_PATH = 'trained_calorie_predictor_model_100epochs.keras'
TEST_DATA_PATH = 'Data/test.csv'
TRAIN_DATA_PATH = 'Data/train.csv'
SUBMISSION_FILE_PATH = 'kaggle_submission.csv'

model = load_model(MODEL_PATH)

kaggle_test_df = pd.read_csv(TEST_DATA_PATH)
submission_ids = kaggle_test_df['id'].copy()

X_kaggle_test = kaggle_test_df.copy()

X_kaggle_test.replace({'male': 1, 'female': 0}, inplace=True)
X_kaggle_test.drop('id', axis=1, inplace=True)

full_train_df_for_scaler = pd.read_csv(TRAIN_DATA_PATH)
full_train_df_for_scaler.replace({'male': 1, 'female': 0}, inplace=True)

columns_to_drop_for_scaler = ['Calories','id']
train_features_for_scaler = full_train_df_for_scaler.drop(columns=columns_to_drop_for_scaler)

X_kaggle_test = X_kaggle_test.reindex(columns=train_features_for_scaler.columns, fill_value=0)

scaler = StandardScaler()
scaler.fit(train_features_for_scaler)
X_kaggle_test_scaled = scaler.transform(X_kaggle_test)

predictions = model.predict(X_kaggle_test_scaled).flatten()

calories_predictions_rounded = np.round(predictions, 3)

submission_df = pd.DataFrame({
    'id': submission_ids,
    'Calories': calories_predictions_rounded
})

submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)

print(submission_df.head())
