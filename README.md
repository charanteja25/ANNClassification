
# ANN-based Customer Churn Prediction

This project implements an Artificial Neural Network (ANN) to predict customer churn for a banking dataset. The model is built using TensorFlow/Keras and includes data preprocessing steps such as one-hot encoding for categorical features and standardization for numerical features.

## Project Structure

The repository contains the following key files:

- `model.h5`: The trained Keras ANN model for churn prediction.
- `label_encoder_gender.pkl`: A Python pickle file containing the `LabelEncoder` used for the 'Gender' column.
- `onehotencode_geo.pkl`: A Python pickle file containing the `OneHotEncoder` used for the 'Geography' column.
- `scalar.pkl`: A Python pickle file containing the `StandardScaler` used for feature scaling.
- `Churn_Modelling.csv`: The original dataset used for training and evaluation.

## Data Preprocessing

The data preprocessing pipeline involves:
1.  Dropping irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
2.  Encoding 'Gender' using `LabelEncoder`.
3.  One-hot encoding 'Geography' using `OneHotEncoder`.
4.  Splitting the dataset into training and testing sets.
5.  Scaling numerical features using `StandardScaler`.

## Model Architecture

The ANN model consists of:
- An input layer connected to a hidden layer with 64 neurons (ReLU activation).
- A second hidden layer with 32 neurons (ReLU activation).
- An output layer with 1 neuron (Sigmoid activation) for binary classification.

## Training

The model was compiled with the Adam optimizer and `binary_crossentropy` loss. Early stopping and TensorBoard callbacks were used during training.

## How to Use (Local Inference)

To use the trained model and encoders locally for inference, you would typically:

1.  Load the preprocessors and the model:
    ```python
    import pandas as pd
    import pickle
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

    # Load preprocessors
    with open('label_encoder_gender.pkl', 'rb') as f: gender_encoder = pickle.load(f)
    with open('onehotencode_geo.pkl', 'rb') as f: geo_encoder = pickle.load(f)
    with open('scalar.pkl', 'rb') as f: scaler = pickle.load(f)

    # Load model
    model = tf.keras.models.load_model('model.h5')
    ```
2.  Preprocess new input data in the same way as the training data.
3.  Make predictions:
    ```python
    # Example (replace with your actual new data)
    new_data = pd.DataFrame([{'CreditScore': 600, 'Geography': 'France', 'Gender': 'Male', 'Age': 35, 'Tenure': 5,
                              'Balance': 60000, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1,
                              'EstimatedSalary': 50000}])

    # Apply label encoding for Gender
    new_data['Gender'] = gender_encoder.transform(new_data['Gender'])

    # Apply one-hot encoding for Geography
    geo_encoded = geo_encoder.transform(new_data[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
    new_data = pd.concat([new_data.drop('Geography', axis=1), geo_df], axis=1)

    # Ensure column order matches training data after dropping original categorical columns
    # This part requires careful handling to match the X_train columns
    # For simplicity, assuming new_data has all other columns in correct order
    
    # Scale numerical features (ensure proper column selection based on your training X)
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    
    # Make prediction
    prediction = model.predict(new_data)
    churn_probability = prediction[0][0]
    churn_status = 'Will Churn' if churn_probability > 0.5 else 'Will Not Churn'

    print(f"Churn Probability: {churn_probability:.2f}")
    print(f"Churn Status: {churn_status}")
    ```

## Repository

Feel free to explore the code and model for customer churn prediction.
