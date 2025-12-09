#Name: Trey Gardner
#Worksheet: 13
#Class CS3080

import numpy as np, pandas as pd, time, tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from fixed_coin_data import depth_chart_live, get_coin_price
from keras.callbacks import EarlyStopping

def create_model(mode='1', coin='btc'):
    print("Creating a new model...")
    global mae
    
    if mode == '1':
        df = pd.read_csv(f'{coin}_order_book_summary.csv')
        X = df[['Mean Bids Price', 'Mean Asks Price']].values
        Y = df['Price After 1min'].values
    else:
        create_xminute_data(mode, coin)  # Ensure the data is created
        df = pd.read_csv(f'{coin}_order_book_{mode}min_summary.csv')
        X = df[['Mean Bids Price', 'Mean Asks Price']].values
        # uncomment if you want other data (make sure you uncomment data fields in depth_chart_live in coin_data.py as well)
        # X = df[['Mean Bids Price', 'Mean Bids Size', 'Mean Asks Price', 'Mean AsksSize']].values
        Y = df[f'Price After {mode}min'].values
    
    # Split data - THIS SHOULD HAPPEN FOR BOTH MODE='1' AND MODE!=1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
    
    # Build model architecture
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)  # Single output for regression
    ])
    
    # Check for GPU
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU for training")
    else:
        print("Using CPU for training")
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Define EarlyStopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',  # metric to monitor
        patience=10,  # epochs with no improvement to wait before stopping
        restore_best_weights=True  # restore model weights from best epoch
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=100,
        validation_split=0.05,
        verbose=1,
        callbacks=[early_stop]
    )
    
    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compare some predictions
    for i in range(5):
        print(f"True: {y_test[i]:.3f}, Predicted: {y_pred[i][0]:.3f}")
    
    # Save model
    model.save(f'{coin}_MLP_{mode}min.keras')
    
    return model


def load_model(mode, coin):
    # Load the saved model
    try:
        model = keras.models.load_model(f'{coin}_MLP_{mode}min.keras')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new MLP model...")
        model = create_model(mode, coin)
    return model


def create_xminute_data(mode, coin):
    df = pd.read_csv(f'{coin}_order_book_summary.csv')
    shift_val = int(mode) - 1  # because data is already shifted by 1 minute
    df[f'Price After {mode}min'] = df['Price After 1min'].shift(-shift_val)
    
    # Shift to create target variable
    df.dropna(inplace=True)
    df.to_csv(f'{coin}_order_book_{mode}min_summary.csv', index=False)


def driver():
    mode = input("Train for and predict next 1 min price or next x min price? Default is 1. Enter a number bigger than equal with 1:\n").strip().lower() or '1'
    coin = input("What coin do you want to predict? Default is btc. (options: btc, eth, shib, etc.):\n").strip().lower() or 'btc'
    model = load_model(mode=mode, coin=coin)
    
    # only applies when saved model is loaded
    global mae
    if 'mae' not in globals():
        mae = 250 if coin == 'btc' else 0.0025  # Default MAE for BTC, adjust for other coins
    
    print("\nStarting live prediction...\n")
    
    while True:
        data = depth_chart_live(coin)
        data = np.array(data).reshape(1, -1)  # Reshape for single prediction
        prediction = model.predict(data, verbose=0)  # Added verbose=0 to reduce output
        print(f"Predicted {coin} price for the next {mode} minute(s):")
        print(f"Prediction\tPrediction - MAE\tPrediction + MAE")
        print(f"{prediction[0][0]:.3f}\t\t{prediction[0][0]-mae:.3f}\t\t\t{prediction[0][0]+mae:.3f}")
        print(f"Actual {coin} price now: {get_coin_price(coin):.3f}")
        print("\n")
        time.sleep(60)


if __name__ == "__main__":
    driver()
