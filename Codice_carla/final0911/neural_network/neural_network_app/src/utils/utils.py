def log(message):
    print(f"[LOG] {message}")

def visualize_training(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2