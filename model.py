import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class QuadraticSolver:
    """
    Neural network for solving quadratic equations ax² + bx + c = 0.
    Predicts roots (x1, x2) given coefficients (a, b, c).
    """
    
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()

    def _build_model(self) -> tf.keras.Model:
        """Constructs a 3-layer neural network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)  # Output: x1, x2
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """
        Trains the model on synthetic quadratic equation data.
        
        Args:
            X: Array of coefficients [a, b, c]
            y: Array of roots [x1, x2]
            epochs: Training iterations
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Normalize data
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            verbose=1
        )

    def predict(self, a: float, b: float, c: float) -> tuple[float, float]:
        """Predicts roots for given coefficients."""
        X = self.scaler.transform([[a, b, c]])
        x1, x2 = self.model.predict(X, verbose=0)[0]
        return round(x1, 4), round(x2, 4)
