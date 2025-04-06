import numpy as np
import tensorflow as tf
from pathlib import Path
import joblib
import logging
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

class QuadraticSolver:
    """Neural network solver for quadratic equations ax² + bx + c = 0."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the solver with a pre-trained model.
        
        Args:
            model_path: Path to .h5 model file (uses built-in model if None)
        """
        self.model, self.scaler = self._load_model_and_scaler(model_path)
        self._plot = None  # For storing plot objects

    def _load_model_and_scaler(self, model_path):
        """Load model and scaler from disk."""
        try:
            model_dir = Path(model_path).parent if model_path else Path(__file__).parent / "model"
            model = tf.keras.models.load_model(model_path or model_dir / "solver.h5")
            scaler = joblib.load(model_dir / "scaler.pkl")
            logging.info("Model loaded successfully")
            return model, scaler
        except Exception as e:
            logging.error(f"Loading failed: {e}")
            raise RuntimeError("Model not found. Train a model first.")

    def solve(self, a: float, b: float, c: float) -> Tuple[float, float]:
        """
        Solve quadratic equation ax² + bx + c = 0.
        
        Returns:
            Predicted roots (x1, x2)
            
        Raises:
            ValueError: If coefficient 'a' is zero
        """
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero")
            
        inputs = self.scaler.transform([[a, b, c]])
        x1, x2 = self.model.predict(inputs, verbose=0)[0]
        return round(x1, 4), round(x2, 4)

    def plot_equation(self, a: float, b: float, c: float, save_path: Optional[str] = None):
        """
        Visualize the equation and its roots.
        
        Args:
            save_path: If provided, saves plot to file instead of displaying
        """
        x = np.linspace(-10, 10, 400)
        y = a*x**2 + b*x + c
        roots = self.solve(a, b, c)
        
        plt.figure()
        plt.plot(x, y, label=f"{a}x² + {b}x + {c}")
        plt.scatter(roots, [0, 0], c='red', label=f"Roots: {roots}")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.legend()
        plt.grid()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @classmethod
    def train(
        cls,
        epochs: int = 200,
        samples: int = 50_000,
        save_dir: str = None
    ) -> "QuadraticSolver":
        """Train and save a new model."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        # Generate synthetic data
        X, y = cls._generate_data(samples)
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X)
        
        # Model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(2)  # Output: x1, x2
        ])
        model.compile(optimizer='adam', loss='mse')

        # Training
        model.fit(X_train, y, epochs=epochs, verbose=1)
        
        # Save artifacts
        save_dir = Path(save_dir) if save_dir else Path(__file__).parent / "model"
        save_dir.mkdir(exist_ok=True)
        
        model.save(save_dir / "solver.h5")
        joblib.dump(scaler, save_dir / "scaler.pkl")
        
        return cls(save_dir / "solver.h5")

    @staticmethod
    def _generate_data(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data."""
        X, y = [], []
        for _ in range(n_samples):
            a = np.random.uniform(0.1, 10)
            b, c = np.random.uniform(-10, 10, size=2)
            D = b**2 - 4*a*c
            
            if D >= 0:  # Only real roots
                x1 = (-b + np.sqrt(D)) / (2*a)
                x2 = (-b - np.sqrt(D)) / (2*a)
                X.append([a, b, c])
                y.append([x1, x2])
                
        return np.array(X), np.array(y)