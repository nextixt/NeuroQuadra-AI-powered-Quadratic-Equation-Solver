import numpy as np
from model import QuadraticSolver

def generate_data(num_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic quadratic equation data.
    Returns:
        X: Array of coefficients [a, b, c]
        y: Array of roots [x1, x2]
    """
    X = []
    y = []
    for _ in range(num_samples):
        a = np.random.uniform(0.1, 10)  # a ≠ 0
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        discriminant = b**2 - 4*a*c
        
        if discriminant >= 0:  # Only real roots
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            x2 = (-b - np.sqrt(discriminant)) / (2*a)
            X.append([a, b, c])
            y.append([x1, x2])
    
    return np.array(X), np.array(y)

def main():
    """Command-line interface for the solver."""
    print("🔢 NeuroQuadra - AI Quadratic Equation Solver")
    
    # Initialize and train
    solver = QuadraticSolver()
    X, y = generate_data()
    solver.train(X, y, epochs=50)
    
    # Interactive mode
    while True:
        try:
            coefs = input("\nEnter coefficients (a b c) or 'q' to quit: ").strip()
            if coefs.lower() == 'q':
                break
                
            a, b, c = map(float, coefs.split())
            if a == 0:
                print("Error: 'a' cannot be zero!")
                continue
                
            x1_pred, x2_pred = solver.predict(a, b, c)
            print(f"\nEquation: {a}x² + {b}x + {c} = 0")
            print(f"AI predicted roots: x₁={x1_pred}, x₂={x2_pred}")
            
            # Calculate exact roots for comparison
            D = b**2 - 4*a*c
            if D >= 0:
                x1_exact = (-b + D**0.5) / (2*a)
                x2_exact = (-b - D**0.5) / (2*a)
                print(f"Exact roots: x₁={x1_exact:.4f}, x₂={x2_exact:.4f}")
            else:
                print("Note: Complex roots (not supported in this version)")
                
        except Exception as e:
            print(f"Error: {e}. Try again!")

if __name__ == "__main__":
    main()
