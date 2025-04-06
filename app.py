Examples:

from quadratic_solver import QuadraticSolver

# Initialize solver (loads pre-trained model)
# Use QuadraticSolver("path/to/model.h5") to specify custom model path
solver = QuadraticSolver()  

# Solve equation: 2x² - 5x + 3 = 0
x1, x2 = solver.solve(2, -5, 3)
print(f"Equation roots: {x1}, {x2}")  # Example output: Equation roots: 1.5, 1.0

# Visualize the equation
solver.plot_equation(2, -5, 3)  # Displays interactive plot

or


from quadratic_solver import QuadraticSolver

# Train a new model with custom parameters:
# - 200 training epochs
# - 50,000 generated samples
# - Saves to "my_model" directory
solver = QuadraticSolver.train(
    epochs=200,
    samples=50_000,
    save_dir="my_model"  # Model will be saved in my_model/ folder
)

# Now we can use the trained solver
# Solve equation: 1x² - 3x + 2 = 0
x1, x2 = solver.solve(1, -3, 2)
print(f"Roots: {x1}, {x2}")  # Expected output: Roots: 2.0, 1.0
