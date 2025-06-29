"""
Mathematical cheatsheet and reference implementations.

This module provides clear, educational implementations of mathematical
concepts needed for understanding chaos theory, from basic algebra to
eigenvectors.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
import sympy as sp


class MathConcept:
    """Base class for mathematical concepts with examples."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def explain(self) -> str:
        """Return explanation of the concept."""
        return f"{self.name}: {self.description}"
    
    def example(self) -> str:
        """Return a worked example."""
        raise NotImplementedError


class PEMDAS(MathConcept):
    """Order of operations: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction."""
    
    def __init__(self):
        super().__init__(
            "PEMDAS",
            "Order of operations in mathematical expressions"
        )
    
    def example(self) -> str:
        return """
        Expression: 2 + 3 * 4^2 - (8 / 2)
        
        Step 1 (Parentheses): 2 + 3 * 4^2 - 4
        Step 2 (Exponents): 2 + 3 * 16 - 4
        Step 3 (Multiplication): 2 + 48 - 4
        Step 4 (Addition/Subtraction): 50 - 4 = 46
        """
    
    def evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Use sympy for safe evaluation
        return float(sp.sympify(expression))


class Exponents(MathConcept):
    """Rules for working with exponents and powers."""
    
    def __init__(self):
        super().__init__(
            "Exponents",
            "Rules for powers and exponential expressions"
        )
    
    def example(self) -> str:
        return """
        Key Rules:
        1. x^a * x^b = x^(a+b)     Example: x^2 * x^3 = x^5
        2. x^a / x^b = x^(a-b)     Example: x^5 / x^2 = x^3
        3. (x^a)^b = x^(a*b)       Example: (x^2)^3 = x^6
        4. x^0 = 1                 Example: 5^0 = 1
        5. x^(-a) = 1/x^a          Example: 2^(-3) = 1/8
        """
    
    def power_rules_demo(self, base: float, exp1: float, exp2: float) -> dict:
        """Demonstrate power rules with specific values."""
        return {
            "multiplication": (base**exp1) * (base**exp2),
            "division": (base**exp1) / (base**exp2) if exp2 != 0 else None,
            "power_of_power": (base**exp1)**exp2,
            "negative_exponent": base**(-exp1)
        }


class LinearEquations(MathConcept):
    """Solving linear equations and systems."""
    
    def __init__(self):
        super().__init__(
            "Linear Equations",
            "Equations of the form ax + b = c"
        )
    
    def example(self) -> str:
        return """
        Single equation: 3x + 5 = 20
        Solution: 
            3x = 20 - 5
            3x = 15
            x = 5
        
        System of equations:
            2x + 3y = 13
            x - y = 2
        
        Solution by substitution:
            x = y + 2
            2(y + 2) + 3y = 13
            2y + 4 + 3y = 13
            5y = 9
            y = 1.8, x = 3.8
        """
    
    def solve_linear(self, a: float, b: float, c: float) -> float:
        """Solve ax + b = c for x."""
        if a == 0:
            raise ValueError("Not a linear equation (a = 0)")
        return (c - b) / a
    
    def solve_system_2x2(self, a1: float, b1: float, c1: float,
                         a2: float, b2: float, c2: float) -> Tuple[float, float]:
        """
        Solve system:
            a1*x + b1*y = c1
            a2*x + b2*y = c2
        """
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            raise ValueError("System has no unique solution")
        
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return x, y


class Quadratics(MathConcept):
    """Quadratic equations and their properties."""
    
    def __init__(self):
        super().__init__(
            "Quadratic Equations",
            "Equations of the form ax^2 + bx + c = 0"
        )
    
    def example(self) -> str:
        return """
        Equation: x^2 - 5x + 6 = 0
        
        Method 1 - Factoring:
            (x - 2)(x - 3) = 0
            x = 2 or x = 3
        
        Method 2 - Quadratic Formula:
            x = (-b ± √(b² - 4ac)) / 2a
            x = (5 ± √(25 - 24)) / 2
            x = (5 ± 1) / 2
            x = 3 or x = 2
        """
    
    def solve_quadratic(self, a: float, b: float, c: float) -> Tuple[complex, complex]:
        """Solve ax² + bx + c = 0 using the quadratic formula."""
        if a == 0:
            raise ValueError("Not a quadratic equation (a = 0)")
        
        discriminant = b**2 - 4*a*c
        sqrt_disc = np.sqrt(discriminant + 0j)  # Force complex for negative discriminant
        
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        
        return x1, x2
    
    def vertex(self, a: float, b: float, c: float) -> Tuple[float, float]:
        """Find the vertex of the parabola y = ax² + bx + c."""
        x_vertex = -b / (2*a)
        y_vertex = a * x_vertex**2 + b * x_vertex + c
        return x_vertex, y_vertex


class Functions(MathConcept):
    """Understanding functions and function composition."""
    
    def __init__(self):
        super().__init__(
            "Functions",
            "Mappings from inputs to outputs"
        )
    
    def example(self) -> str:
        return """
        Function notation: f(x) = x² + 2x + 1
        
        Evaluation: f(3) = 3² + 2(3) + 1 = 9 + 6 + 1 = 16
        
        Composition: If f(x) = x² and g(x) = x + 1, then
            (f ∘ g)(x) = f(g(x)) = f(x + 1) = (x + 1)²
            (g ∘ f)(x) = g(f(x)) = g(x²) = x² + 1
        
        Note: f  g ` g  f in general!
        """
    
    def compose(self, f: callable, g: callable) -> callable:
        """Return the composition f  g."""
        return lambda x: f(g(x))
    
    def iterate(self, f: callable, x0: float, n: int) -> List[float]:
        """Iterate function f starting from x0 for n steps."""
        trajectory = [x0]
        x = x0
        for _ in range(n):
            x = f(x)
            trajectory.append(x)
        return trajectory


class Derivatives(MathConcept):
    """Basic differentiation and its meaning."""
    
    def __init__(self):
        super().__init__(
            "Derivatives",
            "Rate of change of functions"
        )
    
    def example(self) -> str:
        return """
        Derivative = instantaneous rate of change
        
        Power rule: d/dx(x^n) = n*x^(n-1)
        Examples:
            d/dx(x²) = 2x
            d/dx(x³) = 3x²
            d/dx(x) = 1
            d/dx(5) = 0
        
        Chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
        Example: d/dx[(2x + 1)³] = 3(2x + 1)² * 2 = 6(2x + 1)²
        """
    
    def numerical_derivative(self, f: callable, x: float, h: float = 1e-5) -> float:
        """Compute derivative numerically using central difference."""
        return (f(x + h) - f(x - h)) / (2 * h)
    
    def symbolic_derivative(self, expression: str, variable: str = 'x') -> str:
        """Compute derivative symbolically using sympy."""
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)
        derivative = sp.diff(expr, x)
        return str(derivative)


class DifferentialEquations(MathConcept):
    """Introduction to ODEs and their solutions."""
    
    def __init__(self):
        super().__init__(
            "Differential Equations",
            "Equations involving derivatives"
        )
    
    def example(self) -> str:
        return """
        Simple ODE: dy/dx = y
        Solution: y = Ce^x (exponential growth)
        
        System of ODEs (like Lorenz):
            dx/dt = σ(y - x)
            dy/dt = x(ρ - z) - y
            dz/dt = xy - βz
        
        Solved numerically using methods like:
        - Euler's method: x_{n+1} = x_n + h*f(x_n, t_n)
        - Runge-Kutta: More accurate multi-step method
        """
    
    def euler_method(self, f: callable, x0: float, t0: float, 
                     t_final: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Solve ODE dx/dt = f(x, t) using Euler's method."""
        t = np.arange(t0, t_final + h, h)
        x = np.zeros(len(t))
        x[0] = x0
        
        for i in range(1, len(t)):
            x[i] = x[i-1] + h * f(x[i-1], t[i-1])
        
        return t, x


class Vectors(MathConcept):
    """Vector operations and geometry."""
    
    def __init__(self):
        super().__init__(
            "Vectors",
            "Quantities with magnitude and direction"
        )
    
    def example(self) -> str:
        return """
        Vector: v = [3, 4]
        Magnitude: |v| = √(3² + 4²) = 5
        
        Operations:
        - Addition: [1, 2] + [3, 4] = [4, 6]
        - Scalar multiplication: 2 * [1, 2] = [2, 4]
        - Dot product: [1, 2] · [3, 4] = 1*3 + 2*4 = 11
        
        In 3D:
        - Cross product: [1, 0, 0] × [0, 1, 0] = [0, 0, 1]
        """
    
    def magnitude(self, v: np.ndarray) -> float:
        """Compute vector magnitude."""
        return np.linalg.norm(v)
    
    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Return unit vector in direction of v."""
        mag = self.magnitude(v)
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return v / mag
    
    def dot_product(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute dot product of two vectors."""
        return np.dot(v1, v2)
    
    def cross_product(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Compute cross product of two 3D vectors."""
        return np.cross(v1, v2)


class Matrices(MathConcept):
    """Matrix operations and transformations."""
    
    def __init__(self):
        super().__init__(
            "Matrices",
            "Rectangular arrays of numbers representing linear transformations"
        )
    
    def example(self) -> str:
        return """
        Matrix multiplication:
        [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
        [3 4] [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
        
        Matrix-vector multiplication (transformation):
        [2 0] [1]   [2]
        [0 3] [1] = [3]  (scaling transformation)
        
        Special matrices:
        - Identity: I = [[1, 0], [0, 1]]
        - Rotation: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        """
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices."""
        return np.matmul(A, B)
    
    def rotation_matrix_2d(self, theta: float) -> np.ndarray:
        """Create 2D rotation matrix for angle theta (radians)."""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    def transform_points(self, matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply matrix transformation to points."""
        return np.dot(points, matrix.T)


class Eigenvectors(MathConcept):
    """Eigenvectors, eigenvalues, and their significance."""
    
    def __init__(self):
        super().__init__(
            "Eigenvectors and Eigenvalues",
            "Special vectors that don't change direction under transformation"
        )
    
    def example(self) -> str:
        return """
        Definition: Av = λv
        A is matrix, v is eigenvector, λ is eigenvalue
        
        Example:
        A = [[3, 1],     v = [1],    λ = 4
             [1, 3]]         [1]
        
        Check: Av = [[3, 1],  [1]  = [4]  = 4 * [1]  
                     [1, 3]]  [1]    [4]        [1]
        
        Physical meaning:
        - Eigenvectors show invariant directions
        - Eigenvalues show stretching/shrinking factors
        - In chaos: eigenvalues > 1 indicate expansion (sensitivity)
        """
    
    def compute_eigen(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of matrix A."""
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    
    def verify_eigenvector(self, A: np.ndarray, v: np.ndarray, 
                          lambda_val: float, tol: float = 1e-10) -> bool:
        """Verify that v is an eigenvector of A with eigenvalue lambda."""
        Av = np.dot(A, v)
        lambda_v = lambda_val * v
        return np.allclose(Av, lambda_v, atol=tol)


# Essential 12 Curriculum
ESSENTIAL_12 = [
    PEMDAS(),
    Exponents(),
    LinearEquations(),
    Quadratics(),
    Functions(),
    Derivatives(),
    DifferentialEquations(),
    Vectors(),
    Matrices(),
    Eigenvectors(),
    # Two more advanced topics for chaos theory
]


def generate_anki_cards() -> List[dict]:
    """Generate Anki flashcards for the Essential 12."""
    cards = []
    
    for concept in ESSENTIAL_12:
        # Basic definition card
        cards.append({
            "front": f"What is {concept.name}?",
            "back": concept.description,
            "tags": ["math", "chaos-theory", concept.name.lower().replace(" ", "-")]
        })
        
        # Example card
        cards.append({
            "front": f"Give an example of {concept.name}",
            "back": concept.example(),
            "tags": ["math", "chaos-theory", concept.name.lower().replace(" ", "-"), "example"]
        })
        
        # Additional concept-specific cards would go here
        
    return cards


def print_cheatsheet():
    """Print a one-page mathematical cheatsheet."""
    print("=" * 60)
    print("CHAOS THEORY MATHEMATICAL CHEATSHEET".center(60))
    print("=" * 60)
    
    for i, concept in enumerate(ESSENTIAL_12, 1):
        print(f"\n{i}. {concept.name}")
        print("-" * 40)
        print(concept.description)
        print()
        # Print compact example or key formula
        example_lines = concept.example().strip().split('\n')[:4]
        for line in example_lines:
            if line.strip():
                print(f"   {line.strip()}")
    
    print("\n" + "=" * 60)
    print("For interactive examples, run: python -m src.cheatsheet")


if __name__ == "__main__":
    print_cheatsheet()