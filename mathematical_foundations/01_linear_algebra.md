# 🔢 Linear Algebra for Machine Learning

## 📚 Table of Contents
1. [Vectors and Vector Spaces](#vectors-and-vector-spaces)
2. [Vector Norms](#vector-norms)
3. [Matrix Operations](#matrix-operations)
4. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
5. [Matrix Decompositions](#matrix-decompositions)
6. [Applications in ML](#applications-in-ml)

---

## 🎯 Vectors and Vector Spaces

### Definition
A **vector** is an element of a vector space. In machine learning, we typically work with vectors in $\mathbb{R}^n$ (n-dimensional real space).

**Mathematical Notation:**
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

### Vector Operations

#### 1. Vector Addition
$$\mathbf{x} + \mathbf{y} = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{bmatrix}$$

#### 2. Scalar Multiplication
$$c\mathbf{x} = \begin{bmatrix} cx_1 \\ cx_2 \\ \vdots \\ cx_n \end{bmatrix}$$

#### 3. Dot Product (Inner Product)
$$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

**Geometric Interpretation:**
$$\mathbf{x} \cdot \mathbf{y} = ||\mathbf{x}|| \cdot ||\mathbf{y}|| \cos(\theta)$$
where $\theta$ is the angle between vectors $\mathbf{x}$ and $\mathbf{y}$.

### Linear Independence
Vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are **linearly independent** if:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$
implies $c_1 = c_2 = \cdots = c_k = 0$.

**ML Relevance:** Linear independence is crucial for understanding feature redundancy and dimensionality reduction.

---

## 📏 Vector Norms

### Definition
A **norm** is a function that assigns a positive length to each vector in a vector space.

### L^p Norms
The **L^p norm** of a vector $\mathbf{x}$ is defined as:

$$||\mathbf{x}||_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

*This is the formula from your image!*

### Common Norms

#### 1. L¹ Norm (Manhattan Norm)
$$||\mathbf{x}||_1 = \sum_{i=1}^{n} |x_i|$$

**Properties:**
- Promotes sparsity
- Used in Lasso regression
- Robust to outliers

#### 2. L² Norm (Euclidean Norm)
$$||\mathbf{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**Properties:**
- Most common norm
- Differentiable everywhere
- Used in Ridge regression

#### 3. L^∞ Norm (Maximum Norm)
$$||\mathbf{x}||_{\infty} = \max_{i} |x_i|$$

**Properties:**
- Maximum absolute value
- Used in minimax optimization

### Norm Properties
1. **Positivity:** $||\mathbf{x}|| \geq 0$, with equality iff $\mathbf{x} = \mathbf{0}$
2. **Homogeneity:** $||c\mathbf{x}|| = |c| \cdot ||\mathbf{x}||$
3. **Triangle Inequality:** $||\mathbf{x} + \mathbf{y}|| \leq ||\mathbf{x}|| + ||\mathbf{y}||$

**ML Applications:**
- Regularization in regression
- Distance metrics in clustering
- Loss functions in optimization

---

## 🔢 Matrix Operations

### Matrix Multiplication
For matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$:
$$(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

**Properties:**
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$
- **Not commutative:** $AB \neq BA$ (in general)

### Matrix Transpose
$$A^T_{ij} = A_{ji}$$

**Properties:**
- $(A^T)^T = A$
- $(AB)^T = B^T A^T$
- $(A + B)^T = A^T + B^T$

### Matrix Inverse
For a square matrix $A$, its inverse $A^{-1}$ satisfies:
$$AA^{-1} = A^{-1}A = I$$

**Properties:**
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$
- $\det(A^{-1}) = 1/\det(A)$

**Existence:** A matrix is invertible iff its determinant is non-zero.

### Special Matrices

#### 1. Orthogonal Matrix
$$Q^TQ = QQ^T = I$$
**Properties:** Preserves distances and angles

#### 2. Symmetric Matrix
$$A = A^T$$
**Properties:** Real eigenvalues, orthogonal eigenvectors

#### 3. Positive Definite Matrix
$$\mathbf{x}^T A \mathbf{x} > 0 \text{ for all } \mathbf{x} \neq \mathbf{0}$$

---

## 🎭 Eigenvalues and Eigenvectors

### Definition
For a square matrix $A$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:
$$A\mathbf{v} = \lambda\mathbf{v}$$

### Characteristic Equation
$$\det(A - \lambda I) = 0$$

### Properties
1. **Geometric Interpretation:** Eigenvectors are directions that remain unchanged under the linear transformation $A$
2. **Scaling:** The eigenvalue determines how much the eigenvector is scaled
3. **Eigenspace:** The set of all eigenvectors with the same eigenvalue forms a subspace

### Diagonalization
If $A$ has $n$ linearly independent eigenvectors, then:
$$A = PDP^{-1}$$
where:
- $P$ = matrix of eigenvectors
- $D$ = diagonal matrix of eigenvalues

**ML Applications:**
- Principal Component Analysis (PCA)
- Spectral clustering
- Stability analysis of optimization algorithms

---

## 🔍 Matrix Decompositions

### 1. Singular Value Decomposition (SVD)
Every matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed as:
$$A = U\Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ (diagonal matrix of singular values)
- $V \in \mathbb{R}^{n \times n}$ (right singular vectors)

**Properties:**
- $U$ and $V$ are orthogonal matrices
- Singular values are non-negative and ordered: $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$

**ML Applications:**
- Dimensionality reduction
- Recommender systems
- Image compression
- Principal Component Analysis

### 2. LU Decomposition
$$A = LU$$
where:
- $L$ = lower triangular matrix
- $U$ = upper triangular matrix

**Applications:**
- Solving linear systems
- Matrix inversion
- Determinant calculation

### 3. QR Decomposition
$$A = QR$$
where:
- $Q$ = orthogonal matrix
- $R$ = upper triangular matrix

**Applications:**
- Least squares problems
- Eigenvalue algorithms
- Gram-Schmidt orthogonalization

---

## 🤖 Applications in Machine Learning

### 1. Principal Component Analysis (PCA)
**Objective:** Find directions of maximum variance

**Mathematical Formulation:**
1. Center the data: $X_{centered} = X - \mu$
2. Compute covariance matrix: $C = \frac{1}{n}X_{centered}^T X_{centered}$
3. Find eigenvectors of $C$: $C\mathbf{v}_i = \lambda_i\mathbf{v}_i$
4. Principal components = eigenvectors with largest eigenvalues

**Code Example (Conceptual):**
```java
// In your LINEAL.java, you could implement:
public class PCA {
    public static double[][] computePCA(double[][] data, int numComponents) {
        // 1. Center the data
        double[][] centered = centerData(data);
        
        // 2. Compute covariance matrix
        double[][] covariance = computeCovariance(centered);
        
        // 3. Find eigenvalues and eigenvectors
        EigenDecomposition eigen = new EigenDecomposition(covariance);
        
        // 4. Select top k eigenvectors
        return selectTopComponents(eigen, numComponents);
    }
}
```

### 2. Linear Regression
**Normal Equation:**
$$\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}$$

**Derivation:**
1. Minimize: $||\mathbf{y} - X\boldsymbol{\beta}||_2^2$
2. Take derivative: $\frac{\partial}{\partial\boldsymbol{\beta}}(\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta}) = 0$
3. Solve: $-2X^T(\mathbf{y} - X\boldsymbol{\beta}) = 0$
4. Result: $\boldsymbol{\beta} = (X^TX)^{-1}X^T\mathbf{y}$

### 3. Support Vector Machines
**Key Linear Algebra Concepts:**
- Hyperplane separation: $\mathbf{w}^T\mathbf{x} + b = 0$
- Distance to hyperplane: $\frac{|\mathbf{w}^T\mathbf{x} + b|}{||\mathbf{w}||_2}$
- Kernel trick: $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$

### 4. Neural Networks
**Forward Pass:**
$$\mathbf{a}^{(l+1)} = f(W^{(l)}\mathbf{a}^{(l)} + \mathbf{b}^{(l)})$$

**Backpropagation:**
Uses chain rule and matrix derivatives to compute gradients efficiently.

---

## 💡 Key Insights for ML

### 1. Dimensionality and Overfitting
- High-dimensional spaces have counterintuitive properties
- Curse of dimensionality affects distance metrics
- Regularization helps control model complexity

### 2. Geometric Interpretation
- ML algorithms often have geometric interpretations
- Understanding vector spaces helps visualize solutions
- Orthogonality and projections are fundamental concepts

### 3. Computational Considerations
- Matrix operations scale as $O(n^3)$ for $n \times n$ matrices
- Sparse matrices can be exploited for efficiency
- Numerical stability is crucial in implementations

---

## 🚀 Practice Exercises

### Exercise 1: Norm Calculations
Given $\mathbf{x} = [3, -4, 0, 5]$, calculate:
1. $||\mathbf{x}||_1$
2. $||\mathbf{x}||_2$
3. $||\mathbf{x}||_{\infty}$

### Exercise 2: Eigenvalue Problem
Find eigenvalues and eigenvectors of:
$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

### Exercise 3: PCA Implementation
Implement PCA from scratch for a 2D dataset and visualize the principal components.

### Exercise 4: Linear Regression
Derive and implement the normal equation for linear regression.

---

## 📚 Further Reading
1. "Linear Algebra and Its Applications" by Gilbert Strang
2. "Matrix Analysis and Applied Linear Algebra" by Carl Meyer
3. "The Matrix Cookbook" by Petersen & Pedersen
4. Online: MIT 18.06 Linear Algebra Course

---

*Next: [Calculus and Optimization](./02_calculus_optimization.md)*