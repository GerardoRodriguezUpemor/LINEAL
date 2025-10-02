# 📊 Calculus & Optimization for Machine Learning

## 📚 Table of Contents
1. [Single Variable Calculus](#single-variable-calculus)
2. [Multivariable Calculus](#multivariable-calculus)
3. [Gradient Descent](#gradient-descent)
4. [Constrained Optimization](#constrained-optimization)
5. [Convex Optimization](#convex-optimization)
6. [Applications in ML](#applications-in-ml)

---

## 📈 Single Variable Calculus

### Derivatives
The **derivative** of a function $f(x)$ at point $x$ is:
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Geometric Interpretation:** Slope of the tangent line at point $x$.

### Important Derivatives for ML
$$\frac{d}{dx}[x^n] = nx^{n-1}$$
$$\frac{d}{dx}[e^x] = e^x$$
$$\frac{d}{dx}[\ln(x)] = \frac{1}{x}$$
$$\frac{d}{dx}[\sin(x)] = \cos(x)$$
$$\frac{d}{dx}[\cos(x)] = -\sin(x)$$

### Chain Rule
For composite functions:
$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

**ML Example:** Derivative of sigmoid function
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

---

## 🎯 Multivariable Calculus

### Partial Derivatives
For a function $f(x_1, x_2, \ldots, x_n)$, the partial derivative with respect to $x_i$ is:
$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

### Gradient Vector
The **gradient** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ is:
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Properties:**
- Points in direction of steepest increase
- Perpendicular to level curves
- Zero at local extrema

### Hessian Matrix
The **Hessian** contains all second-order partial derivatives:
$$H_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

**Applications:**
- Determines convexity (positive definite ⟹ convex)
- Newton's method for optimization
- Analysis of critical points

### Chain Rule for Multivariable Functions
For $f(g_1(x), g_2(x), \ldots, g_m(x))$:
$$\frac{\partial f}{\partial x} = \sum_{i=1}^{m} \frac{\partial f}{\partial g_i} \frac{\partial g_i}{\partial x}$$

**Critical for Neural Networks:** This is the mathematical foundation of backpropagation!

---

## ⛰️ Gradient Descent

### Basic Gradient Descent
To minimize function $f(\mathbf{x})$:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t)$$

where $\alpha$ is the **learning rate**.

**Algorithm:**
1. Initialize $\mathbf{x}_0$
2. Repeat until convergence:
   - Compute gradient: $\mathbf{g}_t = \nabla f(\mathbf{x}_t)$
   - Update: $\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \mathbf{g}_t$

### Convergence Analysis
For convex functions with Lipschitz continuous gradient:
$$f(\mathbf{x}_{t+1}) - f(\mathbf{x}^*) \leq (1 - \alpha\mu)(f(\mathbf{x}_t) - f(\mathbf{x}^*))$$

where $\mu$ is the strong convexity parameter.

### Variants of Gradient Descent

#### 1. Stochastic Gradient Descent (SGD)
Use single sample or mini-batch:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \nabla f_i(\mathbf{x}_t)$$

**Advantages:**
- Faster per iteration
- Can escape local minima
- Works with large datasets

#### 2. Momentum
$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + (1-\beta)\nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha \mathbf{v}_{t+1}$$

**Effect:** Accelerates convergence and reduces oscillations.

#### 3. Adam Optimizer
Combines momentum with adaptive learning rates:
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla f(\mathbf{x}_t)$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)[\nabla f(\mathbf{x}_t)]^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\hat{\mathbf{m}}_t$$

---

## 🔒 Constrained Optimization

### Lagrange Multipliers
To optimize $f(\mathbf{x})$ subject to constraint $g(\mathbf{x}) = 0$:

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

**Optimality Conditions:**
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) - \lambda \nabla g(\mathbf{x}) = 0$$
$$\frac{\partial \mathcal{L}}{\partial \lambda} = -g(\mathbf{x}) = 0$$

**Geometric Interpretation:** At the optimum, $\nabla f$ and $\nabla g$ are parallel.

### KKT Conditions
For inequality constraints $g_i(\mathbf{x}) \leq 0$:

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_i \lambda_i h_i(\mathbf{x}) + \sum_j \mu_j g_j(\mathbf{x})$$

**KKT Conditions:**
1. **Stationarity:** $\nabla_{\mathbf{x}} \mathcal{L} = 0$
2. **Primal feasibility:** $g_j(\mathbf{x}) \leq 0$, $h_i(\mathbf{x}) = 0$
3. **Dual feasibility:** $\mu_j \geq 0$
4. **Complementary slackness:** $\mu_j g_j(\mathbf{x}) = 0$

**ML Application:** Support Vector Machine optimization

---

## 📐 Convex Optimization

### Convex Sets
A set $C$ is **convex** if:
$$\mathbf{x}, \mathbf{y} \in C \Rightarrow \theta\mathbf{x} + (1-\theta)\mathbf{y} \in C \quad \forall \theta \in [0,1]$$

### Convex Functions
A function $f$ is **convex** if its domain is convex and:
$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**Equivalent Conditions:**
- First-order: $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$
- Second-order: $\nabla^2 f(\mathbf{x}) \succeq 0$ (Hessian is positive semidefinite)

### Properties of Convex Functions
1. **Local minimum is global minimum**
2. **Sum of convex functions is convex**
3. **Composition rules preserve convexity**

### Common Convex Functions in ML
- **Linear:** $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} + b$
- **Quadratic:** $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ (if $A \succeq 0$)
- **Exponential:** $f(x) = e^x$
- **Negative logarithm:** $f(x) = -\log(x)$
- **Norms:** $f(\mathbf{x}) = ||\mathbf{x}||_p$ for $p \geq 1$

---

## 🤖 Applications in Machine Learning

### 1. Linear Regression Loss Function
**Objective:**
$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)})^2$$

**Gradient:**
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)})x_j^{(i)}$$

**Gradient Descent Update:**
$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

### 2. Logistic Regression
**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Derivative:**
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Cross-entropy Loss:**
$$J(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})) + (1-y^{(i)})\log(1-h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}))]$$

**Gradient:**
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)})x_j^{(i)}$$

### 3. Neural Network Backpropagation
**Forward Pass:**
$$z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = g^{(l)}(z^{(l)})$$

**Backward Pass:**
$$\delta^{(L)} = \nabla_{a^{(L)}} J \odot g'^{(L)}(z^{(L)})$$
$$\delta^{(l)} = ((W^{(l+1)})^T\delta^{(l+1)}) \odot g'^{(l)}(z^{(l)})$$

**Parameter Updates:**
$$\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)}(a^{(l-1)})^T$$
$$\frac{\partial J}{\partial b^{(l)}} = \delta^{(l)}$$

### 4. Support Vector Machine Optimization
**Primal Problem:**
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i=1}^{m}\xi_i$$
subject to:
$$y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi_i$$
$$\xi_i \geq 0$$

**Dual Problem (via Lagrange multipliers):**
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{m}\alpha_i - \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_j y^{(i)}y^{(j)}\mathbf{x}^{(i)T}\mathbf{x}^{(j)}$$
subject to:
$$0 \leq \alpha_i \leq C$$
$$\sum_{i=1}^{m}\alpha_i y^{(i)} = 0$$

---

## 💻 Implementation Examples

### Gradient Descent for Linear Regression
```java
public class LinearRegression {
    private double[] theta;
    private double alpha; // learning rate
    
    public void gradientDescent(double[][] X, double[] y, int iterations) {
        int m = X.length;
        int n = X[0].length;
        
        for (int iter = 0; iter < iterations; iter++) {
            double[] gradient = new double[n];
            
            // Compute predictions
            for (int i = 0; i < m; i++) {
                double prediction = predict(X[i]);
                double error = prediction - y[i];
                
                // Accumulate gradients
                for (int j = 0; j < n; j++) {
                    gradient[j] += error * X[i][j];
                }
            }
            
            // Update parameters
            for (int j = 0; j < n; j++) {
                theta[j] -= alpha * gradient[j] / m;
            }
        }
    }
    
    private double predict(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += theta[i] * x[i];
        }
        return sum;
    }
}
```

### Newton's Method Implementation
```java
public class NewtonMethod {
    public static double[] optimize(Function f, Function gradient, Function hessian, 
                                   double[] x0, double tolerance, int maxIterations) {
        double[] x = x0.clone();
        
        for (int iter = 0; iter < maxIterations; iter++) {
            double[] grad = gradient.apply(x);
            double[][] hess = hessian.apply(x);
            
            // Solve: hessian * delta = -gradient
            double[] delta = solveLinearSystem(hess, negate(grad));
            
            // Update: x = x + delta
            for (int i = 0; i < x.length; i++) {
                x[i] += delta[i];
            }
            
            // Check convergence
            if (norm(delta) < tolerance) {
                break;
            }
        }
        
        return x;
    }
}
```

---

## 🧮 Advanced Optimization Topics

### 1. Second-Order Methods
**Newton's Method:**
$$\mathbf{x}_{t+1} = \mathbf{x}_t - [\nabla^2 f(\mathbf{x}_t)]^{-1} \nabla f(\mathbf{x}_t)$$

**Quasi-Newton Methods (BFGS):**
Approximate the Hessian using gradient information.

### 2. Constrained Optimization Algorithms
- **Interior Point Methods**
- **Sequential Quadratic Programming**
- **Augmented Lagrangian Methods**

### 3. Non-Convex Optimization
- **Simulated Annealing**
- **Genetic Algorithms**
- **Particle Swarm Optimization**

---

## 🚀 Practice Exercises

### Exercise 1: Gradient Calculation
Calculate the gradient of:
$$f(x, y) = x^2 + 3xy + 2y^2$$

### Exercise 2: Implement Gradient Descent
Implement gradient descent to minimize:
$$f(x, y) = (x-3)^2 + (y+1)^2$$

### Exercise 3: Logistic Regression Derivation
Derive the gradient of the logistic regression cost function.

### Exercise 4: Constrained Optimization
Use Lagrange multipliers to solve:
$$\min_{x,y} x^2 + y^2 \text{ subject to } x + y = 1$$

---

## 📚 Further Reading
1. "Convex Optimization" by Boyd & Vandenberghe
2. "Numerical Optimization" by Nocedal & Wright
3. "Pattern Recognition and Machine Learning" by Bishop (Chapter 3)
4. Online: Stanford CS229 Lecture Notes on Optimization

---

*Next: [Probability and Statistics](./03_probability_statistics.md)*