package org.upemor.personal.lineal;

import java.util.*;
import java.util.function.Function;

/**
 * LINEAL - Linear Algebra and Machine Learning Mathematics Library
 * 
 * A comprehensive library for understanding and implementing the mathematical
 * foundations of machine learning, including:
 * - Vector and Matrix operations
 * - Norms and Distance metrics  
 * - Optimization algorithms
 * - Statistical functions
 * - Machine Learning algorithms from scratch
 * 
 * @author gera_
 */
public class LINEAL {
    
    // ========================================
    // VECTOR OPERATIONS
    // ========================================
    
    /**
     * Calculates the Lp norm of a vector
     * ||x||_p = (sum(|x_i|^p))^(1/p)
     * 
     * This is the mathematical formula from your image!
     */
    public static double lpNorm(double[] vector, double p) {
        if (p <= 0) {
            throw new IllegalArgumentException("p must be positive");
        }
        
        double sum = 0.0;
        for (double x : vector) {
            sum += Math.pow(Math.abs(x), p);
        }
        
        return Math.pow(sum, 1.0 / p);
    }
    
    /**
     * L1 Norm (Manhattan distance)
     * Used in Lasso regression for sparsity
     */
    public static double l1Norm(double[] vector) {
        return lpNorm(vector, 1.0);
    }
    
    /**
     * L2 Norm (Euclidean distance)  
     * Most common norm in ML
     */
    public static double l2Norm(double[] vector) {
        return lpNorm(vector, 2.0);
    }
    
    /**
     * L-infinity Norm (Maximum norm)
     */
    public static double lInfinityNorm(double[] vector) {
        double max = 0.0;
        for (double x : vector) {
            max = Math.max(max, Math.abs(x));
        }
        return max;
    }
    
    /**
     * Dot product of two vectors
     * Essential for linear models
     */
    public static double dotProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    /**
     * Vector addition
     */
    public static double[] vectorAdd(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }
        
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
    
    /**
     * Scalar multiplication
     */
    public static double[] scalarMultiply(double[] vector, double scalar) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] * scalar;
        }
        return result;
    }
    
    // ========================================
    // MATRIX OPERATIONS
    // ========================================
    
    /**
     * Matrix multiplication
     * Essential for neural networks and linear models
     */
    public static double[][] matrixMultiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int rowsB = B.length;
        int colsB = B[0].length;
        
        if (colsA != rowsB) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication");
        }
        
        double[][] result = new double[rowsA][colsB];
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Matrix transpose
     */
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        
        double[][] result = new double[cols][rows];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        
        return result;
    }
    
    // ========================================
    // ACTIVATION FUNCTIONS
    // ========================================
    
    /**
     * Sigmoid function: 1 / (1 + e^(-x))
     * Used in logistic regression and neural networks
     */
    public static double sigmoid(double x) {
        // Prevent overflow for large negative values
        if (x < -500) return 0.0;
        if (x > 500) return 1.0;
        
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
     * Used in backpropagation
     */
    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    /**
     * ReLU activation: max(0, x)
     */
    public static double relu(double x) {
        return Math.max(0, x);
    }
    
    /**
     * Derivative of ReLU
     */
    public static double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    /**
     * Tanh activation
     */
    public static double tanh(double x) {
        return Math.tanh(x);
    }
    
    // ========================================
    // OPTIMIZATION ALGORITHMS
    // ========================================
    
    /**
     * Simple Gradient Descent for function minimization
     * 
     * @param gradient Function that computes gradient at given point
     * @param x0 Starting point
     * @param learningRate Step size
     * @param iterations Number of iterations
     */
    public static double[] gradientDescent(Function<double[], double[]> gradient,
                                          double[] x0, double learningRate, int iterations) {
        double[] x = Arrays.copyOf(x0, x0.length);
        
        for (int i = 0; i < iterations; i++) {
            double[] grad = gradient.apply(x);
            
            // Update: x = x - alpha * gradient
            for (int j = 0; j < x.length; j++) {
                x[j] = x[j] - learningRate * grad[j];
            }
        }
        
        return x;
    }
    
    // ========================================
    // LINEAR REGRESSION
    // ========================================
    
    /**
     * Linear Regression using Normal Equation
     * theta = (X^T * X)^(-1) * X^T * y
     */
    public static class LinearRegression {
        private double[] theta;
        private boolean fitted = false;
        
        /**
         * Fit the model using normal equation
         */
        public void fit(double[][] X, double[] y) {
            // Add bias term (column of ones)
            double[][] XWithBias = addBiasColumn(X);
            
            // Compute X^T
            double[][] XT = transpose(XWithBias);
            
            // Compute X^T * X
            double[][] XTX = matrixMultiply(XT, XWithBias);
            
            // Compute X^T * y
            double[] XTy = matrixVectorMultiply(XT, y);
            
            // Solve XTX * theta = XTy (simplified - in practice use LU decomposition)
            theta = solveLinearSystem(XTX, XTy);
            fitted = true;
        }
        
        /**
         * Make predictions
         */
        public double predict(double[] x) {
            if (!fitted) {
                throw new IllegalStateException("Model must be fitted before prediction");
            }
            
            // Add bias term
            double[] xWithBias = new double[x.length + 1];
            xWithBias[0] = 1.0; // bias
            System.arraycopy(x, 0, xWithBias, 1, x.length);
            
            return dotProduct(theta, xWithBias);
        }
        
        /**
         * Compute cost function (Mean Squared Error)
         */
        public double cost(double[][] X, double[] y) {
            double[][] XWithBias = addBiasColumn(X);
            double totalError = 0.0;
            
            for (int i = 0; i < X.length; i++) {
                double prediction = dotProduct(theta, XWithBias[i]);
                double error = prediction - y[i];
                totalError += error * error;
            }
            
            return totalError / (2.0 * X.length);
        }
        
        public double[] getTheta() {
            return theta != null ? Arrays.copyOf(theta, theta.length) : null;
        }
    }
    
    // ========================================
    // LOGISTIC REGRESSION
    // ========================================
    
    /**
     * Logistic Regression using Gradient Descent
     */
    public static class LogisticRegression {
        private double[] theta;
        private double learningRate = 0.01;
        private int maxIterations = 1000;
        
        public LogisticRegression(double learningRate, int maxIterations) {
            this.learningRate = learningRate;
            this.maxIterations = maxIterations;
        }
        
        /**
         * Fit the model using gradient descent
         */
        public void fit(double[][] X, double[] y) {
            // Add bias term
            double[][] XWithBias = addBiasColumn(X);
            int features = XWithBias[0].length;
            
            // Initialize theta
            theta = new double[features];
            
            // Gradient descent
            for (int iter = 0; iter < maxIterations; iter++) {
                double[] gradient = computeGradient(XWithBias, y);
                
                // Update theta
                for (int j = 0; j < theta.length; j++) {
                    theta[j] -= learningRate * gradient[j];
                }
            }
        }
        
        /**
         * Predict probability
         */
        public double predictProba(double[] x) {
            double[] xWithBias = new double[x.length + 1];
            xWithBias[0] = 1.0; // bias
            System.arraycopy(x, 0, xWithBias, 1, x.length);
            
            double z = dotProduct(theta, xWithBias);
            return sigmoid(z);
        }
        
        /**
         * Predict class (0 or 1)
         */
        public int predict(double[] x) {
            return predictProba(x) >= 0.5 ? 1 : 0;
        }
        
        /**
         * Compute gradient for logistic regression
         */
        private double[] computeGradient(double[][] X, double[] y) {
            int m = X.length;
            int features = X[0].length;
            double[] gradient = new double[features];
            
            for (int i = 0; i < m; i++) {
                double z = dotProduct(theta, X[i]);
                double prediction = sigmoid(z);
                double error = prediction - y[i];
                
                for (int j = 0; j < features; j++) {
                    gradient[j] += error * X[i][j];
                }
            }
            
            // Average gradient
            for (int j = 0; j < features; j++) {
                gradient[j] /= m;
            }
            
            return gradient;
        }
    }
    
    // ========================================
    // UTILITY METHODS
    // ========================================
    
    /**
     * Add bias column (column of ones) to feature matrix
     */
    private static double[][] addBiasColumn(double[][] X) {
        int rows = X.length;
        int cols = X[0].length;
        
        double[][] result = new double[rows][cols + 1];
        
        for (int i = 0; i < rows; i++) {
            result[i][0] = 1.0; // bias term
            System.arraycopy(X[i], 0, result[i], 1, cols);
        }
        
        return result;
    }
    
    /**
     * Matrix-vector multiplication
     */
    private static double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        
        for (int i = 0; i < rows; i++) {
            result[i] = dotProduct(matrix[i], vector);
        }
        
        return result;
    }
    
    /**
     * Simplified linear system solver (Gaussian elimination)
     * In practice, use LU decomposition or Cholesky for better numerical stability
     */
    private static double[] solveLinearSystem(double[][] A, double[] b) {
        int n = A.length;
        double[][] augmented = new double[n][n + 1];
        
        // Create augmented matrix
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, augmented[i], 0, n);
            augmented[i][n] = b[i];
        }
        
        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }
            
            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;
            
            // Eliminate column
            for (int k = i + 1; k < n; k++) {
                double factor = augmented[k][i] / augmented[i][i];
                for (int j = i; j < n + 1; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
        
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = augmented[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= augmented[i][j] * x[j];
            }
            x[i] /= augmented[i][i];
        }
        
        return x;
    }
    
    /**
     * Print matrix for debugging
     */
    public static void printMatrix(double[][] matrix, String name) {
        System.out.println(name + ":");
        for (double[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println();
    }
    
    /**
     * Print vector for debugging
     */
    public static void printVector(double[] vector, String name) {
        System.out.println(name + ": " + Arrays.toString(vector));
    }
    
    // ========================================
    // DEMONSTRATION AND TESTING
    // ========================================
    
    public static void main(String[] args) {
        System.out.println("🚀 LINEAL - Mathematical Foundations of Machine Learning");
        System.out.println("=========================================================\n");
        
        // Demo 1: Vector Norms
        demonstrateVectorNorms();
        
        // Demo 2: Linear Regression
        demonstrateLinearRegression();
        
        // Demo 3: Logistic Regression
        demonstrateLogisticRegression();
        
        // Demo 4: Optimization
        demonstrateOptimization();
    }
    
    /**
     * Demonstrate different vector norms using your formula
     */
    private static void demonstrateVectorNorms() {
        System.out.println("📏 Vector Norms Demonstration");
        System.out.println("=============================");
        
        double[] vector = {3, -4, 0, 5};
        System.out.println("Vector: " + Arrays.toString(vector));
        
        System.out.printf("L1 Norm (Manhattan): %.3f\n", l1Norm(vector));
        System.out.printf("L2 Norm (Euclidean): %.3f\n", l2Norm(vector));
        System.out.printf("L∞ Norm (Maximum): %.3f\n", lInfinityNorm(vector));
        
        // Demonstrate the general Lp norm formula
        System.out.println("\nGeneral Lp Norms:");
        for (double p = 1.0; p <= 5.0; p += 0.5) {
            System.out.printf("L%.1f Norm: %.3f\n", p, lpNorm(vector, p));
        }
        System.out.println();
    }
    
    /**
     * Demonstrate linear regression
     */
    private static void demonstrateLinearRegression() {
        System.out.println("📈 Linear Regression Demonstration");
        System.out.println("==================================");
        
        // Simple dataset: y = 2x + 1 + noise
        double[][] X = {{1}, {2}, {3}, {4}, {5}};
        double[] y = {3.1, 4.9, 7.2, 9.1, 11.0};
        
        LinearRegression model = new LinearRegression();
        model.fit(X, y);
        
        System.out.println("Training Data:");
        for (int i = 0; i < X.length; i++) {
            System.out.printf("x=%.1f, y=%.1f\n", X[i][0], y[i]);
        }
        
        System.out.println("\nModel Parameters (theta): " + Arrays.toString(model.getTheta()));
        System.out.printf("Cost: %.6f\n", model.cost(X, y));
        
        System.out.println("\nPredictions:");
        for (int i = 0; i < X.length; i++) {
            double prediction = model.predict(X[i]);
            System.out.printf("x=%.1f, actual=%.1f, predicted=%.3f\n", 
                            X[i][0], y[i], prediction);
        }
        System.out.println();
    }
    
    /**
     * Demonstrate logistic regression
     */
    private static void demonstrateLogisticRegression() {
        System.out.println("🎯 Logistic Regression Demonstration");
        System.out.println("====================================");
        
        // Simple binary classification dataset
        double[][] X = {{1, 2}, {2, 3}, {3, 1}, {4, 2}, {5, 4}, {6, 5}};
        double[] y = {0, 0, 0, 1, 1, 1};
        
        LogisticRegression model = new LogisticRegression(0.1, 1000);
        model.fit(X, y);
        
        System.out.println("Training Data:");
        for (int i = 0; i < X.length; i++) {
            System.out.printf("x=[%.1f, %.1f], y=%.0f\n", X[i][0], X[i][1], y[i]);
        }
        
        System.out.println("\nPredictions:");
        for (int i = 0; i < X.length; i++) {
            double proba = model.predictProba(X[i]);
            int prediction = model.predict(X[i]);
            System.out.printf("x=[%.1f, %.1f], actual=%.0f, proba=%.3f, predicted=%d\n", 
                            X[i][0], X[i][1], y[i], proba, prediction);
        }
        System.out.println();
    }
    
    /**
     * Demonstrate gradient descent optimization
     */
    private static void demonstrateOptimization() {
        System.out.println("⛰️ Gradient Descent Demonstration");
        System.out.println("=================================");
        
        // Minimize f(x,y) = (x-3)² + (y+1)²
        // Minimum should be at (3, -1)
        Function<double[], double[]> gradient = x -> new double[]{
            2 * (x[0] - 3),  // ∂f/∂x = 2(x-3)
            2 * (x[1] + 1)   // ∂f/∂y = 2(y+1)
        };
        
        double[] x0 = {0, 0}; // Starting point
        double[] result = gradientDescent(gradient, x0, 0.1, 100);
        
        System.out.println("Minimizing f(x,y) = (x-3)² + (y+1)²");
        System.out.println("Expected minimum: (3, -1)");
        System.out.printf("Starting point: (%.1f, %.1f)\n", x0[0], x0[1]);
        System.out.printf("Found minimum: (%.3f, %.3f)\n", result[0], result[1]);
        
        // Calculate function value at minimum
        double fValue = Math.pow(result[0] - 3, 2) + Math.pow(result[1] + 1, 2);
        System.out.printf("Function value at minimum: %.6f\n", fValue);
        System.out.println();
    }
}
