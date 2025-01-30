import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

##############################################################################
# 1) DATA LOADING
##############################################################################
def load_data(file_path):
    """
    Loads data from a CSV file into X (features) and y (labels).
    Adjust this function based on the structure of your dataset.
    
    Example assumptions:
    - The last column is the label (binary: 0 or 1).
    - The other columns are numerical features.
    """
    data = pd.read_csv(file_path)  # or np.loadtxt(file_path), etc.
    
    # Convert to numpy arrays
    data_np = data.to_numpy()
    
    # Assuming last column is the label
    X = data_np[:, :-1]
    y = data_np[:, -1]
    
    # If your dataset uses labels that are not {0,1}, ensure it is converted:
    # Example: if labels are {-1, +1}, map them to {0,1}
    # y = np.where(y == -1, 0, 1)
    
    return X, y

##############################################################################
# 2) STATISTICAL ANALYSIS & VISUALIZATION
##############################################################################
def analyze_data_distribution(X, y, dataset_name="Dataset"):
    """
    Print out some basic distribution stats and create histograms.
    """
    # Class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n{dataset_name} Class Distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    print(f"Total samples in {dataset_name}: {len(y)}")
    
    # Feature distributions (basic histogram for each feature)
    num_features = X.shape[1]
    for i in range(num_features):
        plt.figure()
        plt.hist(X[:, i], bins=30, alpha=0.7, color='blue')
        plt.title(f"{dataset_name} - Feature {i} distribution")
        plt.xlabel(f"Feature {i}")
        plt.ylabel("Frequency")
        plt.show()

##############################################################################
# 3) LOGISTIC REGRESSION IMPLEMENTATION FROM SCRATCH
##############################################################################
class LogisticRegression:
    """
    Logistic Regression from scratch using batch gradient descent.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None  # Weights will be initialized in fit()
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model on the training data.
        - X: shape (N, d)
        - y: shape (N,)
        """
        # Add bias term (intercept) by concatenating a column of 1s
        ones_col = np.ones((X.shape[0], 1))
        X_bias = np.concatenate([ones_col, X], axis=1)  # shape (N, d+1)
        
        # Initialize weights
        num_features = X_bias.shape[1]
        self.W = np.zeros(num_features)
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Compute predictions (probabilities)
            z = X_bias @ self.W  # shape (N,)
            y_pred = self._sigmoid(z)  # shape (N,)
            
            # Gradient of the loss w.r.t W
            gradient = X_bias.T @ (y_pred - y) / X_bias.shape[0]
            
            # Update weights
            self.W -= self.learning_rate * gradient
    
    def predict(self, X):
        """
        Predict binary labels (0 or 1) for the inputs X.
        - X: shape (N, d)
        """
        # Add bias term
        ones_col = np.ones((X.shape[0], 1))
        X_bias = np.concatenate([ones_col, X], axis=1)
        
        # Compute predicted probabilities
        z = X_bias @ self.W
        probs = self._sigmoid(z)
        
        # Convert probabilities to 0-1 predictions
        return (probs >= 0.5).astype(int)

##############################################################################
# 4) ACCURACY EVALUATION FUNCTION
##############################################################################
def accu_eval(y_true, y_pred):
    """
    Evaluate accuracy (0-1).
    """
    return np.mean(y_true == y_pred)

##############################################################################
# 5) K-FOLD CROSS VALIDATION
##############################################################################
class KFoldCV:
    """
    k-Fold Cross Validation from scratch.
    """
    def __init__(self, k=10):
        self.k = k
    
    def split(self, X, y):
        """
        Generate indices to split data into training and test set for each fold.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # shuffle the indices
        
        fold_sizes = np.full(self.k, n_samples // self.k, dtype=int)
        fold_sizes[: n_samples % self.k] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop

##############################################################################
# 6) EXPERIMENTS
##############################################################################
if __name__ == "__main__":
    # =============================
    # Example on how to use the code
    # =============================
    
    # --- Load your datasets ---
    # Replace with your actual file paths
    dataset_1_path = 'data_file_1.csv'
    dataset_2_path = 'data_file_2.csv'
    
    X1, y1 = load_data(dataset_1_path)
    X2, y2 = load_data(dataset_2_path)
    
    # --- Analyze distributions ---
    analyze_data_distribution(X1, y1, dataset_name="Dataset 1")
    analyze_data_distribution(X2, y2, dataset_name="Dataset 2")
    
    # --- Prepare for experiments ---
    kfold = KFoldCV(k=10)
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    max_iter = 1000
    
    # =============================
    # Example: 10-fold CV on Dataset 1
    # =============================
    for lr in learning_rates:
        print(f"\n--- Logistic Regression on Dataset 1 with LR={lr} ---")
        accuracies = []
        start_time = time.time()
        
        for train_idx, test_idx in kfold.split(X1, y1):
            X_train, y_train = X1[train_idx], y1[train_idx]
            X_test, y_test   = X1[test_idx],  y1[test_idx]
            
            # Create and train model
            model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accu_eval(y_test, y_pred)
            accuracies.append(acc)
        
        end_time = time.time()
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        runtime = end_time - start_time
        
        print(f"Average Accuracy = {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Runtime = {runtime:.4f} seconds")
    
    # =============================
    # Example: 10-fold CV on Dataset 2
    # =============================
    for lr in learning_rates:
        print(f"\n--- Logistic Regression on Dataset 2 with LR={lr} ---")
        accuracies = []
        start_time = time.time()
        
        for train_idx, test_idx in kfold.split(X2, y2):
            X_train, y_train = X2[train_idx], y2[train_idx]
            X_test, y_test   = X2[test_idx],  y2[test_idx]
            
            # Create and train model
            model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            acc = accu_eval(y_test, y_pred)
            accuracies.append(acc)
        
        end_time = time.time()
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        runtime = end_time - start_time
        
        print(f"Average Accuracy = {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Runtime = {runtime:.4f} seconds")
    
    # =========================================================================
    #  Additional ideas for improvement:
    #
    #  1) Feature selection:
    #     - You can try to select a subset of features from X, e.g. X[:, [0,2,4]]
    #       or remove features you think are not informative. Then rerun CV.
    #
    #  2) Feature engineering:
    #     - You can add polynomial features, interaction terms, or other transformations
    #       (like log, sqrt, etc.) to see if they improve accuracy.
    #
    #  3) Grid search / hyperparameter tuning:
    #     - Experiment with different values of max_iter or different regularization
    #       if you add it to your logistic regression.
    #
    #  4) Time measurement:
    #     - As done above, measure how long training and CV take for each set of hyperparameters.
    #
    #  5) Plotting performance:
    #     - For each dataset, create a plot of learning rate vs. average accuracy.
    # =========================================================================
