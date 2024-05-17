import numpy as np


def generate_synthetic_data(n_samples, n_features, shift_intensity=0, sigma_noise=1):
    cov_matrix = np.eye(n_features)
    X_train = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov_matrix, size=n_samples)
    beta = np.random.normal(0, 1, size=(n_features, 1))
    noise_train = np.random.normal(0, sigma_noise, size=(n_samples, 1))
    y_train = X_train @ beta + noise_train
    
    mean_shift = np.full(n_features, shift_intensity)
    X_test = np.random.multivariate_normal(mean=mean_shift, cov=cov_matrix, size=n_samples)
    noise_test = np.random.normal(0, sigma_noise, size=(n_samples, 1))
    y_test = X_test @ beta + noise_test
    
    return X_train, y_train, X_test, y_test

def data_to_text(X, y):
    texts = []
    for features, target in zip(X, y):
        text = f"Given the features: {' '.join([f'x_{i+1}={feature}' for i, feature in enumerate(features)])}, the output value is {target[0]:.2f}."
        texts.append(text)
    return texts