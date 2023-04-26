##Drishya Uniyal
##MT21119

'''In the modified code that I provided, the PCA matrix (i.e., the matrix of eigenvectors) 
is calculated using the power iteration method. The basic idea behind this method is to iteratively 
multiply the data matrix by itself and normalize the resulting vectors until convergence.
Specifically, the power iteration method starts with a random initial vector, multiplies it by 
the data matrix to obtain a new vector, normalizes the new vector, and repeats this process until convergence.

Once the power iteration method has converged, the resulting matrix of eigenvectors is guaranteed to be an 
orthonormal basis for the space spanned by the original data. The eigenvectors are also sorted in descending 
order of eigenvalue, so the first k eigenvectors correspond to the k largest eigenvalues 
(i.e., the eigenvectors that explain the most variance in the data).

To be precise, in the code above, we use the power iteration method to calculate k eigenvectors 
of the covariance matrix. We start with a random matrix of shape (784, k) and multiply it by the 
covariance matrix 100 times, and then normalize the resulting matrix using the QR decomposition to 
ensure that the columns are orthonormal. We then extract the diagonal of the matrix obtained by multiplying 
the eigenvectors by the covariance matrix to obtain the corresponding eigenvalues. Finally, we sort the 
eigenvectors and eigenvalues in descending order of eigenvalue and select the top k eigenvectors that 
explain the most variance in the data.'''


'''We calculate the mean of the data by summing up all the rows of X and dividing by the number of samples. 
This gives us a mean vector of shape (784,). We then center the data by subtracting this mean vector from every 
sample in X.

Note that we still use StandardScaler to standardize the centered data after calculating the covariance matrix. 
This is because the power iteration method used to calculate the eigenvectors doesn't guarantee that they will 
be orthonormal, so we need to standardize the data to ensure that the eigenvectors form an orthonormal basis.'''
import numpy as np
from sklearn.datasets import fetch_openml
import numpy as np
import tensorflow.keras as keras
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def part():
    (X_train, y_train), (X_test, y_test) =  keras.datasets.mnist.load_data()
        # summarize loaded dataset
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
    idx = np.argsort(y_train)
    x_train_sorted = X_train[idx]
    y_train_sorted = X_train[idx]

    for i in range(0,2):
        x_train_ones1 = X_train[y_train == i]
        for j in range(0,5):
            pyplot.subplot(330 + 1 + j)
            pyplot.imshow(x_train_ones1[j], cmap=pyplot.get_cmap('gray'))
        pyplot.show()
        
mnist = fetch_openml("mnist_784")
X = mnist.data / 255.0  # scale the features to [0, 1]
y = mnist.target

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X[(y == '0') | (y == '1')]
y = y[(y == '0') | (y == '1')]

scaler = StandardScaler()
X = scaler.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = np.abs(X_train)
# X_test = np.abs(X_test)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute the covariance matrix
        covariance_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the first n eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data using the mean from the fit method
        X_centered = X - self.mean

        # Project the data onto the principal components
        X_pca = np.dot(X_centered, self.components).real
        return X_pca


def FDA(X, y, n_components=2):
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute the mean vectors of each class
    class_mean_vectors = []
    for c in np.unique(y):
        class_mean_vectors.append(np.mean(X_scaled[y == c], axis=0))

    # Compute the within-class scatter matrix
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for c, mv in zip(np.unique(y), class_mean_vectors):
        class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
        for row in X_scaled[y == c]:
            row, mv = row.reshape(X_scaled.shape[1], 1), mv.reshape(X_scaled.shape[1], 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat

    # Compute the between-class scatter matrix
    overall_mean = np.mean(X_scaled, axis=0)
    S_B = np.zeros((X_scaled.shape[1], X_scaled.shape[1]))
    for i, mean_vec in enumerate(class_mean_vectors):
        n = X_scaled[y == i+1,:].shape[0]
        mean_vec = mean_vec.reshape(X_scaled.shape[1], 1)
        overall_mean = overall_mean.reshape(X_scaled.shape[1], 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    # Compute the eigenvectors and eigenvalues of (S_W)^(-1) S_B
    try:
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    except np.linalg.LinAlgError:
        # If S_W is a singular matrix, add a small positive constant to the diagonal
        S_W += np.eye(X_scaled.shape[1]) * 1e-12
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Sort the eigenvectors by decreasing eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the eigenvectors corresponding to the largest eigenvalues
    W = sorted_eigenvectors[:, :n_components]

    # Project the data onto the FDA subspace
    X_fda = X_scaled.dot(W)

    return X_fda

def part2():
    # Standardize the data
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

    # Calculate the mean of the data
    mean = np.sum(X, axis=0) / X.shape[0]

    # Center the data by subtracting the mean
    X_centered = X - mean

    # Standardize the data
    scaler = StandardScaler()
    X_centered = scaler.fit_transform(X_centered)

    # Calculate the covariance matrix of the centered data using the formula Cov(X) = (X - mean)^T (X - mean) / (n - 1)
    covariance = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    # Calculate the eigenvectors and eigenvalues of the covariance matrix using the power iteration method
    k = 100
    eigenvectors = np.random.rand(X_centered.shape[1], k)
    for i in range(100):
        eigenvectors = np.dot(covariance, eigenvectors)
        eigenvectors, _ = np.linalg.qr(eigenvectors)
    eigenvalues = np.diag(np.dot(np.dot(eigenvectors.T, covariance), eigenvectors))

    # Sort the eigenvectors and eigenvalues in descending order of eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Project the data onto 2, 3, 5, 8, 10, and 15 dimensions
    dimensions = [2, 3, 5, 8, 10, 15]
    X_pca = {}
    for k in dimensions:
        # Select the top k eigenvectors that explain the most variance
        top_k_eigenvectors = eigenvectors[:, :k]

        # Transform the data into the new feature space
        X_pca[k] = np.dot(X_centered, top_k_eigenvectors)

        # Print the shape of the transformed data
        print(f"Transformed shape (k = {k}):", X_pca[k].shape)
        

def part3():
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

    # Calculate the covariance matrix of the centered data using the formula Cov(X) = (X - mean)^T (X - mean) / (n - 1)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    covariance = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

    # Calculate the eigenvectors and eigenvalues of the covariance matrix using the power iteration method
    k_values = [2, 3, 5, 8, 10, 15]
    reconstruction_errors = []
    for k in k_values:
        eigenvectors = np.random.rand(X_centered.shape[1], k)
        for i in range(100):
            eigenvectors = np.dot(covariance, eigenvectors)
            eigenvectors, _ = np.linalg.qr(eigenvectors)
        eigenvalues = np.diag(np.dot(np.dot(eigenvectors.T, covariance), eigenvectors))

        # Sort the eigenvectors and eigenvalues in descending order of eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]

        # Select the top k eigenvectors that explain the most variance
        top_k_eigenvectors = eigenvectors[:, :k]

        # Transform the data into the new feature space
        X_pca = np.dot(X_centered, top_k_eigenvectors)
        X_reconstructed = np.dot(X_pca, top_k_eigenvectors.T) + mean
        reconstruction_error = np.mean(np.sum((X - X_reconstructed)**2, axis=1))
        reconstruction_errors.append(reconstruction_error)

            # Convert class labels to numeric values
#         le = LabelEncoder()
#         y = le.fit_transform(y)

        # Plot the first two principal components of the transformed data
        plt.figure()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='jet')
        plt.colorbar()
        plt.title(f"PCA with k={k}")
        plt.show()

    # Plot the reconstruction error vs. k
    plt.figure()
    plt.plot(k_values, reconstruction_errors, 'o-')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show()

    
def part4():
#     mnist = fetch_openml('mnist_784', version=1)

#     # Extract samples and labels for the 0 and 1 classes
#     X = mnist.data.astype(float)
#     y = mnist.target.astype(int)
#     X = X[(y == 0) | (y == 1)]
#     y = y[(y == 0) | (y == 1)]

#     # Split the dataset into training and testing sets
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_components_list = [2,3,5,8,10,15]
    for n_components in n_components_list:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # lda = LinearDiscriminantAnalysis()
        # lda.fit(X_train_pca, y_train)

        # Apply PCA to the test data and use the trained classifier to make predictions
        X_test_pca = pca.transform(X_test)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_pca, y_train)
        y_pred = lda.predict(X_test_pca)

        # Compute the accuracy of the classifier
        accuracy = np.mean(y_pred == y_test) * 100
        print(f"Classification accuracy with PCA + LDA: {accuracy:.2f}%")
        # Visualize the data in the PCA space

        import matplotlib.pyplot as plt
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('MNIST PCA')
        plt.show()
    
    
def part5():
    mnist = fetch_openml("mnist_784")
    X = mnist.data / 255.0  # scale the features to [0, 1]
    y = mnist.target

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = X[(y == '0') | (y == '1')]
    y = y[(y == '0') | (y == '1')]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train = np.abs(X_train)
    # X_test = np.abs(X_test)

    n_components_list = [2]
    for n_components in n_components_list:
        # Perform PCA
#         pca = PCA(n_components=n_components)
#         pca.fit(X_train)
#         X_train_pca = pca.transform(X_train)
#         X_test_pca = pca.transform(X_test)
        
        X_train_fda = FDA(X_train, y_train, n_components=n_components)
        X_test_fda = FDA(X_test, y_test, n_components=n_components)

        # Plot the first two principal components
        plt.scatter(X_train_fda[:, 0], X_train_fda[:, 1], c=y_train.astype(int), s=8, alpha=0.5)
        plt.title("MNIST FDA")
        plt.colorbar()
        plt.show()

        # Apply LDA to reduce the dimensionality further and classify the data
        lda = LinearDiscriminantAnalysis()
        
        lda.fit(X_train_fda, y_train)
        y_pred1 = lda.predict(X_test_fda)

        accuracy = np.mean(y_pred1 == y_test) * 100
        print(f"Classification accuracy FDA + LDA : {accuracy:.2f}%")
    
def part6():
    mnist = fetch_openml("mnist_784")
    X = mnist.data / 255.0  # scale the features to [0, 1]
    y = mnist.target

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train = np.abs(X_train)
    # X_test = np.abs(X_test)

    n_components_list = [15]
    for n_components in n_components_list:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # lda = LinearDiscriminantAnalysis()
        # lda.fit(X_train_pca, y_train)

        # Apply PCA to the test data and use the trained classifier to make predictions
        X_test_pca = pca.transform(X_test)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_pca, y_train)
        y_pred = lda.predict(X_test_pca)

        X_train_fda = FDA(X_train_pca, y_train, n_components=n_components)
        X_test_fda = FDA(X_test_pca, y_test, n_components=n_components)

    #     # Plot the first two principal components
    #     plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train.astype(int), s=8, alpha=0.5)
    #     plt.title(f"MNIST PCA with {n_components} components")
    #     plt.colorbar()
    #     plt.show()

    #     plt.scatter(X_train_fda[:, 0], X_train_fda[:, 1], c=y_train.astype(int), s=8, alpha=0.5)
    #     plt.title("MNIST FDA")
    #     plt.colorbar()
    #     plt.show()

        # Apply LDA to reduce the dimensionality further and classify the data
    #     lda = LinearDiscriminantAnalysis()
    #     lda.fit(X_train_pfda, y_train)
    #     y_pred = lda.predict(X_test_pca)

        lda.fit(X_train_fda, y_train)
        y_pred1 = lda.predict(X_test_fda)

        # Calculate the classification accuracy
    #     accuracy = np.mean(y_pred == y_test) * 100
    #     print(f"Classification accuracy with {n_components} PCA components: {accuracy:.2f}%")
        accuracy = np.mean(y_pred1 == y_test) * 100
        print(f"Classification accuracy with PCA + FDA + LDA: {accuracy:.2f}%")

        
part()
part2()
part3()
part4()
part5()
part6()




