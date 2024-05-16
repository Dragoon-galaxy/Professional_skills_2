from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Print the correct and incorrect predictions
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print(f"Correct prediction: Predicted {iris.target_names[y_pred[i]]}, Actual {iris.target_names[y_test[i]]}")
    else:
        print(f"Incorrect prediction: Predicted {iris.target_names[y_pred[i]]}, Actual {iris.target_names[y_test[i]]}")
