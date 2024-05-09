from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def fitness(X_train, y_train, X_test, y_test, C = 1.0, kernel = 'rbf', gamma = 0.1):
    # Create a SVM classifier
    clf = SVC(C = C, kernel = kernel, gamma = gamma)
    # Train the classifier
    clf.fit(X_train, y_train)
    # Predict the test set
    y_pred = clf.predict(X_test)
    # Calculate the fitness
    return accuracy_score(y_test, y_pred)


#Path: main.py