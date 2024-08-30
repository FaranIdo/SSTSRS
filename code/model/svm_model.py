from sklearn.svm import SVR


class SVMNDVIModel:
    def __init__(self, kernel="rbf"):
        self.model = SVR(kernel=kernel)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
