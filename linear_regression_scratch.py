import numpy as np

class LinearRegression:
    #here we are intitalizing the learing rate and no of iterations
    def __init__(self, learning_rate=0.01, no_of_iterations=1000):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape  # m = samples, n = features
        self.w = np.zeros(self.n) # this will make every colomn zero 
        self.b = 0 # bias is also zero
        self.X = X # this is storing the x data 
        self.Y = Y # this is y data storage

        for i in range(self.no_of_iterations):
            self.update_weights() #here we are updating the updatawrights method after every iteration

    def update_weights(self):
        Y_prediction = self.predict(self.X) #this is predicting the value of x 
        
        # Gradients
        dw = - (2 * (self.X.T.dot(self.Y - Y_prediction))) / self.m #this is that actual value - predicted value and do the dot product of them and divide it by 
        db = - (2 * np.sum(self.Y - Y_prediction)) / self.m

        # Parameter update
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b
