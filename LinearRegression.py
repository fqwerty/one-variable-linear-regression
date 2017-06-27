from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import ticker
import numpy as np


class LinearRegression(object):
    def __init__(self, theta, x, y):
        self.theta = theta
        self.x = x
        self.y = y
        self.n = theta.size  # number of features
        self.m = y.size  # number of training examples
        self.theta0History = []
        self.theta1History = []

    def hypothesis(self, featuresVector, currentTheta = None):
        """
        calculates the predicted answer. Both arguments have to be a vector
        :param featuresVector: a vector with feature values
        :return: returns predicted answer
        """
        if currentTheta is not None:
            return np.dot(currentTheta, featuresVector)
        else:
            return np.dot(self.theta, featuresVector)

    def J(self,_theta=None):
        if _theta is None:
            theta = self.theta
        else:
            theta = _theta


        """
        calculates the cost function J
        :param localTheta: this will be entered to hypothesis method as an argument
        :return: returns the cost as ndarray
        """
        m = self.x.shape[0]
        sumOfSerie = 0
        for i in range(m):
            sumOfSerie += ((self.hypothesis(self.x[i, :],theta) - self.y[i]) ** 2)

        return (1 / (2 * m)) * sumOfSerie

    def drawCostFunction(self):
        fig = plt.figure()
        axes = fig.add_subplot(1,1,1, projection='3d')
        theta0 = np.arange(-10,15,5)
        theta1 = np.arange(-1,5,1)
        theta0, theta1 = np.meshgrid(theta0,theta1)
        costsList = []
        for j in range(theta0.shape[1]):
            for i in range(theta0.shape[0]):
                currentTheta = np.array([theta0[i,j], theta1[i,j]])



            m = self.x.shape[0]
            sumOfSerie = 0
            for i in range(m):
                sumOfSerie += ((self.hypothesis(self.x[i, :],currentTheta) - self.y[i]) ** 2)

            currentCost =  (1 / (2 * m)) * sumOfSerie
            costsList.append(currentCost)

        axes.plot_surface(theta0,theta1,  costsList)
        plt.show()




    def gradientDescentTillConvergence(self, alpha):
        numberOfThetas = self.n
        previousCost = 0
        sumOfSerie = 0
        while True:
            currentTheta = self.theta.copy()
            for sub in range(numberOfThetas):
                for _super in range(self.m):
                    sumOfSerie += (self.hypothesis(self.x[_super, :], currentTheta = currentTheta) - self.y[_super]) * self.x[_super, sub]
                self.theta[sub] -= (alpha / self.m) * sumOfSerie
                sumOfSerie = 0
            self.theta0History.append(self.theta[0])
            self.theta1History.append(self.theta[1])
            currentCost = self.J()
            #print("Current cost: " + str(currentCost))
            difference = previousCost - currentCost
            #print("Difference: " + str(previousCost - currentCost))
            if abs(difference) < 0.003:
                break
            previousCost = currentCost

    def gradientDescent(self, alpha):
        numberOfThetas = self.n
        sumOfSerie = 0
        currentTheta = self.theta.copy()
        for sub in range(numberOfThetas):
            for _super in range(self.m):
                sumOfSerie += (self.hypothesis(self.x[_super, :], currentTheta = currentTheta) - self.y[_super]) * self.x[_super, sub]
            self.theta[sub] -= (alpha / self.m) * sumOfSerie
            sumOfSerie = 0

    def drawConvergence(self, iterations = 1500):
        costs = np.empty((iterations,))
        alpha = 0.01
        for i in range(iterations):
            self.gradientDescent(alpha)
            currentCost =  self.J()
            costs[i] = currentCost

        fig, axes = plt.subplots()
        axes.plot(np.arange(iterations), costs)
        axes.set_xlabel("iterations")
        axes.set_ylabel("cost")
        axes.grid(True)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        axes.yaxis.set_major_formatter(formatter)


        plt.show()


    def plotCostFunction(self, type="surface"):
        #code to show convergence in surface plot:
#        theta0History, theta1History = np.meshgrid(self.theta0History,self.theta1History)
#        print(theta0History.shape)
#        costHistory = np.empty(theta0History.shape)
#        for j in range(theta0History.shape[1]):
#            for i in range(theta0History.shape[0]):
#                currentTheta0 = theta0History[i,j]
#                currentTheta1 = theta1History[i,j]
#                currentTheta = np.array([currentTheta0,currentTheta1])
#                costHistory[i,j]= self.J(currentTheta)

        theta0 = np.arange(-10,10.5,.5)
        theta1 = np.arange(4,-1.1,-.1)
        theta0, theta1 = np.meshgrid(theta0,theta1)
        z = np.empty(theta0.shape)
        for j in range(theta0.shape[1]):
            for i in range(theta0.shape[0]):
                currentTheta0 = theta0[i,j]
                currentTheta1 = theta1[i,j]
                currentTheta = np.array([currentTheta0,currentTheta1])
                z[i,j]= self.J(currentTheta)

        fig = plt.figure()
        if type == "surface":
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(theta0, theta1, z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)

        elif type =="contour":
            ax = fig.add_subplot(111)
            CS = ax.contour(theta0, theta1, z)
            plt.clabel(CS, inline=1, fontsize=10)
        plt.show()










