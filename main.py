import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


# def drawCostFunction(featureMatrix, originalAnswers):
#    costsVector = np.empty((100,))
#    for i in range(100):
#        for j in range(100):
#            costsVector[i] = J(np.array([[i,j]]), featureMatrix, originalAnswers)
#
#
#    fig = plt.figure()
#    ax = fig.add_subplot()
#    u = range(100)
#    x,y= np.meshgrid(u,u)

#    plt.contour(x,y,costsVector)
#    plt.show()
def testExample():
    x,y = np.loadtxt("mydata1", delimiter=',', unpack=True)
    m = y.size
    x = x[np.newaxis].T
    x = np.c_[np.ones((m,1)), x]
    n = x.shape[1]
    theta  = np.zeros((n,))
    lr = LinearRegression(theta,x,y)
    alpha = 0.001
    iterations = 20000
    lr.gradientDescent(alpha,iterations)
    print("theta0: " + str(theta[0]))
    print("theta1: " + str(theta[1]))




def main():




    itemsMatrix = np.loadtxt("ex1data1.txt", delimiter=',', unpack=True)
    x = itemsMatrix[:-1, :].T
    # y is always a 1D array in this implementation, that is, a row vector. So we won't transpose it.
    y = itemsMatrix[-1, :]
    m = y.size  # m = number of training examples





    # Insert the usual column of 1's into the "x" matrix
    x = np.c_[np.ones((m, 1), dtype=float), x]  # matlab like solution to column inserting
    n = x.shape[1] #number of features
    theta = np.zeros((n,))




#    storedStds, storedMeans = [], []
#    for sub in range(x.shape[1]):  # sub = subscript = column number = feature number
#        currentFeatures = x[:, sub]
#        mean = currentFeatures.mean()
#        std = currentFeatures.std()
#        storedMeans.append(mean)
#        storedStds.append(std)
#        if std == 0: # avoids division by zero and also avoids applying feature normalization to x0, which we normally don't
#            continue
#        x[:, sub] = (x[:, sub] - mean) / std



    alpha = 0.01
    lr = LinearRegression(theta,x,y)
    lr.gradientDescentTillConvergence(alpha)



    #uncomment here to see the result
#    fig, ax = plt.subplots()
#    ax.set_xlabel('Population of City in 10,000s')
#    ax.set_ylabel('Profit in $10,000s')
#    xWithoutOnes = x[:,1]
#    print(xWithoutOnes)
#    ax.plot(xWithoutOnes,y,'x')
#    hresults = theta[0] + theta[1]*x
#    xmin = xWithoutOnes.min()
#    xmax = xWithoutOnes.max()
#    ax.set_xlim([xmin-0.2,xmax+0.2])
#    ax.plot(x,hresults)
#    plt.show()



    lr.plotCostFunction(type="surface")

    #testValues = np.array([[1,1650,3]], dtype=float)
    #normalize it:
    #for i in range(1,testValues.size):
    #    testValues[0,i] = (testValues[0,i]-storedMeans[i])/storedStds[i]

#    print(testValues)
#    print(testValues.shape)
#    print(lr.hypothesis(testValues[0,:]))

    #lr.drawConvergence() # draws convergence

if __name__ == "__main__":

    main()
