import numpy as np
import scipy.optimize as optimize

#Neural network
class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize =2
        self.hiddenLayerSize =3
        self.outputLayerSize = 1

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def getParams(self):
        params = np.concatenate((self.W1.ravel(),self.W2.ravel()))
        return params
                                
    def setParams(self,params):
        W1_start = 0
        W1_end =self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize,self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end],(self.hiddenLayerSize,self.outputLayerSize))
                                
                                
    def computeGradients(self,X,y):
        djdw1,djdw2 = self.costFunctionPrime(X,y)
        return np.concatenate((djdw1.ravel(),djdw2.ravel()))
                                
    def forwardPropagation(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
        
    
    def sigmoid (self,z):
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self,X,y):
        self.yHat = self.forwardPropagation(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self,X,y):
        self.yHat =self.forwardPropagation(X)
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        djdw2 = np.dot(self.a2.T,delta3)
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        djdw1 = np.dot(X.T,delta2)
        return djdw1,djdw2

#Trainer class
class Trainer(object):
    def __init__(self,N):
        self.N = N
        
    def callbackF(self,params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X,self.y))
        
    def costFunctionWrapper(self,params,X,y):
        self.N.setParams(params)
        cost = self.N.costFunction(X,y)
        grad = self.N.computeGradients(X,y)
        return cost,grad
    
    def train(self,X,y):
        self.X =X
        self.y = y
        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter':200,'disp':True}
        _res = optimize.minimize(self.costFunctionWrapper,params0,jac=True,method = 'BFGS',args=(X,y),options= options,callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

#using example 
x =  np.array(([3,5],[5,1],[10,2]),dtype = float)
y= np.array(([75],[82],[93]),dtype = float)
NN = NeuralNetwork()
T = Trainer(NN)
T.train(x,y)

#plot training graph
plot(T.J)
grid(1)
xlabel('Iterations')
ylabel('Cost (J)')

#some code
NN.costFunctionPrime(x,y)
NN.forwardPropagation(x)

#2D graph
hoursSleep = linspace(0,10,100)
hoursStudy = linspace(0,5,100)
hoursSleepNorm = hoursSleep/10
hoursStudyNorm = hoursStudy/5
a,b = meshgrid(hoursSleepNorm,hoursStudyNorm)
allInputs = np.zeros((a.size,2))
allInputs[:,0] = a.ravel()
allInputs[:,1] = b.ravel()
allOutputs = NN.forwardPropagation(allInputs)
yy = np.dot(hoursStudy.reshape(100,1),np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1),np.ones((1,100))).T
CS = contour(xx,yy,100*allOutputs.reshape(100,100))
clabel(CS,inline=1,fontsize=10)
xlabel('Hours sleep')
ylabel('Hours study')

#3D graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection= '3d')
surf = ax.plot_surface(xx,yy,100*allOutputs.reshape(100,100),cmap=cm.jet)
ax.set_xlabel('Hours sleep')
ax.set_ylabel('Hours study')
ax.set_zlabel('Test score')