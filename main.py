import numpy as np
import math
import matplotlib.pyplot as plt

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def xavierWeights(d):
    """
    Xavier initialization, for ReLu
    """
    W = []

    for i in range(1,len(d)):
        n = d[i]
        n_prev = d[i-1]
        w = np.random.randn(d[i-1], d[i]) * np.sqrt(2/ n+n_prev)
        W.append(w)

    return W

def initBias(d):
    """
    Initializes biases to zero, given nueral network archtiecture d.
    """
    B = []

    # option to do xavier Initializing
    for i in range(1,len(d)):
        # n = d[i-1]
        # w = np.random.normal(0, 1/n, d[i])
        # b = np.random.randn(d[i],1) * np.sqrt(2/ n)
        b = np.zeros((d[i], 1))
        B.append(b)


    return B

def gradDescent(W, DW, B, DB, Vw, Vb, gamma = 0.9, alpha = 10E-5):
    """
    Gradient Descent Updating. Expects all dimensions to match
    """
    for i in range(len(W)):


        Vw[i] = gamma* Vw[i] + alpha * DW[i]
        Vb[i] = gamma* Vb[i] + alpha * DB[i]


        W[i] = W[i] - Vw[i]
        B[i] = B[i] - Vb[i]

    return W, B, Vw, Vb

# Helper functions


def ReLU(inp):
    """
    Element wise ReLU for input array [NxK]
    """

    inp = np.maximum(inp, 0)

    return inp

def softmax(inp):
    """
    Takes NxK Array and performs softmax on each row
    """

    maxVal = np.amax(inp, axis=1)
    inp = inp - np.expand_dims(maxVal, 1)
    out = np.exp(np.copy(inp))
    den = np.expand_dims((np.exp(inp)).sum(axis=1), 1)
    out = out/ den

    return out

def compute(x_prev, W, b):
    """
    Expected dimensions
    W: [d^(l-1) x d^(l)]
    x_prev: [N x d^(l-1)]
    b: [d^(l) x 1]

    Output dimensions
    s: [d^(l) x 1]
    """

    s = np.matmul(x_prev, W) + b.transpose()

    return s

def averageCE(p, y):
    """
    Takes prediction and target matrices, of same size [NxK], and computes average loss.
    """

    N, K = p.shape

    # prevent log(0)
    p = p + 1e-10
    avgCE = -(1/N) * np.sum(np.sum(np.multiply(y, np.log(p)), axis=1))

    return avgCE

def gradCE(Y, sL):
    """
    Computes derivative of error with respect to pre-activation on last layer. (dE/dS^(L)).
    Expected dimensions:
    Y  - [NxK]
    sL - [NxK]

    Output dimension:
    delta_L - [NxK]
    """

    xL = softmax(sL)
    delta_L =  xL - Y

    return delta_L

def forward(Xn, W, B):
    """
    Carry out a forward pass. Notation:
    X: The complete Neural Network List of X's which are N x d(l) output matrices at a given layer l
    W: The complete Neural Network List of W's which are d(l-1) x d(l) matrices at a given layer l
    B: The complete Neural Network List of B's which are 1 x d(l) matrices at a given layer l
    """
    # number of links in chain
    L = len(W)
    S = [None]*L
    X = [None]*(L+1)

    # Add data point at X[0]
    x = Xn
    X[0] = x

    # forward prop chain
    for l in range(L-1):

        S[l]  = compute(x, W[l], B[l])
        X[l+1] = ReLU(np.copy(S[l]))

    # Final Activation Layer is Softmax
    S[L-1] = compute(X[L-1], W[-1], B[-1])
    X[L] = softmax(np.copy(S[L-1]))


    return X, S

def backProp(Y, S, W):
    """
    Calculates and returns Delta's for each layer. (Backward Messages)
    D(l): the backward message at a given layer l
    """
    L = len(W)
    D = [None]*L

    n = len(Y) # number of training points

    # first delta_L
    D[L-1] = gradCE(Y, np.copy(S[L-1]))

    # backwards chain to calculate Deltas
    for l in range(L-2, -1, -1):
        grad_relu_sl = gradRelU(np.copy(S[l]))
        D[l] = (np.multiply( grad_relu_sl.transpose() , np.matmul(W[l+1], D[l+1].transpose()))).transpose()

    return D

def gradRelU(sl):

    logical = sl > 0

    return logical*1

def gradWB(W, B, X, D):

    # Weights & Bias gradients
    DW = []
    DB = []


    for l in range(len(W)):
        # len(X) = len(W)+1     ===>    X[l] is X(l-1)
        # len(D) = len(W)       ===>    D[l] is D(l)

        N, K = D[l].shape
        dw = (1/N) * np.matmul(X[l].transpose(), D[l])
        db = (1/N) * np.expand_dims(D[l].sum(axis=0), 1)

        DW.append(dw)
        DB.append(db)


    return DW, DB

def accuracy(X, Y):

    pred = np.argmax(X, axis=1)
    target = np.argmax(Y, axis=1)

    corr = pred==target
    acc = corr.sum()/len(pred)

    return acc



if __name__ == '__main__':

    # Pre-processing Data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)
    n, r, c = testData.shape
    testData = testData.reshape(n, r*c)
    n, r, c = validData.shape
    validData = validData.reshape(n, r*c)
    n, r, c = trainData.shape
    trainData = trainData.reshape(n, r*c)


    bs = len(trainData)
    nb = int(len(trainData)/bs)

    hidden = 100

    # Initializing Weights
    d = [784,hidden, 10]  # layer specification
    W = xavierWeights(d)
    B = initBias(d)



    # Initializing momentum parameters
    gamma = 0.90
    alpha = 10E-3
    epochs = 200

    Vw = []
    Vb = []
    for i in range(len(W)):
        vw = np.ones(W[i].shape) * 10E-5
        vb = np.ones(B[i].shape) * 10E-5
        Vw.append(vw)
        Vb.append(vb)


    trainLoss, valLoss, testLoss = np.zeros((epochs)), np.zeros((epochs)), np.zeros((epochs))
    trainAcc, valAcc, testAcc = np.zeros((epochs)), np.zeros((epochs)), np.zeros((epochs))


    # Training
    for e in range(epochs):

        trainPred, _ = forward(np.copy(trainData), W, B)
        trainLoss[e] = averageCE(np.copy(trainPred[-1]), np.copy(newtrain))
        trainAcc[e] = accuracy(np.copy(trainPred[-1]), np.copy(newtrain))

        valPred, _ = forward(np.copy(validData), W, B)
        valLoss[e] = averageCE(np.copy(valPred[-1]), np.copy(newvalid))
        valAcc[e] = accuracy(np.copy(valPred[-1]), np.copy(newvalid))

        testPred, _ = forward(np.copy(testData), W, B)
        testLoss[e] = averageCE(np.copy(testPred[-1]), np.copy(newtest))
        testAcc[e] = accuracy(np.copy(testPred[-1]), np.copy(newtest))


        print('epoch: {} predSize: {} Train Loss: {} Train Acc: {} Val Acc: {}'.format(e, trainPred[-1].shape, trainLoss[e], trainAcc[e], valAcc[e]))


        for i in range(0, len(trainData), bs):

            # Data Vectors
            # trainData_shuffled, trainTarget_shuffled = shuffle(trainData, newtrain)
            # X_Train, Y_Train = trainData_shuffled[0:bs], trainTarget_shuffled[0:bs]

            X_Train, Y_Train = trainData[i:i+bs], newtrain[i:i+bs]

            # Forward pass - X, S are lists of activations and pre-activations at each layer respectively
            X, S = forward(np.copy(X_Train), W, B)

            acc = accuracy(np.copy(X[-1]), np.copy(Y_Train))
            loss = averageCE(np.copy(X[-1]), np.copy(Y_Train))

            # print('loss: {} acc: {}'.format(loss, acc ))

            # Backward pass
            D = backProp(Y_Train, S, W)
            DW, DB = gradWB(W, B, X, D)

            # Update Weights
            W, B, Vw, Vb = gradDescent(W, DW, B, DB, Vw, Vb, gamma, alpha)


    filename1 = 'loss_Ep{}_Bs{}_Hid{}_e3.png'.format(epochs, bs, hidden)
    filename2 = 'acc_Ep{}_Bs{}_Hid{}_e3.png'.format(epochs, bs, hidden)

    # early stopping point

    bestEpoch = np.argmax(valAcc)


    print('---------------------------------------------------------------------')
    print('Final Metrics: ')
    print('Train Loss: {:.4f} Validation Loss: {:.4f} Test Loss: {:.4f}'.format(trainLoss[-1], valLoss[-1], testLoss[-1]))
    print('Train Accuracy: {:.4f} Validation Accuracy: {:.4f} Test Accuracy: {:.4f}'.format(trainAcc[-1], valAcc[-1], testAcc[-1]))
    print('Early Stopping Point Epoch: ', bestEpoch)
    print('Train Accuracy {:.4f} Validation Accuracy: {:.4f}  Test Accuracy {:.4f}'.format(trainAcc[bestEpoch], valAcc[bestEpoch], testAcc[bestEpoch]))
    print('Train Loss {:.4f} Validation Loss: {:.4f} Test Loss {:.4f}'.format(trainLoss[bestEpoch], valLoss[bestEpoch], testLoss[bestEpoch]))


    plt.plot(trainLoss, 'r', label='Train')
    plt.plot(valLoss, 'g', label='Val')
    plt.plot(testLoss, 'b', label='Test')
    plt.legend(loc = 'upper right')
    plt.title('Loss')
    plt.savefig(filename1)

    plt.clf()

    plt.plot(trainAcc, 'r', label='Train')
    plt.plot(valAcc, 'g', label='Val')
    plt.plot(testAcc, 'b', label='Test')
    plt.legend(loc= 'lower right')
    plt.title('Accuracy')
    plt.savefig(filename2)
