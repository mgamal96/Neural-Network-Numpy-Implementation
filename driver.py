from Neural_Nework import *
import numpy as np

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


if __name__ == '__main__':

    # Data Pre-Processing

    trainX, validX, testX, trainY, validY, testY = loadData()
    trainY, validY, testY = convertOneHot(trainY, validY, testY)
    n, r, c = testX.shape
    testX = testX.reshape(n, r*c)
    n, r, c = validX.shape
    validX= validX.reshape(n, r*c)
    n, r, c = trainX.shape
    trainX = trainX.reshape(n, r*c)


    # Hyer-parameters

    batch_size = len(trainX)
    num_batches = int(len(trainX)/batch_size)
    gamma = 0.90     # Momentum
    alpha = 10E-3
    epochs = 200

    # Loss & Accuracy Arrays

    trainLoss, valLoss, testLoss = np.zeros((epochs)), np.zeros((epochs)), np.zeros((epochs))
    trainAcc, valAcc, testAcc = np.zeros((epochs)), np.zeros((epochs)), np.zeros((epochs))

    # Neural Network Model
    nn = MLP_Neural_Network([784,100, 10]) # archtiecture

    # Evaluation Metrics
    em = Evaluation_Metrics()


    # Train Loop

    for epoch in range(epochs):
        for iter in range(0, batch_size*num_batches, batch_size):

            x, y = trainX[iter:iter+batch_size], trainY[iter:iter+batch_size]
            pred = nn.forward(x)

            loss = em.Cross_Entropy_Loss(pred, y)
            acc = em.Multi_Class_Classificiton_Accuracy(pred, y)

            nn.backward(y)
            nn.update_weights()

            print(loss,round(2), acc.round(2))
