
class MLP_Neural_Network:
    def __init__(self, dims):
        """ Neural Network with integrated optimizer (grad descent with momentum).
        Eg. 
        1. nn = MLP_Neural_Network(dims)
        2. nn.forward(x)
        3. nn.backward()
        4. nn.update_weights(gamma=0.9, alpha=10E-5)
        """

        self.W = self.xavierWeights(d)
        self.B = self.initBias(d)

        self.Vw = []
        self.Vb = []
        for i in range(len(self.W)):
            self.vw = np.ones(self.W[i].shape) * 10E-5
            self.vb = np.ones(self.B[i].shape) * 10E-5
            self.Vw.append(self.vw)
            self.Vb.append(self.vb)

    def forward(self, X0):    # def forward(Xn, W, B):
        """ Forward pass through network.
        Args:
            X0: (np.array) the input to the network, Shape: [N, d(0)]
        Returns:
            X: (list of np.arrays) post-activation outputs at each layer, Shape: [[N, d(0)]...[N, d(l)]]
            S: (list of np.arrays) pre-activation outputs at layer, Shape: [[N, d(0)]...[N, d(l)]]
        """

        # Number of links in chain
        L = len(self.W)
        S = [None]*L
        X = [None]*(L+1)

        # Add data point at X[0]
        X[0] = X0

        # Forward prop chain
        for l in range(L-1):

            S[l]  = self.compute(X[l], self.W[l], self.B[l])
            X[l+1] = self.ReLU(np.copy(S[l]))

        # Final Activation Layer is Softmax
        S[L-1] = self.compute(X[L-1], self.W[-1], self.B[-1])
        X[L] = self.softmax(np.copy(S[L-1]))

        self.X = X
        self.S = S

        return X[-1] # return last ouput

    def backward(self, Y):
        """ Backpropagation function. Calculates and returns Delta's for each layer. (Backward Messages)
        Args:
            Y: (np.array) labels, Shape: [N, d(L)]
            S: (list of np.array) pre-activations at each layer, Shape: [[N, d(0)] ...[N, d(L)]]
            W: (list of np.array) Weights at each layer, Shape: [[d(0), d(1)] ...[d(L-1), d(L)]]
        Returns:
            D: (list of np.array) backward messages, Shape: [[N, d(0)] ...[N, d(L)]]
        """

        # W = self.W
        # S = self.S

        L = len(W)
        D = [None]*L

        n = len(Y) # number of training points

        # first delta_L
        D[L-1] = gradCE(Y, np.copy(self.S[L-1]))

        # backwards chain to calculate Deltas
        for l in range(L-2, -1, -1):
            grad_relu_sl = self.gradRelU(np.copy(self.S[l]))
            D[l] = (np.multiply( grad_relu_sl.transpose() , np.matmul(self.W[l+1], self.D[l+1].transpose()))).transpose()

        self.D = D

    """ Helper Functions """
    def compute(self, x_prev, w, b):
        """ compute output for single layer
        Args:
            x_prev: (np.array) output from previous layer, Shape: [N, d(l-1)]
            w: (np.array) weights connecting layer l-1 to l, Shape: [d(l-1), d(l)]
            b: (np.array) bias at layer l, Shape: [d(l), 1]
        Returns:
            s: (np.array) pre-activation output at layer l
        """

        s = np.matmul(x_prev, w) + b.transpose()

        return s

    def ReLU(self, inp):
        """ Element wise ReLU
        Args:
            inp: (np.array) pre-activation input array for layer l, Shape: [N, d(l-1)]
        Returns:
            out: (np.array) post-activation input array, Shape: [N, d(l-1)]
        """

        out = np.maximum(inp, 0)

        return out

    def softmax(self, x):
        """ Softmax function. Computed on each row in 2D array
        Args:
            x: (np.array) pre-activation input, Shape: [N, d(l-1)]
        Returns:
            out: (np.array) post-activation input, Shape: [N, d(l-1)]
        """

        maxVal = np.amax(x, axis=1)
        x = x - np.expand_dims(maxVal, 1)
        out = np.exp(np.copy(x))
        den = np.expand_dims((np.exp(x)).sum(axis=1), 1)
        out = out/ den

        return out

    def gradCE(self, Y, sL):
        """ Computes derivative of error with respect to pre-activation on last layer. (dE/dS^(L)). Eq.(4) in notes
        Args:
            Y: (np.array) labels, Shape: [N, d(L)]
            sL: (np.array) pre-activations at layer L, Shape: [N, d(L)]

        Returns:
            delta_L: (np.array) backward message at layer L, Shape: [N, d(L)]
        """

        xL = self.softmax(sL)
        delta_L =  xL - Y

        return delta_L

    def gradRelU(self, sl):
        """ Gradient of ReLU(s) with respect to sl. Gradient for ReLU is either identity or zero. ( dsigma(sl)/dsl ) Eq.(21)
        Args:
            sl: (np.array) pre-activation input, Shape: [N, d(l)]
        Returns:
            dsimga: (np.array) derivatives of sigma(sl) with respect to sl, Shape: [N, d(l)]
        """

        logical = sl > 0
        dsigma = logical*1
        return dsimga

    def xavierWeights(self, d):
        """ Xavier initialization, for ReLU
        Args:
            d: (python list) list of layer dimnesions, Shape: [d(0), .. d(l) ..., d(L)]
        Returns:
            W: (list of np.array) initialized at each layer, Shape: [[d(0), d(1)] ... [d(L-1), d(L)] ]
        """
        W = []

        for i in range(1,len(d)):
            n = d[i]
            n_prev = d[i-1]
            w = np.random.randn(d[i-1], d[i]) * np.sqrt(2/ n+n_prev)
            W.append(w)

        return W

    def initBias(d):
        """ Initializes biases to zero, given nueral network archtiecture d.
        Args:
            d: (python list) list of layer dimnesions, Shape: [d(0), .. d(l) ..., d(L)]
        Returns:
            B: (list of np.array) initialized at each layer, Shape: [[d(0), 1] ... [d(L), 1] ]
        """
        B = []


        for i in range(1,len(d)):
            b = np.zeros((d[i], 1))
            B.append(b)


        return B

    """ Optimization"""

    def compute_grads(self):
        """ Computes gradients with respect to all weights and biases. dE/dw = matmul(X.T^(l), D^(l))
        """

        W, B, X, D = self.W, self.B, self.X, self.D
        DW, DB = [], []


        for l in range(len(W)):
            # len(X) = len(W)+1     ==>    X[l] is X(l-1)
            # len(D) = len(W)       ==>    D[l] is D(l)

            N, K = D[l].shape
            dw = (1/N) * np.matmul(X[l].transpose(), D[l])
            db = (1/N) * np.expand_dims(D[l].sum(axis=0), 1)

            DW.append(dw)
            DB.append(db)

        self.DW, self.DB = DW, DB

    def grad_descent_step(self, gamma = 0.9, alpha = 10E-5):
        """ Gradient Descent with momentum. Updates weights and biases
        Args:
            gamma: (float) Hyper-parameter
            alpha: (float) Hyper-parameter
        """

        W, B, DW, DB, Vw, Vb = self.W, self.B, self.DW, self.DB, self.Vw, self.Vb


        for i in range(len(W)):

            Vw[i] = gamma* Vw[i] + alpha * DW[i]
            Vb[i] = gamma* Vb[i] + alpha * DB[i]


            W[i] = W[i] - Vw[i]
            B[i] = B[i] - Vb[i]

        # Update weights and velocities
        self.W, self.B, self.Vw, self.Vb = W, B, Vw, Vb

    def update_weights(self, gamma=0.9, alpha=10E-5):
        """ Compute Gradients and update weights
        """
        compute_grads()
        grad_descent_step(gamma = gamma, alpha = alpha)



class GradDescent:
    def __init__(self):
        pass

    def compute(self, W, B, X, D):
        """ Computes gradients with respect to all weights and biases. dE/dw = matmul(X.T^(l), D^(l))
        Args:
            W: (list of np.array) Weights at each layer, Shape: [ [d(0), d(1)] ... [d(L-1), d(L)] ]
            B: (list of np.array) Bias at each layer, Shape: [ [d(0), 1] ... [d(L), 1] ]
            X: (list of np.array) post-activation outputs, Shape: [ [d(0), 1] ... [d(L), 1] ]
            D: (list of np.array) Backward Messages, Shape: [ [d(0), 1] ... [d(L), 1] ]
        Returns:
            DW: (list of np.array) Derivatves with respect to Weights, Shape: [ [d(0), d(1)] ... [d(L-1), d(L)] ]
            DB: (list of np.array) Derivatves with respect to Biases, Shape: [ [d(0), 1] ... [d(L), 1] ]
        """


        DW = []
        DB = []

        for l in range(len(W)):
            # len(X) = len(W)+1     ==>    X[l] is X(l-1)
            # len(D) = len(W)       ==>    D[l] is D(l)

            N, K = D[l].shape
            dw = (1/N) * np.matmul(X[l].transpose(), D[l])
            db = (1/N) * np.expand_dims(D[l].sum(axis=0), 1)

            DW.append(dw)
            DB.append(db)

        self.DW, self.DB = DW, DB

    def compute_and_step(self, gamma = 0.9, alpha = 10E-5):
        """ Gradient Descent with momentum. Computes gradients and updates weights and biases
        Args:
            gamma: (float) Hyper-parameter
            alpha: (float) Hyper-parameter
        """

        self.compute()
        W, B, DW, DB, Vw, Vb = self.W, self.B, self.DW, self.DB, self.Vw, self.Vb


        for i in range(len(W)):

            Vw[i] = gamma* Vw[i] + alpha * DW[i]
            Vb[i] = gamma* Vb[i] + alpha * DB[i]


            W[i] = W[i] - Vw[i]
            B[i] = B[i] - Vb[i]

        # Update weights and velocities
        self.W, self.B, self.Vw, self.Vb = W, B, Vw, Vb


""" Gradient Descent with momentum.
Args:
    W: (list of np.array) Weights at each layer, Shape: [[d(0), d(1)] ... [d(L-1), d(L)] ]
    B: (list of np.array) Biases at each layer, Shape: [ [d(0), 1] ... [d(L), 1] ]
    DW: (list of np.array) Derivatves with respect to Weights, Shape: [ [d(0), d(1)] ... [d(L-1), d(L)] ]
    DB: (list of np.array) Derivatves with respect to Biases, Shape: [ [d(0), 1] ... [d(L), 1] ]
    Vw: (list of np.array) Weight Velocity at each layer, Shape: [[d(0), d(1)] ... [d(L-1), d(L)] ]
    Vb: (list of np.array) Bias Velocity at each layer, Shape: [ [d(0), 1] ... [d(L), 1] ]
    gamma: (float) Hyper-parameter
    alpha: (float) Hyper-parameter
Returns:
    W: (list of np.array) Updated Weights, Shape: [[d(0), d(1)] ... [d(L-1), d(L)] ]
    B: (list of np.array) Updated Biases, Shape: [ [d(0), 1] ... [d(L), 1] ]
    Vw: (list of np.array) Updated Weight Velocity, Shape: [[d(0), d(1)] ... [d(L-1), d(L)] ]
    Vb: (list of np.array) Updated Bias Velocity, Shape: [ [d(0), 1] ... [d(L), 1] ]
"""


""" Computes gradients with respect to all weights and biases. dE/dw = matmul(X.T^(l), D^(l))
Args:
    W: (list of np.array) Weights at each layer, Shape: [ [d(0), d(1)] ... [d(L-1), d(L)] ]
    B: (list of np.array) Bias at each layer, Shape: [ [d(0), 1] ... [d(L), 1] ]
    X: (list of np.array) post-activation outputs, Shape: [ [d(0), 1] ... [d(L), 1] ]
    D: (list of np.array) Backward Messages, Shape: [ [d(0), 1] ... [d(L), 1] ]
Returns:
    DW: (list of np.array) Derivatves with respect to Weights, Shape: [ [d(0), d(1)] ... [d(L-1), d(L)] ]
    DB: (list of np.array) Derivatves with respect to Biases, Shape: [ [d(0), 1] ... [d(L), 1] ]
"""

"""
Carry out a forward pass. Notation:
X: The complete Neural Network List of X's which are N x d(l) output matrices at a given layer l
W: The complete Neural Network List of W's which are d(l-1) x d(l) matrices at a given layer l
B: The complete Neural Network List of B's which are 1 x d(l) matrices at a given layer l"""
