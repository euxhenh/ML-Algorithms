# import the necessary packages
import numpy as np
from utils import *

class Activation:

    def __call__(self, inp):
        return self.forward(inp)

    def forward(self, inp):
        pass

    def backward(self, inp):
        pass

class VSigmoid(Activation):

    def forward(self, inp):
        return 1 / (1 + np.exp(-inp))

    def backward(self, inp):
        inp = 1 / (1 + np.exp(-inp))
        return inp * (1 - inp)


def VBCE_loss(x, y):
    """
    Binary Cross Entropy Loss for VAE
    """
    epsilon = 10e-8
    loss = np.sum(-y * np.log(x + epsilon) - (1 - y) * np.log(1 - x + epsilon))
    return loss


class VRelu(Activation):

    def forward(self, inp):
        return inp * (inp > 0)

    def backward(self, inp):
        return 1.0 * (inp > 0)


class VLRelu(Activation):
    def __init__(self, model):
        assert model == "wgan" or model == "VAE"
        self.alpha = 0.2 if model == "wgan" else 0.01

    def forward(self, inp):
        return np.maximum(inp, inp * self.alpha)

    def backward(self, inp):
        dx = np.ones_like(inp)
        dx[inp < 0] = self.alpha
        return dx


class VTanh(Activation):

    def forward(self, inp):
        return np.tanh(inp)

    def backward(self, inp):
        return 1.0 - np.tanh(inp) ** 2



def initialize(shape):
    return np.random.normal(0, np.sqrt(6.0 / (sum(shape))),
                            shape)


class VAE(object):
    # This is VAE model that you should implement
    def __init__(self, hidden_units=128, z_units=20, input_dim=784,
                 batch_size=64):
        """
        initialize all parameters in the model.
        Encoding part:
        1. W_input_hidden, b_input_hidden: convert input to hidden
        2. W_hidden_mu, b_hidden_mu:
        3. W_hidden_logvar, b_hidden_logvar
        Sampling:
        1. random_sample
        Decoding part:
        1. W_z_hidden, b_z_hidden
        2. W_hidden_out, b_hidden_out
        """
        self.hidden_units = hidden_units
        self.z_units = z_units
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.W_input_hidden = initialize((input_dim, hidden_units))
        self.b_input_hidden = np.zeros((hidden_units,))
        self.W_hidden_mu = initialize((hidden_units, z_units))
        self.b_hidden_mu = np.zeros((z_units,))
        self.W_hidden_logvar = initialize((hidden_units, z_units))
        self.b_hidden_logvar = np.zeros((z_units,))

        self.W_z_hidden = initialize((z_units, hidden_units))
        self.b_z_hidden = np.zeros((hidden_units,))
        self.W_hidden_out = initialize((hidden_units, input_dim))
        self.b_hidden_out = np.zeros((input_dim,))

        self.lrelu = VLRelu("VAE")
        self.relu = VRelu()
        self.sig = VSigmoid()
        self.tanh = VTanh()

        self.lr = 3e-3

    def encode(self, x):
        """
        input: x is input image with size (batch, indim)
        return: hidden_mu, hidden_logvar, both sizes should be (batch, z_units)
        """
        xk = self.lrelu.forward(x @ self.W_input_hidden + self.b_input_hidden)
        mu = xk @ self.W_hidden_mu + self.b_hidden_mu
        logvar = xk @ self.W_hidden_logvar + self.b_hidden_logvar
        return mu, logvar

    def decode(self, z):
        """
        input: z is the result from sampling with size (batch, z_unit)
        return: out, the generated images from decoder with size (batch, indim)
        """
        zk = self.relu.forward(z @ self.W_z_hidden + self.b_z_hidden)
        return zk @ self.W_hidden_out + self.b_hidden_out

    def forward(self, x, unittest=False):
        """
        combining encode, sampling and decode.
        input: x is input image with size (batch, indim)
        return: out, the generated images from decoder with size (batch, indim)
        """
        # DO NOT modify or delete this line, it is for testing
        # just ignore it, and write you implementation below
        if (unittest):
            np.random.seed(1433125)
        self.mu, self.logvar = self.encode(x)
        self.eps = np.random.normal(0, 1, self.logvar.shape)
        self.zk = self.mu + np.multiply(self.eps, np.sqrt(np.exp(self.logvar)))
        return self.sig.forward(self.decode(self.zk))

    def loss(self, x, out):
        """
        Given the input x (also the ground truth) and out, computing the loss
        (CrossEntropy + KL).
        input: x is the input of the model with size (batch, indim)
               out is the predicted output of the model with size (batch, indim)
        """
        self.ce = VBCE_loss(out, x) / x.shape[0]
        self.kl = np.sum(-1/2 * (1 + self.logvar - self.mu**2 -
                                 np.exp(self.logvar))) / x.shape[0]
        self.l = self.ce + self.kl
        return self.l

    def backward(self, x, pred):
        """
        Given the input x (also the ground truth) and out,
        computing the gradient of parameters.
        input: x is the input of the model with size (batch, indim)
               pred is the predicted output of the model with size (batch, indim)
        return: grad_list = [dW_input_hidden, db_input_hidden,
                            dW_hidden_mu, db_hidden_mu,
                            dW_hidden_logvar, db_hidden_logvar,
                            dW_z_hidden, db_z_hidden,
                            dW_hidden_out, db_hidden_out]
        """
        b = x.shape[0]

        #dlx = -x/pred + (1-x)/(1-pred)
        dlx = (pred - x) / (pred * (1 - pred))
        dec_h = self.relu.forward(self.zk @ self.W_z_hidden + self.b_z_hidden)
        ddec_h = self.relu.backward(dec_h)
        dec_o = self.sig.forward(dec_h @ self.W_hidden_out + self.b_hidden_out)

        # Calc gradient for out
        db_hidden_out = dlx * dec_o * (1 - dec_o)
        dW_hidden_out = dec_h.T @ db_hidden_out

        # Calc gradient for z hidden
        db_z_hidden = (db_hidden_out @ self.W_hidden_out.T) * ddec_h
        dW_z_hidden = self.zk.T @ db_z_hidden
        dlzk = db_z_hidden @ self.W_z_hidden.T

        # Calc repara
        zk = self.mu + np.exp(1/2 * self.logvar) * self.eps
        dlmu = dlzk + self.mu
        dllogvar = dlzk * (1/2 * np.exp(1/2 * self.logvar) * self.eps) +\
            1/2 * (np.exp(self.logvar) - 1)

        enc_h = self.lrelu.forward(
            x @ self.W_input_hidden + self.b_input_hidden)
        denc_h = self.lrelu.backward(enc_h)

        # Calc gradient for mu
        db_hidden_mu = dlmu.copy()
        dW_hidden_mu = enc_h.T @ dlmu

        # Calc gradient for var
        db_hidden_logvar = dllogvar.copy()
        dW_hidden_logvar = enc_h.T @ dllogvar

        db_input_hidden = (dlmu @ self.W_hidden_mu.T) * denc_h + \
            (dllogvar @ self.W_hidden_logvar.T) * denc_h
        dW_input_hidden = x.T @ ((dlmu @ self.W_hidden_mu.T) * denc_h) + \
            x.T @ ((dllogvar @ self.W_hidden_logvar.T) * denc_h)

        db_hidden_out = np.sum(db_hidden_out, axis=0)
        db_z_hidden = np.sum(db_z_hidden, axis=0)
        db_hidden_mu = np.sum(db_hidden_mu, axis=0)
        db_hidden_logvar = np.sum(db_hidden_logvar, axis=0)
        db_input_hidden = np.sum(db_input_hidden, axis=0)
        # Update
        self.b_hidden_out = self.b_hidden_out - self.lr * db_hidden_out/b
        self.W_hidden_out = self.W_hidden_out - self.lr * dW_hidden_out/b
        self.b_z_hidden = self.b_z_hidden - self.lr * db_z_hidden/b
        self.W_z_hidden = self.W_z_hidden - self.lr * dW_z_hidden/b

        self.b_hidden_mu = self.b_hidden_mu - self.lr * db_hidden_mu/b
        self.W_hidden_mu = self.W_hidden_mu - self.lr * dW_hidden_mu/b

        self.b_hidden_logvar = self.b_hidden_logvar - self.lr * db_hidden_logvar/b
        self.W_hidden_logvar = self.W_hidden_logvar - self.lr * dW_hidden_logvar/b
        self.b_input_hidden = self.b_input_hidden - self.lr * db_input_hidden/b
        self.W_input_hidden = self.W_input_hidden - self.lr * dW_input_hidden/b

        return [
            dW_input_hidden, db_input_hidden,
            dW_hidden_mu, db_hidden_mu,
            dW_hidden_logvar, db_hidden_logvar,
            dW_z_hidden, db_z_hidden,
            dW_hidden_out, db_hidden_out
        ]

    def set_params(self, parameter_list):
        """
        TO set parameters with parameter_list
        input: parameter_list = [W_input_hidden, b_input_hidden,
                                W_hidden_mu, b_hidden_mu,
                                W_hidden_logvar, b_hidden_logvar,
                                W_z_hidden, b_z_hidden,
                                W_hidden_out, b_hidden_out]
        """
        self.W_input_hidden = parameter_list[0]
        self.b_input_hidden = parameter_list[1]
        self.W_hidden_mu = parameter_list[2]
        self.b_hidden_mu = parameter_list[3]
        self.W_hidden_logvar = parameter_list[4]
        self.b_hidden_logvar = parameter_list[5]
        self.W_z_hidden = parameter_list[6]
        self.b_z_hidden = parameter_list[7]
        self.W_hidden_out = parameter_list[8]
        self.b_hidden_out = parameter_list[9]


if __name__ == "__main__":
    # x_train is of shape (5000 * 784)
    # We've done necessary preprocessing for you so just feed it into your model.
    x_train = np.load('data.npy')

    epochs = 50
    batch_size = 32

    vae = VAE()

    ces = []
    kls = []
    # Train
    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)

        for b in range(0, indices.shape[0], batch_size):
            out = vae.forward(x_train[indices[b:b+batch_size]])
            _ = vae.backward(x_train[indices[b:b+batch_size]], out)

        out_all = vae.forward(x_train)
        l = vae.loss(x_train, out_all)
        ces.append(vae.ce)
        kls.append(vae.kl)

        print("Epoch {0} :: Reconstruction loss {1} :: Regularization loss {2}".format(epoch+1, ces[-1], kls[-1]))

    from matplotlib import pyplot as plt

    # Generate
    ims = []
    for i in range(100):
        j = np.random.randint(0, x_train.shape[0]-1)
        mu, logvar = vae.encode(x_train[j])
        eps = np.random.normal(0, 1, vae.z_units)
        z = mu + np.multiply(eps, np.sqrt(np.exp(logvar)))
        im = vae.decode(z).reshape(28, 28)
        ims.append(im)
    ims = np.array(ims)
    img_tile(ims, path="images", epoch=300, save=True)

    # Interpolate
    def interpolate(im1, im2, steps=10):
        mu1, logvar1 = vae.encode(im1)
        mu2, logvar2 = vae.encode(im2)

        ims = []

        dmu, dlogvar = mu2 - mu1, logvar2 - logvar1
        for delta in range(steps):
            mu = mu1 + delta/steps * dmu
            logvar = logvar1 + delta/steps * dlogvar
            eps = np.random.normal(0, 1, vae.z_units)

            z = mu + np.multiply(eps, np.sqrt(np.exp(logvar)))
            x = vae.decode(z)

            ims.append(x.reshape(28, 28))
        return np.array(ims)

    ims = interpolate(x_train[17], x_train[23])
    img_save(ims, path="images", epoch=22)

    # Losses
    plt.plot(list(range(1, len(ces)+1)), ces)
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss")
    plt.savefig("images/ce.png")
    plt.show()

    plt.plot(list(range(1, len(kls)+1)), kls)
    plt.xlabel("Epochs")
    plt.ylabel("Regularization Loss")
    plt.title("Regularization Loss")
    plt.savefig("images/kl.png")
    plt.show()
