from utils import *


class Constants:
    epochs = 100
    iterate = 2500
    batch_size = 100
    learning_rate = 0.00005
    n_critic = 5
    c = 0.01
    beta = 0.9  # RMSProp Parameter
    latent = 100  # latent space dimension
    discriminator_hidden_sizes = [784, 512, 256, 1]
    generator_hidden_sizes = [latent, 256, 512, 1024, 784]
    eps = 1e-8
    im_dim = (28, 28)


class MyLRelu:
    def forward(self, x):
        alpha = 0.2
        self.gradient = np.ones_like(x)
        self.gradient[x < 0] = alpha
        return np.maximum(x, x * alpha)

    def backward(self, dout):
        return self.gradient * dout


class MyTanh:
    def forward(self, x):
        self.gradient = 1.0 - np.tanh(x) ** 2
        return np.tanh(x)

    def backward(self, dout):
        return self.gradient * dout


def weights_init(inp_dim, out_dim):
    """
    Function for weights initialization
    :param inp_dim: Input dimension of the weight matrix
    :param out_dim: Output dimension of the weight matrix
    """
    b = np.sqrt(6)/np.sqrt(inp_dim+out_dim)
    return np.random.uniform(-b, b, (inp_dim, out_dim))


def biases_init(dim):
    """
    Function for biases initialization
    :param dim: Dimension of the biases vector
    """
    return np.zeros(dim).astype(np.float32)


class LinearMap:
    def __init__(self, inp_dim, out_dim, beta=0.9, lr=0.00005, eps=1e-8, c=0.01):
        self.beta = beta
        self.lr = lr
        self.eps = eps
        self.c = c

        self.W = weights_init(inp_dim, out_dim)
        self.b = biases_init(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.sdW = np.zeros_like(self.W)
        self.sdb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.batch_size = x.shape[0]
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout / self.batch_size
        self.db = np.mean(dout, axis=0)
        return dout @ self.W.T

    def step(self):
        self.sdW = self.beta * self.sdW + (1 - self.beta) * (self.dW)**2
        self.W -= self.lr * self.dW / (np.sqrt(self.sdW) + self.eps)
        self.sdb = self.beta * self.sdb + (1 - self.beta) * (self.db)**2
        self.b -= self.lr * self.db / (np.sqrt(self.sdb) + self.eps)

    def clip(self):
        self.W = np.clip(self.W, -self.c, self.c)
        self.b = np.clip(self.b, -self.c, self.c)


class Discriminator(object):
    def __init__(self):
        """
        Initialize your weights, biases and anything you need here.
        """
        self.shapes = Constants.discriminator_hidden_sizes
        self.c = Constants.c
        self.lr = Constants.learning_rate
        self.beta = Constants.beta
        self.eps = Constants.eps

        self.linear0 = LinearMap(self.shapes[0], self.shapes[1],
                                 beta=self.beta, lr=self.lr,
                                 eps=self.eps, c=self.c)
        self.lrelu0 = MyLRelu()
        self.linear1 = LinearMap(self.shapes[1], self.shapes[2],
                                 beta=self.beta, lr=self.lr,
                                 eps=self.eps, c=self.c)
        self.lrelu1 = MyLRelu()
        self.linear2 = LinearMap(self.shapes[2], self.shapes[3],
                                 beta=self.beta, lr=self.lr,
                                 eps=self.eps, c=self.c)

    def __call__(self, x):
        return self.forward(x)

    def __getattr__(self, attr):
        if attr == 'W0':
            return self.linear0.W
        elif attr == 'b0':
            return self.linear0.b
        elif attr == 'W1':
            return self.linear1.W
        elif attr == 'b1':
            return self.linear1.b
        elif attr == 'W2':
            return self.linear2.W
        elif attr == 'b2':
            return self.linear2.b

    def forward(self, x):
        """
        Forward pass for discriminator
        :param x: Input for the forward pass with shape (batch_size, 28, 28, 1)
        :return Output of the discriminator with shape (batch_size, 1)
        """
        x = x.reshape(x.shape[0], -1)
        self.batch_size = x.shape[0]
        x1 = self.linear0.forward(x)
        lx1 = self.lrelu0.forward(x1)
        x2 = self.linear1.forward(lx1)
        lx2 = self.lrelu1.forward(x2)
        x3 = self.linear2.forward(lx2)
        return x3

    def backward(self, logit, inp, image_type):
        """
        Backward pass for discriminator
        :param logit: logit value with shape (batch_size, 1)
        :param inp: input image with shape (batch_size, 28, 28, 1).
        :image_type: Integer value -1 or 1 depending on whether
                                    it is a real or fake image.
        """
        dd = np.ones((self.batch_size, 1)) * image_type

        dx3 = self.linear2.backward(dd)
        ldx2 = self.lrelu1.backward(dx3)
        dx2 = self.linear1.backward(ldx2)
        ldx1 = self.lrelu0.backward(dx2)
        _ = self.linear0.backward(ldx1)

        self.linear0.step()
        self.linear1.step()
        self.linear2.step()

    def weight_clipping(self):
        """
        Weight clipping for discriminator's weights and biases.
        """
        self.linear0.clip()
        self.linear1.clip()
        self.linear2.clip()


class Generator(object):

    def __init__(self):
        self.shapes = Constants.generator_hidden_sizes
        self.lr = Constants.learning_rate
        self.beta = Constants.beta
        self.eps = Constants.eps
        self.w, self.h = Constants.im_dim
        self.latent = self.shapes[0]

        self.linear0 = LinearMap(self.shapes[0], self.shapes[1],
                                 beta=self.beta, lr=self.lr, eps=self.eps)
        self.lrelu0 = MyLRelu()
        self.linear1 = LinearMap(self.shapes[1], self.shapes[2],
                                 beta=self.beta, lr=self.lr, eps=self.eps)
        self.lrelu1 = MyLRelu()
        self.linear2 = LinearMap(self.shapes[2], self.shapes[3],
                                 beta=self.beta, lr=self.lr, eps=self.eps)
        self.lrelu2 = MyLRelu()
        self.linear3 = LinearMap(self.shapes[3], self.shapes[4],
                                 beta=self.beta, lr=self.lr, eps=self.eps)
        self.tanh3 = MyTanh()

    def __call__(self, z):
        return self.forward(z)

    def __getattr__(self, attr):
        if attr == 'W0':
            return self.linear0.W
        elif attr == 'b0':
            return self.linear0.b
        elif attr == 'W1':
            return self.linear1.W
        elif attr == 'b1':
            return self.linear1.b
        elif attr == 'W2':
            return self.linear2.W
        elif attr == 'b2':
            return self.linear2.b
        elif attr == 'W3':
            return self.linear3.W
        elif attr == 'b3':
            return self.linear3.b

    def forward(self, z):
        """
        Forward pass for generator
        :param z: Input for the forward pass with shape (batch_size, 100)
        :returns Linear output after the hidden layers with shape
                                                    (batch_size, 784)
                 Output of the generator with shape (batch_size, 28, 28)
        """
        self.batch_size = z.shape[0]
        z0 = self.linear0.forward(z)
        lz0 = self.lrelu0.forward(z0)
        z1 = self.linear1.forward(lz0)
        lz1 = self.lrelu1.forward(z1)
        z2 = self.linear2.forward(lz1)
        lz2 = self.lrelu2.forward(z2)
        z3 = self.linear3.forward(lz2)
        tz3 = self.tanh3.forward(z3)
        return tz3.reshape(z.shape[0], self.w, self.h, 1)

    def backward(self, fake_logit, fake_input, discriminator):
        """
        Backward pass for generator
        :param fake_logit: Logit output from the discriminator
                            with shape (batch_size, 1)
        :param fake_input: Fake images generated by the discriminator
                            with shape (batch_size, 28, 28)
        :param discriminator: discriminator object
        """

        dd = -np.ones((self.batch_size, 1))
        dx3 = discriminator.linear2.backward(dd)
        ldx2 = discriminator.lrelu1.backward(dx3)
        dx2 = discriminator.linear1.backward(ldx2)
        ldx1 = discriminator.lrelu0.backward(dx2)
        x0 = discriminator.linear0.backward(ldx1)

        tdz3 = self.tanh3.backward(x0)
        dz3 = self.linear3.backward(tdz3)
        ldz2 = self.lrelu2.backward(dz3)
        dz2 = self.linear2.backward(ldz2)
        ldz1 = self.lrelu1.backward(dz2)
        dz1 = self.linear1.backward(ldz1)
        ldz0 = self.lrelu0.backward(dz1)
        _ = self.linear0.backward(ldz0)

        self.linear0.step()
        self.linear1.step()
        self.linear2.step()
        self.linear3.step()


class WGAN(object):
    def __init__(self, digits):
        self.d = Discriminator()
        self.g = Generator()
        self.epochs = Constants.epochs
        self.iterate = Constants.iterate
        self.batch_size = Constants.batch_size
        self.n_critic = Constants.n_critic
        self.x_train, self.y_train, _ = mnist_reader(digits)
        self.w, self.h = Constants.im_dim
        self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                            self.w, self.h, 1)
        self.lossd = []
        self.lossg = []
        self.losses = []

    def linear_interpolation(self):
        """
        Generate linear interpolation between two data points.
        """
        z = np.random.normal(0, 1, (1, w.g.latent))
        imi = w.g(z)
        k = np.random.normal(0, 1, (1, w.g.latent))
        imj = w.g(k)

        diff = z - k

        ims = []
        for i in range(10):
            ims.append(w.g(k + diff*i/10)[0, :, :, 0])

        def plot_images(images, cols=3, cmap='gray'):
            rows = (len(images) + cols - 1) / cols
            for i, image in enumerate(images):
                plt.subplot(rows, cols, i+1)
                plt.imshow(image, cmap=cmap)
                plt.xticks([])
                plt.yticks([])

            plt.show()

        plot_images(ims, cols=10)

    def train(self):
        indices = np.arange(self.x_train.shape[0])
        for iteration in range(self.iterate):
            loss = 0
            for critic in range(self.n_critic):
                # Real
                random_inds = np.random.choice(indices, self.batch_size)
                o = self.d.forward(self.x_train[random_inds])
                self.d.backward(None, None, -1)
                self.d.weight_clipping()
                loss += np.mean(o)

                # Fake
                z = np.random.normal(0, 1, (self.batch_size, self.g.latent))
                o = self.d.forward(self.g(z))
                self.d.backward(None, None, 1)
                self.d.weight_clipping()
                loss -= np.mean(o)

            z = np.random.normal(0, 1, (self.batch_size, self.g.latent))
            _ = self.d.forward(self.g(z))
            self.g.backward(None, None, self.d)

            self.losses.append(loss)
            if iteration % 20 == 0:
                img_tile(self.g(z), path='iterations',
                         epoch=iteration, save=True)
                print("Iteration {0} :: Loss {1:.2f}".format(iteration, loss))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    w = WGAN([1, 2, 3])
    w.train()
