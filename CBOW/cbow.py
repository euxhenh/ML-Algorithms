import numpy as np
import matplotlib.pylab as plt

seed = 10417617


def weight_init(x_len, y_len):
    b = np.sqrt(6.0/(x_len+y_len))
    return np.random.normal(-b, b, (x_len, y_len))


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CBOW:
    """
    This class is an implementation of the continous bag of words model
    by first principles.
    CBOW model tries to predict the center words from surrounding words.
    """

    def __init__(self, text, window, n, learning_rate=1e-4):
        """
        Initialize self.n, self.text,self.window,self.lr,self.vocab,
                    self.word2index, self.V and self.U
        n = desired size of word vectors
        window = size of window
        vocab = vocabulary of all words in the dataset
        word2index = index for each word in the vocabulary
        input vector matrix of shape (n, len(vocab))
        output vector matrix of shape (len(vocab),n)
        """
        self.n = n
        self.text = text
        self.window = window
        self.lr = learning_rate

        self.vocab = np.sort(np.unique(self.text))

        self.word2index = dict([(w, i) for i, w in enumerate(self.vocab)])

        self.V = weight_init(n, self.vocab.shape[0])
        self.U = weight_init(self.vocab.shape[0], n)

    def one_hot_vector(self, index):
        """
        Function for one hot vector encoding.
        :input: a word index
        :return: one hot encoding
        """
        oh = np.zeros((self.vocab.shape[0]))
        oh[index] = 1
        return oh

    def get_vector_representation(self, one_hot=None):
        """"
        Function for vector representation from one-hot encoding.
        :input: one hot encoding
        :return: word vector
        """
        return np.squeeze(self.V[:, np.argwhere(one_hot == 1)])

    def get_average_context(self, left_context=None, right_context=None):
        """
        Function for average vector generation from surrounding
        context of current word.
        :input: surrounding context (left/right)
        :return: averaged vector representation
        """
        lv = [np.squeeze(self.V[:, self.word2index[x]]) for x in left_context]
        rv = [np.squeeze(self.V[:, self.word2index[x]]) for x in right_context]
        return np.mean(lv+rv, axis=0)

    def get_score(self, avg_vector=None):
        """
        Function for product score given an averaged vector
        in current context of the center word.
        :input: averaged vector
        :return: product score with matrix U.
        """
        return np.squeeze(self.U @ avg_vector.reshape(-1, 1))

    def softmax(self, x):
        """
        Function to return the softmax output of given function.
        """
        mu = np.max(x)
        e = np.exp(x - mu)
        return e / np.sum(e)

    def compute_cross_entropy_error(self, y, y_hat):
        """
        Given one hot encoding and the output of the softmax function,
        this function computes the cross entropy error.
        ---------
        :input y: one_hot encoding of the current center word
        :input y_hat: output of the softmax function
        :return: cross entropy error
        """
        return -y * np.log(y_hat + 1e-5)

    def compute_EH(self, error):
        """
        Function to compute the value of EH, the sum of the
        output vectors of all words in the vocabulary,
        weighted by their prediction error.
        ---------
        :input: error
        :return: value of EH
        """
        return np.sum(self.U * error.reshape(-1, 1), axis=0)

    def update_U(self, error, avg_vector):
        """
        Given the cross entropy error occured in the current sample,
        this function updates the U matrix.
        :return self.U
        """
        self.U -= self.lr * (error.reshape(-1, 1) @ avg_vector.reshape(1, -1))
        return self.U

    def update_V(self, error, left_context, right_context):
        """
        Similarly, this function updates the V matrix.
        :return self.V
        """
        b = len(left_context) + len(right_context)
        eh = self.compute_EH(error)
        for w in left_context:
            index = self.word2index[w]
            self.V[:, index] -= self.lr * eh * b
        for w in right_context:
            index = self.word2index[w]
            self.V[:, index] -= self.lr * eh * b
        return self.V

    def fit(self, epoch):
        """
        Learn the values of V and U vectors with given window size.
        """
        tloss = 0
        for i, word in enumerate(self.text[self.window:-self.window]):
            l_context = self.text[i:i+self.window]
            r_context = self.text[i+self.window+1:i+2*self.window+1]
            avg_context = self.get_average_context(l_context, r_context)
            score = self.get_score(avg_context)
            softscore = self.softmax(score)
            oh = self.one_hot_vector(self.word2index[word])
            loss = self.compute_cross_entropy_error(oh, softscore)
            tloss += np.mean(loss)
            _ = self.update_U(loss, avg_context)
            _ = self.update_V(loss, l_context, r_context)
        tloss /= (self.text.shape[0] - (2 * self.window))
        print(f"Epoch {epoch} :: Loss {tloss}")


if __name__ == "__main__":
    window = 2
    n = 100
    lr = 1e-4
    epochs = 50
    with open("data.txt", "r") as f:
        text = f.read()
    text = np.array(text.split(' '))
    text = text[np.where(text != '')]

    cbow = CBOW(text=text, window=window, n=n, learning_rate=lr)

    for epoch in range(epochs):
        cbow.fit(epoch)

    def word2vec(word):
        index = cbow.word2index[word]
        return (cbow.V[:, index] + cbow.U[index, :]) / 2

    vecs = []
    for word in cbow.vocab:
        vecs.append(word2vec(word))
    vecs = np.array(vecs)

    def sortclosest(word):
        dists = []
        v = word2vec(word)
        for w in vecs:
            dists.append(cos(v, w))
        dists = np.array(dists)
        return np.flip(np.argsort(dists))

    d = cbow.vocab[sortclosest("stock")]
    print(d[:6])
    d = cbow.vocab[sortclosest("mortgage")]
    print(d[:6])
    d = cbow.vocab[sortclosest("dollar")]
    print(d[:6])

    v_fall = word2vec("fall")
    ind_fall = cbow.word2index["fall"]
    v_rise = word2vec("rise")
    ind_rise = cbow.word2index["rise"]
    v_low = word2vec("low")
    ind_low = cbow.word2index["low"]

    add_obj_i = None
    add_obj_val = -1

    mul_obj_i = None
    mul_obj_val = -1

    for i, w in enumerate(vecs):
        if i == ind_fall or i == ind_rise or i == ind_low:
            continue

        lobj = cos(w, v_rise - v_fall + v_low)
        if lobj > add_obj_val:
            add_obj_i = i
            add_obj_val = lobj

        lobj = cos(w, v_rise) * cos(w, v_low) / (cos(w, v_fall) + 1e-8)
        if lobj > mul_obj_val:
            mul_obj_i = i
            mul_obj_val = lobj

    print("ADD", cbow.vocab[add_obj_i])
    print("MUL", cbow.vocab[mul_obj_i])
