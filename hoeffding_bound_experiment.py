"""
    By Euxhen Hasanaj
    Dec. 20, 2018

    Illustrating the difference between a single bin and multiple bins w.r.t
    the Hoeffding Inequality.Flip 1000 coins 10 times each. c_1 is the first
    coin flipped, c_rand is a random coin, and c_min is the  coin  with  the
    minimum frequency of heads. Let v_1, v_rand, v_min be  the  fraction  of
    heads for each coin respectively.

    a) Find mu for the three coins.
    b) Repeat the experiment 1,000 times and  plots the  histograms  of  the
       distributions of v_1, v_rand, v_min.
    c) Plot estimates for P[|v - mu| > eps] as a function of  eps,  together
       with the Hoeffding bound 2*exp(-2*eps^2*N).
"""
import random
import numpy as np
from matplotlib import pyplot as plt

def experiment(coins=1000, trials_per_coin=10):
    """
        Single experiment. Returns v_1, v_rand, v_min.
    """
    c_rand = random.randint(0, coins-1) # Choose random coin
    v_1, v_rand, v_min = 0, 0, 0
    head_min = trials_per_coin + 1 # Will hold min coin value

    heads = 0
    # First coin
    for c in range(trials_per_coin):
        heads += random.randint(0, 1)
    v_1 = heads / trials_per_coin

    # Remaining coins
    for coin in range(coins - 1):
        heads = 0
        for _ in range(trials_per_coin):
            heads += random.randint(0, 1)

        if coin == c_rand:  # Random coin encountered
            v_rand = heads / trials_per_coin

        if heads < head_min: # Minimum coin encountered
            v_min = heads / trials_per_coin
            head_min = heads

    return v_1, v_rand, v_min

def fact(n):
    """ Factorial """
    prod = 1
    for x in range(1, n+1):
        prod *= x
    return prod

def nchoosek(n, k):
    """ Combination """
    return fact(n) / (fact(k) * fact(n - k))

def get_mu(coins=1000, trials_per_coin=10):
    """
        a) Find mu for the three coins

        For v_1, and v_rand, mu is equal to 0.5.
        For v_min we have to do some calculations after which mu = 0.038.

        Basically, for v_min we need to find the expectation of the probability
        distribution:
            P(x_min = k)
            = 1 - P(x > k on all trials) - P(x_min < k)
            = 1 - [1 - P(x = k)^trials] - P(x_min < k)
            = P(x = k)^trials - P(x_min < k).

        After some mathematical nonsense, this boils down to the following:
        Assume probs is a list of probabilities for each x = 0, 1, ..., 10.
        Calculate another array 'sums', where for  each  i = 0, 1, ..., 10,
        sums[i] holds the sum of all probs[j] for j = i, i+1, ..., 10,  and
        for each i > 10, sums[i] = 0.

        Then P(x_min=k) = sums[k]^m - sums[k+1]^m.
        From here mu_min = sum_0^10 (x/10 * P(x=k)).
    """

    # Calculate P(x=k for a single coin and 10 tosses)
    # Since a coin has two outputs, we are dealing with a binomial distribution.
    # P(x=k) = nchoosek(10, k) * p^k * (1-p)^(10-k)
    # Since p = 0.5, this simplifies to P(x=k) = nchoosek(10, k) * (0.5)^10
    buff = 0.5**trials_per_coin
    probs = []
    for x in range(0, trials_per_coin+1):
        probs.append(nchoosek(trials_per_coin, x) * buff)
    # sums[i] will hold the sum of probs[j] for j = i, i+1, ..., 10.
    sums = [probs[-1]]
    for x in range(trials_per_coin-1, -1, -1):
        sums.append(probs[x] + sums[-1])
    sums.reverse()
    sums.append(0)

    mu_min = 0
    for x in range(trials_per_coin+1):
        mu_min += x / trials_per_coin * (sums[x]**coins - sums[x+1]**coins)

    return 0.5, 0.5, mu_min


def plot_histograms(v_1s, v_rands, v_mins):
    """
        b) Plot the histograms of the distributions of v_1, v_rand, v_min.
    """
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(v_1s)
    ax1.set_title('Distribution of v_1')
    ax2.hist(v_rands)
    ax2.set_title('Distribution of v_rand')
    ax3.hist(v_mins)
    ax3.set_title('Distribution of v_min')

    plt.show()

def hoeff_bound(epspace, N):
    return 2 * np.exp(-2 * np.multiply(epspace, epspace) * N)

def plot_hoeffding_bound(v_1s, v_rands, v_mins):
    """
        c) Plot estimates for P[|v - mu| > eps] as a function of eps, together
           with the Hoeffding bound 2*exp(-2*eps^2*N).
    """
    mu_1, mu_rand, mu_min = get_mu()
    assert len(v_1s) == len(v_rands) and len(v_1s) == len(v_mins)
    trials = len(v_1s)

    v_1s    = sorted(np.abs(np.subtract(v_1s, mu_1)))
    v_rands = sorted(np.abs(np.subtract(v_rands, mu_rand)))
    v_mins  = sorted(np.abs(np.subtract(v_mins, mu_min)))

    epspace = np.linspace(0, 1, 100)
    hoeffding_bound = hoeff_bound(epspace, trials)

    p_1s = []
    index = 0

    for eps in epspace:
        if index != trials:
            while eps >= v_1s[index]:
                index += 1
                if index == trials:
                    break
        p_1s.append((trials-index) / trials)

    plt.plot(epspace, hoeffding_bound, color='red')
    plt.plot(epspace, p_1s, color='green')
    plt.show()

def main(trials=1000, coins=1000):
    v_1s, v_rands, v_mins = [], [], []
    for trial in range(trials):
        v_1, v_rand, v_min = experiment(coins)
        v_1s.append(v_1)
        v_rands.append(v_rand)
        v_mins.append(v_min)

    plot_hoeffding_bound(v_1s, v_rands, v_mins)
    plot_histograms(v_1s, v_rands, v_mins)

if __name__ == '__main__':
    main()
