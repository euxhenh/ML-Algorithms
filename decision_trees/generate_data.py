import random

# Built in functions
def conjunction(x):
    d = len(x)
    for feature in range(0, d, 3):
        if x[feature] == 0:
            return -1
    else:
        return 1

def generate_data(n, d, f):
    """
    n - number of data points to generate
    d - dimension of input space to generate data from
    f - function to compute y
    """
    calX = []
    calY = []

    for i in range(n):
        x = []
        for feature in range(d):
            x.append(random.randint(0, 1))
        calX.append(x)
        calY.append(f(x)) 
    return calX, calY

def print_data_to_file(name):
    fl = open(name, "w")
    calX, calY = generate_data(100, 6, conjunction)
    for i, x in enumerate(calX):
        for feature in x:
            fl.write(str(feature) + " ")
        fl.write(str(calY[i]) + "\n")
    fl.close()

if __name__ == "__main__":
    print_data_to_file("data")
