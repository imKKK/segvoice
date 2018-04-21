import matplotlib.pyplot as plt
import numpy as np
from seg import generate_mix, segment


def plot():

    A = generate_mix()
    B = segment('train_A.mdl', 'mix.wav','seg')
    plt.figure(figsize=(10, 3))

    for i in range(len(B)):
        idx = len(B)-i-1
        plt.barh(2, B[idx][1], color='g')
        plt.barh(2, B[idx][0], color='w')

    for i in range(len(A)):
        idx = len(A)-i-1
        plt.barh(1, A[idx][1], color='r' if A[idx][2] == 'A' else 'g')

    plt.savefig("demo.jpg") 


if __name__ == '__main__':
    plot()
