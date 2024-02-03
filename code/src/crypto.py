import math
import numpy as np
from scipy.linalg import pascal, invpascal
from scipy.integrate import odeint
from .tools import imread, mat_join_blocks, mat_split_blocks


def hyperchaotic(x, t, a, b, c, d, e, f, g, h):
    return np.array(
        [
            -a * x[0] + a * x[4] - b * x[4] * x[5] * x[6],
            -c * x[1] - d * x[5] + 1 * x[0] * x[5] * x[6],
            -a * x[2] + a * x[4] - g * x[0] * x[1] * x[6],
            -a * x[3] + e * x[0] + 1 * x[0] * x[1] * x[2],
            -a * x[4] + e * x[6] - 1 * x[1] * x[2] * x[3],
            -e * x[5] + e * x[4] + 1 * x[2] * x[3] * x[4],
            -b * x[6] + f * x[1] - h * x[3] * x[4] * x[5],
        ]
    )


def get_hyperchaotic_initials(v):
    xt0 = np.zeros(7, dtype=np.float64)
    xt0[0] = (np.sum(v) + v.size) / ((1 << 23) + v.size)
    for i in range(6):
        xt0[i + 1] = 1e7 * xt0[i] % 1
    return xt0


def get_hyperchaotic_sequence(xt0, hp, size):
    t = np.linspace(0, 100, math.ceil(size / 7))
    x = odeint(hyperchaotic, xt0, t, hp).flatten()
    return x[x.size - size :]


def get_sorted_indexes(vector):
    return np.array([i[0] for i in sorted(enumerate(vector), key=lambda i: i[1])])


def encryption(G, hp, rounds=2):
    inits = []
    E = G.copy()
    for _ in range(rounds):
        V = E.flatten()
        xt0 = get_hyperchaotic_initials(V)
        inits.append(xt0)
        S = get_hyperchaotic_sequence(xt0, hp, V.size)
        SS = get_sorted_indexes(S)
        B = V[SS]
        D = B.reshape(G.shape)
        DS = mat_split_blocks(D, (4, 4))
        ES = np.array([np.mod(np.matmul(i, pascal(4)), 256) for i in DS])
        E = mat_join_blocks(ES)
    return E, inits


def decryption(E, inits, hp, rounds=2):
    G = E.copy()
    for i in range(rounds):
        ES = mat_split_blocks(G, (4, 4))
        DS = np.array([np.mod(np.matmul(i, invpascal(4)), 256) for i in ES])
        D = mat_join_blocks(DS)
        U = D.flatten()
        xt0 = inits[rounds - i - 1]
        S = get_hyperchaotic_sequence(xt0, hp, U.size)
        SS = get_sorted_indexes(S)
        O = np.zeros(E.size)
        O[SS] = U
        G = O.reshape(E.shape)
    return G


if __name__ == "__main__":
    key = (15, 5, 0.5, 25, 10, 4, 0.1, 1.5)

    img = imread("../pictures/Lena512.bmp")
    enc = encryption(img, key)
    dec = decryption(enc, key)

    print((img == dec).all())
