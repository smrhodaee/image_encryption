import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2


def getargs(args):
    res = {}
    if len(sys.argv) <= len(args):
        print(
            f"Usage: {sys.argv[0]}",
            " ".join(f"<{k.replace('_', '-')}>" for k in args.keys()),
        )
        exit(1)
    for i, k in enumerate(args.keys(), 1):
        exec(f"res['{k}'] = {args[k]}(sys.argv[{i}])")
    return res


def subplots(nrows, ncols, label):
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), num=label)
    fig.set_layout_engine("constrained")
    return axs


def imscatter(ax, title, x, y):
    ax.scatter(x.flatten(), y.flatten(), marker=".")
    ax.axis([0, 255, 0, 255])
    ax.set_title(title)


def imread(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def imshow(ax, title, img):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)


def histshow(ax, title, img):
    ax.set_title(title)
    ax.hist(img.flatten(), bins=256, color="gray")


def mean_execute_time(run_count, func, *args):
    lst_time = np.zeros(run_count)
    for i in range(run_count):
        tic = cv2.getTickCount()
        res = func(*args)
        lst_time[i] = (cv2.getTickCount() - tic) / cv2.getTickFrequency()
    return np.mean(lst_time)


def psnr(original, compressed):
    max_pixel = 255
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10((max_pixel**2) / mse)


def salt_pepper_noise(image, prob):
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > (1 - prob):
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def image_entropy(img):
    x = np.array(img, dtype=float)
    xh, _ = np.histogram(x.flatten(), img.shape[0])
    xh = xh / np.sum(xh)
    i = np.where(xh)[0]
    return -np.sum(xh[i] * np.log2(xh[i]))


def adjancy_corr_pixel_rand(plain_img, enc_img):
    plain_img = plain_img.astype(np.float64)
    enc_img = enc_img.astype(np.float64)
    m, n = plain_img.shape
    m -= 1
    n -= 1
    k = 100
    s = np.random.choice(m * n, k)
    x, y = np.unravel_index(s, (m, n))
    return (
        [
            [
                "V",
                np.corrcoef(plain_img[x, y], plain_img[x, y + 1])[0, 1],
                np.corrcoef(enc_img[x, y], enc_img[x, y + 1])[0, 1],
            ],
            [
                "H",
                np.corrcoef(plain_img[x, y], plain_img[x + 1, y])[0, 1],
                np.corrcoef(enc_img[x, y], enc_img[x + 1, y])[0, 1],
            ],
            [
                "D",
                np.corrcoef(plain_img[x, y], plain_img[x + 1, y + 1])[0, 1],
                np.corrcoef(enc_img[x, y], enc_img[x + 1, y + 1])[0, 1],
            ],
        ],
        x,
        y,
    )


def uaci_npcr(c1, c2):
    m, n = c1.shape
    c1 = c1.astype(float)
    c2 = c2.astype(float)
    h = lambda i, j: 0 if c2[i, j] == c1[i, j] else 1
    e = lambda i, j: abs(c2[i, j] - c1[i, j])
    uaci = sum(e(i, j) for j in range(n) for i in range(m)) / (255 * m * n)
    npcr = sum(h(i, j) for j in range(n) for i in range(m)) / (m * n)
    return 100 * uaci, 100 * npcr


def mat_split_blocks(mat, block_shape):
    m, n = mat.shape
    bm, bn = block_shape
    assert m % bm == 0
    assert n % bn == 0

    om, on = m // bm, n // bn
    out = [0] * om

    for i in range(om):
        out[i] = [0] * on
        for j in range(on):
            out[i][j] = np.zeros(block_shape)
            for k in range(bm):
                for l in range(bn):
                    out[i][j][k][l] = mat[bm * i + k][bn * j + l]

    return np.array(out)


def mat_join_blocks(blocks):
    m, n, bm, bn = blocks.shape
    om, on = m * bm, n * bn
    out = np.zeros((om, on))
    for i in range(m):
        for j in range(n):
            for k in range(bm):
                for l in range(bn):
                    out[bm * i + k][bn * j + l] = blocks[i][j][k][l]

    return out
