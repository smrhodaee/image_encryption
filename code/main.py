import numpy as np
import matplotlib.pyplot as plt
from src.crypto import encryption, decryption, get_hyperchaotic_initials
from src.tools import *
from tabulate import tabulate
import inspect


def showplot():
    plt.waitforbuttonpress()
    plt.close("all")


class ImageEncryptionTest:
    def __init__(self, img_path) -> None:
        self.plain_img = imread(img_path)
        self.m, self.n = self.plain_img.shape
        self.hp = (15, 5, 0.5, 25, 10, 4, 0.1, 1.5)
        self.enc_img, self.inits = encryption(self.plain_img, self.hp)
        self.dec_img = decryption(self.enc_img, self.inits, self.hp)
        diff_plain_dec = np.sum(self.plain_img.astype(int) - self.dec_img.astype(int))
        print("PlainImage - DecryptedImage =", diff_plain_dec)
        print(
            "MAE (PlainImage, ChiperImage) = ",
            np.mean(np.abs(self.plain_img - self.enc_img)),
        )

    def histogram_analysis(self):
        axs = subplots(2, 3, "Simulation Result ...")
        imshow(axs[0, 0], "Plain Image", self.plain_img)
        imshow(axs[0, 1], "Cipher Image", self.enc_img)
        imshow(axs[0, 2], "Decrypted Image", self.dec_img)
        histshow(axs[1, 0], "Histogram of Plain Image", self.plain_img)
        histshow(axs[1, 1], "Histogram of Cipher Image", self.enc_img)
        histshow(axs[1, 2], "Histogram of Decrypted Image", self.dec_img)
        showplot()
        print("\n")

    def chi_square_test(self):
        # TODO: impl
        pass

    def information_entropy_analysis(self):
        print("PlainImage  Entropy =", image_entropy(self.plain_img))
        print("EncImage Entropy =", image_entropy(self.enc_img))
        print("\n")

    def speed_test(self):
        run_count = 5
        tenc = mean_execute_time(run_count, encryption, self.plain_img, self.hp)
        tdec = mean_execute_time(run_count, decryption, self.enc_img, self.inits, self.hp)
        print(f"{run_count} Runs --> Average Encryption Time = {tenc}")
        print(f"{run_count} Runs --> Average Decryption Time = {tdec}")

    def key_sensitivity_analysis(self):
        hp2 = (15.0001, 5, 0.5, 25, 10, 4, 0.1, 1.5)

        dec1 = decryption(self.enc_img, self.inits, self.hp)
        dec2 = decryption(self.enc_img, self.inits, hp2)

        axs = subplots(1, 4, "Images ...")
        imshow(axs[0], "Plain Image", self.plain_img)
        imshow(axs[1], "Enc Image", self.enc_img)
        imshow(axs[2], "Decrypted2 Image", dec2)
        imshow(axs[3], "Decrypted1 Image", dec1)
        showplot()

    def differential_attack_analysis(self):
        plain_img1bit = self.plain_img.copy()
        pos1 = np.random.randint(self.m)
        pos2 = np.random.randint(self.n)
        print(f"Before: Pixel at({pos1},{pos2}) = {plain_img1bit[pos1, pos2]}")
        plain_img1bit[pos1, pos2] = np.uint8((plain_img1bit[pos1, pos2] + 1) % 256)
        enc_img1bit, _ = encryption(plain_img1bit, self.hp)
        print(f"After: Pixel at ({pos1},{pos2}) = {plain_img1bit[pos1, pos2]}")
        uaci, npcr = uaci_npcr(self.enc_img, enc_img1bit)
        print(f"NPCR = {npcr}   UACI = {uaci}")

    def correlation_coefficient_analysis(self):
        cc, x, y = adjancy_corr_pixel_rand(self.plain_img, self.enc_img)
        print(tabulate(cc, ["Type", "Plain Image", "Chiper Image"], "grid"))
        axs = subplots(2, 4, "Correlation Coefficient Result ...")
        imshow(axs[0, 0], "Plain Image", self.plain_img)
        imshow(axs[1, 0], "Cipher Image", self.enc_img)
        imscatter(
            axs[0, 1],
            "Plain Gray (X, Y + 1)",
            self.plain_img[x, y],
            self.plain_img[x, y + 1],
        )
        imscatter(
            axs[1, 1],
            "Chiper Gray (X, Y + 1)",
            self.enc_img[x, y],
            self.enc_img[x, y + 1],
        )
        imscatter(
            axs[0, 2],
            "Plain Gray (X + 1, Y)",
            self.plain_img[x, y],
            self.plain_img[x + 1, y],
        )
        imscatter(
            axs[1, 2],
            "Chiper Gray (X + 1, Y)",
            self.enc_img[x, y],
            self.enc_img[x + 1, y],
        )
        imscatter(
            axs[0, 3],
            "Plain Gray (X + 1, Y + 1)",
            self.plain_img[x, y],
            self.plain_img[x + 1, y + 1],
        )
        imscatter(
            axs[1, 3],
            "Chiper Gray (X + 1, Y + 1)",
            self.enc_img[x, y],
            self.enc_img[x + 1, y + 1],
        )
        showplot()

    def cropping_attack(self):
        axs = subplots(2, 5, "Cropping attack ...")
        for i in range(5):
            crop = 2 ** (5 - i)
            crop_size = int(self.plain_img.shape[0] / crop)
            enc_img_croped = self.enc_img.copy()
            enc_img_croped[:crop_size, :crop_size] = 0
            dec_img_croped = decryption(enc_img_croped, self.inits, self.hp)
            psnr_crop = psnr(self.plain_img, dec_img_croped)
            print(f"PSNR of 1/{crop} cropped cipher image =", psnr_crop)
            imshow(axs[0, i], f"1/{crop} Chiper", enc_img_croped)
            imshow(axs[1, i], f"1/{crop} Decrypted", dec_img_croped)
        showplot()

    def salt_paper_noise_attack(self):
        axs = subplots(2, 4, "Salt and pepper noise attack ...")
        for i in range(4):
            noise_level = 5 / (10 ** (4 - i))
            if noise_level == 0.5:
                noise_level = 0.1
            enc_img_noised = salt_pepper_noise(self.enc_img, noise_level)
            dec_img_noised = decryption(enc_img_noised, self.inits, self.hp)
            psnr_noised = psnr(self.plain_img, dec_img_noised)
            print(
                f"Noise Level = {noise_level}, PSNR of nosiy cipher image = {psnr_noised}"
            )
            imshow(axs[0, i], f"{noise_level} Chiper", enc_img_noised)
            imshow(axs[1, i], f"{noise_level} Decrypted", dec_img_noised)
        showplot()


def main(img_path):
    iet = ImageEncryptionTest(img_path)

    methods = {"names": [], "funcs": []}

    def menu():
        for i, name in enumerate(methods["names"], 1):
            print(f"{i}.{name}")

    methods["names"].append("Menu")
    methods["funcs"].append(menu)
    for name, func in inspect.getmembers(iet, predicate=inspect.ismethod):
        if "__" not in name:
            methods["names"].append(name.replace("_", " ").capitalize())
            methods["funcs"].append(func)
    methods["names"].append("Exit")
    methods["funcs"].append(exit)

    menu()
    while True:
        try:
            ch = int(input("Enter Your Choice: "))
            print("\n")
            if ch <= 0 or ch > len(methods["names"]):
                raise Exception("Bad Choice")
            methods["funcs"][ch - 1]()
        except Exception as e:
            print("Error: ", e)


def test(img_path):
    iet = ImageEncryptionTest(img_path)
    iet.differential_attack_analysis()
    # iet.key_sensitivity_analysis()
    # iet.speed_test()
    # iet.cropping_attack()
    # iet.salt_paper_noise_attack()
    iet.correlation_coefficient_analysis()
    # iet.histogram_analysis()
    # iet.information_entropy_analysis()


if __name__ == "__main__":
    args = getargs({"img_path": "str"})
    main(**args)
    # test(**args)
