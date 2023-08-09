# encoding: utf8
import cv2
import numpy as np
import glob

#_detecotor = cv2.AKAZE_create()
#_detecotor = cv2.ORB_create()
#_detecotor = cv2.BRISK_create()
_detecotor = cv2.KAZE_create()


def make_codebook(images, code_book_size, save_name):
    bow_trainer = cv2.BOWKMeansTrainer(code_book_size)
    count = 1
    for img in images:
        f = calc_feature(img)
        if f is not None:
            if not np.isnan(f).all():
                bow_trainer.add(f)
                count += 1

    code_book = bow_trainer.cluster()
    np.savetxt(save_name, code_book)


def calc_feature(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Image not found: {filename}')
        return None

    kp, discriptor = _detecotor.detectAndCompute(img, None)
    if discriptor is None:
        print(f'No descriptor found for {filename}')
        return None

    return np.array(discriptor, dtype=np.float32)


def make_bof(code_book_name, images, hist_name, file_name):
    code_book = np.loadtxt(code_book_name, dtype=np.float32)
    label_to_int = {"Avocado": 0, "Banana": 1, "Blueberry": 2, "Cauliflower": 3, "Cherry Wax Red": 4, "Corn": 5,
                    "Corn Husk": 6, "Eggplant": 7, "Fig": 8, "Ginger Root": 9, "Grape White": 10,
                    "Kohlrabi": 11, "Lemon": 12,  "Lychee": 13, "Onion White": 14, "Pepper Green": 15,
                    "Pineapple": 16, "Raspberry": 17, "Walnut": 18, "Watermelon": 19}

    knn = cv2.ml.KNearest_create()
    knn.train(code_book, cv2.ml.ROW_SAMPLE, np.arange(len(code_book), dtype=np.float32))

    hists = []
    label_list = []
    for img in images:
        f = calc_feature(img)
        if np.isnan(f).all() == False:
            png_name = img.split('/')
            label = png_name[4].split('_')
            idx = knn.findNearest(f, 1)[1]
            h = np.zeros(len(code_book))
            for i in idx:
                h[int(i)] += 1

            hists.append(h)
            label_list.append(label_to_int[label[0]])
        np.savetxt(hist_name, hists, fmt=str("%d"))
        np.savetxt(file_name, label_list, fmt='%d')


def main():
    data_root = "../data/Fruit360/Categories20"
    files = glob.glob(data_root + "/*/*.jpg")
    make_codebook(files, 200, "codebookA_200.txt")
    make_bof("./codebookA_200.txt", files, "./histogramA_200.txt", "./labelA_200.txt")

    #data_root = "../data/Fruit360/Categories20_rotated"
    #files = glob.glob(data_root + "/*/*.jpg")
    #make_codebook(files, 200, "codebookB_200.txt")
    #make_bof("./codebookB_200.txt", files, "./histogramB_200.txt", "./labelB_200.txt")


if __name__ == '__main__':
    main()
