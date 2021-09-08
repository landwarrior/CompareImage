import sys

import cv2
import numpy as np

def main():
    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])
    img_size = (1000, 1000)
    # 比較にために同じサイズにする
    image1 = cv2.resize(image1, img_size)
    image2 = cv2.resize(image2, img_size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    print(f"1に近いほど一致している: {cv2.compareHist(hist1, hist2, 0)}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.ORB_create()
    (target_kp, target_des) = detector.detectAndCompute(image1, None)
    (comparing_kp, comparing_des) = detector.detectAndCompute(image2, None)
    matches = bf.match(target_des, comparing_des)
    dist = [m.distance for m in matches]
    print(f"0に近いほど一致している: {sum(dist) / len(dist)}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        # 1つ目には必ず自身のファイル名が入っている
        print('please give 2 args: image paths')
        exit()
    main()
