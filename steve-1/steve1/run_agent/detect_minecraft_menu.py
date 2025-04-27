import argparse
import cv2
import numpy as np

colors = {
    'bright': np.array([198, 198, 198]),
    'dark': np.array([63, 63, 63]),
}

c_uplefts = [(329, 103), (406, 103)]
c_info = {
    'width': 5,
    'height': 7,
    'offsets': [
                (1, 0), (2, 0), (3, 0),
        (0, 1),                         (4, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),                         (4, 5),
                (1, 6), (2, 6), (3, 6)
        ]
}

f_uplefts = [(347, 103), (424, 103)]
f_info = {
    'width': 4,
    'height': 7,
    'offsets': [
                        (2, 0), (3, 0),
                (1, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6)
        ]
}

g_uplefts = [(364, 105), (441,105)]
g_info = {
    'width': 5,
    'height': 6,
    'offsets': [
                (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1),                         (4, 1),
        (0, 2),                         (4, 2),
                (1, 3), (2, 3), (3, 3), (4, 3),
                                        (4, 4),
        (0, 5), (1, 5), (2, 5), (3, 5)
        ]
}

def detect_letter(img, ul, info):
    x = ul[0]
    y = ul[1]
    for dy in range(info['height']):
        for dx in range(info['width']):
            if (dx, dy) in info['offsets']:
                if not np.array_equal(img[y+dy][x+dx], colors['dark']):
                    return False
            else:
                if not np.array_equal(img[y+dy][x+dx], colors['bright']):
                    return False
    return True

def detect_letters(img):
    cfg1 = [(c_uplefts[0], c_info), (f_uplefts[0], f_info), (g_uplefts[0], g_info)]
    cfg2 = [(c_uplefts[1], c_info), (f_uplefts[1], f_info), (g_uplefts[1], g_info)]
    count1 = 0
    for cfg in cfg1:
        if detect_letter(img, cfg[0], cfg[1]):
            count1 += 1
    count2 = 0
    for cfg in cfg2:
        if detect_letter(img, cfg[0], cfg[1]):
            count2 += 1
    return (count1, count2)

def main(imagefile):
   image = cv2.imread(imagefile, cv2.IMREAD_UNCHANGED)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   img = np.array(image)
   print(img.shape, img.dtype)
   print(detect_letters(img))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagefile", type=str, required=True, help="Path to the .png file")
    args = parser.parse_args()
    main(imagefile=args.imagefile)
