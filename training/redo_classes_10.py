#!/usr/bin/env python3
import os
import os.path
from detect_sign import *

out_dir = './yogyakarta_fixed_dataset'


def move_image_rename_and_get_classes(dir):
    cnt = 1
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(".jpg") and '10_' in f]:
            print("No of images:", cnt, end=' ')
            cnt = cnt + 1
            path = os.path.join(dirpath, filename)

            # print(filename)

            image_class_code = filename.split('_')[0]

            # Make boundary file
            boundary = detect_sign_by_path(
                path, is_yolo=True, category=2)
            if boundary[0] is not None:
                f = open(os.path.join(out_dir, "{}.txt".format(
                    filename.split('.')[0])), 'w')
                f.write("{} {} {} {} {}".format(image_class_code,
                                                boundary[0], boundary[1], boundary[2], boundary[3]))
                f.close()


move_image_rename_and_get_classes(out_dir)
