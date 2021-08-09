#!/usr/bin/env python3
import os
import os.path
from detect_sign import *

dataset_path = './yogyakarta_dataset/Survei rambu'
out_dir = './yogyakarta_processed_dataset'

CATEGORY_PERINGATAN = 1
CATEGORY_LARANGAN = 2
CATEGORY_PERINTAH = 3


def category_check(path):
    global CATEGORY_PERINGATAN, CATEGORY_LARANGAN, CATEGORY_PERINTAH
    if "1. Rambu Peringatan" in path:
        return CATEGORY_PERINGATAN
    elif "2. Rambu Larangan" in path:
        return CATEGORY_LARANGAN
    elif "3. Rambu Perintah" in path:
        return CATEGORY_PERINTAH
    return False


def move_image_rename_and_get_classes(dir):
    paths = []
    boundary = []
    classes = []
    counts = []
    cnt = 1

    # skip = 219

    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            print("No of images:", cnt, end=' ')
            cnt = cnt + 1

            # if cnt <= skip:
            #     continue

            path = os.path.join(dirpath, filename)

            category = category_check(path)
            if category:
                print('(Category {})'.format(category))
            else:
                print('Invalid Category')
                continue

            image_class = " ".join(path.split('/')[-2].split()[1:])

            # paths.append(path)
            # boundary.append(detect_sign_by_path(path, is_yolo=True))

            if image_class not in classes:
                classes.append(image_class)
                counts.append(0)

            image_class_code = classes.index(image_class)
            counts[image_class_code] = counts[image_class_code] + 1

            # Make boundary file
            boundary = detect_sign_by_path(
                path, is_yolo=True, category=category)
            if boundary[0] is not None:
                f = open(os.path.join(out_dir, "{}_{}.txt".format(
                    image_class_code, filename.split('.')[0])), 'w')
                f.write("{} {} {} {} {}".format(image_class_code,
                                                boundary[0], boundary[1], boundary[2], boundary[3]))
                f.close()

            # Copy image and rename
            command = "cp {} {}".format(path.replace(' ', '\\ ').replace('(', '\\(').replace(')', '\\)'), os.path.join(out_dir, "{}_{}".format(
                image_class_code, filename)))
            # print(command)
            os.system(command)

            # if cnt == 30:
            #     return paths, boundary, classes, counts

    return paths, boundary, classes, counts


paths, boundary, classes, counts = move_image_rename_and_get_classes(
    dataset_path)

f_classes = open(os.path.join(out_dir, "classes.txt"), 'w')
f_counts = open(os.path.join(out_dir, "counts.txt"), 'w')
# f_paths = open(os.path.join(out_dir, "paths.txt"), 'w')
# f_boundary = open(os.path.join(out_dir, "boundary.txt"), 'w')

f_classes.write("\n".join(classes))
f_counts.write("\n".join([str(i) for i in counts]))
# f_paths.write(str(paths))
# f_boundary.write(str(boundary))

f_classes.close()
f_counts.close()
# f_paths.close()
# f_boundary.close()
