#!/usr/bin/env python3
import os
import os.path
from types import new_class

dataset_path = './yogyakarta_processed_dataset'
out_dir = './yogyakarta_fixed_dataset'
real_classes = open(os.path.join(dataset_path, 'classes.txt'), 'r').readlines()
real_classes = [i.replace('\n', '') for i in real_classes]


def get_classes_with_more_than_50():
    f = open(os.path.join(dataset_path, 'classes.txt'), 'r')
    f1 = open(os.path.join(dataset_path, 'counts.txt'), 'r')
    f = f.readlines()
    f1 = f1.readlines()
    f1 = [int(i) for i in f1]

    g = open(os.path.join(out_dir, 'classes.txt'), 'w')
    g1 = open(os.path.join(out_dir, 'counts.txt'), 'w')

    for i in range(len(f1)):
        if f1[i] > 50:
            g.write(f[i])
            g1.write(str(f1[i]) + '\n')


def move_image_rename_and_get_classes():
    new_classes = open(os.path.join(out_dir, 'classes.txt'), 'r').readlines()
    new_classes = [i.replace('\n', '') for i in new_classes]

    cnt = 1
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            # print("No of images:", cnt, end=' ')
            cnt = cnt + 1
            path = os.path.join(dirpath, filename)

            image_class = filename.split('_')[0]

            try:
                new_class_code = new_classes.index(
                    real_classes[int(image_class)])
                b = open(path.replace('.jpg', '.txt'), 'r')
                b = b.read()
                b = b.split()
                b[0] = str(new_class_code)
                b = ' '.join(b)

                new_file_name = filename.replace('.jpg', '').split('_')
                new_file_name[0] = str(new_class_code)
                new_file_name = '_'.join(new_file_name)

                print(new_file_name)

                c = open(os.path.join(out_dir, new_file_name + '.txt'), 'w')
                c.write(b)
                c.close()

                os.system("cp {} {}".format(
                    path, os.path.join(out_dir, new_file_name + '.jpg')))
            except:
                continue


get_classes_with_more_than_50()
move_image_rename_and_get_classes()
