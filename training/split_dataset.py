#!/usr/bin/env python3
import os
from sklearn.model_selection import train_test_split

dataset_path = './yogyakarta_fixed_dataset'
out_dir = './yogyakarta_traffic_sign_dataset'


def split_dataset():
    bucket = [[] for i in range(11)]
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            idx = int(filename.split('_')[0])
            bucket[idx].append(filename)

    train = []
    test = []

    for b in bucket:
        tr, ts = train_test_split(b, test_size=0.2)
        train = train + tr
        test = test + ts

    f_train = open(os.path.join(out_dir, 'data/train.txt'), 'w')
    f_test = open(os.path.join(out_dir, 'data/test.txt'), 'w')
    for t in train:
        print(t)
        f_train.write('data/obj/' + t + '\n')
        os.system("cp {} {}".format(os.path.join(dataset_path, t),
                                    os.path.join(out_dir, 'data/obj/' + t)))
        os.system("cp {} {}".format(os.path.join(dataset_path, t.replace('.jpg', '.txt')),
                                    os.path.join(out_dir, 'data/obj/' + t.replace('.jpg', '.txt'))))
    for t in test:
        print(t)
        f_test.write('data/test/' + t + '\n')
        os.system("cp {} {}".format(os.path.join(dataset_path, t),
                                    os.path.join(out_dir, 'data/test/' + t)))
        os.system("cp {} {}".format(os.path.join(dataset_path, t.replace('.jpg', '.txt')),
                                    os.path.join(out_dir, 'data/test/' + t.replace('.jpg', '.txt'))))
    f_train.close()
    f_test.close()


def move_classes_txt():
    os.system("cp {} {}".format(os.path.join(dataset_path,
                                             "classes.txt"), os.path.join(out_dir, "obj.names")))


split_dataset()
move_classes_txt()
