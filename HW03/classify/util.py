import errno
import sys
import csv
import random


def verify(f_name):
    try:
        f = open(f_name)
    except IOError as e:
        if e.errno == errno.ENOENT:
            sys.stderr.write('\'{}\' is not a valid file name!\n'.format(f_name))
            sys.exit(1)
    else:
        f.close()

def get_rows(file_name, false=0, shuffle=False):
    with open(file_name, newline='') as f:
        if shuffle:
            h = f.readline()
            fs = f.readlines()
            random.shuffle(fs)
            fs.insert(0, h)
            reader = csv.DictReader(fs)
        else:
            reader = csv.DictReader(f)
        for row in reader:
            yield dict((k, int(v)) if int(v) else (k, false) for k,v in row.items())

def get_headers(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        return reader.__next__()[:-1]
