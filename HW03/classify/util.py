import errno
import sys
import csv


def verify(f_name):
    try:
        f = open(f_name)
    except IOError as e:
        if e.errno == errno.ENOENT:
            sys.stderr.write('\'{}\' is not a valid file name!\n'.format(f_name))
            sys.exit(1)
    else:
        f.close()

def get_rows(file_name):
    with open(file_name, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k:int(v) for k,v in row.items()}

def get_headers(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        return reader.__next__()[:-1]
