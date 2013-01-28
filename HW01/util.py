def verify(f_name):
    try:
        f = open(f_name)
    except IOError as e:
        if e.errno == errno.ENOENT:
            sys.stderr.write('\'{}\' is not a valid file name!\n'.format(f_name))
            sys.exit(1)
    else:
        f.close()
