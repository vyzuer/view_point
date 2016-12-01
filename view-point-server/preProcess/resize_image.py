import os, sys, glob
import Image

maxsize = 640.0

def resize_image(infile, outfile):
    if infile != outfile:
        try:
            im = Image.open(infile)
            print im.size
            width, height = im.size
            ratio = min(maxsize/width, maxsize/height)
            width = int(width*ratio)
            height = int(height*ratio)
            im = im.resize((width, height), Image.ANTIALIAS) 
            im.save(outfile)
        except IOError:
            print "cannot create file for '%s'" % infile


def read_files(db_path, dump_path):
    # read the dataset
    for infile in glob.glob(os.path.join(db_path, '*.jpg')):
        dir_name, file_name = os.path.split(infile)
        outfile = dump_path + file_name
        resize_image(infile, outfile)

if len(sys.argv) != 3:
    print "Usage : dataset_path dump_path"
    sys.exit(0)

db_path = sys.argv[1]
dump_path = sys.argv[2]

read_files(db_path, dump_path)


