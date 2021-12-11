import argparse

import os
import sys

# create the parser
my_parser = argparse.ArgumentParser(
    prog="myls",
    usage="%(prog)s [options] path",
    description="list of content of a folder",
    epilog="Enjoy the program :)",
)

# add the arguments
my_parser.add_argument(
    "Path", metavar="path", type=str, help="the path to the list"
)

# execute the parse_arge() method
args = my_parser.parse_args()

input_path = args.Path

if not os.path.isdir(input_path):
    print("the path specidfied does not exist")
    sys.exit()

print("\n".join(os.listdir(input_path)))