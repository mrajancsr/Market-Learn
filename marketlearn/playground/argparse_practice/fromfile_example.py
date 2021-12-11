# using argparse library from a file
import argparse

parser = argparse.ArgumentParser(
    prog="fromfile_example",
    usage="%(prog)s [options] @file_name",
    fromfile_prefix_chars="@",
    description="getting arguments from file",
    epilog="enjoy python * - *",
)
parser.add_argument("a", help="a first argument")

parser.add_argument("b", help="a second argument")
parser.add_argument("c", help="a third argument")

parser.add_argument("d", help="a fourth argument")
parser.add_argument("e", help="a fifth argument")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="an optional argument"
)

args = parser.parse_args()

print("if u read this line, that means u provided all the parameters")

test = args.a
test2 = args.b
test3 = args.c
test4 = args.d
test5 = args.e

print(" ".join(c for c in [test, test2, test3, test4, test5]))
