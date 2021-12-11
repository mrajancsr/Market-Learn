from collections import namedtuple
import argparse

MultiSelectionOption = namedtuple(
    "MultipleSelectionOption", "label value queue"
)


def get_options_from_file(file_name):
    with open(file_name) as f:
        options = [line.strip() for line in f.readlines()]
    return [MultiSelectionOption(o, o, "") for o in options]


parser = argparse.ArgumentParser(
    prog="testargs2",
    usage="%(prog)s [options] file_name",
    description="getting filename from arguments",
    epilog="enjoy python * - *",
)

parser.add_argument("--filename", action="store", type=str, required=True)

args = parser.parse_args()

test = get_options_from_file(args.filename)

# call withi these arguments
# python testargs2.py --filename /Users/raj/Documents/QuantResearch/Home/
# market-learn/marketlearn/playground/data/args2.csv

print(test)
