import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--callee_path", type=str)
args = parser.parse_args()

print('callee: {}'.format(args.callee_path))