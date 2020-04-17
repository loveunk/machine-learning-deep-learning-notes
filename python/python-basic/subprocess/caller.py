import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print('caller: {} {}'.format(__file__, args.path))

    theproc0 = subprocess.Popen([sys.executable, "-u", "callee.py", "--callee_path", "000"])
    theproc1 = subprocess.Popen([sys.executable, "-u", "callee.py", "--callee_path", "111"])
    theproc0.communicate()
    theproc0.communicate()
