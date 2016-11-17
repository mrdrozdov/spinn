import argparse
import os
from spinn.data.sst import load_sst_data


command = """
curl -o sst.zip http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip sst.zip
rm sst.zip
mv trees sst
mv sst ./datasets/

sed -i -e 's/)/\ )/g' ./datasets/sst/dev.txt
sed -i -e 's/)/\ )/g' ./datasets/sst/test.txt
sed -i -e 's/)/\ )/g' ./datasets/sst/train.txt
"""


def download():
    os.system(command)


def demo():
    examples, _ = load_sst_data.load_data('sst/dev.txt')
    print(examples[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    args = parser.parse_args()

    if args.command == 'download':
        download()
    elif args.command == 'demo':
        demo()
    else:
        raise Exception("Invalid command.")
