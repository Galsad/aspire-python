#!/opt/anaconda2/bin/python

import numpy as np
import argparse
import mrcfile


def rand_vol(n, output):
    """
    random a volume of size n x n x n and saves it in output as mrcfile
    :param n: size of volume
    :param output: the path the volume is going to be saved to
    :return: None
    """

    vol = np.random.uniform(-np.pi, np.pi, n * n * n).reshape([n, n, n])
    with mrcfile.new(output, overwrite=True) as mrc:
        mrc.set_data(vol.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate a random volume mrc file")
    parser.add_argument("volume size", metavar="n", type=int,
                        help="size of volume")
    parser.add_argument("output", metavar="output", type=str,
                        help="output path of the volume")

    args = parser.parse_args()
    rand_vol(args.n, args.output)
