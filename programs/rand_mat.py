#!/opt/anaconda2/bin/python
import numpy as np
import pickle
import argparse


def randrot(n, to_file=None, seed=0):
    """
    randomize n rot matrices
    :param n: number of rotations to generate
    :return:
    """
    np.random.seed(seed)
    qs = np.random.uniform(0, 1, 4 * n).reshape([n, 4])
    rots = []
    # normalize each vector
    for i in range(n):
        qs[i] /= np.sqrt(sum(np.square(qs[i])))
        rots.append(q_to_rot(qs[i]))

    # in case used from python
    if to_file is None:
        return np.array(rots)

    # in case used from command line
    if to_file is not None:
        f = open(to_file, "wb")
        pickle.dump(rots, f)


def q_to_rot(q):
    """
    returns a matrix represents the quaternion
    :param q: unit vector of size 4 (quaternion)
    :return: 3x3 matrix
    """
    a, b, c, d = q[0], q[1], q[2], q[3]
    return np.array(
        [a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d,
         2 * a * c + 2 * b * d,
         2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2,
         -2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d, 2 * a * b + 2 * c * d,
         a ** 2 - b ** 2 - c ** 2 + d ** 2]).reshape([3, 3])


def load_rot_mat(rot_file):
    """
    load all rotation matrices from rot file and return them
    :param rot_file: file contains rotation matrices
    :return: list of all rotation matrices in rot_file
    """
    try:
        rots = pickle.load(open(rot_file, "rb"))
        return rots
    except IOError:
        raise "File specified doe's not exist!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="random rotation matrices and save them to a pickle file")
    parser.add_argument('number_of_matrices', metavar='n', type=int,
                        help='amount of matrices to random')
    parser.add_argument('output', metavar='output', type=str,
                        help='output file')

    args = parser.parse_args()
    randrot(args.n, args.output)
