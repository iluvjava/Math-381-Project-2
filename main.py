

import numpy as np
from string import ascii_letters

ENCODING = "utf8"

class TransitionMatrix:

    def __init__(self, filename):
        """
            creates a transitional matrix for the given text.
            This is the ordering the states:

            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '<None>'
        :param filename:
            The file path
        """
        self.matrix = np.zeros((55, 55))
        characters = ascii_letters + " '"

        with open(filename, 'r', encoding=ENCODING) as f:
            for line in f.readlines():
                line = list(line.strip())
                for i, c2 in enumerate(line[1:]):
                    c1 = line[i]
                    indx1 = characters.find(c1)
                    indx2 = characters.find(c2)
                    self.matrix[indx1, indx2] += 1

        for i in range(self.matrix.shape[0]):
            s = np.sum(self.matrix[i])
            assert s > 0, "The given text is not long enough to contain all all characters. "
            if s > 0:
                self.matrix[i] /= s

# test code
if __name__ == '__main__':
    try:
        t = TransitionMatrix('allNone_case.txt')
    except:
        print("Error is Expected.")

    t = TransitionMatrix('tester.txt')
    matrix = t.matrix
    print("Matrix shape =", matrix.shape)
    print(matrix)
    print("Matrix row sums =", np.sum(matrix, axis=1).ravel()) # not guarantee add up to 1 all the time




# simple case
'''
if __name__ == '__main__':
    t = transition_matrix('simple_case.txt')
    matrix = t.matrix
    if (matrix[0][0] == 1):
        print("PASS SIMPLE CASE")
    else:
        print("FAIL SIMPLE CASE")
#ascii case
if __name__ == '__main__':
    t = transition_matrix('ascii_case.txt')
    matrix = t.matrix
    real_matrix = np.zeros((54, 54))
    for i in range(53):
        real_matrix[i][i+1] = 1
    if np.array_equal(matrix, real_matrix):
        print("PASS ASCII CASE")
    else:
        print("FAIL ASCII CASE")'''





