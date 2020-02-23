
"""
Group 4, this is for the Project 2.


"""
import numpy as np
from string import ascii_letters
from typing import List

from os import listdir
from os.path import isfile

ENCODING = "utf8"

# A list of authors' directory:
CHARLES_DICKENS = "data/Charles Dickens"

def file_readlines(filepath:str):
    """
    Function reads lines from the file, with the new line character stripped off from
    the line

    :param filename:
        The file path.
    :return:
    A array of string. Each string is a line in the file.
    """
    with open(filepath, 'r', encoding=ENCODING) as f:
        return [l.strip() for l in f.readlines()]

class TransitionMatrix:
    characters = ascii_letters + " '"

    def __init__(self, lines:List[List[str]]=None):
        """
            creates a transitional matrix for the given text.
            This is the ordering the states:
            abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '<None>
        :param filename:
            The file path
        :param lines:
            Provide strings is also ok, it will take an array of lines to construct the matrix.
        """
        n = len(TransitionMatrix.characters)
        Characters = TransitionMatrix.characters
        self.matrix = np.zeros((n,n))

        for line in lines:
            line = list(line)
            if len(line) == 0:
                continue
            for i, c2 in enumerate(line[1:]):
                c1 = line[i]
                indx1 = Characters.find(c1)
                indx2 = Characters.find(c2)
                self.matrix[indx1, indx2] += 1

        for i in range(self.matrix.shape[0]):
            s = np.sum(self.matrix[i])
            if s > 0:
                self.matrix[i] /= s

    def missing_states(self):
        """
            Sometimes not all the letters in the alphebet are used,
            This method will return a list of characters that are not
            used in the lines of text.

        :return:
            A list letters that didn't appear in the lines of text.
        """
        res = []
        for I, RowSum in enumerate(self.matrix.sum(axis=1)):
            if RowSum == 0.0:
                res.append(TransitionMatrix.characters[I])
        return res


    def __repr__(self):
        return str(self.matrix)


"""
Files for an author and transitional matrix for the author. 
* Transitional Matrix classified by each files in the folder
* For each file, there will be several transitional matrices for parts of the files. 
"""
class Author:

    def __init__(self, dir:str):
        FilePathList = []
        for filename in listdir(dir):
            filepath = dir + "/" + filename
            if isfile(filepath):
                FilePathList.append(filepath)

        assert len(FilePathList) > 0, f"There is no file under the directory: {dir}"

        FilePath2Lines = {}
        for f in FilePathList:
            FilePath2Lines[f] = file_readlines(f)
        self.__FilePathToLines = FilePath2Lines

        self.__MatrixEachFile = None
        self.__MatrixAllFiles = None
        self.__AuthorWorks = list(self.__FilePathToLines.items())

    def get_fp2lines(self):
        return self.__FilePathToLines

    def list_of_works(self):
        return list(self.__FilePathToLines.keys())

    def list_of_works_content(self):
        return list(self.__FilePathToLines.values())

    def matrix_eachfile(self):
        """
        :param partition:
            For each file, the user has the option to partition the matrix.
            For this case, there will be multiple transitional matrices for a single given file.
        :return:
            A list of the instances for the TransitionalMatrix, each corresponds to a file of the author.
        """
        if self.__MatrixEachFile is not None:
            return self.__MatrixEachFile
        self.__MatrixEachFile = [TransitionMatrix(lines) for lines in self.list_of_works_content()]
        return self.__MatrixEachFile

    def matrix_allfiles(self):
        """
            combine all the lines in the file into one single work.
            Then create the transitional matrix for this authors, treating all his works as one big line of text.
        """
        if self.__MatrixAllFiles is not None:
            return self.MatriAllFiles
        alllines = []
        for writing in self.list_of_works_content():
            alllines += writing
        self.__MatrixAllFiles = TransitionMatrix(alllines)
        return self.__MatrixAllFiles



def test_authors():
    instance = Author(CHARLES_DICKENS)
    print(len(instance.get_fp2lines()))
    print(instance.get_fp2lines().keys())
    print(instance.matrix_eachfile())
    print("-----Here is a list of numpy matrix from author: ")
    for M in instance.matrix_eachfile():
        print(M)
        print(f"The missing states for the transitional matrix:{M.missing_states()}")

    print("-------Here is the transitional matrix for all lines of works from the author:")
    print(instance.matrix_allfiles())

def main():
    try:
        t = TransitionMatrix('allNone_case.txt')
    except:
        print("Error is Expected.")
    t = TransitionMatrix('tester.txt')
    matrix = t.matrix
    print("Matrix shape =", matrix.shape)
    print([[round(matrix[I, J], 5) for I in range(55)] for J in range(55)])
    print("Matrix row sums =", np.sum(matrix, axis=1).ravel()) # not guarantee add up to 1 all the time


# test code
if __name__ == '__main__':
    # main()
    test_authors()




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





