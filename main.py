
"""
Group 4, this is for the Project 2.


"""
import numpy as np
from string import ascii_letters
from typing import List
import re

from os import listdir
from os.path import isfile

ENCODING = "utf8"

# A list of authors' directory:
CHARLES_DICKENS = "data/Charles Dickens"
MARK_TWAIN = "data/Mark Twain"

np.set_printoptions(threshold=np.inf, precision=2, linewidth=1000)


def process_text(filepath: str):
    """
    Function reads lines from the file, with the new line character stripped off from
    the line

    :param filepath:
        The file path.
    :return:
    A array of string. Each string is a line in the file.
    """
    with open(filepath, 'r', encoding=ENCODING) as f:
        return [l.strip() for l in f.readlines()]

def trim_line(s):
    """
        This function trims off all the punctuations in the line and collpase the
        spaces in the text.
    :param s:
        Single line of string, should be trimmed
    :return:
        A new line of string that is trimmed.
    """
    s = s.strip()
    NonAlphabet = '''!()-[]{};:"\,<>./?@#$%^&*_~=0123456789+`|'''
    Astrophe = "'"
    Res = ""
    for char in s:
        if char in NonAlphabet:
            Res = Res + ' '
        elif char == Astrophe:
            continue # Strip off apostrophe.
        else:
            Res += char
    return re.sub(' +', ' ', Res.lower())

def get_tm27(lines):
    """
        Function takes the path of a file and returns the transition
        matrix based on the 26 letters in the alphabet,
        The last states of the matrix is space.
        * All spaces are collapsed.
        * All punctuations are ignored.
        * The apostrophe is stripped off from the text.
    :param lines:
        An array of lines that is in the file.
    :return:
    """
    matrix = np.zeros((27, 27))
    characters = ascii_letters[0:26] + " "
    for line in lines:
        trimmed_line = trim_line(line)
        line = list(trimmed_line)
        for i, c2 in enumerate(line[1:]):
            c1 = line[i]
            indx1 = characters.find(c1)
            indx2 = characters.find(c2)
            if (indx1 == -1 or indx2 == -1):
                continue # Skip non alphabetical characters.
            matrix[indx1, indx2] += 1
    for i in range(27):
        s = np.sum(matrix[i])
        if s > 0:
            matrix[i] /= s
    return matrix

def get_tm55(lines):
    """
        This function creates a matrix of 55 states.
        All the letters in lower cases and capitalized letters.
        It will also mark the apostrophe as the observable state.
        All other characters will be put into a hidden state.
    :param lines:
        A array of lines read from the file.
    :return:
        np matrix.
    """
    characters = ascii_letters + " '"
    n = len(characters)
    Characters = TransitionMatrix.characters
    npmatrix = np.zeros((n, n))
    for line in lines:
        line = list(line)
        if len(line) == 0:
            continue
        for i, c2 in enumerate(line[1:]):
            c1 = line[i]
            indx1 = Characters.find(c1)
            indx2 = Characters.find(c2)
            npmatrix[indx1, indx2] += 1

    for i in range(npmatrix.shape[0]):
        s = np.sum(npmatrix[i])
        if s > 0:
            npmatrix[i] /= s
    return npmatrix

def get_2ndtm(lines):
    pass

# class TransitionMatrix:
#     characters = ascii_letters + " '"
#
#     def __init__(self, lines:List[List[str]]=None):
#         pass
#
#     def missing_states(self):
#         """
#             Sometimes not all the letters in the alphabet are used,
#             This method will return a list of characters that are not
#             used in the lines of text.
#         :return:
#             A list letters that didn't appear in the lines of text.
#         """
#         res = []
#         for I, RowSum in enumerate(self.npmatrix.sum(axis=1)):
#             if RowSum == 0.0:
#                 res.append(TransitionMatrix.characters[I])
#         return res
#
#     def __repr__(self):
#         return str(self.npmatrix)


"""
Files for an author and transitional matrix for the author. 
* Transitional Matrix classified by each files in the folder
* For each file, there will be several transitional matrices for parts of the files. 
* All the matrices will be in the same order as the list of works.
"""
class Author:

    def __init__(self, dir:str, matrixfunction=get_tm27: function):
        FilePathList = []
        for filename in listdir(dir):
            filepath = dir + "/" + filename
            if isfile(filepath):
                FilePathList.append(filepath)

        assert len(FilePathList) > 0, f"There is no file under the directory: {dir}"

        FilePath2Lines = {}
        for f in FilePathList:
            FilePath2Lines[f] = process_text(f)
        self.__FilePathToLines = FilePath2Lines

        self.__TMFunction = matrixfunction

        self.__NpMatrices = None # a list of np matrices for each works of the author
        self.__AggregateMatrix = None # Instance of transition matrix.
        self.__AuthorItems = list(self.__FilePathToLines.items())

    def get_fp2lines(self):
        return self.__FilePathToLines

    def list_of_works(self):
        return list(self.__FilePathToLines.keys())

    def list_of_works_content(self):
        return list(self.__FilePathToLines.values())

    def get_matrices(self):
        """
            This function returns a transition matrix for each work of the author.
            each work of the author is a file in the author's folder.
        :return:
            A list of np matrix.
        """
        res = None
        if self.__NpMatrices is not None:
            res = self.__NpMatrices
        else:
            self.__NpMatrices = [self.__TMFunction(lines) for lines in self.list_of_works_content()]
            res = self.__NpMatrices
        return res

    def aggregate_matrix(self):
        """
            combine all the lines in the folder into one single work.
            Then create the transitional matrix for this author,
            treating all his works as one text.
            * Results are stored after first time computing it.
        :return:
            An npmatrix
        """
        if self.__AggregateMatrix is not None:
            return self.__AggregateMatrix
        alllines = []
        for writing in self.list_of_works_content():
            alllines += writing
        self.__AggregateMatrix = self.__TMFunction(alllines)
        return self.__AggregateMatrix

    def centroid_matrix(self):
        """
            This function returns the centroid matrix.
            * A centroid matrix is the average for each of all the matrices
            from this authors.
            * It's not necessarily a transition matrix anymore.
        :return:
            A numpy matrix.
        """
        TransitionMatrices = self.get_matrices()
        N = len(TransitionMatrices)
        CentroidMatrix = np.zeros(TransitionMatrices[0].shape)
        for TM in TransitionMatrices:
            CentroidMatrix += TM
        CentroidMatrix /= N
        return CentroidMatrix

    def distance_list(self, norm=2, mode=1):
        """
            The function will calculate the distance for all the transition matrix
            from each file.
        :param mode:
            mode == 1:
                Using the centroid as the center of this author.
            mode != 1:
                Using the aggregate matrix as the center of this author.
        :return:
            A map; the key is the name of the file and the
            value is the distance from teh centroid matrix.
        """
        DistanceMap = {}
        Center = self.centroid_matrix() if mode == 1 else self.aggregate_matrix()
        for Writing, Matrix in zip(self.list_of_works(), self.get_matrices()):
            DistanceMap[Writing] = np.linalg.norm(Matrix - Center, norm)
        return DistanceMap

    def distance_to(self, m2, norm=2, mode=1):
        """
            The function return the distance of this author to a given transition
            matrix of the same size.
            * How the distance is calculated depends on the input
            parameters.
        :param norm:
            The matrix to norm to calculate the distances.
        :param mdoe:
            mode==1:
                Distance from the centroid of the author.
            mode != 1:
                Distance from the aggregate matrix of the author.
        :return:
            A float.
        """
        m1 = self.centroid_matrix() if mode == 1 else self.aggregate_matrix()
        return np.linalg.norm(m1 - m2, norm)


def dis_between_authors(author1, author2, norm=2, mode=1):
    """
        This function returns 1 number to represent the distance between 2 author's
        works.
    :param author1:
        An instance of an Author class.
    :param author2:
        An instance of an Author class.
    :param norm:
        The matrix norm to use to measure distance.
    :param mode:
        mode == 1:
            Measure the distance by the difference between 2 centroids.
        mode != 1:
            Measure the distance by the difference between 2 aggregate matrix.
    :return:
        a float.
    """
    author1 = author1.centroid_matrix() if mode == 1 else author1.aggregate_matrix()
    return author2.distance_to(author1, norm, mode)


def test_authors():
    Author1 = Author(CHARLES_DICKENS)
    print(len(Author1.get_fp2lines()))
    print(Author1.get_fp2lines().keys())
    print("-----Here is a list of numpy matrix from author: ")
    for M in Author1.get_matrices():
        print(M)

    print("------ This is the centroid matrix-----")
    print(Author1.centroid_matrix())

    print("-------This is the aggregate matrix -----")
    print(Author1.aggregate_matrix())

    print("-------List of euclidean distances from the centroid of the author:----")
    print(Author1.distance_list())

    print("------- List of euclidean distances from the aggregate matrix of the author: ---")
    print(Author1.distance_list(mode=2))

    print("----- one norm distance from the centroid of this author: ----")
    print(Author1.distance_list(norm=1))

    print("----- infinity from the centroid for this author: ----")
    print(Author1.distance_list(norm = np.inf))

    print("------ Creating another author and compare the old author to the new author. ")
    Author2 = Author(MARK_TWAIN)

    print("------ The distance between 2 centroid of the authors is: ----")
    print(f"norm={2}: {dis_between_authors(Author1, Author2)}")
    print(f"norm={1}: {dis_between_authors(Author1, Author2, norm=1)}")


def main():
    pass


if __name__ == '__main__':
    main()
    test_authors()
    pass




