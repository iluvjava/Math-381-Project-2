
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
MARK_TWAIN = "data/Mark Twain"


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
        self.npmatrix = np.zeros((n, n))

        for line in lines:
            line = list(line)
            if len(line) == 0:
                continue
            for i, c2 in enumerate(line[1:]):
                c1 = line[i]
                indx1 = Characters.find(c1)
                indx2 = Characters.find(c2)
                self.npmatrix[indx1, indx2] += 1

        for i in range(self.npmatrix.shape[0]):
            s = np.sum(self.npmatrix[i])
            if s > 0:
                self.npmatrix[i] /= s

    def missing_states(self):
        """
            Sometimes not all the letters in the alphabet are used,
            This method will return a list of characters that are not
            used in the lines of text.
        :return:
            A list letters that didn't appear in the lines of text.
        """
        res = []
        for I, RowSum in enumerate(self.npmatrix.sum(axis=1)):
            if RowSum == 0.0:
                res.append(TransitionMatrix.characters[I])
        return res


    def __repr__(self):
        return str(self.npmatrix)


"""
Files for an author and transitional matrix for the author. 
* Transitional Matrix classified by each files in the folder
* For each file, there will be several transitional matrices for parts of the files. 
* All the matrices will be in the same order as the list of works.
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

        self.__MatrixEachFile = None # Stored as an instance of transition matrix.
        self.__MatrixAllFiles = None # Instance of transition matrix.
        self.__AuthorWorksList = list(self.__FilePathToLines.items())

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
        Res = None
        if self.__MatrixEachFile is not None:
            res = [M.npmatrix for M in self.__MatrixEachFile]
        self.__MatrixEachFile = [TransitionMatrix(lines) for lines in self.list_of_works_content()]
        return [M.npmatrix for M in self.__MatrixEachFile]

    def aggregate_matrix(self):
        """
            combine all the lines in the folder into one single work.
            Then create the transitional matrix for this author,
            treating all his works as one text.
            * Results are stored after first time computing it.
        :return:
            An npmatrix
        """
        if self.__MatrixAllFiles is not None:
            return self.__MatrixAllFiles.npmatrix

        alllines = []
        for writing in self.list_of_works_content():
            alllines += writing
        self.__MatrixAllFiles = TransitionMatrix(alllines)
        return self.__MatrixAllFiles.npmatrix

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
        return np.linalg(m1 - m2, norm)





def dis_between_authors(author1, author2, norm = 2, mode=1):
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
    
    pass

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

    print("------ Creating another author and compare the old author to the new author. ")

def main():
    pass

# test code
if __name__ == '__main__':
    main()
    test_authors()
    pass




