"""
Group 4, this is for the Project 2.


"""
import numpy as np
import matplotlib.pyplot as plt

from string import ascii_letters
from typing import List, Callable, Type
import re
import enum
import math
from os import listdir
from os.path import isfile

__all__ = ["Author", "dis_between_authors", "get_tm55", "get_tm27", "get_2ndtm",
           "CHARLES_DICKENS",
           "MARK_TWAIN", "CentroidOption", "MatrixMetric", "AuthorMetric"]

# A list of authors' directory:
CHARLES_DICKENS = "data/Charles Dickens"
MARK_TWAIN = "data/Mark Twain"
ENCODING = "utf8"
np.set_printoptions(threshold=10, precision=2, linewidth=1000)


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


def trim_line(s: str):
    """
        This function trims off all the punctuations in the line and collapse the
        spaces in the text.
        * The return string is going to contains only alphabetical letters
        with single space between it.
    :param s:
        Single line of string, should be trimmed
    :return:
        A new line of string that is trimmed.
    """
    s = s.strip()
    characters = ascii_letters + " '"
    NonAlphabet = '''!()-[]{};:"\,<>./?@#$%^&*_~=0123456789+`|'''
    Astrophe = "'"
    Res = ""
    for char in s:
        if char in NonAlphabet:
            Res = Res + ' '
        elif char == Astrophe:
            continue  # Strip off apostrophe.
        else:
            Res += char if char in characters else ""
    return re.sub(' +', ' ', Res.lower())


def get_tm27(lines: List[str]):
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
                continue  # Skip non alphabetical characters.
            matrix[indx1, indx2] += 1
    for i in range(27):
        s = np.sum(matrix[i])
        if s > 0:
            matrix[i] /= s
    return matrix


def get_tm55(lines: List[str]):
    """
        This function creates a matrix of 55 states given the raw inputs from the file
        as an array of strings.
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
    Characters = characters
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


def get_2ndtm(lines: List[str]):
    """
        Given the content of the file separated by lines, this function will return the
        26^2 by 26^2 transition matrix.
        * It's a second order transition matrix based on the letters of the alphabet.
        * Spaces will be included as the last states of the matrix.
    :param lines:
        The content of the file represented in the an array of lines.
    :return:
        The np matrix.
    """
    Alphabet = ascii_letters[0:26] + " "
    l = len(Alphabet)
    n = l ** 2
    npmatrix = np.zeros((n, n))

    def s(letter):
        return Alphabet.find(letter)

    for Line in lines:
        Line = trim_line(Line)
        for I in range(len(Line) - 3):
            i = s(Line[I]) * l + s(Line[I + 1])
            j = s(Line[I + 2]) * l + s(Line[I + 3])
            npmatrix[i, j] += 1

    for i in range(npmatrix.shape[0]):
        s = np.sum(npmatrix[i])
        if s > 0:
            npmatrix[i] /= s
    return npmatrix


def get_3ndtm_nonsquare(lines: List[str]):
    """
       Non square 3rd order matrix.
    :param lines:
        The content of the file represented in the an array of lines.
    :return:
        The np matrix.
    """
    Alphabet = ascii_letters[0:26] + " "
    l = len(Alphabet)
    n = l ** 2
    npmatrix = np.zeros((l ** 3, l))

    def s(letter):
        return Alphabet.find(letter)

    for Line in lines:
        Line = trim_line(Line)
        for I in range(len(Line) - 3):
            i = s(Line[I]) * l ** 2 + s(Line[I + 1]) * l + s(Line[I + 2])
            j = s(Line[I + 3])
            npmatrix[i, j] += 1

    for i in range(npmatrix.shape[0]):
        s = np.sum(npmatrix[i])
        if s > 0:
            npmatrix[i] /= s
    return npmatrix


def get_3rdtm(lines: List[str]):
    raise RuntimeError("Not yet Implemented. ")


class CentroidOption(enum.Enum):
    """
    An enum class to represent the options for center of the author cloud.
    """
    AggregateMatrix = 1 # Taking the average among all works of the author.
    AverageMatrix = 2 # Treating all works as one single block of text.


class MatrixMetric(enum.Enum):
    """
    An enum class to represent the options of measuring distance between matrices.
    """
    OneNorm = 1 # Matrix 1 norm
    TwoNorm = 2 # Matrix Euclidean distance
    # WeightedNorm = 3 # Matrix weighted by PD matrix.
    # HighPower2Norm = 4 # Raising matrix to high power and take the 2 norm.
    Vectorized1Norm = 5


class AuthorMetric(enum.Enum):
    """
    THis is a enum class consists of method that can be used to determine the distance
    between the cloud of an author to a transition matrix.
    ! This option only specifies how to determing the distance of an author with a transition
    matrix!
    """
    MinimumDis = 1 # The distance between the author and a given transition matrix is the minimum distance
    # of any work of the author to that transition matrix.

    AverageDis = 2 # Taking the average distance of the given transition matrix with respect to
    # All the matrices of the author.

    CentroidDis = 3 # This metric take the matrix norm on the difference of 2 centroids of the author.


MM = Type[MatrixMetric]
CO = Type[CentroidOption]
def dis(Matrix1, Matrix2, Metric:MM, WeightVec1 = None, WeightVec2 = None):
    """
        This function returns the distance between 2 matrices, given
        the type of Metric space and the weights.
    :param Matrix1:
        A numpy matrix.
    :param Matrix2:
        A numpy matrix.
    :param WeightVec1:
        A numpy vector
    :param WeightVec2:
        A numpy vector.
    :return:
        A float.
    """
    if Metric == MatrixMetric.OneNorm:
        return np.linalg.norm(Matrix1 - Matrix2, 1)
    elif Metric == MatrixMetric.TwoNorm:
        return np.linalg.norm(Matrix1 - Matrix2)
    # elif Metric == MatrixMetric.WeightedNorm:
    #     raise RuntimeError("WeightedNorm not yet implemented. ")
    # elif Metric == MatrixMetric.HighPower2Norm:
    #     return np.linalg.norm(Matrix1**10 - Matrix2**10)
    elif Metric == MatrixMetric.Vectorized1Norm:
        return np.linalg.norm(np.matrix.ravel(Matrix1) - np.matrix.ravel(Matrix2), 1)
    else:
        raise RuntimeError("Invalid Matrix metric space. ")


class Author:
    """
    Files for an author and transitional matrix for the author.
    * Transitional Matrix classified by each files in the folder
    * For each file, there will be several transitional matrices for parts of the files.
    * All the matrices will be in the same order as the list of works.
    """
    CentroidType = CentroidOption.AggregateMatrix
    MMetricType = MatrixMetric.TwoNorm
    AMetricType = AuthorMetric.CentroidDis

    def __init__(self, dir: str, matrixfunction: Callable = get_tm27):
        """
            Create an instance of an author by specifying:
                * A directory containing all text files written by the author.
        :param dir:
            The directory of the folder.
        :param matrixfunction:
            A function you want to use to genereate the transition matrices for the authors.
        """
        FilePathList = []
        for filename in listdir(dir):
            filepath = dir + "/" + filename
            if isfile(filepath):
                FilePathList.append(filepath)

        assert len(FilePathList) > 0, f"There is no file under the directory: {dir}"

        FilePath2Lines = {}
        for f in FilePathList:
            FilePath2Lines[f] = process_text(f)
        # A map that maps the file path to array of lines containing the content of the file.
        self.__FilePathToLines = FilePath2Lines
        self.__AuthorName = dir.split("/")[-1]
        self.__TMFunction = matrixfunction
        self.__NpMatrices = None  # a list of np matrices for each works of the author
        self.__AggregateMatrix = None  # Instance of transition matrix.
        self.__AuthorItems = list(self.__FilePathToLines.items())

    def get_fp2lines(self):
        """
            A map where the key is the path of the file of an author's work, and the
            value is a list of string, representing the raw content of the
            work written by the author.
            * The text in the line is un-processed.
        :return:
        """
        return self.__FilePathToLines

    def list_of_works(self):
        return [work.split("/")[-1] for work in list(self.__FilePathToLines.keys())]

    def list_of_works_content(self):
        return list(self.__FilePathToLines.values())

    def name(self):
        return self.__AuthorName

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

    def __aggregate_matrix(self):
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

    def __average_matrix(self):
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

    def get_center(self):
        """
            Returns a matrix representing the center of the author,
            The type of matrix depends on the global options for the authors.
        :return:
            An Np matrix.
        """
        Center = None
        if Author.CentroidType == CentroidOption.AggregateMatrix:
            Center = self.__aggregate_matrix()
        elif Author.CentroidType == CentroidOption.AverageMatrix:
            Center = self.__average_matrix()
        else:
            raise RuntimeError("Unspecified Centroid Option.")
        return Center

    def work_distances(self):
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
        Center = self.get_center()
        for Writing, Matrix in zip(self.list_of_works(), self.get_matrices()):
            DistanceMap[Writing] = dis(Matrix, Center, Metric=Author.MMetricType, WeightVec1=None, WeightVec2=None)
        return DistanceMap

    def author_cloud(self):
        """
            * The average distance.
            * the standard deviations of the distance.
            * A map describing the distance from the center, mapping
            author's works to a distance represented in a float value.
        :return:
            2 items:
            1. [<average distance>, <standard deviation>]
            2. A map, string to float.
        """
        DistanceList = self.work_distances()
        Sum = 0
        Squaresum = 0
        L = len(DistanceList)
        for Distance in DistanceList.values():
            Sum += Distance
            Squaresum += Distance**2
        Sum/=L
        Squaresum/=L
        return [Sum, math.sqrt(Squaresum - Sum**2)], DistanceList

    def distance_to(self, m2):
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
        centroid = self.get_center()
        if Author.AMetricType == AuthorMetric.CentroidDis:
            return dis(centroid, m2, Metric=Author.MMetricType, WeightVec1=None, WeightVec2=None)
        temp = [dis(m1, m2, Metric=Author.MMetricType) for m1 in self.get_matrices()]
        if Author.AMetricType == AuthorMetric.MinimumDis:
            return min(temp)
        if Author.AMetricType == AuthorMetric.AverageDis: # Not sure if symmetry property is preserved.
            return sum(temp)/len(temp)
        raise("Invalid Author Metric. ")

    def __repr__(self):
        s = "-------------------AUTHOR INFO---------------------\n"
        s += f"Author's Name: {self.__AuthorName} \n"
        s += "Distances of his works from the center:\n"
        Cloud, DistanceList = self.author_cloud()
        s += f"Average Distance from the center: {Cloud[0]}\n"
        s += f"Standard deviation of the distances: {Cloud[1]}\n"
        TitleMaxLength = max(len(W) for W in DistanceList.keys())
        for Work, dis in DistanceList.items():
            s += f"{(Work+':').ljust(TitleMaxLength)} {'{:10.4f}'.format(dis)} \n"
        s += f"Matrix Norm used: {Author.MMetricType}\n"
        s += f"Centroid Matrix is: {Author.CentroidType}\n"
        s += f"Function used to generate transition matrix: {self.__TMFunction.__name__}\n"
        return s


def dis_between_authors(author1, author2):
    """
        This function returns 1 number to represent the distance between 2 author's
        works.
    :param author1:
        An instance of an Author class.
    :param author2:
        An instance of an Author class.
    :param metric
        A type of author metric. 
    :return:
        a float.
    """
    metric = Author.AMetricType
    if metric == AuthorMetric.CentroidDis:
        return author2.distance_to(author1.get_center)
    temp = [author1.distance_to(Author2Works) for Author2Works in author2.get_matrices()]
    if metric == AuthorMetric.AverageDis:
        return sum(temp)/len(temp)
    if metric == AuthorMetric.MinimumDis:
        return min(temp)
    raise RuntimeError("Invalid AuthorMetricOption")


def visualize_author(theauthor, theplot):
    """
        Function will use matplot lib to plot out the transition matrix of the author.
        It will show the plot, and it will also save the image into the root directory.
        *
    :param theauthor:
        An instance of the class Author.
    :return:
        An instance of matplot.
    """
    pass # TODO: IMPLEMENT THIS SHIT.


def main():
    pass


if __name__ == '__main__':

    pass
