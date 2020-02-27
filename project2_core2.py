from project2_core import *


AUTHORS_FOLDER = "data"
from os import listdir
from os.path import isfile, isdir
from typing import Callable



class Authors:
    """
        This class models a bunch of authors.
    """
    def __init__(self, MatrixGeneratingFunction: Callable,
                 CentroidType=CentroidOption.AggregateMatrix,
                 Metric=MatrixMetric.TwoNorm):
        AuthorsFolders = []
        for d in listdir(AUTHORS_FOLDER):
            d = AUTHORS_FOLDER + "/" + d
            if isdir(d):
                AuthorsFolders.append(d)
        assert len(AuthorsFolders) > 0, "Author's Folder is empty"
        self.__AuthorList = []
        Author.CentroidType = CentroidType
        Author.MetricType = Metric
        for AuthorDir in AuthorsFolders:
            self.__AuthorList.append(Author(AuthorDir, MatrixGeneratingFunction))
        self.__AuthorNameList = [A.name() for A in self.__AuthorList]

    def change_metric(self):
        pass # TODO: IMPLEMENT THIS.

    def __repr__(self):
        s = ""
        for author in self.__AuthorList:
            s += str(author)


        N = len(self.__AuthorList)
        A = self.__AuthorList
        s += "--------------ALL AUTHORS TOGETHER---------------------"
        s += "Author's Distance matrix with triangular part printed:\n"
        s += " | " + " | ".join(self.__AuthorNameList) + " |\n"
        for I in range(N - 1):
            for J in range(I + 1, N):
                s += '{:10.4f}'.format(dis_between_authors(A[I], A[J]))
            s += "\n"
        s += f"The metric is: {Author.MetricType}"
        return s


if __name__ == "__main__":
    print(Authors(get_tm27, Metric=MatrixMetric.HighPower2Norm))

    print("===================================================")
    print(Authors(get_tm27, Metric=MatrixMetric.Vectorized1Norm))

    print("===================================================")
    print(Authors(get_2ndtm, Metric=MatrixMetric.Vectorized1Norm))


    print(Authors(get_tm27, Metric=MatrixMetric.HighPower2Norm, CentroidType=CentroidOption.AverageMatrix))

    print("===================================================")
    print(Authors(get_tm27, Metric=MatrixMetric.Vectorized1Norm, CentroidType=CentroidOption.AverageMatrix))

    print("===================================================")
    print(Authors(get_2ndtm, Metric=MatrixMetric.Vectorized1Norm, CentroidType=CentroidOption.AverageMatrix))

