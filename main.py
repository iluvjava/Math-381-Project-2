from core import *


AUTHORS_FOLDER = "data"
from os import listdir
from os.path import isfile, isdir
from typing import Callable, Type

class Authors:
    """
        This class models a bunch of authors.
    """
    def __init__(self, MatrixGeneratingFunction: Callable,
                 CentroidType:Type[CentroidOption]=CentroidOption.AggregateMatrix,
                 Mmtric:Type[MatrixMetric]=MatrixMetric.TwoNorm,
                 Ametric:Type[AuthorMetric]=AuthorMetric.CentroidDis,
                 IgnoreSpecialNoun:bool=False):
        AuthorsFolders = []
        for d in listdir(AUTHORS_FOLDER):
            d = AUTHORS_FOLDER + "/" + d
            if isdir(d):
                AuthorsFolders.append(d)
        assert len(AuthorsFolders) > 0, "Author's Folder is empty"
        self.__AuthorList = []
        Author.CentroidType = CentroidType
        Author.MMetricType = Mmtric
        Author.AMetricType = Ametric
        self.__IgnoreSpecialNoun = IgnoreSpecialNoun
        for AuthorDir in AuthorsFolders:
            self.__AuthorList.append(Author(AuthorDir, MatrixGeneratingFunction, IgoreSpecialNoun=IgnoreSpecialNoun))
        self.__AuthorNameList = [A.name() for A in self.__AuthorList]

    def change_mmetric(self, m: Type[MatrixMetric]):
        Author.MMetricType = m

    def change_amatric(self, m: Type[AuthorMetric]):
        Author.AMetricType = m

    def __repr__(self):
        s = ""
        for author in self.__AuthorList:
            s += str(author)


        N = len(self.__AuthorList)
        A = self.__AuthorList
        s += "--------------ALL AUTHORS TOGETHER---------------------\n"
        s += "Author's Distance matrix with triangular part printed:\n"
        s += " | " + " | ".join(self.__AuthorNameList) + " |\n"
        for I in range(N - 1):
            for J in range(I + 1, N):
                s += '{:10.4f}'.format(dis_between_authors(A[I], A[J]))
            s += "\n"
        s += f"The matrix metric is: {Author.MMetricType}\n"
        s += f"The Author metric is: {Author.AMetricType}\n"
        s += "All special nouns are IGNORED\n" if self.__IgnoreSpecialNoun else ""
        s += "======================================================"
        return s


if __name__ == "__main__":

    for f in [get_2ndtm, get_tm27]:
        theinstance = Authors(f,IgnoreSpecialNoun=True)
        for am in AuthorMetric:
            for mmt in MatrixMetric:
                theinstance.change_mmetric(mmt)
                theinstance.change_amatric(am)
                print(theinstance)
        for f in [get_2ndtm, get_tm27]:
            theinstance = Authors(f, IgnoreSpecialNoun=False)
            for am in AuthorMetric:
                for mmt in MatrixMetric:
                    theinstance.change_mmetric(mmt)
                    theinstance.change_amatric(am)
                    print(theinstance)
    s =\
"""
    ================================
    | How to interpret the output? |
    ================================
    Each authors cloud information is printed in blocks, with 
    the option printed. 
    
    All the author together has information on distances between 
    all pairs of author, for example, if the following is printed: 
    -------------------------------------------------------------
    | Charles Dickens | Mark Twain | Mark Twain's Lost Text |
    3.9300    4.3990
    3.3338
    -------------------------------------------------------------
    It means that, the distance between "Charles Dickens" and "Mark Twain" is 3.9300,
    the distance between "Charles Dickens" and "Mark Twain's Lost Text" is 4.3990;
    and the distance between "Mark Twain" and "Mark Twain's Lost text" is 3.3338.
    
    The list can be longer, each i th row is basically the distance from work 
    i to i+1, i+2 all the way to the end of the list of authors.  
"""
    print(s)
