from core import *


AUTHORS_FOLDER = "data"
from os import listdir
from os.path import isfile, isdir
from typing import Callable, Type

class Authors:
    """
        This class models a bunch of authors.
        self.__AuthorList:
            This is an array of authors created from the root folder, it contains
            all instance of the author, with generating function and a
            ignore case option specified by the self.prepare() function.
    """
    def __init__(self, MatrixGeneratingFunction: Callable,
                 IgnoreSpecialNoun:bool=False):
        self.__AuthorList = None
        self.prepare(MatrixGeneratingFunction, IgnoreSpecialNoun)
        self.__IgnoreSpecialNoun = IgnoreSpecialNoun

    def prepare(self, MatrixGeneratingFxn: Callable, IgnoreSpecialNoun: bool):
        """
            This function set up all the authors in the folders.
            * It buffers all the text for each of the authors.
            * Specifies Matrix generationfunction and Ignore special nouns option.
            * Please repeatedly call the function cause that will me it slower than necessary.
            * This function must be called in the __init__ to establish the field
            of the class, such as the transition matrix generating function, ignore case option, and the
            author list.

        :param MatrixGeneratingFxn:
            This is the matrix function you want to use
            to generate the transition matrices of the authors.
        :param IgnoreSpecialNoun:
            This is set to true if you want to remove
            all special nouns in the text.
        :return:
            None
        """
        AuthorsFolders = []
        for d in listdir(AUTHORS_FOLDER):
            d = AUTHORS_FOLDER + "/" + d
            if isdir(d):
                AuthorsFolders.append(d)
        assert len(AuthorsFolders) > 0, f"Author's Folder is empty: AUTHORS_fOLDER:{AUTHORS_FOLDER}"

        self.__AuthorList = []
        self.__IgnoreSpecialNoun = IgnoreSpecialNoun
        self.__GeneratingFunction = MatrixGeneratingFxn
        for AuthorDir in AuthorsFolders:
            self.__AuthorList.append(Author(AuthorDir, MatrixGeneratingFxn, IgnoreSpecialNoun=IgnoreSpecialNoun))

    def produce_results(self,
                CentroidType:Type[CentroidOption]=CentroidOption.AggregateMatrix,
                Mmtric:Type[MatrixMetric]=MatrixMetric.TwoNorm,
                Ametric:Type[AuthorMetric]=AuthorMetric.CentroidDis,):
        """
            This function takes a series of options on metric of the authors and the matrices and
            centroid type.
            * I will return a string, which is a verbal description of all the
            author's works relative to each other.
        :param CentroidType:
            An instance of the class: CentroidType
        :param Mmtric:
            An instance of the class: MatrixMetric
        :param Ametric:
            An instance of the class Ametric.
        :return:
            A string that is a report of all the works of the authors.
        """
        # Specify all the metrics for all the authors
        Author.CentroidType = CentroidType
        Author.MMetricType = Mmtric
        Author.AMetricType = Ametric
        NameList = [Author.name() for Author in self.__AuthorList]
        MaxNameLen = max(len(n) for n in NameList)
        NameList = [Name.ljust(MaxNameLen) for Name in NameList]
        Res = f"||{'|'.join(NameList)}||\n"

        DistanceTriangle = []
        for I, EachAuthor in enumerate(self.__AuthorList):
            Info1, Info2 = EachAuthor.author_cloud()
            Row = [Info1[0]]
            for J in range(I + 1, len(self.__AuthorList)):
                Row.append(dis_between_authors(EachAuthor, self.__AuthorList[J]))
            DistanceTriangle.append(Row)

        for I, EachRow in enumerate(DistanceTriangle):
            EachRow = [f"{{:<{MaxNameLen + 1}.4f}}".format(Value) for Value in EachRow]
            Res += "  " + " "*I*(MaxNameLen + 1) + "".join(EachRow) + "\n"

        Res += "".join(str(Author) for Author in self.__AuthorList)
        Res += "="*50 + "\n"
        return Res


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


def print_experiment(
        MatrixGeneratingFxn: Callable,
        IgnoreSpecialNouns: bool,
        CentroidType: CentroidOption,
        MatrixMetric: MatrixMetric,
        AuthorMetric: AuthorMetric
):
    """
        Given all the options, this function will print out everything and run the experiment based on that option.
    :param MatrixGeneratingFxn:
    :param IgnoreSpecialNouns:
    :param CentroidType:
    :param MatrixMetric:
    :param AuthorMetric:
    :return:
    """
    TheInstance = Authors(MatrixGeneratingFunction=MatrixGeneratingFxn,
                          IgnoreSpecialNoun = IgnoreSpecialNouns)
    print(TheInstance.produce_results(CentroidType=CentroidType,
                Mmtric=MatrixMetric,
                Ametric=AuthorMetric))
    return


def print_cross_compare_experiement(Author1Directory:str,
                                    Author2Directory:str,
                                    MatrixGeneratingFxn: Callable = get_tm27,
                                    IgnoreSpecialNouns: bool = True,
                                    MatrixMetric: MatrixMetric = MatrixMetric.TwoNorm):
    """
        This function takes in 2 instances of author and cross compare then and then print all
        the report on each of the individual works, average distances to the opposite centroid.
    :param Author1Directory:
        A string representing the folder that contains all the text files of the author
    :param Author2Directory:
        A string, representing another author.
    :return:
        None
    """
    assert Author2Directory != Author1Directory, "You should not cross compare the same author. "

    Author.MMetricType = MatrixMetric

    Author_A, Author_B= Author(Author1Directory, MatrixGeneratingFxn, IgnoreSpecialNouns), \
                      Author(Author2Directory, MatrixGeneratingFxn, IgnoreSpecialNouns)
    Author1, Author2 = Author_A.cross_distance_stats(Author_B)
    def print_author(A, B):
        for K, V in A[0].items():
            print(f"{K}: {V}")
        print("-"*40)
        print(f"The average distance is: {A[1]}, The Standard Deviation is: {A[2]}")

    print("Here is a list of distances when we compare each individuals works of "
         f"{Author_A.name()} to {Author_B.name()}'s centroid")
    print("+"*40)
    print_author(Author1, Author2)
    print("Here is a list of distances when we compare each individuals works of "
          f"{Author_B.name()} to {Author_A.name()}'s centroid")
    print("+"*40)
    print_author(Author2, Author1)
    print("="*40)
    print("Here is a list the cloud info for both author: ")
    print("="*40)
    print(Author_A)
    print(Author_B)
    return




if __name__ == "__main__":

    # print_experiment(MatrixGeneratingFxn=get_tm27,
    #     IgnoreSpecialNouns=True,
    #     CentroidType=CentroidOption.AggregateMatrix,
    #     MatrixMetric=MatrixMetric.TwoNorm,
    #     AuthorMetric=AuthorMetric.CentroidDis)
    print(
        print_cross_compare_experiement("data/Charles Dickens", "data/Mark Twain"))

