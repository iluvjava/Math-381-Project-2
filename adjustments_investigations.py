"""
This files investigate some of the special thing discovered
when writing the paper.


TODO: DO THIS SHIT
* Compering the Huckleburry and the prince from Twain's
writings.
"""


from core import *
import matplotlib.pyplot as plt


def f1():
    Novel1, Novel2 = "Adventures of Huckleberry Finn.txt", \
                     "The Prince and The Pauper.txt"
    Mark = Author(MARK_TWAIN, IgnoreSpecialNoun=True)
    for Work, Matrix in zip(Mark.list_of_works(), Mark.get_matrices()):
        print(Work)
        if Work == Novel1:
            Novel1 = Matrix
        if Work == Novel2:
            Novel2 = Matrix
    print("The distance between \"Adventrues of Huckleberry Finn.txt\""
          "and \"The Prince and The Pauper.txt\"is: ")
    print(dis(Novel1, Novel2, MatrixMetric.TwoNorm))
    print("Ignoring Special Nouns and using matrix 2 norm")

def f2():
    """
        Comparing thespecial nouns impact on Huckleberry Finn,
        in tm27.
    :return:
    """
    A1 = Author(MARK_TWAIN, matrixfunction=get_tm27, IgnoreSpecialNoun=True)
    A1WorkDict = A1.work_matrix_dict()
    WorkName = "Adventures of Huckleberry Finn.txt"
    FinnMatrix = A1WorkDict[WorkName]
    CentroidMatrix = A1.get_center()

    DifferenceMatrix = FinnMatrix - CentroidMatrix
    plt.imshow(DifferenceMatrix)
    plt.colorbar()
    plt.title("Difference_Matrix")
    plt.savefig("Difference_Matrix")
    plt.clf()

    plt.imshow(CentroidMatrix)
    plt.colorbar()
    plt.title("CentroidMatrix")
    plt.savefig("CentroidMatrix")
    plt.clf()

    plt.imshow(CentroidMatrix)
    plt.colorbar()
    plt.title("FinnMatrix")
    plt.savefig("FinnMatrix")
    plt.clf()


    pass

if __name__ == "__main__":
    f2()
    pass