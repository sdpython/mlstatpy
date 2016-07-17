# -*- coding: cp1252 -*-
import os.path
import copy
import matplotlib
import pylab
import random
import math


def TestROC_sort(a, b):
    if a[0] < b[0]:
        return -1
    elif a[0] == b[0]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    test = TestROC("output_sia.txt")
    print(test)

    #test.DrawROC ( [1000])

    #test.DrawROC ( [10, 100, 1000, 5000] )
    print("computing rate..............................")
    rate, inte, mmm = test.ROC_point_intervalle(
        0.1, 100, read=True, bootstrap=500)
    print("rate = \t", "%3.2f" % (rate * 100), "%")
    print("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (
        inte[0] * 100, inte[1] * 100))
    print("intervalle min,max = \t", "[%3.2f, %3.2f]" % (
        mmm[0] * 100, mmm[1] * 100))
    print("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" %
          (mmm[2] * 100, mmm[3] * 100, mmm[4] * 100))

    rate, inte, mmm = test.ROC_AUC(0.1, 100, bootstrap=200)
    print("AUC= \t", "%3.2f" % (rate))
    print("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (inte[0], inte[1]))
    print("intervalle min,max = \t", "[%3.2f, %3.2f]" % (mmm[0], mmm[1]))
    print("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" %
          (mmm[2] * 100, mmm[3] * 100, mmm[4] * 100))

    test.DrawROC([100], read=True, bootstrap=100)
