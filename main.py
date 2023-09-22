import numpy
import scipy
import csv


def getData(file_name):
    matrix = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        for row in csv_reader:
            matrix_row = [int(x) for x in row]
            matrix.append(matrix_row)

    return matrix


def getData2(file_name):
    matrix = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')

        for row in csv_reader:
            matrix_row = [float(x) for x in row]

    return matrix_row



a = getData2('Dados/a.csv')

M = getData('Dados/M.csv')
N = getData('Dados/N.csv')


alpha = 1.0  # Scalar multiplier
beta = 0.0   # Scalar multiplier for y
trans = 'N'  # No transpose of M

MN_result = scipy.linalg.blas.sgemm(alpha=1.0, a=M, b=N)
aM_result = scipy.linalg.blas.dgemv(alpha=alpha, a=M, x=a, trans=1)

print("a * M:")
print(aM_result)

print("\nM * N: ")
print(MN_result)






