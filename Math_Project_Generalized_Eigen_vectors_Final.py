import numpy as np
import scipy.linalg as la
from sympy import symbols, Eq, solve


def three_Same(eigvals, A_lambdaX):
    # from here the actual eigen vector finding starts
    # v1 = [0,0,0]
    temp = A_lambdaX[0]

    x, y, z = symbols('x y z')
    v1_eq1 = Eq(temp[0][0] * x + temp[0][1] * y + temp[0][2] * z, 0)
    v1_eq2 = Eq(temp[1][0] * x + temp[1][1] * y + temp[1][2] * z, 0)
    v1_eq3 = Eq(temp[2][0] * x + temp[2][1] * y + temp[2][2] * z, 0)

    sol = solve((v1_eq1, v1_eq2, v1_eq3), (x, y, z))
    print("Note:-Make sure the eigen vector is part of the column space of A-lambda*I, if not, it is the final eigen "
          "vector we can get for the A matrix for that specific eigen value")
    print("The relation between x1,x2,x3 for first eigen vector is:-")
    print(sol)

    print("The first eigen vector will be? (for eigen value - ", eigvals[0], ")")
    v1 = []

    for i in range(0, 3):
        ele = float(input())
        v1.append(ele)

    print("The first eigen vector is:- ", v1)

    temp2 = A_lambdaX[1]

    x1, y1, z1 = symbols('x y z')
    v2_eq1 = Eq(temp2[0][0] * x1 + temp2[0][1] * y1 + temp2[0][2] * z1 - v1[0], 0)
    v2_eq2 = Eq(temp2[1][0] * x1 + temp2[1][1] * y1 + temp2[1][2] * z1 - v1[1], 0)
    v2_eq3 = Eq(temp2[2][0] * x1 + temp2[2][1] * y1 + temp2[2][2] * z1 - v1[2], 0)

    sol2 = solve((v2_eq1, v2_eq2, v2_eq3), (x1, y1, z1))
    print("The relation between x1,x2,x3 for second eigen vector is:-")
    print(sol2)

    print("The second eigen vector will be ? (for eigen value - ", eigvals[1], ")")
    v2 = []

    for i in range(0, 3):
        ele = float(input())
        v2.append(ele)
    print("The second eigen vector is:- ", v2)

    temp = A_lambdaX[2]

    x2, y2, z2 = symbols('x y z')
    v3_eq1 = Eq(temp[0][0] * x2 + temp[0][1] * y2 + temp[0][2] * z2 - v2[0], 0)
    v3_eq2 = Eq(temp[1][0] * x2 + temp[1][1] * y2 + temp[1][2] * z2 - v2[1], 0)
    v3_eq3 = Eq(temp[2][0] * x2 + temp[2][1] * y2 + temp[2][2] * z2 - v2[2], 0)

    sol = solve((v3_eq1, v3_eq2, v3_eq3), (x2, y2, z2))
    print("The relation between x1,x2,x3 for third eigen vector is:-")
    print(sol)
    print("The third eigen vector will be ? (for eigen value - ", eigvals[2], ")")

    v3 = []

    for i in range(0, 3):
        ele = float(input())
        v3.append(ele)
    print("The third eigen vector is:- ", v3)


def two_same(eigvals,A_lambdaX):
    print("2 same")
    if eigvals[0] == eigvals[1]:
        # from here the actual eigen vector finding starts
        # v1 = [0,0,0]
        temp = A_lambdaX[0]

        x, y, z = symbols('x y z')
        v1_eq1 = Eq(temp[0][0] * x + temp[0][1] * y + temp[0][2] * z, 0)
        v1_eq2 = Eq(temp[1][0] * x + temp[1][1] * y + temp[1][2] * z, 0)
        v1_eq3 = Eq(temp[2][0] * x + temp[2][1] * y + temp[2][2] * z, 0)

        sol = solve((v1_eq1, v1_eq2, v1_eq3), (x, y, z))
        print(
            "Note:-Make sure the eigen vector is part of the column space of A-lambda*I, if not, it is the final eigen "
            "vector we can get for the A matrix for that specific eigen value")
        print("The relation between x1,x2,x3 for first eigen vector is:-")
        print(sol)

        print("The first eigen vector will be? (for eigen value - ", eigvals[0], ")")
        v1 = []

        for i in range(0, 3):
            ele = float(input())
            v1.append(ele)

        print("The first eigen vector is:- ", v1)

        temp2 = A_lambdaX[1]

        x1, y1, z1 = symbols('x y z')
        v2_eq1 = Eq(temp2[0][0] * x1 + temp2[0][1] * y1 + temp2[0][2] * z1 - v1[0], 0)
        v2_eq2 = Eq(temp2[1][0] * x1 + temp2[1][1] * y1 + temp2[1][2] * z1 - v1[1], 0)
        v2_eq3 = Eq(temp2[2][0] * x1 + temp2[2][1] * y1 + temp2[2][2] * z1 - v1[2], 0)

        sol2 = solve((v2_eq1, v2_eq2, v2_eq3), (x1, y1, z1))
        print("The relation between x1,x2,x3 for second eigen vector is:-")
        print(sol2)

        print("The second eigen vector will be ? (for eigen value - ", eigvals[1], ")")
        v2 = []

        for i in range(0, 3):
            ele = float(input())
            v2.append(ele)
        print("The second eigen vector is:- ", v2)

        temp = A_lambdaX[2]

        x2, y2, z2 = symbols('x y z')
        v3_eq1 = Eq(temp[0][0] * x2 + temp[0][1] * y2 + temp[0][2] * z2, 0)
        v3_eq2 = Eq(temp[1][0] * x2 + temp[1][1] * y2 + temp[1][2] * z2, 0)
        v3_eq3 = Eq(temp[2][0] * x2 + temp[2][1] * y2 + temp[2][2] * z2, 0)

        sol = solve((v3_eq1, v3_eq2, v3_eq3), (x2, y2, z2))
        print("The relation between x1,x2,x3 for third eigen vector is:-")
        print(sol)
        print("The third eigen vector will be ? (for eigen value - ", eigvals[2], ")")

        v3 = []

        for i in range(0, 3):
            ele = float(input())
            v3.append(ele)
        print("The third eigen vector is:- ", v3)
    elif eigvals[1] == eigvals[2]:
        # from here the actual eigen vector finding starts
        # v1 = [0,0,0]
        temp = A_lambdaX[0]

        x, y, z = symbols('x y z')
        v1_eq1 = Eq(temp[0][0] * x + temp[0][1] * y + temp[0][2] * z, 0)
        v1_eq2 = Eq(temp[1][0] * x + temp[1][1] * y + temp[1][2] * z, 0)
        v1_eq3 = Eq(temp[2][0] * x + temp[2][1] * y + temp[2][2] * z, 0)

        sol = solve((v1_eq1, v1_eq2, v1_eq3), (x, y, z))
        print(
            "Note:-Make sure the eigen vector is part of the column space of A-lambda*I, if not, it is the final eigen "
            "vector we can get for the A matrix for that specific eigen value")
        print("The relation between x1,x2,x3 for first eigen vector is:-")
        print(sol)

        print("The first eigen vector will be? (for eigen value - ", eigvals[0], ")")
        v1 = []

        for i in range(0, 3):
            ele = float(input())
            v1.append(ele)

        print("The first eigen vector is:- ", v1)

        temp2 = A_lambdaX[1]

        x1, y1, z1 = symbols('x y z')
        v2_eq1 = Eq(temp2[0][0] * x1 + temp2[0][1] * y1 + temp2[0][2] * z1, 0)
        v2_eq2 = Eq(temp2[1][0] * x1 + temp2[1][1] * y1 + temp2[1][2] * z1, 0)
        v2_eq3 = Eq(temp2[2][0] * x1 + temp2[2][1] * y1 + temp2[2][2] * z1, 0)

        sol2 = solve((v2_eq1, v2_eq2, v2_eq3), (x1, y1, z1))
        print("The relation between x1,x2,x3 for second eigen vector is:-")
        print(sol2)

        print("The second eigen vector will be ? (for eigen value - ", eigvals[1], ")")
        v2 = []

        for i in range(0, 3):
            ele = float(input())
            v2.append(ele)
        print("The second eigen vector is:- ", v2)

        temp = A_lambdaX[2]

        x2, y2, z2 = symbols('x y z')
        v3_eq1 = Eq(temp[0][0] * x2 + temp[0][1] * y2 + temp[0][2] * z2 - v2[0], 0)
        v3_eq2 = Eq(temp[1][0] * x2 + temp[1][1] * y2 + temp[1][2] * z2 - v2[1], 0)
        v3_eq3 = Eq(temp[2][0] * x2 + temp[2][1] * y2 + temp[2][2] * z2 - v2[2], 0)

        sol = solve((v3_eq1, v3_eq2, v3_eq3), (x2, y2, z2))
        print("The relation between x1,x2,x3 for third eigen vector is:-")
        print(sol)
        print("The third eigen vector will be ? (for eigen value - ", eigvals[2], ")")

        v3 = []

        for i in range(0, 3):
            ele = float(input())
            v3.append(ele)
        print("The third eigen vector is:- ", v3)
    elif eigvals[0] == eigvals[2]:
        # from here the actual eigen vector finding starts
        # v1 = [0,0,0]
        temp = A_lambdaX[0]

        x, y, z = symbols('x y z')
        v1_eq1 = Eq(temp[0][0] * x + temp[0][1] * y + temp[0][2] * z, 0)
        v1_eq2 = Eq(temp[1][0] * x + temp[1][1] * y + temp[1][2] * z, 0)
        v1_eq3 = Eq(temp[2][0] * x + temp[2][1] * y + temp[2][2] * z, 0)

        sol = solve((v1_eq1, v1_eq2, v1_eq3), (x, y, z))
        print(
            "Note:-Make sure the eigen vector is part of the column space of A-lambda*I, if not, it is the final eigen "
            "vector we can get for the A matrix for that specific eigen value")
        print("The relation between x1,x2,x3 for first eigen vector is:-")
        print(sol)

        print("The first eigen vector will be? (for eigen value - ", eigvals[0], ")")
        v1 = []

        for i in range(0, 3):
            ele = float(input())
            v1.append(ele)

        print("The first eigen vector is:- ", v1)

        temp2 = A_lambdaX[1]

        x1, y1, z1 = symbols('x y z')
        v2_eq1 = Eq(temp2[0][0] * x1 + temp2[0][1] * y1 + temp2[0][2] * z1, 0)
        v2_eq2 = Eq(temp2[1][0] * x1 + temp2[1][1] * y1 + temp2[1][2] * z1, 0)
        v2_eq3 = Eq(temp2[2][0] * x1 + temp2[2][1] * y1 + temp2[2][2] * z1, 0)

        sol2 = solve((v2_eq1, v2_eq2, v2_eq3), (x1, y1, z1))
        print("The relation between x1,x2,x3 for second eigen vector is:-")
        print(sol2)

        print("The second eigen vector will be ? (for eigen value - ", eigvals[1], ")")
        v2 = []

        for i in range(0, 3):
            ele = float(input())
            v2.append(ele)
        print("The second eigen vector is:- ", v2)

        temp = A_lambdaX[2]

        x2, y2, z2 = symbols('x y z')
        v3_eq1 = Eq(temp[0][0] * x2 + temp[0][1] * y2 + temp[0][2] * z2 - v1[0], 0)
        v3_eq2 = Eq(temp[1][0] * x2 + temp[1][1] * y2 + temp[1][2] * z2 - v1[1], 0)
        v3_eq3 = Eq(temp[2][0] * x2 + temp[2][1] * y2 + temp[2][2] * z2 - v1[2], 0)

        sol = solve((v3_eq1, v3_eq2, v3_eq3), (x2, y2, z2))
        print("The relation between x1,x2,x3 for third eigen vector is:-")
        print(sol)
        print("The third eigen vector will be ? (for eigen value - ", eigvals[2], ")")

        v3 = []

        for i in range(0, 3):
            ele = float(input())
            v3.append(ele)
        print("The third eigen vector is:- ", v3)


def all_diff(eigvals,A_lambdaX):
    # from here the actual eigen vector finding starts
    # v1 = [0,0,0]
    temp = A_lambdaX[0]

    x, y, z = symbols('x y z')
    v1_eq1 = Eq(temp[0][0] * x + temp[0][1] * y + temp[0][2] * z, 0)
    v1_eq2 = Eq(temp[1][0] * x + temp[1][1] * y + temp[1][2] * z, 0)
    v1_eq3 = Eq(temp[2][0] * x + temp[2][1] * y + temp[2][2] * z, 0)

    sol = solve((v1_eq1, v1_eq2, v1_eq3), (x, y, z))
    print("Note:-Make sure the eigen vector is part of the column space of A-lambda*I, if not, it is the final eigen "
          "vector we can get for the A matrix for that specific eigen value")
    print("The relation between x1,x2,x3 for first eigen vector is:-")
    print(sol)

    print("The first eigen vector will be? (for eigen value - ", eigvals[0], ")")
    v1 = []

    for i in range(0, 3):
        ele = float(input())
        v1.append(ele)

    print("The first eigen vector is:- ", v1)

    temp2 = A_lambdaX[1]

    x1, y1, z1 = symbols('x y z')
    v2_eq1 = Eq(temp2[0][0] * x1 + temp2[0][1] * y1 + temp2[0][2] * z1, 0)
    v2_eq2 = Eq(temp2[1][0] * x1 + temp2[1][1] * y1 + temp2[1][2] * z1, 0)
    v2_eq3 = Eq(temp2[2][0] * x1 + temp2[2][1] * y1 + temp2[2][2] * z1, 0)

    sol2 = solve((v2_eq1, v2_eq2, v2_eq3), (x1, y1, z1))
    print("The relation between x1,x2,x3 for second eigen vector is:-")
    print(sol2)

    print("The second eigen vector will be ? (for eigen value - ", eigvals[1], ")")
    v2 = []

    for i in range(0, 3):
        ele = float(input())
        v2.append(ele)
    print("The second eigen vector is:- ", v2)

    temp = A_lambdaX[2]

    x2, y2, z2 = symbols('x y z')
    v3_eq1 = Eq(temp[0][0] * x2 + temp[0][1] * y2 + temp[0][2] * z2, 0)
    v3_eq2 = Eq(temp[1][0] * x2 + temp[1][1] * y2 + temp[1][2] * z2, 0)
    v3_eq3 = Eq(temp[2][0] * x2 + temp[2][1] * y2 + temp[2][2] * z2, 0)

    sol = solve((v3_eq1, v3_eq2, v3_eq3), (x2, y2, z2))
    print("The relation between x1,x2,x3 for third eigen vector is:-")
    print(sol)
    print("The third eigen vector will be ? (for eigen value - ", eigvals[2], ")")

    v3 = []

    for i in range(0, 3):
        ele = float(input())
        v3.append(ele)
    print("The third eigen vector is:- ", v3)


# Main code
print("The original matrix is:- ")
A = np.array([[6, -2, -1], [3, 1, -1], [2, -1, 2]])  # 1st matrix
# A = np.array([[3, 2, 0], [0, 3, 4], [0, 0, 3]]) # 2nd matrix
# A = np.array([[4, 1, 0], [1, 4, 1], [4, -4, 7]]) # 3rd matrix

print(A)

eigvals, eigvecs = la.eig(A)
eigvals = eigvals.real.round(0).astype(float)
print("The eigen values are:- ")
print(eigvals)

A_lambdaX = []
i = 0
while i < 3:
    temp = eigvals[i]
    A_lambdaX.append(np.array(
        [[A[0][0] - temp, A[0][1], A[0][2]], [A[1][0], A[1][1] - temp, A[1][2]],
         [A[2][0], A[2][1], A[2][2] - temp]]))
    i += 1

print("The A - lambda*X")
i = 0
while i < 3:
    print(i + 1, ")", end="")
    print(A_lambdaX[i])
    i += 1

eig_dict = {}
for item in eigvals:
    if (item in eig_dict):
        eig_dict[item] += 1
    else:
        eig_dict[item] = 1

print(eig_dict)
if eigvals[0] == eigvals[1] == eigvals[2] and (
        3 > (3 - np.linalg.matrix_rank(A_lambdaX[0], tol=None, hermitian=False))):
    three_Same(eigvals, A_lambdaX)
    print("Thank you for using the interactive eigen values and eigen vector generator")
    exit
elif (eig_dict.get(eigvals[0]) == 2 and (2 > (3 - np.linalg.matrix_rank(A_lambdaX[0])))) or (eig_dict.get(eigvals[1]) == 2 and (2 > (3 - np.linalg.matrix_rank(A_lambdaX[1])))) or (eig_dict.get(eigvals[2]) == 2 and (2 > (3 - np.linalg.matrix_rank(A_lambdaX[2])))):
    two_same(eigvals, A_lambdaX)
    print("Thank you for using the interactive eigen values and eigen vector generator")
    exit
else:
    all_diff(eigvals, A_lambdaX)
    print("Thank you for using the interactive eigen values and eigen vector generator")
    exit


    # first A matrix
    # v1 = [1,1,1]
    # v2 = [0,0,-1]
    # v3 = [0,-1,2]

    # second A matrix
    # v1 = [1,0,0]
    # v2 = [0,0.5,0]
    # v3 = [1,0,0.125]

    # third A matrix
    # v1 = [1,1,0]
    # v2 = [0,1,2]
    # v3 = [0,0,1]
