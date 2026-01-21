def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        return "Matrices are not multipliable"

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

print(matrix_multiply(A, B))
