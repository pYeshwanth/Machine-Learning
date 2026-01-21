def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))

matrix = [
    [1, 2, 3],
    [4, 5, 6]
]

print("Transpose:")
print(transpose_matrix(matrix))
