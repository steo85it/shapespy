import numpy as np


def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def angle_btw(v1, v2, axis=1, method='arctan'):
    # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    from numpy import arctan, pi, signbit
    from numpy.linalg import norm

    # print(v1.shape, v2.shape)

    u1 = v1 / norm(v1, axis=axis)[:, np.newaxis]
    u2 = v2 / norm(v2, axis=axis)[:, np.newaxis]

    if method == 'arctan':
        y = u1 - u2
        x = u1 + u2

        a0 = 2 * arctan(norm(y, axis=axis) / norm(x, axis=axis))

        signa0 = signbit(a0)
        signpia0 = signbit(np.pi - a0)

        a0 = np.where(signa0, 0., a0)
        a0 = np.where(signpia0, np.pi, a0)  # TODO check if correct

        # if (not signbit(a0)) or signbit(pi - a0):
        #     return a0
        # elif signbit(a0):
        #     return 0.0
        # else:
        #     return pi

    elif method == 'arccos':  # this seems to be a factor 2 faster (maybe more singularities?)
        a0 = np.arccos(np.einsum('ik,ik->i', u1, u2))
    else:
        print("** Accepted methods are arctan or arccos.")
        exit()

    return a0

import numpy as np


def vertex_faces(vertices, triangles):
    # Create an empty list to store vertex faces
    vertex_faces = [[] for _ in range(len(vertices))]

    # Iterate through triangles and populate vertex_faces
    for i, triangle in enumerate(triangles):
        for vertex_index in triangle:
            vertex_faces[vertex_index].append(i)

    # Convert the lists of faces to NumPy arrays
    vertex_faces = [np.array(faces) for faces in vertex_faces]

    # Determine the maximum length of arrays in the list
    max_length = max(len(arr) for arr in vertex_faces)

    # Create an empty array filled with NaN values
    result = np.full((len(vertex_faces), max_length), np.nan)

    # Populate the result array with values from the list
    for i, arr in enumerate(vertex_faces):
        result[i, :len(arr)] = arr

    return result


