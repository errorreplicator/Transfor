import numpy as np
from sklearn.utils import shuffle


X_train = np.array([[1,2],[3,4],[5,6],[3,4],[1,1],[44,22]])
y_train = np.array([22,1,2,3,4,15])

X, y = shuffle(X_train,y_train)

print(X)
print(y)



# arr = [[1,2],[45,101],[22,222], [1,2]]
#
# arr = set(tuple(i) for i in arr)
#
# print(arr)
#
# arr = [list(i) for i in arr]
# num_test = np.array([0,0])
#
# for element in test:
#     print(element)
#     if np.isin(num_test,element):
#         pass
#     else:
#         np.append(num_test,element,axis=0)
# print(num_test)