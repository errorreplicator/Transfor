import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

X_train = np.array([[1,4],[3,2],[5,6],[3,4],[1,7],[5,10]])
y_train = np.array([22,1,2,3,4,15])


for pair in X_train:
    print(pair[0],tf.one_hot(pair[0],11))
    print(pair[1],tf.one_hot(pair[1],11))
    print(50*"*")



# X, y = shuffle(X_train,y_train)
#
# print(X)
# print(y)



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