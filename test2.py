import numpy as np
import tensorflow as tf

import time

X_train = np.array([
                    [1,4],
                    [3,2],
                    [5,6],
                    [3,4],
                    [1,7],
                    [5,10]
                    ])

print([0 for x in range(12)])

corp_size = 12000
one_hot_matrix = np.zeros((corp_size,corp_size))
one_hot_list =[]
for i in range(corp_size):
    one_hot_list.append([0 for x in range(corp_size)])


#************************ Vanila *******************************


start_time = time.time()

for idx,x in enumerate(one_hot_matrix):
    x[idx] = 1

end_time = time.time()

print("First run: ", end_time - start_time)

#**************************int iteration  *********************

start_time = time.time()
for x in range(1,corp_size):
    one_hot_matrix[x][x] = 1

end_time = time.time()

print("2nd run: ", end_time - start_time)
#******************** plain list ********************************
# start_time = time.time()
# for idx, x in enumerate(one_hot_list):
#     x[idx] = 1
#
# end_time = time.time()
#
# print("3rd run: ", end_time - start_time)
#


#******************** plain list ********************************
start_time = time.time()

for x in range(1,corp_size+1):
    one_hot_list[x][x] = 1

end_time = time.time()

print("4thrun: ", end_time - start_time)


# print(one_hot_list)




# print(X_train[:,[0]].reshape(6,))
# print(tf.one_hot(X_train[:,[0]].reshape(6,),11))

