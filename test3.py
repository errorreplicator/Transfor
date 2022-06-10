import numpy as np


x = np.array([19,10,12,15,17,13,29])


print(x)
print(np.argpartition(x,3)[:3])
print(x[np.argpartition(x,3)[:3]])
print(100*"#")
print(x)
print(np.argpartition(x,-3)[-3:])
print(x[np.argpartition(x,-3)[-3:]])





# print(x)
# print(np.argpartition(x,-3)[-3:])
# print(np.argpartition(x,3))
# print(np.argpartition(x,0))
# print(np.argsort(x))
#[3,9,0,5 ,7 ,2 ,9 ,]