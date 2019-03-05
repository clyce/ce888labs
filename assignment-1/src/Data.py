import numpy as np


with open('data/1/X_train.txt') as data1_x_file:
    X_1_train_ = np.array([[float(x) for x in line.split()]
                           for line in data1_x_file.readlines()])
with open('data/1/y_train.txt') as data1_x_file:
    y_1_train_ = np.array([int(line)
                           for line in data1_x_file.readlines()])
with open('data/1/X_test.txt') as data1_x_file:
    X_1_test_ = np.array([[float(x) for x in line.split()]
                          for line in data1_x_file.readlines()])
with open('data/1/y_test.txt') as data1_x_file:
    y_1_test_ = np.array([int(line)
                          for line in data1_x_file.readlines()])

X_1_ = np.concatenate((X_1_train_, X_1_test_))
y_1_ = np.concatenate((y_1_train_, y_1_test_))

X_1 = [[X_1_[0]]]
y_1 = [y_1_[0]]
for i in range(1, len(y_1_)):
    if y_1_[i] == y_1_[i-1]:
        X_1[-1].append(X_1_[i])
    else:
        X_1.append([X_1_[i]])
        y_1.append(y_1_[i])

X_1_steps = max(map(lambda c: len(c), X_1))
X_1 = [np.tile(np.array(t), (1 + int(X_1_steps / len(t)), 1))[:X_1_steps]
       for t
       in X_1]

X_1 = np.array(X_1)
y_1 = np.array(y_1)
