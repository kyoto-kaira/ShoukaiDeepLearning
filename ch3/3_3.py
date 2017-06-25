import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(123)

d = 2
N = 10
mean = 5

#randn: 標準正規分布に従い乱数を生成する　平均0, 分散1

x1 = rng.randn(N, d) + np.array([0.0])
x2 = rng.randn(N, d) + np.array([mean, mean])
x = np.concatenate((x1, x2), axis=0) #x1, x2を縦方向(0軸方向)に連結

w = np.zeros(d)
b = 0

def step(x):
    return x > 0

def y(x):
    return step(np.dot(w, x) + b)

def t(i): #正解ラベル
    if i<N:
        return 0
    else:
        return 1

while True:
    classified = True
    for i in range(N*2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i) - y(x[i]))
        w += delta_w
        b += delta_b
        classified *= all(delta_w==0) * (delta_b==0)
    print("w:", w, " b:", b)
    if classified:
        break

plt.scatter([x[0] for x in x1], [x[1] for x in x1], marker="x")
plt.scatter([x[0] for x in x2], [x[1] for x in x2], marker="o")
line_x = np.linspace(-1, 6, 100)
line_y = -line_x * w[0]/w[1] -b/w[1]
plt.plot(line_x, line_y)
plt.show()

print(w)