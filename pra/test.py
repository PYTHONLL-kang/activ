import matplotlib.pyplot as plt

f, axes = plt.subplots(3, 4)
f.set_size_inches((10, 7))
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# [0, 1] 위치 막대 그래프
axes[0, 1].bar(['x', 'y', 'z'], [15, 13, 18], color = ['r', 'g', 'y'], alpha = 0.4)

# [1, 3] 위치 선 그래프
axes[1][3].plot(range(5), [2, 8, 6, 3, 7])

# [2, 0] 위치 scatter 그래프(색깔 다르게 2개 겹치기)
axes[2, 0].scatter(range(5), [2, 8, 6, 3, 7], color = 'red', s = 10)
axes[2, 0].scatter([0.5, 1.5, 2.5, 3.5, 4.5], [4, 5, 4, 2, 6], color = 'purple', s = 10)

plt.show()