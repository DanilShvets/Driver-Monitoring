import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# xArray = np.array([])
# # xArray = []
# f1 = open("xArray.txt", "r")
# for x in f1:
#     x_arr = np.append(xArray, x[0])
# print(x_arr)

# x_arr_read = []
# f_x = open('xData.txt')
# with f_x as infile:
#     for line in infile:
#         line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
#         x_arr_read.append([float(v) for v in line.split(', ')])
# x_arr = x_arr_read[0]
# f_x.close()
# # print(x_arr)
#
# y_arr_read = []
# f_y = open('yData.txt')
# with f_y as infile:
#     for line in infile:
#         line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
#         y_arr_read.append([float(v) for v in line.split(', ')])
# y_arr = y_arr_read[0]
# f_y.close()
# # print(y_arr)

start = 5300
finish = 6000

z_data_read = []
f_z_d = open('zData.txt')
with f_z_d as infile:
    for line in infile:
        line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
        z_data_read.append([float(v) for v in line.split(', ')])
z_data = z_data_read[0]
z_data = np.array(z_data[start:finish])
f_z_d.close()


z_arr_read = []
f_z = open('zArray.txt')
with f_z as infile:
    for line in infile:
        line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
        z_arr_read.append([float(v) for v in line.split(', ')])
z_arr = z_arr_read[0]
z_arr = np.array(z_arr[start:finish])
f_z.close()
# print(z_arr)


# xAxisArray = np.linspace(0, 0.1*len(z_data[:1200]), len(z_data[:1200]))

print(finish)
print(start)
xAxisArray = np.linspace(0.1 * start, 0.1 * finish, num=(finish-start))
print(len(xAxisArray))

# xAxisArray = np.linspace(0, 0.1*len(x_arr), len(x_arr))
# print(xAxisArray)


fig, ax = plt.subplots()
# ax.plot(xAxisArray, x_arr, label='Отклонение по X', alpha=0.7)
# ax.plot(xAxisArray, y_arr, label='Отклонение по Y', alpha=0.7)
# ax.plot(xAxisArray, z_arr, label='Отклонение по Z', alpha=0.7)

ax.plot(xAxisArray, z_arr[:len(xAxisArray)], label='Резкое вождение', alpha=0.7)
ax.plot(xAxisArray, z_data, label='Спокойное вождение', alpha=0.7, color='r')

plt.rcParams['font.size'] = '16'
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Данные с гироскопа (ось Z)')
plt.xlabel('Время, с', fontsize=16)
plt.ylabel("Угловая скорость, рад/с", fontsize=16)
plt.xticks()
plt.grid(True)
# plt.ylim(-22, 25)
ax.legend()
plt.show()
