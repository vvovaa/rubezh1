import time
import matplotlib.pyplot as plt
import numpy as np


def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = int(dx / 2.0)
    ystep = 1 if y0 < y1 else -1
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        coord = (y, x) if steep else (x, y)
        line.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return line


def measure_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time


# Генерация миллиона точек
points = np.random.randint(0, 100, size=(10000, 4))

# Отрисовка отрезков с использованием алгоритма Брезенхема
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)

for point in points:
    line = bresenham_line(*point)
    line = np.array(line)
    plt.plot(line[:, 0], line[:, 1], label='Bresenham')

plt.title('Bresenham Algorithm')

# Отрисовка отрезков с использованием Matplotlib
plt.subplot(1, 2, 2)

for point in points:
    plt.plot([point[0], point[2]], [point[1], point[3]], label='Matplotlib')

plt.title('Matplotlib')

# Отображение графиков
plt.tight_layout()
plt.show()

# Измерение времени выполнения для отрезков
segment_times_bresenham = []
segment_times_matplotlib = []

for i in range(2, 8):
    sample_size = 10 ** i
    sample_points = points[:sample_size]

    # Измерение времени для своей реализации Брезенхема
    bresenham_time = measure_time(bresenham_line, *sample_points[0])
    segment_times_bresenham.append(bresenham_time)

    # Измерение времени для реализации matplotlib
    matplotlib_time = measure_time(plt.plot, [sample_points[0, 0], sample_points[0, 2]],
                                   [sample_points[0, 1], sample_points[0, 3]])
    segment_times_matplotlib.append(matplotlib_time)

# Построение графика для времени выполнения отрезков
plt.figure(figsize=(10, 6))
plt.plot(range(2, 8), segment_times_bresenham, label='Bresenham')
plt.plot(range(2, 8), segment_times_matplotlib, label='Matplotlib')
plt.xlabel('Log10(Sample Size)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.title('Segment Drawing Time Comparison')
plt.show()
