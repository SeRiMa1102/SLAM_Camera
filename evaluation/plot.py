import matplotlib.pyplot as plt
import csv

def load_csv(filepath):
    x_vals = []
    y_vals = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2:
                continue
            x, y = int(row[0]), float(row[1])
            x_vals.append(x)
            y_vals.append(y)
    return x_vals, y_vals

# Загрузка данных
_, y1 = load_csv("/home/rinat/SLAM_Camera/displacement.csv")
_, y2 = load_csv("/home/rinat/SLAM_Camera/displacement_proccessed.csv")

# Начать с 150-го отсчёта
start_index = 150
y1_trimmed = y1[start_index:]
y2_trimmed = y2[start_index:]
y2_shifted = [val + 0 for val in y2_trimmed]

# Новая ось X: 0, 1, 2, ...
x_new = list(range(len(y1_trimmed)))

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(x_new, y1_trimmed, label="Raw displacement", linewidth=1.5)
plt.plot(x_new, y2_shifted, label="Processed (trimmed)", linewidth=1.5)
plt.xlabel("Frame (starting from 150)")
plt.ylabel("Displacement")
plt.title("Displacement from frame 150 (renumbered from 0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
