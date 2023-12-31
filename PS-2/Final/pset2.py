# -*- coding: utf-8 -*-
"""pset2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NTSGN_kTT62hCsRZM1SIwE9yfuLtEtPL

#**<font color='red'>MAKE A COPY TO YOUR GOOGLE DRIVE </font>**
Please make a copy, by navigating to **File > Save a copy in Drive**. This is important as you may lose your progress otherwise.

Once completed, please download as a Python (.py) file and submit on Gradescope (navigate to **File > Download > Download .py**). The file name should be **pset2.py**.


Make sure as you are working through the notebook, you run all the cells as you go along (otherwise your code may produce errors or not work properly). Additionally, you're code cells may de-load if the session has been running for too long; please periodically restart your runtime / run all your cells.

In completing the assignment, please complete the functions / classes / variables. Do not change the names of existing functions / classes / variables and their signature / input (or re-assign them) as this is what gradescope parses for. Lastly, do not import additional libraries beyond the ones provided (as gradescope will reject them).

# Problem Set 2 (110 pts)

In this problem set, you'll explore the NumPy library, sensor signals, and sensor fusion. It is highly encouraged to read the [documentation](https://numpy.org/doc/) for the NumPy library.
"""

### DO NOT CHANGE ###
# importing the libraries (do not remove or add libraries)
from typing import List, Set, Dict, Tuple, Optional
import numpy as np

"""## NumPy Core (40 pts)

This section will explore some of the core functionalities within the NumPy library.

Implement the following functions (each 3 pts).


*   `function1` (3 pt): $4\sin(\pi x)+2$
*   `function2` (3 pt): $\log_2(x)+2\log_{10}(x^2)$
*   `function3` (3 pt): $4e^x + 9x^2$

**Hint**: Use the numpy library. Documentation on [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html), [`np.log2`](https://numpy.org/doc/stable/reference/generated/numpy.log2.html), [`np.log10`](https://numpy.org/doc/stable/reference/generated/numpy.log10.html), [`np.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html), and [python math operations](https://www.programiz.com/python-programming/operators) will be helpful.
"""

def function1(x: float) -> np.float64:
    return 4 * np.sin(np.pi * x) + 2

def function2(x: float) -> np.float64:
    return np.log2(x) + 2 * np.log10(x**2)

def function3(x: float) -> np.float64:
    return 4 * np.exp(x) + 9 * x**2

"""`dot_product` (5 pts): Given two numeric numpy arrays, `v1` and `v2`, return the dot product. If the two arrays are the wrong dimension or size, return `None` instead. **Hint**: Documentation on [`np.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html), [`np.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html), and [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) will be helpful. This question is very similar to `array_prod` from a previous pset."""

def dot_product(v1: np.ndarray, v2: np.ndarray) -> Optional[np.float64]:
    if v1.ndim != 1 or v2.ndim != 1 or v1.shape != v2.shape:
        return None
    else:
        return np.dot(v1, v2)

"""`calculate_sigma_metric` (5 pts): Assume the values for `x` and `y` will be 1-dimensional numeric numpy arrays of the same length and dimension. Complete the function `calculate_metric` that calculates this metric defined below:

\begin{align}
x &= [x_1, x_2, ..., x_n]\\
y &= [y_1, y_2, ..., y_n]\\
M &= \sum_{i=0}^n (x_i - y_i)^2
\end{align}

**Hint**: Documention on [numpy operations](https://www.pluralsight.com/guides/overview-basic-numpy-operations), [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html), and [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) will be helpful. Additionally, here's a [resource](https://www.mathsisfun.com/algebra/sigma-notation.html) on the $\sum$ (sigma) operator.

**Note**: The above metric is often used to represent the total sum of error in many machine learning models.
"""

def calculate_sigma_metric(x: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sum(np.power(x - y, 2))

"""`matrix_splice` (15 pts): Given a 2D numeric numpy matrix `A`, a row index `r`, and a column index `c`, return the sum of the row vector of `A` at index `r` and the column vector of `A` at index `c` (following python indexing rules, i.e., starts at zero and supports positive and negative indexing). Edge cases to handle are invalid index values for the matrix `A` and summing vectors of different size (this happens when the width and height of the matrix are different) - the return value for these cases should be `None`.

**Example 1**:
```
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
r = 1
c = 1

matrix_splice(A, r, c) = [6, 10, 14]
```
Explanation: Get the row at index 1, which is `[4, 5, 6]`, get the column at index 1, which is `[2, 5, 8]`, and sum the vectors (i.e., along each component), so `[4 + 2, 5 + 5, 6 + 8]`, which is `[6, 10, 14]`

**Example 2**:
```
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
r = 1
c = 1

matrix_splice(A, r, c) = None
```
Explanation: Get the row at index 1, which is `[4, 5, 6]` and get the column at index 1, which is `[2, 5]`. The dimensions of the resulting vectors are different so we can't add them together. Thus, we return `None`.

**Example 3**:
```
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
r = 3
c = 1

matrix_splice(A, r, c) = None
```
Explanation: There's no row at index 3, as the index range for matrix `A` is 0 to 2. Thus, we have an invalid value for `r` and will return `None`.

**Hint**: Documentation on [numpy operations](https://www.pluralsight.com/guides/overview-basic-numpy-operations), [numpy array splicing](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/), [`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html), [`np.ndarray.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html), and [`np.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html) will be helpful. The syntax for indexing with numpy is different from normal Python.
"""

def matrix_splice(A: np.ndarray, r: int, c: int) -> Optional[np.ndarray]:
    if A.ndim != 2:
        return None

    if not (-A.shape[0] <= r < A.shape[0]) or not (-A.shape[1] <= c < A.shape[1]):
        return None

    row_vector = A[r, :]
    column_vector = A[:, c]

    return row_vector + column_vector

"""`matrix_rotation` (6 pts): Given a matrix `A`, rotate it by $-90^{\circ}$ (i.e., 90 degrees counter-clockwise). **Hint**: this can be done with indexing and / or the transpose function.

**Example**:
```
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

matrix_rotation(A) = np.array([
    [3, 6, 9],
    [2, 5, 8],
    [1, 4, 7],
])
```

"""

def matrix_rotation(A: np.ndarray) -> np.ndarray:
    return np.transpose(A)[::-1]

"""## NumPy Sensor Fusion (15 pts)

In this section, you'll build some functions for sensor fusion with numpy. Pay close attention to the shapes of the input and output.

`avg_sensor_fusion` (5 pts). You are working with a robot that is navigating a warehouse. To help it navigate better, it uses multiple RADAR sensors to determine how far objects are in front of it. Let say you have $n$ RADAR sensors collecting data every second up to time $t$ seconds. Your `sensor_data` will be a `n` by `t` numpy matrix (more precisely, it has shape `(n, t)`).

You want to create a function `avg_sensor_fusion` that does sensor fusion by averaging the sensor readings at each time step (just simple average, no special weights). Your final result will be a vector of length `t` representing the average sensor readings at each time step (more precisely, it's shape should be `(t,)`). Implement the function `avg_sensor_fusion`.

**Hint**: Pay attention to the shape of the input / output. Use [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and the `axis` parameter.
"""

def avg_sensor_fusion(sensor_data: np.ndarray) -> np.ndarray:
    return np.mean(sensor_data, axis=0)

"""`odd_one_out` (10 pts). You are working with a robot that is navigating a warehouse. To help it navigate better, it uses multiple RADAR sensors to determine how far objects are in front of it. Let say you have $n$ RADAR sensors collecting data every second up to time $t$ seconds.

In your regular maintenance of the robot, you regularly assess the sensor data to see if any of the sensors are not working properly. There are multiple methods by which you can deem if a sensor is an odd one out. For this problem, we'll compare a single, potentially faulty sensor, called `sensor_i`, whose data is represented by a vector of size `t` (more precisely, has shape `(t,)`), against a collection of `n` working sensors, called `sensor_data`, whose data is represented by a `n` by `t` numpy matrix (more precisely, it has shape `(n, t)`). At each time step, we'll compare the average of the `n` working sensors reading with the value collected by the single, potentially faulty sensor. If the absolute delta or absolute difference between the average and the individual sensor reading is greater than some tolerance `atol`, we will count that as a faultly behavior for that time step; if it's within the tolerance `atol`, then we will count that as correct behavior for that time step. If the percentage of the time that we see faulty behavior is greater than or equal to some fault threshold `fault_threshold` (which will be some number between 0 to 1), then we will deem that the individual sensor is faulty (return `True`), otherwise the sensor will be deemed functional (return `False`). Implement this in the function `odd_one_out`.
"""

def odd_one_out(sensor_i: np.ndarray, sensor_data: np.ndarray, atol: float, fault_threshold: float) -> bool:
    average_readings = avg_sensor_fusion(sensor_data)
    absolute_differences = np.abs(sensor_i - average_readings)
    faulty_behavior_percentage = np.mean(absolute_differences > atol)

    return faulty_behavior_percentage >= fault_threshold

"""## Interpreting IMU Data (20 pts)

In this section, you'll relate IMU data to potential situations that an autonomous vehicle might use to navigate on the road.

For IMU data, it's important to know the orientation. For this section, please refer to this diagram from the Society of Automotive Engineers (SAE) to see the orientation - this orientation will not change.

![](https://drive.google.com/uc?export=view&id=1rfJjE70Zbp78Q95roTdQ_Arp7JFxFLNo)
<center>Diagram showing vehicle axes of motion from SAE J670</center>

The above diagram provide the X, Y, and Z axes for the accelerometer. For the gyroscope, it is:
- Roll is along the X-axis (Gryoscope X)
- Pitch is along the Y-axis (Gryoscope Y)
- Yaw is along the Z-axis (Gryoscope Z)

Some additional notes:
- The positive direction for the accelerometer is indicated by direction of the arrowhead:
    - Forward is the positive direction for the X-axis
    - Left is the positive direction for the Y-axis
    - Up is the positive direction for the Z-axis
- The positive direction for rotation / gyroscope is in the counter-clockwise direction (where the clock's face is in the direction of the arrowhead):
    - For the X-axis, the positive direction is counter-clockwise when you look at the car from the front
    - For the Y-axis, the positive direction is counter-clockwise when you look at the car from the left side
    - For the Z-axis, the positive direction is counter-clockwise when you look at the car from the top
- A positive accelerometer output would indicate the car is increasing speed in the forward direction or slowing down in the reverse direction
- A zero accelerometer output would indicate the car is moving at constant speed
- A negative accelerometer output would indicate the car is increasing speed in the reverse direction or slowing down in the forward direction
- A positive gyroscope output would indicate the car is rotating in the positive direction
- A zero gyroscope output would indicate the car is not rotating
- A negative gyroscope output would indicate the car is rotating in the negative direction

You have data from the IMU sensors on four autonomous cars. You plot each of the IMU measurements (accelerometer and gyroscope along the X, Y, and Z axis over time) from the four cars, with each column of graphs representing one of the four vehicles. The columns of graphs are labelled `A`, `B`, `C`, and `D`. Also, we use `ACC` as shorthand for the accelerometer data and `GYRO` as shorthand for the gyroscope data. The black, horizontal line represents the value zero (so values above it are positive, values below it are negative). Each car experienced a different situation as described below - based on the situation, match each car to their respective IMU plots. The answers should be one of the choices amongst `A`, `B`, `C`, or `D` as capital string.  

**Hint**: Think about how a car would move and rotate along the various axes and how it would look over time.
"""

### DO NOT CHANGE ###
#@title IMU Measurement Data
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    t = 100
    time_vec = np.array(range(0, t))
    rng = np.random.default_rng(seed=4)
    fig, ax = plt.subplots(6, 4, figsize=(30, 20))

    plot_to_ax = {}
    for i, option in enumerate(["A", "B", "C", "D"]):
        for j, imu_measurement in enumerate(["ACC X", "ACC Y", "ACC Z", "GYRO X", "GYRO Y", "GYRO Z"]):
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].set_xlim((0, t-1))
            ax[j, i].set_ylim((-5, 5))
            ax[j, i].set_title(f"{option}: {imu_measurement}", fontsize=24)
            ax[j, i].axhline(y = 0, c = "black", linewidth = 0.5)
            plot_to_ax[(option, imu_measurement)] = ax[j, i]

    # A: This car hit a speed bump while traveling at fast speed.
    a_acc_x = rng.normal(1.5, 0.25, t)
    a_acc_y = rng.normal(0, 0.5, t)
    a_acc_z = np.concatenate([
        rng.normal(0, 0.5, 25),
        np.linspace(0, 3.5, 5) + rng.normal(0, 0.1, 5),
        rng.normal(3, 0.25, 5),
        np.linspace(3, -1, 10) + rng.normal(0, 0.25, 10),
        rng.normal(-0.6, 0.25, 25),
        rng.normal(0, 0.5, 30)
        ])
    a_gyro_x = rng.normal(0, 0.5, t)
    a_gyro_y = np.concatenate([
        rng.normal(0, 0.5, 25),
        np.linspace(0, -4, 5) + rng.normal(0, 0.2, 5),
        rng.normal(-3.5, 0.25, 5),
        np.linspace(-3.75, 1, 10) + rng.normal(0, 0.25, 10),
        rng.normal(0.5, 0.25, 25),
        rng.normal(0, 0.5, 30)
        ])
    a_gyro_z = rng.normal(0, 0.25, t)

    plot_to_ax[("A", "ACC X")].plot(time_vec, a_acc_x, c = 'r', linewidth = 3)
    plot_to_ax[("A", "ACC Y")].plot(time_vec, a_acc_y, c = 'r', linewidth = 3)
    plot_to_ax[("A", "ACC Z")].plot(time_vec, a_acc_z, c = 'r', linewidth = 3)
    plot_to_ax[("A", "GYRO X")].plot(time_vec, a_gyro_x, c = 'r', linewidth = 3)
    plot_to_ax[("A", "GYRO Y")].plot(time_vec, a_gyro_y, c = 'r', linewidth = 3)
    plot_to_ax[("A", "GYRO Z")].plot(time_vec, a_gyro_z, c = 'r', linewidth = 3)

    # B: This car is moving at a constant speed on a straight highway.
    b_acc_x = rng.normal(0, 0.2, t)
    b_acc_y = rng.normal(0, 0.5, t)
    b_acc_z = rng.normal(0, 0.25, t)
    b_gyro_x = rng.normal(0, 0.2, t)
    b_gyro_y = rng.normal(0, 0.3, t)
    b_gyro_z = rng.normal(0, 0.4, t)

    plot_to_ax[("B", "ACC X")].plot(time_vec, b_acc_x, c = 'r', linewidth = 3)
    plot_to_ax[("B", "ACC Y")].plot(time_vec, b_acc_y, c = 'r', linewidth = 3)
    plot_to_ax[("B", "ACC Z")].plot(time_vec, b_acc_z, c = 'r', linewidth = 3)
    plot_to_ax[("B", "GYRO X")].plot(time_vec, b_gyro_x, c = 'r', linewidth = 3)
    plot_to_ax[("B", "GYRO Y")].plot(time_vec, b_gyro_y, c = 'r', linewidth = 3)
    plot_to_ax[("B", "GYRO Z")].plot(time_vec, b_gyro_z, c = 'r', linewidth = 3)

    # C: This car had to make a sharp left turn at an intersection.
    c_acc_x = np.concatenate([
        rng.normal(0, 0.2, 10),
        rng.normal(0, 0.5, 15),
        rng.normal(-0.75, 0.5, 10),
        rng.normal(-2.25, 0.5, 10),
        rng.normal(-0.75, 0.5, 10),
        rng.normal(0, 0.5, 15),
        rng.normal(0, 0.2, 30),
        ])
    c_acc_y = np.concatenate([
        rng.normal(0, 0.75, 25),
        np.linspace(0, 4, 10) + rng.normal(0, 0.5, 10),
        rng.normal(3.5, 0.35, 10),
        np.linspace(4, -0.05, 15) + rng.normal(0, 0.75, 15),
        rng.normal(0.5, 0.25, 25),
        rng.normal(0, 0.5, 15)
        ])
    c_acc_z = rng.normal(0, 0.75, t)
    c_gyro_x = rng.normal(0, 1, t)
    c_gyro_y = rng.normal(0, 0.3, t)
    c_gyro_z = np.concatenate([
        rng.normal(0.5, 0.5, 25),
        np.linspace(0, 2.5, 10) + rng.normal(0, 0.5, 10),
        rng.normal(2.5, 0.35, 10),
        np.linspace(3, -0.5, 20) + rng.normal(0, 0.75, 20),
        rng.normal(0.5, 0.25, 20),
        rng.normal(0, 0.5, 15)
        ])

    plot_to_ax[("C", "ACC X")].plot(time_vec, c_acc_x, c = 'r', linewidth = 3)
    plot_to_ax[("C", "ACC Y")].plot(time_vec, c_acc_y, c = 'r', linewidth = 3)
    plot_to_ax[("C", "ACC Z")].plot(time_vec, c_acc_z, c = 'r', linewidth = 3)
    plot_to_ax[("C", "GYRO X")].plot(time_vec, c_gyro_x, c = 'r', linewidth = 3)
    plot_to_ax[("C", "GYRO Y")].plot(time_vec, c_gyro_y, c = 'r', linewidth = 3)
    plot_to_ax[("C", "GYRO Z")].plot(time_vec, c_gyro_z, c = 'r', linewidth = 3)

    # D: This car had to brake suddenly to avoid a potential car crash.
    d_acc_x = np.concatenate([
        rng.normal(0, 0.2, 10),
        rng.normal(0, 0.5, 15),
        rng.normal(-3.5, 0.5, 10),
        rng.normal(-2.25, 0.5, 10),
        rng.normal(-3.75, 0.25, 10),
        rng.normal(0, 0.5, 15),
        rng.normal(0, 0.2, 30),
        ])
    d_acc_y = rng.normal(0, 0.5, t)
    d_acc_z = rng.normal(0, 0.5, t)
    d_gyro_x = rng.normal(0, 0.5, t)
    d_gyro_y = rng.normal(0, 0.5, t)
    d_gyro_z = rng.normal(0, 0.5, t)

    plot_to_ax[("D", "ACC X")].plot(time_vec, d_acc_x, c = 'r', linewidth = 3)
    plot_to_ax[("D", "ACC Y")].plot(time_vec, d_acc_y, c = 'r', linewidth = 3)
    plot_to_ax[("D", "ACC Z")].plot(time_vec, d_acc_z, c = 'r', linewidth = 3)
    plot_to_ax[("D", "GYRO X")].plot(time_vec, d_gyro_x, c = 'r', linewidth = 3)
    plot_to_ax[("D", "GYRO Y")].plot(time_vec, d_gyro_y, c = 'r', linewidth = 3)
    plot_to_ax[("D", "GYRO Z")].plot(time_vec, d_gyro_z, c = 'r', linewidth = 3)

    plt.tight_layout()

"""Note: the data displayed is for demonstrative purposes - real world IMU data may be more noisy and harder to parse. Furthermore, if the plots are not visible, make sure to run the code cell associated with "IMU Measurement Data"

`car1` (5 pts). This car is moving forward at a constant speed on a straight highway.

`car2` (5 pts). This car is driving straight and had to make a sharp left turn at an intersection.

`car3` (5 pts). This car was driving straight and hit a speed bump while traveling at constant, fast speed.

`car4` (5 pts). This car was driving straight and had to brake suddenly to avoid a potential car crash.
"""

#@title Your Answers
car1 = "B" #@param ["", "A", "B", "C", "D"]

car2 = "A" #@param ["", "A", "B", "C", "D"]

car3 = "C" #@param ["", "A", "B", "C", "D"]

car4 = "D" #@param ["", "A", "B", "C", "D"]

"""## Drone Sensor Fusion (30 pts)

In this section, you'll solve some problems involving LiDAR and sensor fusion.

**Context**

You have a drone that is surveying a national park to assess how high different natural structures, such as trees, mountains, and rock formations, are. The drone is equipped with various sensors. As you work through the problems, you will be expose to how different sensors can work in tandem to provide measurements about the drone's environment.

For this section, please use $c = 3 * 10^8$ meters per second for the speed of light in your calculations. In terms of numeric precision, please have 4 decimal point precision (if applicable) (e.g., 3.1415 has 4 decimal point precision).

`measurement1` (5 pts). You have a drone that is leveled perfectly / parallel with the ground. The ground below the drone has no structures (just the ground / grass). The LiDAR sensor shoots a pulse of light straight down and receieves the light back in 0.0000015 seconds. How far is the drone from the ground (in meters)?

**Hint**: See the below graphic - the solid, red line represents the pulse from the LiDAR sensor. From physics, we can model distance travelled as speed multiplied by time taken. Remember, that the light for LiDAR needs to travel the distance twice, once to the ground and then back to the drone.

![](https://drive.google.com/uc?export=view&id=1kL9Gai-YiURi9kQuOTBLeebKh5OFIt4L)
"""

#@title Measurement 1
measurement1 = "225" #@param {type:"string"}

"""`measurement2` (5 pts). You have a drone that is leveled perfectly / parallel with the ground. The ground below the drone has a tree. The LiDAR sensor shoots a pulse of light straight down and receieves the light back in 0.00000066 seconds. You can use a similar calculation from previous part to determine how far the drone is from the tree (in meters).

However, you are more interested in determining the height of the tree. Thus, you can pair your LiDAR reading with GPS data. The GPS data will tell you the drone's position as an $(X, Y, Z)$ coordinate (the coordinate position will be relative to the drone's launchpoint on the ground which will serve as the $(0, 0, 0)$ coordinate and its units are meters). The GPS sensor indicates the drone is at position $(1000, 500, 175)$.

How tall is the tree below the drone (in meters)?

**Hint**: See the below graphic - the solid, red line represents the pulse from the LiDAR sensor. The $Z$-coordinate from the GPS tells you the height of the drone from the ground. With the distance between the drone and the tree and the height of the drone from the ground, you can calculate the height of the tree.

![](https://drive.google.com/uc?export=view&id=1gc-RmYvWTvHkPeFTjcDeAjDLqjL0gM-u)
"""

#@title Measurement 2
measurement2 = "76" #@param {type:"string"}

"""`measurement3` (10 pts). You have a drone that is leveled perfectly / parallel with the ground. The ground below the drone has trees. The LiDAR sensor rotates where it sends the light beam / pulse to measure the forest side to side. The LiDAR shoots a pulse of light to the left at an angle $\theta = 60°$, hitting a tree, and receieves the light back in 0.0000014 seconds. The GPS sensor indicates the drone is at position $(2000, 750, 195)$.

How tall is the tree that the LiDAR pulse hit (in meters)?

**Hint**: See the below graphic - the solid, red line represents the pulse from the LiDAR sensor; you're interested in the height of the tree the pulse hit. This problem builds on the calculations from the previous parts. However, now the distance derived from the LiDAR is at an angle, but you can get the vertical component of it by using the properties of right triangles (via sine, cosine, and/or tangent) - here's a [refresher tutorial](https://www.mathsisfun.com/algebra/trig-finding-side-right-triangle.html).

![](https://drive.google.com/uc?export=view&id=1InASHqUzFdqKCBnRx-JHyzHn7EzS4AYE)
"""

#@title Measurement 3
measurement3 = "90" #@param {type:"string"}

"""`measurement4` (10 pts). Your drone is experiencing some turbulence - the IMU sensor on board was able to detect some rotation from the drone. The drone is now at an angle $\varphi = 36°$ to the right (as shown below). The ground below the drone has trees. The LiDAR shoots a pulse of light to the left at an angle $\theta =24°$ (relative to it's new orientation), hitting a tree, and receieves the light back in 0.0000024 seconds. The GPS sensor indicates the drone is at position $(-100, 70, 240)$.

Combining information from the LiDAR, GPS, and IMU, how tall is the tree that the LiDAR pulse hit (in meters)?

**Hint**: See the below graphic - the solid, red line represents the pulse from the LiDAR sensor; you're interested in the height of the tree the pulse hit. This problem builds on the calculations from the previous parts. However, now the distance derived from the LiDAR is at an angle, both from the angle the LiDAR shot the pulse and the drone body itself from turbulence. You still want the vertical component of it by using the properties of right triangles (via sine, cosine, and/or tangent) but now you have to factor in the angle of the drone's orientation. The pink dash line and the non-vertical black dash are actually parallel / along the same axis of the drone - with this information and the properties of angles along parallel lines, you should be able to create a right triangle to properly model the situation. Here's a refresher on the [properties of angles along parallel lines](https://thirdspacelearning.com/gcse-maths/geometry-and-measure/angles-in-parallel-lines/#:~:text=re%20still%20stuck.-,What%20are%20angles%20in%20parallel%20lines%3F,angle%20around%20the%20intersecting%20transversal.)

![](https://drive.google.com/uc?export=view&id=13sT9ruL0orrVuJOzn-nDKv9OZ54FlmRD)
"""

#@title Measurement 4
measurement4 = "60" #@param {type:"string"}

"""## Problem Set Survey (5 pts)

Please fill out the survey questions (the first five are each worth 1 point; the last one is optional).


1.   `TIME` (1 pt): approximately how many hours did you spend on the problem set? Please use decimals to express partial hours (e.g., a value of `2.5` means two and half hours).
2.   `DIFFICULTY` (1 pt): on a scale of 1-10, how difficult was this problem set with 1 being very easy and 10 being very hard?
3.   `FAVORITE_PART` (1 pt): What was your favorite topic / learning from the unit (i.e., between the last pset and this pset)? This should contain at least 20 words.
4.   `WENT_WELL` (1 pt): What went well? Describe what went well with the course so far (this can be about the lecture, assignments, and/or other course content). This should contain at least 20 words.
5.   `CHALLENGING` (1 pt): What was challenging? Describe what was challenging or didn't go well with the course so far (this can be about the lecture, assignments, and/or other course content). This should contain at least 20 words.
6.   `COMMENTARY` (0 pt): If there is anything else you'd like to share with course staff, please add it here. If not, no need to change / edit the default string.
"""

#@title Problem Set Survey Questions
TIME = "3" #@param {type: "string"}

DIFFICULTY = "6" #@param ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

FAVORITE_PART = "My favorite part was calculating the height of the trees based on the angles and the length of one of the sides" #@param {type:"string"}

WENT_WELL = "I think the use of numpy was the part that went better because I have previous knowledge of numpy for dealing with datasets" #@param {type:"string"}

CHALLENGING = "I think that the most challenging part was remembering and applying the formulas to calculate the side of a triangle having only the angles and one other side" #@param {type:"string"}

COMMENTARY = "" #@param {type:"string"}

"""**<font color='red'>To submit, please download as a Python (.py) file and submit on Gradescope (navigate to File > Download > Download .py). Please use the correct file name and comment out any test / extraneous code to avoid any compile and parser issues </font>**"""