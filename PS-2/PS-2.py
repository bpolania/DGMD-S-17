import numpy as np
from typing import Optional

# NumPy Core

def function1(x: float) -> np.float64:
    return 4 * np.sin(np.pi * x) + 2

def function2(x: float) -> np.float64:
    return np.log2(x) + 2 * np.log10(x**2)

def function3(x: float) -> np.float64:
    return 4 * np.exp(x) + 9 * x**2

def dot_product(v1: np.ndarray, v2: np.ndarray) -> Optional[np.float64]:
    if v1.ndim != 1 or v2.ndim != 1 or v1.shape != v2.shape:
        return None
    else:
        return np.dot(v1, v2)
    
def calculate_metric(x: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sum(np.power(x - y, 2))

def matrix_splice(A: np.ndarray, r: int, c: int) -> Optional[np.ndarray]:
    if A.ndim != 2:
        return None
    
    if not (-A.shape[0] <= r < A.shape[0]) or not (-A.shape[1] <= c < A.shape[1]):
        return None

    row_vector = A[r, :]
    column_vector = A[:, c]
    
    return row_vector + column_vector

def matrix_rotation(A: np.ndarray) -> np.ndarray:
    return np.transpose(A)[::-1]

## function1 test cases:
def test_function1():
    # Test for x = 0
    assert np.isclose(function1(0), 2.0)

    # Test for x = 0.5, where sin(pi * x) = 1
    assert np.isclose(function1(0.5), 6.0)

    # Test for x = 1, where sin(pi * x) = 0
    assert np.isclose(function1(1), 2.0)

    # Test for negative x = -0.5, where sin(pi * x) = -1
    assert np.isclose(function1(-0.5), -2.0)

    # Test for x = 1/3, an arbitrary value
    assert np.isclose(function1(1/3), 2 + 4*np.sin(np.pi/3))

    print("All function 1 tests passed")

test_function1()

## function2 test cases:
def test_function2():
    # Test for x = 1, log of 1 is always 0
    assert np.isclose(function2(1), 0)

    # Test for x = 10, log2(10) = ~3.32 and 2*log10(10^2) = 4
    assert np.isclose(function2(10), 3.321928094887362 + 4)

    # Test for x = 2, log2(2) = 1 and 2*log10(2^2) = ~1.20
    assert np.isclose(function2(2), 1 + 2*0.6020599913279624)

    print("All function 2 tests passed")

test_function2()

## function3 test cases:
def test_function3():
    # Test for x = 0, e^0 = 1
    assert np.isclose(function3(0), 4)

    # Test for x = 1, e^1 = e ~ 2.71828
    assert np.isclose(function3(1), 4 * np.exp(1) + 9)

    # Test for x = -1, e^-1 = 1/e ~ 0.367879
    assert np.isclose(function3(-1), 4 / np.exp(1) + 9)

    print("All function 3 tests passed")

test_function3()

## dot_product test cases
def test_dot_product():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    assert np.isclose(dot_product(v1, v2), 32)

    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5])
    assert dot_product(v1, v2) is None

    print("All dot product tests passed")

test_dot_product()

## calculate_metric test cases:
def test_calculate_metric():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert np.isclose(calculate_metric(x, y), 27)

    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert np.isclose(calculate_metric(x, y), 0)

    print("All calculate metric tests passed")

test_calculate_metric()

def test_matrix_splice():\
    # 1.
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert matrix_splice(A, 3, 0) is None
    assert matrix_splice(A, 3, 1) is None
    assert matrix_splice(A, 0, 3) is None
    assert matrix_splice(A, 0, -4) is None

    comparison = matrix_splice(A, 0, 0) == np.array([2,6,10])
    equal_arrays = comparison.all()
    assert (equal_arrays)
    comparison = matrix_splice(A, 1, 1) == np.array([6,10,14])
    equal_arrays = comparison.all()
    assert (equal_arrays)


    B = np.array([[1, 2, 3], [4, 5, 6]])

    # The sum of the first row is 6 and the sum of the first column is 5
    # assert np.isclose(matrix_splice(B, 0, 0), 11)

    print("All calculate matrix splice passed")

test_matrix_splice()

## test_matrix test cases:
def test_matrix_rotation():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_rotated = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
    assert np.array_equal(matrix_rotation(A), A_rotated)

    B = np.array([[1, 2], [3, 4]])
    B_rotated = np.array([[2, 4], [1, 3]])
    assert np.array_equal(matrix_rotation(B), B_rotated)

    C = np.array([])
    C_rotated = np.array([])
    assert np.array_equal(matrix_rotation(C), C_rotated)

    print("All calculate matrix rotation passed")

test_matrix_rotation()

# NumPy Sensor Fusion

def avg_sensor_fusion(sensor_data: np.ndarray) -> np.ndarray:
    return np.mean(sensor_data, axis=0)

## test_avg_sensor_fusion test cases:
def test_avg_sensor_fusion():
    sensor_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    average_data = np.array([4, 5, 6])
    assert np.array_equal(avg_sensor_fusion(sensor_data), average_data)

    sensor_data = np.array([[7, 5, 2], [9, 2, 1], [5, 5, 5], [10, 21, 14]])
    average_data = np.array([7.75, 8.25, 5.5])
    assert np.array_equal(avg_sensor_fusion(sensor_data), average_data)

    print("All test avg sensor fusion tests passed")

## odd_one_out test cases:
def odd_one_out(sensor_i: np.ndarray, sensor_data: np.ndarray, atol: float, fault_threshold: float) -> bool:
    
    average_readings = avg_sensor_fusion(sensor_data)
    
    absolute_differences = np.abs(sensor_i - average_readings)
    
    faulty_behavior_percentage = np.mean(absolute_differences > atol)
    
    return faulty_behavior_percentage >= fault_threshold

test_avg_sensor_fusion()

## test_odd_one_out test cases:
def test_odd_one_out():
    sensor_i = np.array([1, 2, 3, 4, 5])
    sensor_data = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert odd_one_out(sensor_i, sensor_data, 0.1, 0.2) == False

    sensor_i = np.array([1, 2, 3, 4, 10])
    assert odd_one_out(sensor_i, sensor_data, 0.1, 0.2) == True

    sensor_data = np.array([[7, 5, 2], [9, 2, 1], [5, 5, 5], [10, 21, 14]])
    
    sensor_i = np.array([7, 5, 3])
    assert odd_one_out(sensor_i, sensor_data, 10, 0.7) == False

    sensor_i = np.array([20, 20, 20])
    assert odd_one_out(sensor_i, sensor_data, 10, 0.7) == True


    print("All test odd one out tests passed")

test_odd_one_out()


