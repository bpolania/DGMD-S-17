import numpy as np
from typing import List
from typing import Optional

def signal_threshold(signal: float) -> bool:
    return signal > 75

def array_prod(lst1: List[int], lst2: List[int]) -> Optional[int]:
    return sum(j*i for i,j in zip(lst1, lst2))

def collatz_steps(x: int) -> int:
    steps = 0
    if x <= 0:
      return 0
    while x != 1:
      if x%2 == 0:
        x = x/2
      else:
        x = 3*x + 1
      steps += 1
    return steps

class Animal:
    def __init__(self, name: str, color: str, age: int) -> None:
        self.name = name
        self.color = color
        self.age = age

    def generate_id(self) -> str:
        return f"{self.name}_{self.color}_{self.age}"

    def is_older_than(self, age: int) -> bool:
        return self.age > age
    
class Tiger(Animal):
    def __init__(self, name: str, color: str, age: int, top_speed: int) -> None:
      super().__init__(name, color, age)
      self.top_speed = top_speed

    def age_speed_ratio(self) -> float:
      return self.age/self.top_speed

def matrix_vector_product(A, v):
    return np.dot(A, v)

# signal_threshold

def test_signal_threshold():
    # Case where signal is less than 75
    assert not signal_threshold(float(50)), "Failed on case with signal=50"
    
    # Case where signal is exactly 75
    assert not signal_threshold(float(75)), "Failed on case with signal=75"
    
    # Case where signal is more than 75
    assert signal_threshold(float(100)), "Failed on case with signal=100"
    
    # Case where signal is a floating point number less than 75
    assert not signal_threshold(74.9), "Failed on case with signal=74.9"
    
    # Case where signal is a floating point number more than 75
    assert signal_threshold(float(75.1)), "Failed on case with signal=75.1"
    
    print("All test cases pass")

test_signal_threshold()

def test_array_prod():
    # Test case 1: Regular case with positive numbers
    assert array_prod([1, 2, 3], [4, 5, 6]) == 32, "Test case 1 failed"

    # Test case 2: Case with zero
    assert array_prod([0, 2, 3], [4, 5, 6]) == 28, "Test case 2 failed"

    # Test case 3: Case with negative numbers
    assert array_prod([-1, -2, -3], [-4, -5, -6]) == 32, "Test case 3 failed"

    # Test case 4: Case with empty lists
    assert array_prod([], []) == 0, "Test case 4 failed"

    # Test case 5: Case with unequal length lists
    try:
        array_prod([1, 2, 3], [4, 5])
    except ValueError as e:
        assert str(e) == "Input lists must have the same length", "Test case 5 failed"

    print("All test cases pass")

test_array_prod()

def test_collatz_steps():
    # Test case 1: If the input is 1, the function should return 0
    assert collatz_steps(1) == 0, "Test case 1 failed"

    # Test case 2: If the input is 16, the function should return 4 (16 -> 8 -> 4 -> 2 -> 1)
    assert collatz_steps(16) == 4, "Test case 2 failed"

    # Test case 3: If the input is 6, the function should return 8 (6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1)
    assert collatz_steps(6) == 8, "Test case 3 failed"

    # Test case 4: If the input is negative or zero, the function should return 0
    assert collatz_steps(-5) == 0, "Test case 4 failed"
    assert collatz_steps(0) == 0, "Test case 5 failed"

    assert collatz_steps(123) ==46, "Test case 3 failed"

    print("All test cases pass")

test_collatz_steps()

def test_Animal():
    # Test case 1: Checking the correct id generation
    clifford = Animal('Clifford', 'red', 5)
    assert clifford.generate_id() == 'Clifford_red_5', "Test case 1 failed"

    # Test case 2: Checking the age comparison
    assert clifford.is_older_than(4) == True, "Test case 2 failed"
    assert clifford.is_older_than(5) == False, "Test case 3 failed"
    assert clifford.is_older_than(6) == False, "Test case 4 failed"

    # Test case 3: Checking the attribute assignments
    assert clifford.name == 'Clifford', "Test case 5 failed"
    assert clifford.color == 'red', "Test case 6 failed"
    assert clifford.age == 5, "Test case 7 failed"

    print("All test cases pass")

test_Animal()

def test_Tiger():
    # Test case 1: Checking the correct id generation
    tiger = Tiger('Tony', 'orange', 10, 60)
    assert tiger.generate_id() == 'Tony_orange_10', "Test case 1 failed"

    # Test case 2: Checking the age comparison
    assert tiger.is_older_than(9) == True, "Test case 2 failed"
    assert tiger.is_older_than(10) == False, "Test case 3 failed"
    assert tiger.is_older_than(11) == False, "Test case 4 failed"

    # Test case 3: Checking the attribute assignments
    assert tiger.name == 'Tony', "Test case 5 failed"
    assert tiger.color == 'orange', "Test case 6 failed"
    assert tiger.age == 10, "Test case 7 failed"
    assert tiger.top_speed == 60, "Test case 8 failed"

    # Test case 4: Checking the age_speed_ratio method
    assert tiger.age_speed_ratio() == 10/60, "Test case 9 failed"

    print("All test cases pass")

test_Tiger()

import numpy as np
from typing import List

def matrix_vector_product(A, v):
    return np.dot(A, v)

def test_matrix_vector_product():
    A : List[List[float]] = [
        [1, 5],
        [3, 2],
        [6, 8]
    ]
    B : List[List[float]] = [
        [2, 8],
        [6, 4]
    ]
    v : List[float] = [2, 5]

    # Test case 1: Matrix-vector multiplication Av
    Av_expected = np.dot(A, v)
    assert np.allclose(matrix_vector_product(A, v), Av_expected), "Test case 1 failed"
    print(Av_expected)
    print("---")
    # Test case 2: Matrix-matrix multiplication AB
    AB_expected = np.dot(A, B)
    assert np.allclose(matrix_vector_product(A, B), AB_expected), "Test case 2 failed"
    print(AB_expected)
    print("---")
    # Test case 3: Matrix-matrix-vector multiplication ABv
    ABv_expected = np.dot(np.dot(A, B), v)
    assert np.allclose(matrix_vector_product(matrix_vector_product(A, B), v), ABv_expected), "Test case 3 failed"
    print(ABv_expected)

    print("All test cases pass")

test_matrix_vector_product()



