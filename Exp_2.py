# To implement AND OR gate using Perceptron.

def activation_fn(x):
   return 1 if x >= 0 else 0 # Step function

def perceptron(inputs, weights, bias):
    total = sum(i * w for i, w in zip(inputs, weights)) + bias
    return activation_fn(total)

# Input combinations
input_data = [(0, 0), (0, 1), (1, 0), (1, 1)]

# AND Gate
and_weights = [1, 1]
and_bias = -1.5

print("AND Gate:")

for inputs in input_data:
    output = perceptron(inputs, and_weights, and_bias)
    print(f"Input: {inputs} -> Output: {output}")

# OR Gate
or_weights = [1, 1]
or_bias = -0.5

print("\nOR Gate:")

for inputs in input_data:
    output = perceptron(inputs, or_weights, or_bias)
    print(f"Input: {inputs} -> Output: {output}")

    