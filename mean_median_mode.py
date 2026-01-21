import random

numbers = [random.randint(100, 150) for _ in range(100)]

total = 0
for value in numbers:
    total += value
mean = total / len(numbers)

numbers_sorted = sorted(numbers)
n = len(numbers_sorted)

if n % 2 == 0:
    median = (numbers_sorted[n // 2 - 1] + numbers_sorted[n // 2]) / 2
else:
    median = numbers_sorted[n // 2]

frequency = {}
for value in numbers:
    if value in frequency:
        frequency[value] += 1
    else:
        frequency[value] = 1

max_count = 0
mode = None
for key in frequency:
    if frequency[key] > max_count:
        max_count = frequency[key]
        mode = key

print("Numbers:", numbers)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
