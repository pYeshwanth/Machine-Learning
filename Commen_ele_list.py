def common_elements_count(list1, list2):
    return len(set(list1) & set(list2))
l1 = [1, 2, 3, 4, 5]
l2 = [3, 4, 5, 6, 7]

print("Common elements count:", common_elements_count(l1, l2))
