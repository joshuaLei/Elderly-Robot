array = [(200, 300), (230, 500), (200, 540), (201, 900)]

lst = []

for i, j in array:
    lst.append(int(j))
    print(j)

max = max(lst)
min = min(lst)

val = max - min

print(val)