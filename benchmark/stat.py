'''
This file is used to statistic the repeated paramters.
'''

with open("sample.txt", "r") as f:
    lines = [line for line in f]

print(len(lines))
print(len(set(lines)))

totcount = len(lines)

keys = []

for ni in range(totcount-1):
    count = 0
    for nj in range(ni+1, totcount):
        if lines[ni] == lines[nj]:
            if lines[ni] not in keys:
                keys.append(lines[ni])
            count += 1
    
    if count == 1:
        print(lines[ni], count)

# print(set(keys))
print(keys)