import os

for i in range(1, 101):
    if not os.path.exists(i):
        os.makedirs(str(i))
