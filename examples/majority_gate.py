from qca import majority

print("Majority truth table:")
for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            print(a, b, c, "->", majority(a, b, c))
