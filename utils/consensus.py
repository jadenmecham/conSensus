from itertools import combinations

def uniqueSensorCombos(listOfSensors):
    "returns a list of unique sensor combinations from a list of sensors. excludes empty set, includes full set"
    combos = []
    for i in range(1, len(listOfSensors) + 1):
        for combo in combinations(listOfSensors, i):
            combos.append(list(combo))
    return combos