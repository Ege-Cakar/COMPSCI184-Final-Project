x=4
def hi():
    global x
    x=3+7
    return 5+3

def bye():
    return hi()

print(bye())
print(x)