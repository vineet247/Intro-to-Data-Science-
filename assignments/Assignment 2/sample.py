def func1():
    a = 0
    while True:
        a += 1
        return a

def func2():
	a = 0
	while True:
		a += 1
		yield a

def func3():
    for i in range(5):
	       yield i

for f in func3():
    print (f)

f1 = func1()
f2 = func2()
for i in range(5):
    returnval = f1
    yieldval = next(f2)
