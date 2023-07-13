def myPrint(x, *y, **kwargs):
    print(x, type(y), type(kwargs))
    print(y, kwargs)
    
myPrint(1, 2, 3, a=3)
