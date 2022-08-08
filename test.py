def foo(*args):
    for a, b in zip(*args):
        print(f'a={a}, b={b}')


foo([1, 2, 3], [1, 3, 2])
