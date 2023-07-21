t = int(input())

for i in range(t):
    l = input()
    p, e = l.split()
    p = int(p)
    e = int(e)
    if p-e<10:
       print('NO')
    else:
       print('YES')