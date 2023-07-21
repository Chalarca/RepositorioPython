t = int(input())
i=1


for i in range(t):

    n = int(input())

    r = int( ((n+1)*(n+2)*(n+3) /6 )-((n+1)*(n+3)/2)+( (n+1)/2 ) )

    print(str(r))