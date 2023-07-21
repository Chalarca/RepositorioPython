
while True:
    car = input()
    if car == '0 0':
        break

    set1 = input()
    set2 = input()
    set1 = set1.split()
    set2 = set2.split()
    
    set1 = [int(i) for i in set1]
    set2 = [int(i) for i in set2]
    
    resta1 = set(set1)-set(set2)
    resta2 = set(set2)-set(set1)
    union = set(set1+set2)
    print(len(resta1))
    
    inter = (union-resta1)-resta2
    print(len(inter))
    print(len(resta2))
    print(len(union))
    
    