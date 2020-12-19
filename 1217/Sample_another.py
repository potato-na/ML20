# original Sample
pnum = [x for x in range(2,100) if all([x%y!=0 for y in range(2,x)])]
print(pnum)

# map 
pnum = [x for x in range(2, 100) if all(map(lambda y: x%y!=0, range(2, x)))]
print(pnum)

# filter
pnum = [x for x in range(2, 100) if not list(filter(lambda y: x%y==0, range(2, x)))]
print(pnum)


