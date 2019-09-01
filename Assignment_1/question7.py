n = 20
arr = []
for i in range(1,n):
    for j in range(1,n):
        for k in range(1,n):
            if(i**2+j**2 == k**2):
                temp = sorted([i,j,k])
                if(temp not in arr):
                    arr.append(temp)
print(*arr,sep = "\n")
    
