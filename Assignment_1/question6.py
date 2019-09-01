# Code to print table of sine, cosine and tangent values
import math
import numpy as np
val = np.arange(0,3,0.1)
i = 0
k = len(val)
print("{0:^8} {1:^8} {2:^8} {3:^8}".format("Degrees"+chr(176),"Sine","Cosine","Tangent"))

while(i<k):
    print("{0:^8.1f} {1:^8.2f} {2:^8.2f} {3:^8.2f}"
    .format(val[i],math.sin(val[i]),math.cos(val[i]),math.tan(val[i])))
    i += 1