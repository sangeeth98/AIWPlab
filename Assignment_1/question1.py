import math
degree = float(input("Enter angle in degrees: "))
radian = round(degree*(math.pi/180),2)
sine = round(math.sin(radian),2)
cosine = round(math.cos(radian),2)
print("Angle: ", radian, "\nSin(angle): ",sine,"\nCos(angle): ",cosine)
