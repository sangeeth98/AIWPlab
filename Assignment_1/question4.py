# Input in Farenheit -> f
f = float(input("Enter temperature in farenheit: "))
# Temperature conversion to Celcius -> c
c = ((f-32)*5)/9
print("%.2f%sF = %.2f%sC" %(f,chr(176),c,chr(176)))
