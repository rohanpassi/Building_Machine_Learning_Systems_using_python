import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
	return sp.sum((f(x) - y)**2)


data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
# print(data[:10])
# print(data.shape)

x = data[:, 0]
y = data[:, 1]

# print(sp.sum(sp.isnan(y)))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# plot the (x, y) points with dots of size 10
plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i' %w for w in range(10)])
plt.autoscale(tight=True)
# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')



# Linear Model
f1 = sp.poly1d(sp.polyfit(x, y, 1))
print("Error with deg=1 %s" % error(f1, x, y))

# Polynomial with deg=2
f2 = sp.poly1d(sp.polyfit(x, y, 2))
print("Error with deg=2 %s" % error(f2, x, y))

# Polynomial with deg=3
f3 = sp.poly1d(sp.polyfit(x, y, 3))
print("Error with deg=3 %s" % error(f3, x, y))

# Polynomial with deg=10
f10 = sp.poly1d(sp.polyfit(x, y, 10))
print("Error with deg=10 %s" % error(f10, x, y))

# Polynomial with deg=53
f53 = sp.poly1d(sp.polyfit(x, y, 53))
print("Error with deg=53 %s" % error(f53, x, y))

# calculate the inflection point in hours
inflection = int(3.5 * 7 * 24)

# data before inflection point
xa = x[:inflection]
ya = y[:inflection]

# data after inflection point
xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 2))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

print("Error inflection = %f" % (fa_error + fb_error))
