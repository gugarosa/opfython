import opfython.math.distribution as d

# Generating a bernoulli distribution
b = d.bernoulli_distribution(prob=0.5, size=100)
print(b)