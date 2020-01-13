import opfython.math.distribution as d

# Generating a Bernoulli distribution
b = d.generate_bernoulli_distribution(prob=0.5, size=100)
print(b)

# Generating a Lévy distribution
l = d.generate_levy_distribution(beta=0.1, size=100)
print(l)
