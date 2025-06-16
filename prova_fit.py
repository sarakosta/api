import functions as f
from scipy.stats import norm
from scipy.stats import kstest

G_c = f.create_network("data/gcontrolled.csv")
plant_degrees_c, pollinator_degrees_c = f.degree(G_c)

mu, sigma = norm.fit(plant_degrees_c)

print(f"Media stimata (mu): {mu:.4f}")
print(f"Deviazione standard stimata (sigma): {sigma:.4f}")

# Test KS per la distribuzione normale con i parametri stimati
D, p_value = kstest(plant_degrees_c, 'norm', args=(mu, sigma))

print(f"KS statistic: {D:.4f}, p-value: {p_value:.4f}")