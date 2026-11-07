import pandas as pd

orig = pd.read_csv(r"C:\Users\sonny\OneDrive\Documents\Coding\Credit Defualt\sample_2005\sample_orig_2005.txt", sep="|", header=None, low_memory=False)
perf = pd.read_csv(r"C:\Users\sonny\OneDrive\Documents\Coding\Credit Defualt\sample_2005\sample_perf_2005.txt", sep="|", header=None, low_memory=False)

print(orig.shape, perf.shape)
print(orig.head())
print(orig.dtypes)
print(perf.head())
print(perf.dtypes)