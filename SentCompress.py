import matplotlib.pyplot as plt
import numpy as np

# Data
times = [
    [0.7601838111877441, 0.7283258438110352, 0.7653648853302002],
    [1.3353729248046875, 1.367460012435913, 1.3761591911315918],
    [1.8698029518127441, 1.8760230541229248, 1.9444069862365723],
    [2.3365108966827393, 2.3506429195404053, 2.3415539264678955],
    [2.8326361179351807, 2.8010590076446533, 2.823704957962036],
]

# Compute averages and standard deviations for error bars
means = [np.mean(level) for level in times]
std_devs = [np.std(level) for level in times]

# X-axis labels for compression levels
compression_levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(compression_levels, means, yerr=std_devs, capsize=5, color='skyblue', alpha=0.8, label='Average Time')

# Labels and title
plt.xlabel('Compression Levels')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Compression Level Times')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
