import matplotlib.pyplot as plt

data =[]
with open("infer_time.txt") as f:
    data.append(f.readline())
x = range(1, len(data) + 1)

# Plotting the data
plt.plot(x, data)

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Plot')

# Save the plot as an image
plt.savefig('sample_plot.png')

# Display the plot (optional)
plt.show()
