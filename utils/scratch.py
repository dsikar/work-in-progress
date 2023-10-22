import numpy as np, matplotlib.pyplot as plt
p=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
i=0
pdf=[[1,1,1,1,1,1,1,1,1,1,1]]

pl=plt.plot(p,pdf[i])

# Basic set up - heads every time.
for i in range(1,10):
    pdf.append([])
    for j in range (0,11):
        pdf[i].append(pdf[i-1][j]*p[j])
    pl=plt.plot(p,pdf[i])
plt.show()

import numpy as np
from scipy.stats import norm

def gaussian_pdf(x, mu, sigma2):

    sigma = np.sqrt(sigma2)
    return norm.pdf(x, mu, sigma)

x_values = np.linspace(0, 10, 1000)
mu = 5
sigma2 = 4
pdf_values = gaussian_pdf(x_values, mu, sigma2)
print(pdf_values)