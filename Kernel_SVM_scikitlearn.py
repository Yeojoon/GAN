import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC


def XOR(a, b):
    
    threshold = 0
    if a >= threshold and b >= threshold:
        return int(0)
    elif a >= threshold and b < threshold:
        return int(1)
    elif a < threshold and b >= threshold:
        return int(1)
    else:
        return int(0)

    
xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                     np.linspace(-3, 3, 50))
rng = np.random.RandomState(0)
X = rng.randn(200, 2)
vec_XOR = np.vectorize(XOR)
Y = vec_XOR(X[:, 0], X[:, 1])

# fit the model
plt.figure(figsize=(10, 5))


clf = SVC()
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
Z = clf.predict(np.vstack((xx.ravel(), yy.ravel())).T)
Z = Z.reshape(xx.shape)
    
image = plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
#contours = plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=['k'])
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.colorbar(image)
plt.title('XOR with Kernel SVM', fontsize=12)
    
plt.tight_layout()
plt.show()
