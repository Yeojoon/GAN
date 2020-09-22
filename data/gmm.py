import numpy as np
import matplotlib.pyplot as plt


def generate_cov(eigvecs, eigvals):
    # constructs covariance matrix via PDP^-1

    # turn into numpy vecs
    eigvecs = [np.array(vec)[None] for vec in eigvecs]
    
    # generate P, D, and P inv
    P = np.concatenate(eigvecs).T
    D = np.diag(eigvals)
    Pinv = np.linalg.inv(P)
    
    # produce cov matrix
    return P @ D @ Pinv
    

def visualize_eig(eigvecs, eigvals):
    v0, v1 = eigvecs # save for later
    L0, L1 = eigvals

    Cov = generate_cov(eigvecs, eigvals)
    
    # sample
    points = np.random.multivariate_normal([0, 0], Cov, size=2048)

    # plot points and eigenvectors
    M0, M1 = L0 ** 0.5, L1 ** 0.5
    plt.scatter(points[:,0], points[:,1])
    plt.plot([0, M0 * v0[0]], [0, M0 * v0[1]], linewidth=3, \
             color="red", zorder=10)
    plt.plot([0, M1 * v1[0]], [0, M1 * v1[1]], linewidth=3, \
             color="lime", zorder=10)
    plt.axis("equal")
    plt.show()
    

class GaussianMixtureModel2D:
    CUSTOM_MEANS = [(-3, 3), (0, 0), (2, 3), (5, 1), (7, 3)]

    ADJ = 3  # adjustment factor for tuning custom covs
    
    CUSTOM_COVS = [generate_cov([[1,1],[1,-1]],[4/ADJ,1/ADJ]),
                  generate_cov([[1,-1],[1,1]],[4/ADJ,1/ADJ]),
                  generate_cov([[1,1],[1,-1]],[4/ADJ,0.6/ADJ]),
                  generate_cov([[1,1],[1,-1]],[6/ADJ,1/ADJ]),
                  generate_cov([[.05,-1],[1,.05]],[2/ADJ,2.1/ADJ])]

    CUSTOM_WEIGHTS = [1, 2, 2.5, 1, 1]


    def __init__(self, means=None, covs=None, weights=None, seed=None):
        self._means = means
        self._covs = covs
        # uses its own rng instance, separately seeded
        self._rng = np.random.RandomState(seed=seed)

        # convert weights to pdf
        total_weight = sum(weights)
        self._pdf = [weight / total_weight for weight in weights]


    # samples points from the mixture
    # NB: right now this goes point by point, so it's pretty slow, averaging
    # around ~0.1s/1k samples, although this should be fast enough
    def sample(self, num_pts=None):
        if num_pts is None:
            return self._sample()
        pts = [self._sample() for _ in range(num_pts)]
        return np.array(pts)


    # samples single value
    def _sample(self):
        mean_idx = self._rng.choice(list(range(len(self._means))), p=self._pdf)
        mean, cov = self._means[mean_idx], self._covs[mean_idx]
        return self._rng.multivariate_normal(mean, cov)


    # helper to visualize the gaussian mixture
    def display_sample(self, num_pts=None, overlay_mesh=False):
        if num_pts is None:
            num_pts = 1 << 12
        data = self.sample(num_pts)
        plt.scatter(data[:,0], data[:,1], s=3, c="C0")
        
        for mean, cov in zip(self._means, self._covs):
            (e1, e2), v = np.linalg.eig(cov)
            v1, v2 = v[:,0], v[:,1]

            (e1, v1), (e2, v2) = max([(e1,v1),(e2,v2)]), min([(e1,v1),(e2,v2)])
            plt.arrow(mean[0], mean[1], (e1 ** 0.5) * v1[0], (e1 ** 0.5) * v1[1], \
                      width=0.05, color="C3")
            plt.arrow(mean[0], mean[1], (e2 ** 0.5) * v2[0], (e2 ** 0.5) * v2[1], \
                      width=0.05, color="C1")

        if overlay_mesh:
            min_x, max_x = data[:,0].min(), data[:,0].max()
            min_y, max_y = data[:,1].min(), data[:,1].max()
            x_range = max_x - min_x
            y_range = max_y - min_y
            adj = 0.1
            min_x, max_x = min_x - (adj * x_range), max_x + (adj * x_range)
            min_y, max_y = min_y - (adj * y_range), max_y + (adj * y_range)
            x_range = max_x - min_x
            y_range = max_y - min_y
            resolution = 100
            if x_range > y_range: # x dominant
                x_res = resolution
                y_res = int(resolution * (y_range / x_range))
            else: # y dominant
                y_res = resolution
                x_res = int(resolution * (x_range / y_range))
            xi, yi = np.mgrid[:x_res, :y_res]
            xi = (x_range * xi / x_res) + min_x
            yi = (y_range * yi / y_res) + min_y
            density = np.vectorize(lambda x, y : self.get_density((x, y)))
            zi = density(xi, yi)
            plt.pcolormesh(xi, yi, zi, shading="gouraud", cmap=plt.cm.Purples, zorder=-10)
        plt.axis("equal")
        plt.show()


    # computes exact density, for use in computing kl divergence
    def get_density(self, x):
        def multivariate_pdf(mean, cov, x):
            x = np.array(x)[:,None]
            mean = np.array(mean)[:,None]
            return (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))) * \
                   np.exp((-1 / 2) * (np.linalg.solve(cov, x - mean).T @ (x - mean)))
        return sum(weight * multivariate_pdf(mean, cov, x) \
                   for weight, mean, cov in zip(self._pdf, self._means, self._covs))


    # attempts to automatically pick nice parameters for a GMM
    # tightness determines how grouped together the means will be,
    # with a smaller value resulting in more spread apart means, and
    # a larger value resulting in tightly clustered, overlapping means
    def get_nice_model(tightness=3, n_clusters=4, seed=None):
        X_BOUNDS, Y_BOUNDS = (-5, 5), (-3, 3)
        X_BOUNDS = tuple(xb * (3 / tightness) for xb in X_BOUNDS)
        Y_BOUNDS = tuple(yb * (3 / tightness) for yb in Y_BOUNDS)
        rng = np.random.RandomState(seed=seed)
        assert n_clusters >= 1
        dist = lambda p1, p2 : (p2[0] - p1[0])**2 + (p2[1] - p1[1]) ** 2
        means = [(rng.uniform(*X_BOUNDS), rng.uniform(*Y_BOUNDS))]
        for _ in range(1, n_clusters):
            best_mean = (rng.uniform(*X_BOUNDS), rng.uniform(*Y_BOUNDS))
            best_dist = min(dist(best_mean, cur_mean) for cur_mean in means)
            for _ in range(9): # try 10 times
                new_mean = (rng.uniform(*X_BOUNDS), rng.uniform(*Y_BOUNDS))
                new_dist = min(dist(new_mean, cur_mean) for cur_mean in means)
                if new_dist > best_dist:
                    best_dist = new_dist
                    best_mean = new_mean
            means.append(best_mean)
            
        covs = []
        for _ in range(n_clusters):
            main_angle = rng.uniform(-np.pi, np.pi)
            off_angle = main_angle + (np.pi / 2)
            main_mag = rng.uniform(1.1, 2.0)
            off_mag = rng.uniform(0.15, 0.85)
            covs.append(generate_cov([[np.cos(main_angle), np.sin(main_angle)],
                                      [np.cos(off_angle), np.sin(off_angle)]],
                                     [main_mag, off_mag]))
            
        weights = [rng.uniform(1, 3) for _ in range(n_clusters)]
        
        return GaussianMixtureModel2D(
            means=means,
            covs=covs,
            weights=weights,
            seed=seed)


    def get_custom_model(seed=None):
        GMM2D = GaussianMixtureModel2D
        return GMM2D(means=GMM2D.CUSTOM_MEANS,
                     covs=GMM2D.CUSTOM_COVS,
                     weights=GMM2D.CUSTOM_WEIGHTS,
                     seed=seed)

GMM2D = GaussianMixtureModel2D
