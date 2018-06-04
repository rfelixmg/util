import numpy as np


def suggest_sensible_lengthscale(x_data):
    from scipy.spatial import distance
    lengthscale = np.mean(distance.pdist(x_data, 'euclidean'))
    return lengthscale

def suggest_good_intial_inducing_points(x_data, num_inducing):
    from sklearn import cluster

    print(":: starting k-means")
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_inducing, batch_size=num_inducing*10)
    kmeans.fit(x_data)
    new_inducing = kmeans.cluster_centers_
    print(":: ending k-means")
    return new_inducing




def process_validation(dataset, datafake, options):
    from util.storage import Container
    from sklearn.cross_validation import train_test_split
    import copy

    XS_train, XS_val, ys_train, ys_val = train_test_split(dataset.train.X,
                                                          dataset.train.Y,
                                                          test_size=options.validation_split,
                                                          random_state=42)
    XU_train, XU_val, yu_train, yu_val = train_test_split(datafake.X,
                                                          datafake.Y,
                                                          test_size=options.validation_split,
                                                          random_state=42)

    ys_train = np.atleast_2d(ys_train).T
    ys_val = np.atleast_2d(ys_val).T
    yu_train = np.atleast_2d(yu_train).T
    yu_val = np.atleast_2d(yu_val).T

    dataset.train.X = np.r_[XS_train, XU_train]
    dataset.train.Y = np.r_[ys_train, yu_train]
    dataset.train.seen = Container({'X': XS_train,
                                    'Y': ys_train})
    dataset.train.unseen = Container({'X': XU_train,
                                      'Y': yu_train})

    dataset.val.X = np.r_[XS_val, XU_val]
    dataset.val.Y = np.r_[ys_val, yu_val]
    dataset.val.seen = Container({'X': XS_val,
                                  'Y': ys_val})
    dataset.val.unseen = Container({'X': XU_val,
                                    'Y': yu_val})

    if options.gp_kmeans_indpoints:
        try:
            Z_indpoints = np.load(options.gp_kmeans_indpoints)
            print(":: Centroids for Pseudo-Matrix (Z) found!!!")
        except:
            print(":: (warning) Centroids for Pseudo-Matrix (Z) not found!!!")
            xids = np.arange(0, dataset.train.X.shape[0])
            np.random.shuffle(xids)
            x_data = dataset.train.X[xids[:int(options.gp_indpoints*10)]].copy()
            Z_indpoints = suggest_good_intial_inducing_points(x_data=x_data, num_inducing=options.gp_indpoints)
            np.save(options.gp_kmeans_indpoints, Z_indpoints)

    else:
        xids = np.arange(0, dataset.train.X.shape[0])
        np.random.shuffle(xids)
        Z_indpoints = dataset.train.X[xids[:options.gp_indpoints]].copy()

    if options.gp_lengthscale:
        length = options.gp_lengthscale
    else:
        xids = np.arange(0, dataset.train.X.shape[0])
        np.random.shuffle(xids)
        x_data = dataset.train.X[xids[:int(options.gp_indpoints * 10)]].copy()
        length = suggest_sensible_lengthscale(x_data)

    return dataset, Z_indpoints, length
