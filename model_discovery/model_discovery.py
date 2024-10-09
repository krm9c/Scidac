import numpy as np
import pysindy as ps

hbaromega_min = 0
hbaromega_max = 8

data = np.load("../data/processed_extrapolation.npy", allow_pickle=True)
X = data[()]["data"][hbaromega_min:hbaromega_max, 1:].T
N_Max = data[()]["Nmax"].reshape([-1])  # not sure why this dtype is messed up
N_Max = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])

differentiation_method = ps.FiniteDifference(order=2)
feature_library = ps.PolynomialLibrary(degree=1)
# feature_library = ps.FourierLibrary()
optimizer = ps.STLSQ(threshold=1e-8)
feature_names = ["x%d" % i for i in range(hbaromega_min, hbaromega_max)]
# feature_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]

model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=feature_names,
)
model.fit(X, t=N_Max)

# print(model.complexity)
# model.print()

X0 = X[:, 0]
model.simulate(X0.T, N_Max)
