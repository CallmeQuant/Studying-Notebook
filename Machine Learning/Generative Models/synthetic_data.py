def synthetic_data(dataset_name, n_samples=4000, noise=0.05, dtype=np_dtype):
    """
    Function to create synthetic data 
    """
    scaler = MinMaxScaler()
    if dataset_name == "Moons":
        x, _ = make_moons(n_samples, noise=noise)
    elif dataset_name == "EightGaussians":
        sq_2 = np.sqrt(2)
        c_s = 5.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / sq_2, 1. / sq_2),
                   (1. / sq_2, -1. / sq_2),
                   (-1. / sq_2, 1. / sq_2),
                   (-1. / sq_2, -1. / sq_2)]
        centers = [(c_s * x_1, c_s * x_2) for x_1, x_2 in centers]
        x = []
        for i in range(n_samples):
            p = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            p[0] += center[0]
            p[1] += center[1]
            x.append(p)
        x = np.array(x)
    elif dataset_name == "SwissRoll":
        x = make_swiss_roll(n_samples=n_samples, noise=0.7)[0][:, [0, 2]]
    elif dataset_name == "PinWheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_wheels = 7
        num_per_wheel = n_samples // num_wheels
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_wheels, endpoint=False)

        feats = np.random.randn(num_wheels * num_per_wheel, 2) * np.array([radial_std, tangential_std])
        feats[:, 0] += 1.
        labels = np.repeat(np.arange(num_wheels), num_per_wheel)
        theta = rads[labels] + rate * np.exp(feats[:, 0])
        rot_mat = np.stack([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)])
        rot_mat = np.reshape(rot_mat.T, (-1, 2, 2))
        x = np.random.permutation(np.einsum("ti, tij->tj", feats, rot_mat))

    x = scaler.fit_transform(x)
    x = x.astype(dtype)
    return x

# Params to make plot resembled to ICML format 
custom_params = {'font.family': 'serif',
 'figure.figsize': (3.25, 2.0086104634371584),
 'figure.constrained_layout.use': True,
 'figure.autolayout': False,
 'savefig.bbox': 'tight',
 'savefig.pad_inches': 0.015,
 'font.size': 18,
 'axes.labelsize': 15,
 'legend.fontsize': 6,
 'xtick.labelsize': 6,
 'ytick.labelsize': 6,
 'axes.titlesize': 18,
 'figure.dpi': 150}

# Create dataset dictionary
dataset_names = ["Moons", "EightGaussians", "SwissRoll", "PinWheel"]
datasets = {name: make_synthetic_data(name) for name in dataset_names}

plt.rcParams.update(custom_params)
fig, axes = plt.subplots(1, len(dataset_names), figsize=(6 * len(dataset_names), 6),
                        gridspec_kw={'width_ratios': [1] * len(dataset_names)})

for i in range(len(dataset_names)):
    name = dataset_names[i]
    x = datasets[name]
    axes[i].scatter(x[:, 0], x[:, 1], c="#ff7300", s=2)  #007aff
    axes[i].set(title=f"{name}", xlabel="$\mathbf{x}_1$", ylabel="$\mathbf{x}_2$")
plt.show()
