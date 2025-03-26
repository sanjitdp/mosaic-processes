import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.fftpack import ifft2
import warnings

warnings.filterwarnings("ignore")

plt.style.use("math.mplstyle")


def sample_poisson_lines(lambda_intensity, window_width, window_height, seed=None):
    """Sample lines from a Poisson line process"""
    if seed is not None:
        np.random.seed(seed)

    radius = np.sqrt((window_width / 2) ** 2 + (window_height / 2) ** 2)

    perimeter = 2 * np.pi * radius
    expected_lines = lambda_intensity * perimeter
    n_lines = np.random.poisson(expected_lines)

    thetas = np.random.uniform(0, np.pi, n_lines)
    rs = np.random.uniform(-radius, radius, n_lines)

    window_half_width = window_width / 2
    window_half_height = window_height / 2

    lines_intersecting = []

    print(f"Filtering {n_lines} sampled lines...")
    for r, theta in tqdm(zip(rs, thetas), total=n_lines):
        nx, ny = np.cos(theta), np.sin(theta)
        dx, dy = -ny, nx
        px, py = r * nx, r * ny

        intersections = []

        if dx != 0:
            t = (-window_half_width - px) / dx
            y = py + t * dy
            if abs(y) <= window_half_height:
                intersections.append((-window_half_width, y))

        if dx != 0:
            t = (window_half_width - px) / dx
            y = py + t * dy
            if abs(y) <= window_half_height:
                intersections.append((window_half_width, y))

        if dy != 0:
            t = (-window_half_height - py) / dy
            x = px + t * dx
            if abs(x) <= window_half_width:
                intersections.append((x, -window_half_height))

        if dy != 0:
            t = (window_half_height - py) / dy
            x = px + t * dx
            if abs(x) <= window_half_width:
                intersections.append((x, window_half_height))

        if len(intersections) >= 2:
            lines_intersecting.append((r, theta))

    return lines_intersecting


def generate_gaussian_field(shape, length_scale=1.0, alpha=2.0, seed=None):
    """Generate a Gaussian random field using spectral method"""
    if seed is not None:
        np.random.seed(seed)

    ny, nx = shape

    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kxx, kyy = np.meshgrid(kx, ky)

    k_radial = np.sqrt(kxx**2 + kyy**2)
    k_radial[0, 0] = 1e-12

    power_spectrum = k_radial ** (-alpha / 2.0) * np.exp(
        -((k_radial * length_scale) ** 2)
    )
    power_spectrum[0, 0] = 0

    phase = np.random.uniform(0, 2 * np.pi, size=shape)

    fft_coeff = power_spectrum * (np.cos(phase) + 1j * np.sin(phase))

    Z = np.real(ifft2(fft_coeff))

    Z = (Z - np.mean(Z)) / np.std(Z)

    return Z


def run_simulation(
    lambda_intensity,
    window_width,
    window_height,
    length_scale=0.5,
    mean_sigma=1.0,
    seed=None,
    plot_lines=True,
    grid_size=1000,
):
    """
    Run a complete simulation with Gaussian fields filling regions

    Parameters:
    - lambda_intensity: Intensity of the Poisson line process
    - window_width, window_height: Dimensions of the window
    - length_scale: Length scale for the Gaussian field within each region
    - mean_sigma: Standard deviation for region means ~ N(0, mean_sigma²)
    - seed: Random seed for reproducibility
    - plot_lines: Whether to draw the lines on the plot
    - grid_size: Resolution of the grid

    Returns:
    - fig, ax: The matplotlib figure and axes
    """
    master_seed = seed if seed is not None else np.random.randint(0, 10000)
    np.random.seed(master_seed)

    print("Sampling Poisson lines...")
    lines = sample_poisson_lines(
        lambda_intensity, window_width, window_height, seed=master_seed
    )

    window_half_width = window_width / 2
    window_half_height = window_height / 2

    x = np.linspace(-window_half_width, window_half_width, grid_size)
    y = np.linspace(-window_half_height, window_half_height, grid_size)
    X, Y = np.meshgrid(x, y)

    print("Assigning grid points to regions...")
    signatures = np.empty(X.shape, dtype="U" + str(len(lines)))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point_x, point_y = X[i, j], Y[i, j]
            signature = ""

            for r, theta in lines:
                nx, ny = np.cos(theta), np.sin(theta)
                px, py = r * nx, r * ny
                distance = (point_x - px) * nx + (point_y - py) * ny
                signature += "1" if distance > 1e-10 else "0"

            signatures[i, j] = signature

    unique_signatures = np.unique(signatures)
    print(f"Found {len(unique_signatures)} unique regions")

    all_fields = np.zeros(X.shape)
    all_fields[:] = np.nan

    region_means = {}

    print("Generating fields for each region...")
    for idx, sig in enumerate(tqdm(unique_signatures)):
        mask = signatures == sig

        if np.sum(mask) < 5:
            print(f"Skipping tiny region with only {np.sum(mask)} points")
            continue

        region_mean = np.random.normal(0, mean_sigma)
        region_means[sig] = region_mean

        region_seed = master_seed + idx + 1
        field = generate_gaussian_field(
            shape=X.shape, length_scale=length_scale * 10, alpha=4.0, seed=region_seed
        )

        all_fields[mask] = field[mask] + region_mean

    scale = 8 / max(window_width, window_height)
    fig, ax = plt.subplots(figsize=(window_width * scale, window_height * scale))

    ax.set_xlim([-window_half_width, window_half_width])
    ax.set_ylim([-window_half_height, window_half_height])

    vmin = np.nanmin(all_fields)
    vmax = np.nanmax(all_fields)

    print("Rendering visualization...")
    c = ax.pcolormesh(X, Y, all_fields, vmin=vmin, vmax=vmax)

    if plot_lines:
        for r, theta in lines:
            nx, ny = np.cos(theta), np.sin(theta)
            dx, dy = -ny, nx
            px, py = r * nx, r * ny

            intersections = []

            if dx != 0:
                t = (-window_half_width - px) / dx
                y = py + t * dy
                if abs(y) <= window_half_height:
                    intersections.append((-window_half_width, y))

            if dx != 0:
                t = (window_half_width - px) / dx
                y = py + t * dy
                if abs(y) <= window_half_height:
                    intersections.append((window_half_width, y))

            if dy != 0:
                t = (-window_half_height - py) / dy
                x = px + t * dx
                if abs(x) <= window_half_width:
                    intersections.append((x, -window_half_height))

            if dy != 0:
                t = (window_half_height - py) / dy
                x = px + t * dx
                if abs(x) <= window_half_width:
                    intersections.append((x, window_half_height))

            if len(intersections) >= 2:
                x1, y1 = intersections[0]
                x2, y2 = intersections[1]
                ax.plot([x1, x2], [y1, y2], "k-")

    rect = Rectangle(
        (-window_half_width, -window_half_height),
        window_width,
        window_height,
        fill=False,
        color="k",
        linestyle="-",
    )
    ax.add_patch(rect)

    fig.colorbar(c, ax=ax, shrink=0.7)

    if plot_lines:
        ax.set_title(f"Mosaic process with lines ($\\lambda={lambda_intensity})$")
    else:
        ax.set_title(f"Mosaic process only ($\\lambda={lambda_intensity})$")

    ax.set_aspect("equal")

    plt.tight_layout()

    return fig, ax, region_means


if __name__ == "__main__":
    lambda_intensity = 0.3
    window_width = 10
    window_height = 8
    length_scale = 0.5
    mean_sigma = 1.5

    fig, ax, region_means = run_simulation(
        lambda_intensity=lambda_intensity,
        window_width=window_width,
        window_height=window_height,
        length_scale=length_scale,
        mean_sigma=mean_sigma,
        seed=42,
        plot_lines=True,
        grid_size=500,
    )

    region_mean_values = list(region_means.values())
    print(f"\nRegion mean statistics:")
    print(f"Number of regions: {len(region_mean_values)}")
    print(f"Mean of region means: {np.mean(region_mean_values):.4f}")
    print(f"Std dev of region means: {np.std(region_mean_values):.4f}")

    plt.savefig("simulation_with_lines.png", bbox_inches="tight")
    plt.show()

    fig, ax, _ = run_simulation(
        lambda_intensity=lambda_intensity,
        window_width=window_width,
        window_height=window_height,
        length_scale=length_scale,
        mean_sigma=mean_sigma,
        seed=42,
        plot_lines=False,
        grid_size=500,
    )

    plt.savefig("simulation_without_lines.png", bbox_inches="tight")
    plt.show()

    fig, ax, _ = run_simulation(
        lambda_intensity=lambda_intensity,
        window_width=window_width,
        window_height=window_height,
        length_scale=length_scale,
        mean_sigma=3.0,
        seed=42,
        plot_lines=True,
        grid_size=500,
    )

    plt.savefig("simulation_high_mean_sigma.png", bbox_inches="tight")
    plt.show()
