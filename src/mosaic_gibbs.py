import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
from scipy.fftpack import ifft2
import warnings

warnings.filterwarnings("ignore")

plt.style.use("math.mplstyle")


def is_line_intersecting(r, theta, window_width, window_height):
    """Check if a line with parameters (r, theta) intersects the window."""
    window_half_width = window_width / 2
    window_half_height = window_height / 2

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

    return len(intersections) >= 2


def calculate_line_distance(line1, line2, window_width, window_height):
    """
    Calculate a distance measure between two lines.

    Parameters:
    - line1, line2: Tuples of (r, theta) line parameters
    - window_width, window_height: Dimensions of the window

    Returns:
    - Distance measure between the lines
    """
    r1, theta1 = line1
    r2, theta2 = line2

    angle_diff = np.abs(np.mod(theta1 - theta2, np.pi))
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff

    if angle_diff < 0.1:
        d_r = np.abs(r1 - r2)

        cos_angle = np.cos(angle_diff) if angle_diff > 0 else 1.0

        return d_r / cos_angle

    return np.abs(r1 - r2) + angle_diff * min(window_width, window_height) / np.pi


def pairwise_interaction(line1, line2, beta, window_width, window_height):
    """
    Calculate pairwise interaction energy between two lines.

    Parameters:
    - line1, line2: Tuples of (r, theta) line parameters
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - window_width, window_height: Dimensions of the window

    Returns:
    - Interaction energy
    """
    r1, theta1 = line1
    r2, theta2 = line2

    angle_diff = np.abs(np.mod(theta1 - theta2, np.pi))
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff

    distance = calculate_line_distance(line1, line2, window_width, window_height)

    if distance < 0.001:
        distance = 0.001

    if angle_diff < 0.1:
        interaction = 1.0 / (distance**1.5)
    else:
        interaction = np.exp(-distance)

    return beta * interaction


def total_energy(lines, beta, window_width, window_height):
    """
    Calculate total energy of the line configuration.

    Parameters:
    - lines: List of (r, theta) line parameters
    - beta: Interaction parameter
    - window_width, window_height: Dimensions of the window

    Returns:
    - Total energy of the configuration
    """
    energy = 0.0
    n = len(lines)

    for i in range(n):
        for j in range(i + 1, n):
            energy += pairwise_interaction(
                lines[i], lines[j], beta, window_width, window_height
            )

    return energy


def sample_gibbs_lines(
    lambda_intensity,
    beta,
    window_width,
    window_height,
    n_iterations=1000,
    seed=None,
    burn_in=200,
):
    """
    Sample lines from a Gibbs line process with interaction parameter beta.

    Parameters:
    - lambda_intensity: Base intensity of the process
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - window_width, window_height: Dimensions of the window
    - n_iterations: Number of MCMC iterations
    - seed: Random seed for reproducibility
    - burn_in: Number of initial iterations to discard

    Returns:
    - List of (r, theta) parameters for the sampled lines
    """
    if seed is not None:
        np.random.seed(seed)

    radius = np.sqrt((window_width / 2) ** 2 + (window_height / 2) ** 2)

    perimeter = 2 * np.pi * radius
    expected_lines = lambda_intensity * perimeter
    n_lines = max(1, np.random.poisson(expected_lines))

    lines = []

    if beta > 5.0:
        num_horizontal = max(1, int(np.ceil(np.sqrt(expected_lines) / 2)))
        num_vertical = max(1, int(np.ceil(np.sqrt(expected_lines) / 2)))

        spacing_h = window_height / (num_horizontal + 1)
        for i in range(num_horizontal):
            y = -window_height / 2 + (i + 1) * spacing_h
            if abs(y) < window_height / 2:
                lines.append((y, 0.0))

        spacing_v = window_width / (num_vertical + 1)
        for i in range(num_vertical):
            x = -window_width / 2 + (i + 1) * spacing_v
            if abs(x) < window_width / 2:
                lines.append((x, np.pi / 2))
    else:
        for _ in range(n_lines):
            while len(lines) < n_lines:
                theta = np.random.uniform(0, np.pi)
                r = np.random.uniform(-radius, radius)
                if is_line_intersecting(r, theta, window_width, window_height):
                    lines.append((r, theta))
                    if len(lines) >= n_lines:
                        break

    if not lines:
        lines.append((0.0, 0.0))

    n_current = len(lines)

    current_energy = total_energy(lines, beta, window_width, window_height)

    print(f"Running Metropolis-Hastings for {n_iterations} iterations...")
    for iteration in tqdm(range(n_iterations)):
        move_type = np.random.choice(["birth", "death", "modify"], p=[0.2, 0.2, 0.6])

        n_current = len(lines)

        if move_type == "birth":
            if beta > 5.0 and np.random.random() < 0.7:
                theta_new = np.random.choice([0.0, np.pi / 2])
            else:
                theta_new = np.random.uniform(0, np.pi)

            r_new = np.random.uniform(-radius, radius)

            if is_line_intersecting(r_new, theta_new, window_width, window_height):
                lines.append((r_new, theta_new))

                new_energy = total_energy(lines, beta, window_width, window_height)

                log_acceptance_ratio = np.log(lambda_intensity) - (
                    new_energy - current_energy
                )

                if np.log(np.random.random()) < log_acceptance_ratio:
                    current_energy = new_energy
                else:
                    lines.pop()

        elif move_type == "death" and n_current > 1:
            idx_to_remove = np.random.randint(0, n_current)

            removed_line = lines.pop(idx_to_remove)

            new_energy = total_energy(lines, beta, window_width, window_height)

            log_acceptance_ratio = -np.log(lambda_intensity) - (
                new_energy - current_energy
            )

            if np.log(np.random.random()) < log_acceptance_ratio:
                current_energy = new_energy
            else:
                lines.insert(idx_to_remove, removed_line)

        elif move_type == "modify" and n_current > 0:
            idx_to_modify = np.random.randint(0, n_current)
            original_line = lines[idx_to_modify]

            r_orig, theta_orig = original_line

            if beta > 5.0 and np.random.random() < 0.3:
                theta_new = np.random.choice([0.0, np.pi / 2])

                if theta_new == 0.0:
                    r_new = r_orig * np.sin(theta_orig) if theta_orig != 0 else r_orig
                else:
                    r_new = (
                        r_orig * np.cos(theta_orig)
                        if theta_orig != np.pi / 2
                        else r_orig
                    )
            else:
                r_new = r_orig + np.random.normal(0, radius / 10)
                theta_new = theta_orig + np.random.normal(0, np.pi / 10)

                theta_new = np.mod(theta_new, np.pi)

            if is_line_intersecting(r_new, theta_new, window_width, window_height):
                lines[idx_to_modify] = (r_new, theta_new)

                new_energy = total_energy(lines, beta, window_width, window_height)

                log_acceptance_ratio = -(new_energy - current_energy)

                if np.log(np.random.random()) < log_acceptance_ratio:
                    current_energy = new_energy
                else:
                    lines[idx_to_modify] = original_line

    lines_intersecting = []
    for r, theta in lines:
        if is_line_intersecting(r, theta, window_width, window_height):
            lines_intersecting.append((r, theta))

    if not lines_intersecting:
        lines_intersecting.append((0.0, 0.0))
        print("Warning: No intersecting lines found. Adding a default line.")

    print(f"Final number of lines: {len(lines_intersecting)}")
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
    beta,
    window_width,
    window_height,
    length_scale=0.5,
    mean_sigma=1.0,
    seed=None,
    plot_lines=True,
    grid_size=1000,
    n_iterations=1000,
):
    """
    Run a complete simulation with Gaussian fields filling regions

    Parameters:
    - lambda_intensity: Base intensity of the line process
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - window_width, window_height: Dimensions of the window
    - length_scale: Length scale for the Gaussian field within each region
    - mean_sigma: Standard deviation for region means ~ N(0, mean_sigma²)
    - seed: Random seed for reproducibility
    - plot_lines: Whether to draw the lines on the plot
    - grid_size: Resolution of the grid
    - n_iterations: Number of MCMC iterations for the Gibbs process

    Returns:
    - fig, ax: The matplotlib figure and axes
    """
    master_seed = seed if seed is not None else np.random.randint(0, 10000)
    np.random.seed(master_seed)

    print("Sampling from Gibbs line process...")
    lines = sample_gibbs_lines(
        lambda_intensity=lambda_intensity,
        beta=beta,
        window_width=window_width,
        window_height=window_height,
        n_iterations=n_iterations,
        seed=master_seed,
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
        interaction_type = "repulsion" if beta > 0 else "attraction"
        ax.set_title(
            f"Mosaic process with Gibbs lines ($\\lambda={lambda_intensity}$, $\\beta={beta}$ - {interaction_type})"
        )
    else:
        ax.set_title(
            f"Mosaic process only ($\\lambda={lambda_intensity}$, $\\beta={beta}$)"
        )

    ax.set_aspect("equal")

    plt.tight_layout()

    return fig, ax, region_means


if __name__ == "__main__":
    lambda_intensity = 250.0
    beta = 100.0
    window_width = 10
    window_height = 8
    length_scale = 0.5
    mean_sigma = 1.5

    fig, ax, region_means = run_simulation(
        lambda_intensity=lambda_intensity,
        beta=beta,
        window_width=window_width,
        window_height=window_height,
        length_scale=length_scale,
        mean_sigma=mean_sigma,
        seed=42,
        plot_lines=True,
        grid_size=500,
        n_iterations=2000,
    )

    region_mean_values = list(region_means.values())
    print(f"\nRegion mean statistics:")
    print(f"Number of regions: {len(region_mean_values)}")
    print(f"Mean of region means: {np.mean(region_mean_values):.4f}")
    print(f"Std dev of region means: {np.std(region_mean_values):.4f}")

    plt.savefig("simulation_with_lines_strong_repulsion.png", bbox_inches="tight")
    plt.show()

    # # Run simulation with attraction
    # fig, ax, region_means = run_simulation(
    #     lambda_intensity=lambda_intensity,
    #     beta=-5.0,  # Negative beta for attraction
    #     window_width=window_width,
    #     window_height=window_height,
    #     length_scale=length_scale,
    #     mean_sigma=mean_sigma,
    #     seed=42,
    #     plot_lines=True,
    #     grid_size=500,
    #     n_iterations=2000,
    # )

    # plt.savefig("simulation_with_lines_attraction.png", bbox_inches="tight")
    # plt.show()

    # # Run simulation without lines
    # fig, ax, _ = run_simulation(
    #     lambda_intensity=lambda_intensity,
    #     beta=beta,
    #     window_width=window_width,
    #     window_height=window_height,
    #     length_scale=length_scale,
    #     mean_sigma=mean_sigma,
    #     seed=42,
    #     plot_lines=False,
    #     grid_size=500,
    #     n_iterations=2000,
    # )

    # plt.savefig("simulation_without_lines.png", bbox_inches="tight")
    # plt.show()
