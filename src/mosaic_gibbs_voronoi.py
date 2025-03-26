import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from tqdm import tqdm
from scipy.fftpack import ifft2
from scipy.spatial import Voronoi, voronoi_plot_2d
import warnings

warnings.filterwarnings("ignore")

plt.style.use("math.mplstyle")


def pairwise_interaction(point1, point2, beta, min_distance=0.001):
    """
    Calculate pairwise interaction energy between two points.

    Parameters:
    - point1, point2: Tuples of (x, y) point coordinates
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - min_distance: Minimum distance to prevent numerical issues

    Returns:
    - Interaction energy
    """
    x1, y1 = point1
    x2, y2 = point2

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    if distance < min_distance:
        distance = min_distance

    # Repulsion or attraction based on beta's sign
    interaction = np.exp(-distance)

    return beta * interaction


def total_energy(points, beta):
    """
    Calculate total energy of the point configuration.

    Parameters:
    - points: List of (x, y) point coordinates
    - beta: Interaction parameter

    Returns:
    - Total energy of the configuration
    """
    energy = 0.0
    n = len(points)

    for i in range(n):
        for j in range(i + 1, n):
            energy += pairwise_interaction(points[i], points[j], beta)

    return energy


def sample_gibbs_points(
    lambda_intensity,
    beta,
    window_width,
    window_height,
    n_iterations=1000,
    seed=None,
    burn_in=200,
):
    """
    Sample points from a Gibbs point process with interaction parameter beta.

    Parameters:
    - lambda_intensity: Base intensity of the process (points per unit area)
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - window_width, window_height: Dimensions of the window
    - n_iterations: Number of MCMC iterations
    - seed: Random seed for reproducibility
    - burn_in: Number of initial iterations to discard

    Returns:
    - List of (x, y) coordinates for the sampled points
    """
    if seed is not None:
        np.random.seed(seed)

    window_half_width = window_width / 2
    window_half_height = window_height / 2

    # Expected number of points based on intensity and area
    area = window_width * window_height
    expected_points = max(3, int(lambda_intensity * area))

    # Initialize with a random configuration
    points = []
    for _ in range(expected_points):
        x = np.random.uniform(-window_half_width, window_half_width)
        y = np.random.uniform(-window_half_height, window_half_height)
        points.append((x, y))

    n_current = len(points)
    current_energy = total_energy(points, beta)

    print(f"Running Metropolis-Hastings for {n_iterations} iterations...")
    for iteration in tqdm(range(n_iterations)):
        move_type = np.random.choice(["birth", "death", "modify"], p=[0.3, 0.3, 0.4])

        if move_type == "birth":
            # Add a new point
            x_new = np.random.uniform(-window_half_width, window_half_width)
            y_new = np.random.uniform(-window_half_height, window_half_height)

            points.append((x_new, y_new))
            new_energy = total_energy(points, beta)

            # Calculate acceptance ratio (using log for numerical stability)
            log_acceptance_ratio = np.log(lambda_intensity * area / (n_current + 1)) - (
                new_energy - current_energy
            )

            if np.log(np.random.random()) < log_acceptance_ratio:
                current_energy = new_energy
                n_current += 1
            else:
                points.pop()

        elif move_type == "death" and n_current > 1:
            # Remove a random point
            idx_to_remove = np.random.randint(0, n_current)
            removed_point = points.pop(idx_to_remove)

            new_energy = total_energy(points, beta)

            log_acceptance_ratio = np.log(n_current / (lambda_intensity * area)) - (
                new_energy - current_energy
            )

            if np.log(np.random.random()) < log_acceptance_ratio:
                current_energy = new_energy
                n_current -= 1
            else:
                points.insert(idx_to_remove, removed_point)

        elif move_type == "modify" and n_current > 0:
            # Modify a random point's position
            idx_to_modify = np.random.randint(0, n_current)
            original_point = points[idx_to_modify]

            # Propose a new position with a small displacement
            x_orig, y_orig = original_point
            step_size = min(window_width, window_height) / 10
            x_new = x_orig + np.random.normal(0, step_size)
            y_new = y_orig + np.random.normal(0, step_size)

            # Keep the point within the window
            x_new = np.clip(x_new, -window_half_width, window_half_width)
            y_new = np.clip(y_new, -window_half_height, window_half_height)

            points[idx_to_modify] = (x_new, y_new)
            new_energy = total_energy(points, beta)

            log_acceptance_ratio = -(new_energy - current_energy)

            if np.log(np.random.random()) < log_acceptance_ratio:
                current_energy = new_energy
            else:
                points[idx_to_modify] = original_point

    print(f"Final number of points: {len(points)}")
    return points


def generate_voronoi_regions(points, window_width, window_height):
    """
    Generate Voronoi regions from a set of points.

    Parameters:
    - points: List of (x, y) coordinates
    - window_width, window_height: Dimensions of the window

    Returns:
    - Voronoi object and clipped regions as polygons
    """
    window_half_width = window_width / 2
    window_half_height = window_height / 2

    # Add points at the corners and far outside to ensure bounded regions
    buffer = max(window_width, window_height)
    corner_points = [
        (-window_half_width - buffer, -window_half_height - buffer),
        (-window_half_width - buffer, window_half_height + buffer),
        (window_half_width + buffer, -window_half_height - buffer),
        (window_half_width + buffer, window_half_height + buffer),
        (-window_half_width - buffer, 0),
        (window_half_width + buffer, 0),
        (0, -window_half_height - buffer),
        (0, window_half_height + buffer),
    ]

    extended_points = points + corner_points

    # Create Voronoi tessellation
    vor = Voronoi(extended_points)

    # Clip Voronoi regions to the window
    clipped_regions = []
    window_boundary = [
        (-window_half_width, -window_half_height),
        (window_half_width, -window_half_height),
        (window_half_width, window_half_height),
        (-window_half_width, window_half_height),
    ]

    # Only process the regions for the original points (not the corner points)
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        # Skip any unbounded regions (should be none with our corner points)
        if -1 in vertices_idx:
            continue

        polygon_vertices = [vor.vertices[v] for v in vertices_idx]

        # Clip to window
        clipped_polygon = clip_polygon_to_window(
            polygon_vertices,
            -window_half_width,
            window_half_width,
            -window_half_height,
            window_half_height,
        )

        if clipped_polygon and len(clipped_polygon) >= 3:
            clipped_regions.append((i, clipped_polygon))

    return vor, clipped_regions


def clip_polygon_to_window(polygon, x_min, x_max, y_min, y_max):
    """
    Clip a polygon to a rectangular window.

    Parameters:
    - polygon: List of (x, y) vertices
    - x_min, x_max, y_min, y_max: Window boundaries

    Returns:
    - Clipped polygon as a list of vertices
    """

    def clip_edge(polygon, x_min=None, x_max=None, y_min=None, y_max=None):
        if not polygon:
            return []

        result = []
        for i in range(len(polygon)):
            current = polygon[i]
            prev = polygon[i - 1] if i > 0 else polygon[-1]

            # Check if current point is inside the clipping boundary
            current_inside = True
            if x_min is not None and current[0] < x_min:
                current_inside = False
            if x_max is not None and current[0] > x_max:
                current_inside = False
            if y_min is not None and current[1] < y_min:
                current_inside = False
            if y_max is not None and current[1] > y_max:
                current_inside = False

            # Check if previous point is inside the clipping boundary
            prev_inside = True
            if x_min is not None and prev[0] < x_min:
                prev_inside = False
            if x_max is not None and prev[0] > x_max:
                prev_inside = False
            if y_min is not None and prev[1] < y_min:
                prev_inside = False
            if y_max is not None and prev[1] > y_max:
                prev_inside = False

            # If we're crossing a boundary, add the intersection point
            if current_inside != prev_inside:
                if x_min is not None and (
                    (prev[0] < x_min and current[0] >= x_min)
                    or (current[0] < x_min and prev[0] >= x_min)
                ):
                    t = (
                        (x_min - prev[0]) / (current[0] - prev[0])
                        if current[0] != prev[0]
                        else 0
                    )
                    y = prev[1] + t * (current[1] - prev[1])
                    result.append((x_min, y))

                if x_max is not None and (
                    (prev[0] > x_max and current[0] <= x_max)
                    or (current[0] > x_max and prev[0] <= x_max)
                ):
                    t = (
                        (x_max - prev[0]) / (current[0] - prev[0])
                        if current[0] != prev[0]
                        else 0
                    )
                    y = prev[1] + t * (current[1] - prev[1])
                    result.append((x_max, y))

                if y_min is not None and (
                    (prev[1] < y_min and current[1] >= y_min)
                    or (current[1] < y_min and prev[1] >= y_min)
                ):
                    t = (
                        (y_min - prev[1]) / (current[1] - prev[1])
                        if current[1] != prev[1]
                        else 0
                    )
                    x = prev[0] + t * (current[0] - prev[0])
                    result.append((x, y_min))

                if y_max is not None and (
                    (prev[1] > y_max and current[1] <= y_max)
                    or (current[1] > y_max and prev[1] <= y_max)
                ):
                    t = (
                        (y_max - prev[1]) / (current[1] - prev[1])
                        if current[1] != prev[1]
                        else 0
                    )
                    x = prev[0] + t * (current[0] - prev[0])
                    result.append((x, y_max))

            # Add current point if it's inside
            if current_inside:
                result.append(current)

        return result

    # Clip against each edge of the window
    clipped = polygon
    clipped = clip_edge(clipped, x_min=x_min)
    clipped = clip_edge(clipped, x_max=x_max)
    clipped = clip_edge(clipped, y_min=y_min)
    clipped = clip_edge(clipped, y_max=y_max)

    return clipped


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
    plot_voronoi=True,
    grid_size=500,
    n_iterations=1000,
):
    """
    Run a complete simulation with Gaussian fields filling Voronoi regions

    Parameters:
    - lambda_intensity: Base intensity of the point process (points per unit area)
    - beta: Interaction parameter (positive for repulsion, negative for attraction)
    - window_width, window_height: Dimensions of the window
    - length_scale: Length scale for the Gaussian field within each region
    - mean_sigma: Standard deviation for region means ~ N(0, mean_sigma²)
    - seed: Random seed for reproducibility
    - plot_voronoi: Whether to draw the Voronoi edges on the plot
    - grid_size: Resolution of the grid
    - n_iterations: Number of MCMC iterations for the Gibbs process

    Returns:
    - fig, ax: The matplotlib figure and axes
    - region_means: Dictionary of region means
    """
    master_seed = seed if seed is not None else np.random.randint(0, 10000)
    np.random.seed(master_seed)

    print("Sampling from Gibbs point process...")
    points = sample_gibbs_points(
        lambda_intensity=lambda_intensity,
        beta=beta,
        window_width=window_width,
        window_height=window_height,
        n_iterations=n_iterations,
        seed=master_seed,
    )

    window_half_width = window_width / 2
    window_half_height = window_height / 2

    print("Generating Voronoi regions...")
    vor, clipped_regions = generate_voronoi_regions(points, window_width, window_height)

    x = np.linspace(-window_half_width, window_half_width, grid_size)
    y = np.linspace(-window_half_height, window_half_height, grid_size)
    X, Y = np.meshgrid(x, y)

    print("Assigning grid points to regions...")
    region_indices = np.ones(X.shape, dtype=int) * -1

    # For each grid point, find the closest Voronoi site
    for i in range(grid_size):
        for j in range(grid_size):
            point = (X[i, j], Y[i, j])

            min_dist = float("inf")
            closest_idx = -1

            for idx, (x, y) in enumerate(points):
                dist = (point[0] - x) ** 2 + (point[1] - y) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

            region_indices[i, j] = closest_idx

    unique_regions = np.unique(region_indices)
    print(f"Found {len(unique_regions)} unique regions")

    all_fields = np.zeros(X.shape)
    all_fields[:] = np.nan

    region_means = {}

    print("Generating fields for each region...")
    for region_idx in tqdm(unique_regions):
        if region_idx == -1:  # Skip points outside the window
            continue

        mask = region_indices == region_idx

        if np.sum(mask) < 5:
            print(f"Skipping tiny region with only {np.sum(mask)} points")
            continue

        region_mean = np.random.normal(0, mean_sigma)
        region_means[region_idx] = region_mean

        region_seed = master_seed + int(region_idx) + 1
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

    if plot_voronoi:
        # Plot the Voronoi edges within the window
        for region_idx, vertices in clipped_regions:
            vertices_array = np.array(vertices)
            ax.plot(vertices_array[:, 0], vertices_array[:, 1], "k-", linewidth=1)

        # Plot the Gibbs points
        ax.plot([p[0] for p in points], [p[1] for p in points], "ko", markersize=3)

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

    if plot_voronoi:
        interaction_type = "repulsion" if beta > 0 else "attraction"
        ax.set_title(
            f"Mosaic process with Voronoi regions ($\\lambda={lambda_intensity}$, $\\beta={beta}$ - {interaction_type})"
        )
    else:
        ax.set_title(
            f"Mosaic process only ($\\lambda={lambda_intensity}$, $\\beta={beta}$)"
        )

    ax.set_aspect("equal")

    plt.tight_layout()

    return fig, ax, region_means


if __name__ == "__main__":
    lambda_intensity = 1.0  # Points per unit area (adjust as needed)
    beta = 0.0  # Repulsion strength
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
        plot_voronoi=True,
        grid_size=500,
        n_iterations=2000,
    )

    region_mean_values = list(region_means.values())
    print(f"\nRegion mean statistics:")
    print(f"Number of regions: {len(region_mean_values)}")
    print(f"Mean of region means: {np.mean(region_mean_values):.4f}")
    print(f"Std dev of region means: {np.std(region_mean_values):.4f}")

    plt.savefig("simulation_with_voronoi_strong_repulsion.png", bbox_inches="tight")
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
    #     plot_voronoi=True,
    #     grid_size=500,
    #     n_iterations=2000,
    # )

    # plt.savefig("simulation_with_voronoi_attraction.png", bbox_inches="tight")
    # plt.show()

    # # Run simulation without showing Voronoi edges
    # fig, ax, _ = run_simulation(
    #     lambda_intensity=lambda_intensity,
    #     beta=beta,
    #     window_width=window_width,
    #     window_height=window_height,
    #     length_scale=length_scale,
    #     mean_sigma=mean_sigma,
    #     seed=42,
    #     plot_voronoi=False,
    #     grid_size=500,
    #     n_iterations=2000,
    # )

    # plt.savefig("simulation_without_voronoi_edges.png", bbox_inches="tight")
    # plt.show()
