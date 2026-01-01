import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to array_geometry.yaml")
    p.add_argument("--annotate", action="store_true", help="Draw mic indices")
    args = p.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)

    geom = np.asarray(d["array_geometry"], dtype=float)  # (M,3)
    M = geom.shape[0]
    x, y, z = geom[:, 0], geom[:, 1], geom[:, 2]

    print("Geometry shape:", geom.shape)
    print("x min/max:", x.min(), x.max())
    print("y min/max:", y.min(), y.max())
    print("z min/max:", z.min(), z.max())

    # 1) Y-Z plane (most relevant since x==0)
    plt.figure()
    plt.scatter(y, z)
    plt.title(f"Microphone Array Geometry (Y-Z plane) - {M} mics")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)

    if args.annotate:
        for i in range(M):
            plt.text(y[i], z[i], str(i), fontsize=8)

    # 2) 3D plot (just to confirm)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)
    ax.set_title(f"Microphone Array Geometry (3D) - {M} mics")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    if args.annotate:
        for i in range(M):
            ax.text(x[i], y[i], z[i], str(i), fontsize=7)

    plt.show()


if __name__ == "__main__":
    main()
