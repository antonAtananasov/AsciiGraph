from AsciiPlotter import *


def test():
    """
    Description
    -----------
    Observe
    """
    plotter = AsciiPlotter()  # ((161, 75))  # (canvasSize=(9, 19))

    print(f"ASCII rsolution {plotter.canvasSize[0]}x{plotter.canvasSize[1]}")
    print(f"x in range [{plotter.bounds[0][0]},{plotter.bounds[0][1]}]")
    print(f"y in range [{plotter.bounds[1][0]},{plotter.bounds[1][1]}]")
    print("")
    eqs = [
        "y^2+x^2<=10^2",
        "y!=3x+3(3)",
        "3sin(x/3)>=y",
        r"\left(\left(\frac{x}{7}\right)^{2}+\left(\frac{y}{7}\right)^{2}-1\right)^{3}<\left(\frac{x}{7}\right)^{2}\left(\frac{y}{7}\right)^{3}",
    ]
    colors = [
        plotter.colors.RED,
        plotter.colors.BLUE,
        plotter.colors.YELLOW,
        plotter.colors.GREEN,
    ]

    # individual===============================================
    for i in range(len(eqs)):
        eq = eqs[i]
        color = colors[i]
        print("", tex2py(eq))
        print(plotter.plotCartesianAsciiEquations([eq], colors=[color]))

    # overlay===============================================
    plot = plotter.plotCartesianAsciiEquations(
        eqs, system=False, contourOnTop=False, intersect=None, colors=colors
    )
    for eq in eqs:
        print("[", tex2py(eq))
    print(plot)

    # overlay + contour + intersect===============================================
    plot = plotter.plotCartesianAsciiEquations(eqs, colors=colors)
    for eq in eqs:
        print(":", tex2py(eq))
    print(plot)

    # system===============================================
    plot = plotter.plotCartesianAsciiEquations(eqs, system=True, colors=colors)
    for eq in eqs:
        print("|", tex2py(eq))
    print(plot)
    # cartesian points===============================================
    pts = [
        (0, 0),
        (4, 6),
        (-2, 3),
        (plotter.bounds[0][0] + 3, plotter.bounds[1][0] + 3),
        (
            random.randrange(plotter.bounds[0][0], plotter.bounds[0][1]),
            random.randrange(plotter.bounds[1][0], plotter.bounds[1][1]),
        ),
    ]
    chars = ["O", "A", "B", "C", "R"]
    for i in range(len(pts)):
        print(":", f"{chars[i]}{pts[i]}")
    print(plotter.plotCartesianAsciiPoints(pts, chars, colors=colors))

    # polar=================================================
    eqs = [
        r"r/6=\sin(1.2(\theta+0pi))^{2}+\cos(6(\theta+0pi))^{3}",
        r"r/6=\sin(1.2(\theta+2pi))^{2}+\cos(6(\theta+2pi))^{3}",
        r"r/6=\sin(1.2(\theta+4pi))^{2}+\cos(6(\theta+4pi))^{3}",
        r"r/6=\sin(1.2(\theta+6pi))^{2}+\cos(6(\theta+6pi))^{3}",
        r"r/6=\sin(1.2(\theta+8pi))^{2}+\cos(6(\theta+8pi))^{3}",
        r"r/6=\sin(1.2(\theta+10pi))^{2}+\cos(6(\theta+10pi))^{3}",
        r"r/4<1+2\sin\left(3\theta\right)",
    ]
    polarPlot = plotter.plotPolarAsciiEquations(
        eqs,
        colors=[
            plotter.colors.RED,
            plotter.colors.RED,
            plotter.colors.RED,
            plotter.colors.RED,
            plotter.colors.RED,
            plotter.colors.RED,
            plotter.colors.BLUE,
        ],
    )
    for i in range(len(eqs)):
        eq = eqs[i]
        print(":", tex2py(eq))
    print(polarPlot)
    # polar=================================================
    for i in range(len(pts)):
        print(":", pts[i])
    pts = [(2, 30), (4, 150), (10, -150)]
    for pt in pts:
        print(":", pt)
    print(plotter.plotPolarAsciiPoints(pts, colors=colors))


if __name__ == "__main__":
    import time

    start_time = time.time_ns()
    test()
    print("Done in", (time.time_ns() - start_time) / 1e9, "s")
