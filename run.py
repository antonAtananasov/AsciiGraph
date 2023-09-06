from AsciiPlotter import *


def test():
    """
    Description
    -----------
    Observe
    """
    plotter = AsciiPlotter((231, 115))  # (canvasSize=(9, 19))

    print(f"ASCII rsolution {plotter.canvasSize[0]}x{plotter.canvasSize[1]}")
    print(f"x in range [{plotter.bounds[0][0]},{plotter.bounds[0][1]}]")
    print(f"y in range [{plotter.bounds[1][0]},{plotter.bounds[1][1]}]")
    print("")
    eqs = [
        "y^2+x^2<=10^2",
        "y!=3x+3(3)",
        "3sin(x/3)>=y",
        r"\left(\left(\frac{x}{7}\right)^{2}+\left(\frac{y}{7}\right)^{2}-1\right)^{3}<\left(\frac{x}{7}\right)^{2}\left(\frac{y}{7}\right)^{3}",
        # r"y=x^x",
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
        color = colors[i % len(colors)]
        print("", tex2py(eq))
        print(plotter.plotCartesianAsciiEquations([eq], colors=[color]))

    # # overlay===============================================
    # plot = plotter.plotCartesianAsciiEquations(
    #     eqs, system=False, contourOnTop=False, intersect=None, colors=colors
    # )
    # for eq in eqs:
    #     print("[", tex2py(eq))
    # print(plot)

    # # overlay + contour + intersect===============================================
    # plot = plotter.plotCartesianAsciiEquations(eqs, colors=colors)
    # for eq in eqs:
    #     print(":", tex2py(eq))
    # print(plot)

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

    plotter = AsciiPlotter((np.int_(np.array((231, 115)) / 1)), ((-7, 7), (-7, 7)))
    eqs = [
        r"((x/7)^2*sqrt(abs(abs(x)-3)/(abs(x)-3))+(y/3)^2*sqrt(abs(y+3/7*sqrt(33))/(y+3/7*sqrt(33)))-1)=0",
        r"(abs(x/2)-((3*sqrt(33)-7)/112)*x^2-3+sqrt(1-(abs(abs(x)-2)-1)^2)-y)=0",
        r"(9*sqrt(abs((abs(x)-1)*(abs(x)-.75))/((1-abs(x))*(abs(x)-.75)))-8*abs(x)-y)=0",
        r"(3*abs(x)+.75*sqrt(abs((abs(x)-.75)*(abs(x)-.5))/((.75-abs(x))*(abs(x)-.5)))-y)=0",
        r"(2.25*sqrt(abs((x-.5)*(x+.5))/((.5-x)*(.5+x)))-y)=0",
        r"(6*sqrt(10)/7+(1.5-.5*abs(x))*sqrt(abs(abs(x)-1)/(abs(x)-1))-(6*sqrt(10)/14)*sqrt(4-(abs(x)-1)^2)-y)=0"
        # r"\left(\left(\frac{x}{2}\right)-\frac{\left(3\sqrt{33}-7\right)}{112}x^{2}-3+\sqrt{1-\left(\left(\left(x\right)-2\right)-1\right)^{2}}-y\right)=0",
        # r"\left(9s\left(\left(1-\left(x\right)\right)\left(\left(x\right)-.75\right)\right)-8\left(x\right)-y\right)=0",
        # r"\left(3\left(x\right)+.75s\left(\left(.75-\left(x\right)\right)\left(\left(x\right)-.5\right)\right)-y\right)=0",
        # r"\left(2.25s\left(\left(.5-x\right)\left(x+.5\right)\right)-y\right)=0",
        # r"\left(\frac{6\sqrt{10}}{7}+\left(1.5-.5\left(x\right)\right)s\left(\left(x\right)-1\right)-\frac{6\sqrt{10}}{14}\sqrt{4-\left(\left(x\right)-1\right)^{2}}-y\right)=0",
    ]
    s = tex2py(r"\sqrt{\frac{\left|x\right|}{x}}")
    for i in range(len(eqs)):
        # eqs[i] = tex2py(eqs[i]).replace("sqrt", "F")
        # eqs[i] = tex2py(eqs[i]).replace("s", s)
        # eqs[i] = tex2py(eqs[i]).replace("F", "sqrt")
        print(":", eqs[i])

    print(plotter.plotCartesianAsciiEquations(eqs, intersect=None))
    print("Done in", (time.time_ns() - start_time) / 1e9, "s")
