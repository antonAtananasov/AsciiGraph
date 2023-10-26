from AsciiPlotter import *
import sys

SCALE = 1  # reduce this to change resolution fast
NATIVE_RESOLUTION = (231, 115)

# =========================================

DEFAULT_BOUNDS = ((-12, 12), (-12, 12))
DEFAULT_CANVAS_SIZE = (19, 9)

PLOT_TYPES = ["cartesian", "polar"]

HELP_ARGS = {
    "h": "Help",
    "b": f"Bounds for the coordinate system. Default: '{DEFAULT_BOUNDS}'. Format: '((min_x, max_x),(min_y, max_y))'",
    "s": f"Size / Resolution in (rows,columns) in ASCII symbols. Default: {DEFAULT_CANVAS_SIZE}",
    "t": f"Plot type {{cartesian | polar}}. Default: {PLOT_TYPES[0]}",
    "e": r"Equation to draw. Supports some LaTeX tegs Examples: y^2+x^2<=10^2, y!=3x+3(3), 3sin(x/3)>=y, r\left(\left(\frac{x}{7}\right)^{2}+\left(\frac{y}{7}\right)^{2}-1\right)^{3}<\left(\frac{x}{7}\right)^{2}\left(\frac{y}{7}\right)^{3}, y-tan(x)",
}
PARAM_ARGS = {"b", "s", "t", "e"}  # those args which take in a value
SIMPLE_ARGS = {"h"}


def main():
    bounds = DEFAULT_BOUNDS
    canvasSize = DEFAULT_CANVAS_SIZE
    plotType = PLOT_TYPES[0]
    equation = ""
    args = sys.argv
    for i in range(len(args)):
        if args[i].startswith("-"):
            #
            simpleArgsCount = 0
            for arg in args[i][1:]:
                if arg in SIMPLE_ARGS:
                    simpleArgsCount += 1
            #
            for j in range(len(args[i][1:])):
                option = args[i][1:][j]
                if option in PARAM_ARGS:
                    argValue = args[i + j + 1 - simpleArgsCount]
                    if option == "b":
                        try:
                            inputValue = [
                                float(b)
                                for b in argValue.replace("(", "")
                                .replace(")", "")
                                .replace(" ", "")
                                .split(",")
                            ]
                            boundsX, boundsY = tuple(inputValue[:2]), tuple(
                                inputValue[2:]
                            )
                            bounds = (boundsX, boundsY)
                        except:
                            raise ValueError
                    elif option == "s":
                        try:
                            inputValue = [
                                int(b)
                                for b in argValue.replace("(", "")
                                .replace(")", "")
                                .replace(" ", "")
                                .split(",")
                            ]
                            canvasSize = tuple(inputValue)
                        except:
                            raise ValueError
                    elif option == "t":
                        try:
                            for t in PLOT_TYPES:
                                if argValue in t:
                                    plotType = t
                        except:
                            raise ValueError
                    elif option == "e":
                        try:
                            equation = argValue
                        except:
                            raise ValueError
                elif option in SIMPLE_ARGS:
                    if option == "h":
                        print("List of available options:")
                        for k, v in HELP_ARGS.items():
                            print("\t", f"-{k}", "\t", v)

    plotter = AsciiPlotter(canvasSize, bounds)
    if len(equation):
        if plotType == "cartesian":
            print(plotter.plotCartesianAsciiEquations([equation]))
        elif plotType == "polar":
            print(plotter.plotPolarAsciiEquations([equation]))
    else:
        print("There is no given valid equation.")


def test(nativeSize: tuple, scale: float) -> None:
    """
    Description
    -----------
    Observe
    """
    plotter10x10 = AsciiPlotter(
        (np.int_(np.array(nativeSize) * scale))
    )  # (canvasSize=(9, 19))

    print(f"ASCII rsolution {plotter10x10.canvasSize[0]}x{plotter10x10.canvasSize[1]}")
    print(f"x in range [{plotter10x10.bounds[0][0]},{plotter10x10.bounds[0][1]}]")
    print(f"y in range [{plotter10x10.bounds[1][0]},{plotter10x10.bounds[1][1]}]")
    print("")
    eqs = [
        "y^2+x^2<=10^2",
        "y!=3x+3(3)",
        "3sin(x/3)>=y",
        r"\left(\left(\frac{x}{7}\right)^{2}+\left(\frac{y}{7}\right)^{2}-1\right)^{3}<\left(\frac{x}{7}\right)^{2}\left(\frac{y}{7}\right)^{3}",
        "y-tan(x)"
        # r"y=x^x",
    ]
    colors = [
        ConsoleColors.RED,
        ConsoleColors.BLUE,
        ConsoleColors.YELLOW,
        ConsoleColors.GREEN,
    ]

    # individual===============================================
    for i in range(len(eqs)):
        eq = eqs[i]
        color = colors[i % len(colors)]
        print("", tex2py(eq))
        print(plotter10x10.plotCartesianAsciiEquations([eq], colors=[color]))

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
    for eq in eqs:
        print("|", tex2py(eq))
    plot = plotter10x10.plotCartesianAsciiEquations(eqs, system=True, colors=colors)
    print(plot)
    # cartesian points===============================================
    pts = [
        (0, 0),
        (4, 6),
        (-2, 3),
        (plotter10x10.bounds[0][0] + 3, plotter10x10.bounds[1][0] + 3),
        (
            random.randrange(plotter10x10.bounds[0][0], plotter10x10.bounds[0][1]),
            random.randrange(plotter10x10.bounds[1][0], plotter10x10.bounds[1][1]),
        ),
    ]
    chars = ["O", "A", "B", "C", "R"]
    for i in range(len(pts)):
        print(":", f"{chars[i]}{pts[i]}")
    print(plotter10x10.plotCartesianAsciiPoints(pts, chars, colors=colors))

    # polar=================================================
    eqs = [
        r"r/6=\sin(1.2(\theta+0pi))^{2}+\cos(6(\theta))^{3}",
        r"r/6=\sin(1.2(\theta+2pi))^{2}+\cos(6(\theta+2pi))^{3}",
        r"r/6=\sin(1.2(\theta+4pi))^{2}+\cos(6(\theta+4pi))^{3}",
        r"r/6=\sin(1.2(\theta+6pi))^{2}+\cos(6(\theta+6pi))^{3}",
        r"r/6=\sin(1.2(\theta+8pi))^{2}+\cos(6(\theta+8pi))^{3}",
        r"r/6=\sin(1.2(\theta+10pi))^{2}+\cos(6(\theta+10pi))^{3}",
        r"r/4<1+2\sin\left(3\theta\right)",
    ]
    for i in range(len(eqs)):
        eq = eqs[i]
        print(":", tex2py(eq))
    polarPlot = plotter10x10.plotPolarAsciiEquations(
        eqs,
        colors=[
            ConsoleColors.RED,
            # ConsoleColors.RED,
            # ConsoleColors.RED,
            # ConsoleColors.RED,
            # ConsoleColors.RED,
            # ConsoleColors.RED,
            ConsoleColors.BLUE,
            ConsoleColors.YELLOW,
            ConsoleColors.GREEN,
            ConsoleColors.BEIGE,
            ConsoleColors.GREEN2,
        ],
    )
    print(polarPlot)
    eqs = ["r<=6", "t>rad(30)", "t<2pi-rad(30)"]
    for i in range(len(eqs)):
        eq = eqs[i]
        print("|", tex2py(eq))
    polarPlot = plotter10x10.plotPolarAsciiEquations(
        eqs, system=True, contourOnTop=True
    )
    print(polarPlot)
    # polar=================================================
    pts = [(2, 30), (4, 150), (10, -150)]
    for pt in pts:
        print(";", pt)
    print(plotter10x10.plotPolarAsciiPoints(pts, colors=colors))

    plotter7x7 = AsciiPlotter(
        (np.int_(np.array(nativeSize) * scale)), ((-7, 7), (-7, 7))
    )
    eqs = [
        r"((x/7)^2*sqrt(abs(abs(x)-3)/(abs(x)-3))+(y/3)^2*sqrt(abs(y+3/7*sqrt(33))/(y+3/7*sqrt(33)))-1)=0",
        r"(abs(x/2)-((3*sqrt(33)-7)/112)*x^2-3+sqrt(1-(abs(abs(x)-2)-1)^2)-y)=0",
        r"(9*sqrt(abs((abs(x)-1)*(abs(x)-.75))/((1-abs(x))*(abs(x)-.75)))-8*abs(x)-y)=0",
        r"(3*abs(x)+.75*sqrt(abs((abs(x)-.75)*(abs(x)-.5))/((.75-abs(x))*(abs(x)-.5)))-y)=0",
        r"(2.25*sqrt(abs((x-.5)*(x+.5))/((.5-x)*(.5+x)))-y)=0",
        r"(6*sqrt(10)/7+(1.5-.5*abs(x))*sqrt(abs(abs(x)-1)/(abs(x)-1))-(6*sqrt(10)/14)*sqrt(4-(abs(x)-1)^2)-y)=0",
    ]
    for i in range(len(eqs)):
        print(":", eqs[i])
    print(plotter7x7.plotCartesianAsciiEquations(eqs, intersect=None))


if __name__ == "__main__":
    try:
        import time

        start_time = time.time_ns()
        main() if len(sys.argv) > 1 else test(NATIVE_RESOLUTION, SCALE)

        print("Done in", (time.time_ns() - start_time) / 1e9, "s")

    except Exception as e:
        if type(e) in [ValueError]:
            print("You or the programmer did something wrong:", type(e).__name__)
        else:
            print("Oops, I did not expect this error. No message, sorry.")
