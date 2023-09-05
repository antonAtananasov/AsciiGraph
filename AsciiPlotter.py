import numpy as np
from latex2python import tex2py


def mux(input: int) -> int:
    """
    Returns
    -------
    int(np.log2(input))
    """
    return int(np.log2(input))


def demux(input: int) -> int:
    """
    Returns
    -------
    int(2**input)
    """
    return int(2**input)


def split(txt: str, seps: [str]) -> [str]:
    """
    Description
    -----------
    str.split() but with an array of separators
    """
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


class AsciiPlotter:
    def coordLine(self, size: int, bounds: (float, float)):
        """
        Description
        -----------
        Creates a 1D space (number line) of complex numbers

        Returns
        -------
        Shorthand for np.complex128(np.linspace(...)) -> np.complex128
        """
        return np.complex128(np.linspace(bounds[0], bounds[1], size))

    def coordLineTile(self, size: (int, int), bounds: (float, float)):
        """
        Description
        -----------
        Automatically create a 2D space by vertically tiling a 1D space

        Returns
        -------
        Shorthand for np.tile(self.coordLine(...), ...)
        """
        return np.tile(
            self.coordLine(size[0], bounds),
            [size[1], 1],
        )

    def X(self):
        """
        Description
        -----------
        Automatically create a 2D space with left-to-right gradient from minimum x-axis bound[0][0]
        to maximum x-axis bound[0][1] (using np.linspace). The length of the array is the
        horizontal canvasSize[0] by the vertical canvasSize[1] (same as Y()).
        """
        return self.coordLineTile(np.int_(self.canvasSize) + 1, self.bounds[0])

    def Y(self):
        """
        Description
        -----------
        Automatically create a 2D space with bottom-to-top gradient from minimum y-axis bound[1][0]
        to maximum y-axis bound[1][1] (using np.linspace). The length of the array is the
        horizontal canvasSize[0] by the vertical canvasSize[1] (same as X()).
        """
        return self.coordLineTile(
            np.int_(self.canvasSize[::-1]) + 1, self.bounds[1]
        ).T[::-1]

    def __init__(
        self,
        canvasSize: (int, int) = (40, 20),
        bounds: ((float, float), (float, float)) = ((-12, 12), (-12, 12)),
    ):
        """
        Parameters
        ----------
        canvasSize: (int, int)
            The number of (columns, lines) of the working space.
            It governs the size of the numpy arrays for the calculations
            and the output arrays and strings from functions.
        bounds: ((float, float), (float, float))
            Governs the extent of the coordinate plane used for the calculations.
            format: ((min_x, max_x),(min_y, max_y))
        """
        self.canvasSize = canvasSize
        self.bounds = bounds
        self.functions = self.globals = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "arcsin": np.arcsin,
            "arccos": np.arccos,
            "arctan": np.arctan,
            "hypot": np.hypot,
            "arctan2": np.arctan2,
            "degrees": np.degrees,
            "radians": np.radians,
            "deg": np.degrees,
            "rad": np.radians,
            "deg2rad": np.deg2rad,
            "rad2deg": np.rad2deg,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "arcsinh": np.arcsinh,
            "arccosh": np.arccosh,
            "arctanh": np.arctanh,
            "round": np.round,
            "around": np.around,
            "rint": np.rint,
            "fix": np.fix,
            "floor": np.floor,
            "ceil": np.ceil,
            "trunc": np.trunc,
            "exp": np.exp,
            "log": np.log,
            "lg": np.log10,
            "sinc": np.sinc,
            "cbrt": np.cbrt,
            "absolute": np.absolute,
            "sign": np.sign,
            "sqrt": np.sqrt,
            "pow": np.power,
            "e": np.e,
            "pi": np.pi,
        }  # the list of available mathematical operations, constants, etc.

        self.functions.update({"x": self.X()})
        self.functions.update({"y": self.Y()})
        self.marchingSqMask = (
            {  # used for bitwise operations with charsetMode (see below)
                np.array2string(np.bool_([[1, 1], [1, 1]])): 0b0000000000000001,
                np.array2string(np.bool_([[1, 1], [0, 1]])): 0b0000000000000010,
                np.array2string(np.bool_([[1, 1], [1, 0]])): 0b0000000000000100,
                np.array2string(np.bool_([[1, 1], [0, 0]])): 0b0000000000001000,
                np.array2string(np.bool_([[1, 0], [1, 1]])): 0b0000000000010000,
                np.array2string(np.bool_([[1, 0], [0, 1]])): 0b0000000000100000,
                np.array2string(np.bool_([[1, 0], [1, 0]])): 0b0000000001000000,
                np.array2string(np.bool_([[1, 0], [0, 0]])): 0b0000000010000000,
                np.array2string(np.bool_([[0, 1], [1, 1]])): 0b0000000100000000,
                np.array2string(np.bool_([[0, 1], [0, 1]])): 0b0000001000000000,
                np.array2string(np.bool_([[0, 1], [1, 0]])): 0b0000010000000000,
                np.array2string(np.bool_([[0, 1], [0, 0]])): 0b0000100000000000,
                np.array2string(np.bool_([[0, 0], [1, 1]])): 0b0001000000000000,
                np.array2string(np.bool_([[0, 0], [0, 1]])): 0b0010000000000000,
                np.array2string(np.bool_([[0, 0], [1, 0]])): 0b0100000000000000,
                np.array2string(np.bool_([[0, 0], [0, 0]])): 0b1000000000000000,
            }
        )
        self.charset = [  # the characters used to represent different marching square states as a block of text in the console
            "+",
            "\\",
            "/",
            "-",
            "\\",
            "\\",
            "|",
            "/",
            "/",
            "|",
            "/",
            "\\",
            "-",
            "/",
            "\\",
            ":",
        ]
        self.charsetMode = {  # different modes select relevant marching squares and replace them with a symbol from the charset having the same index as the multiplexed value of the square frommarchingSqMask
            "<=": 0b1111111111111110,
            "!=": 0b1000000000000001,
            ">=": 0b0111111111111111,
            "=": 0b0111111111111110,
            "<": 0b1000000000000000,
            ">": 0b0000000000000001,
        }

    def newCanvasMatrix(self, canvasSize: (int, int) = (-1, -1)) -> np.ndarray:
        """
        Description
        -----------
        Used to create a blank canvas full of spaces with swappable characters
        to be populated by other functions

        Parameters
        ----------
        canvasSize: (int, int)
            *See canvasSize in the class description above*
            Additionally, setting one or both of the integers
            to a negative number defaults it to the class's canvasSize.
        Returns
        -------
        A np.ndarray of shape canvasSize, full of space strings (' ')
        """
        canvasSize = list(canvasSize)
        if canvasSize[0] <= 0:
            canvasSize[0] = self.canvasSize[0]
        if canvasSize[1] <= 0:
            canvasSize[1] = self.canvasSize[1]

        canvasMatrix = np.zeros((canvasSize[1], canvasSize[0]), str)
        canvasMatrix[canvasMatrix == ""] = " "
        return canvasMatrix

    def strToExpr(self, s: str) -> str:
        """
        Description
        -----------
        Parses an expression from simple format or LaTeX, quick to type in the command line to a python expression.
        Useful for copy-pasting stuff from Desmos Graphing Calculator or some documents and for typing simply by hand.
        If x and y are not used in the expression, a line equivalent
        to the y-level of the result is assumed and equalities are ignored.

        Parameters
        ----------
        s: str
            The equation or inequality to parse into usable format

        Returns
        -------
        A tuple (string, charsetMode). The charsetMode is selected using the equation type.
          Writing an expression without comparison or equal sign enables all symbols to fill the canvas.
        """
        if not "x" in s and not "y" in s:
            s = f"y={split(s,[charsetModeString for charsetModeString, charsetModeMask in self.charsetMode.items()])[0]}"

        s = tex2py(s)
        for charsetModeString, charsetMask in self.charsetMode.items():
            if charsetModeString in s:
                left, right = s.split(charsetModeString)
                s = f"{left}-({right})"
                return s, charsetMask
        return s, 0xFFFF

    def calculateCartesianPlane(
        self,
        equation: str,
        returnType: type = np.bool_,
        signsOnly: bool = True,
        signPositiveBias: bool = True,
    ) -> np.ndarray:
        """
        Description
        -----------
        Evaluates the expression using numpy functions and constants.

        Parameters
        ----------
        equation: str
            A text or LaTeX expression (strToExpr is used internally)
        returnType: type
            The format/type of the numpy typed array expected as a return of
            the function by the user (numpy.bool_, numpy.int16, numpy.float64, numpy.complex128, etc.)
        signsOnly: bool
            Whether the function should return only the sign of the values calculated in the coordinate plane or the actual values
        signPositiveBias: bool
            If signsOnly is set to True, to convert the signs to a binary array for a marching square loop,
            a number is added to remove the neutral zeros. (numpy.sign(0) returs 0). If signPositiveBias is
            set to True 1 is added and -1's become 0's and 1's remain 1's. If set to False, 1's become 0's and -1's become 1's

        Returns
        -------
        The type specified in returnType or numpy.ndarray
        """
        eq = self.strToExpr(equation)[0]
        plane = eval(eq, self.functions)
        bias = 1 if signPositiveBias else -1
        return (
            returnType(
                (np.sign(plane) + (1 if returnType == np.bool_ else 0) * bias)
                * bias
            )
            if signsOnly
            else returnType(plane)
        )

    def planeToBool(
        self, plane: np.ndarray, charsetMask: int = 0b0111111111111110
    ) -> np.bool_:
        """
        Description
        -----------
        Applies marching squares to determine if a given position on the canvas should belong to the graph of an equation.

        Parameters
        ----------
        plane: numpy.ndarray
            A calculated 2D array of values (from calculateCartesianPlane(...))
        charsetMask : int
            Default is equation. If specified, it uses the given mask to
            suppress unwanted features (marching square states) in the graph.

        Returns
        -------
        A numpy boolean 2D array like a bitwise mask of the graph of the given plane; numpy.bool_
        """
        boolMatrix = np.bool_(np.zeros(self.canvasSize)).T
        for i in np.arange(len(plane) - 1):
            for j in np.arange(len(plane[0]) - 1):
                square = plane[i : i + 2, j : j + 2]
                strSquare = np.array2string(square)
                charCandidate = self.marchingSqMask[strSquare]
                boolMatrix[i][j] = charCandidate & charsetMask > 0

        return boolMatrix

    def cartesianPlaneToStrMatrix(
        self,
        plane: np.ndarray,
        charsetMask: int = 0b0111111111111110,
    ) -> np.ndarray:
        """
        Description
        -----------
        Generates a 2D array of characters that make up the ascii graph of the given plane.

        Parameters
        ----------
        plane: np.ndarray
            A calculated 2D array of values (from calculateCartesianPlane(...))
        charsetMask: int
            Default is equation. If specified, it uses the given mask to
            suppress unwanted features (marching square states) in the graph.

        Returns
        -------
        [[str,str,...],[str,str,...],...] a 2D array of characters that make up the ascii graph of the given plane.
        """
        canvasMatrix = self.newCanvasMatrix()
        for i in np.arange(len(plane) - 1):
            for j in np.arange(len(plane[0]) - 1):
                square = plane[i : i + 2, j : j + 2]
                strSquare = np.array2string(square)
                char = " "
                charCandidate = self.marchingSqMask[strSquare]
                if charCandidate & charsetMask > 0:
                    char = self.charset[mux(charCandidate)]
                    canvasMatrix[i][j] = char
        return canvasMatrix

    def strMatrixToStr(self, strMatrix: np.ndarray) -> str:
        """
        Description
        -----------
        Stitches the rectangular array of characters to a block of string.

        Parameters
        ----------
        strMatrix: np.ndarray
            See cartesianEqToStrMatrix above
        """
        canvasRows = ["".join(char) for char in strMatrix]
        outputCanvas = "\n".join(canvasRows)
        return outputCanvas

    def overlayStrMatrices(
        self,
        strMatrices: [str],
        contourOnTop: bool = False,
        intersect: str = None,
    ):
        """
        Description
        -----------
        Overlays the second strMatrix over the first one and returns a combined graph of the two

        Parameters
        ----------
        strMatrix1: str
            See cartesianEqToStrMatrix above
        strMatrix2: str
            See cartesianEqToStrMatrix above

        contourOnTop: bool
            If set to True, the contour characters of the first matrix will be drawn on top of the second.

        intersect: str
            if contours from the two matrices occupy the same spot, it will be filled with the specified character

        Returns
        -------
        See cartesianEqToStrMatrix above

        """
        contourChars = self.charset[1:-1] + [intersect]
        newStrMatrix = self.newCanvasMatrix()
        for k in np.arange(len(strMatrices) - 1):
            strMatrix1, strMatrix2 = strMatrices[k : k + 2]
            for i in np.arange(self.canvasSize[1]):
                for j in np.arange(self.canvasSize[0]):
                    if strMatrix2[i][j] == " ":
                        newStrMatrix[i][j] = strMatrix1[i][j]
                    else:
                        if intersect != None:
                            if strMatrix1[i][j] in contourChars:
                                if strMatrix2[i][j] in contourChars:
                                    newStrMatrix[i][j] = intersect
                                else:
                                    newStrMatrix[i][j] = (
                                        strMatrix1[i][j]
                                        if contourOnTop
                                        else strMatrix2[i][j]
                                    )
                            else:
                                newStrMatrix[i][j] = strMatrix2[i][j]
                        else:
                            newStrMatrix[i][j] = strMatrix2[i][j]

        return newStrMatrix

    def maskStrMatrix(
        self, strMatrix: np.ndarray, booleanMask: np.bool_
    ) -> np.ndarray:
        """
        Description
        -----------
        Uses the booleanMask to remove characters from the strMatrix where the values at a specific index are False

        Parameters
        ----------
        strMatrix: numpy.ndarray
            See cartesianEqToStrMatrix above
        booleanMask: np.bool_
            See planeToBool above

        Returns
        -------
        See cartesianEqToStrMatrix above
        """
        newStrMatrix = self.newCanvasMatrix()
        for i in np.arange(self.canvasSize[1]):
            for j in np.arange(self.canvasSize[0]):
                if booleanMask[i][j]:
                    newStrMatrix[i][j] = strMatrix[i][j]
                else:
                    newStrMatrix[i][j] = " "
        return newStrMatrix

    def compileCartesianEquation(self, eq: str) -> str:
        """
        Parameters
        ----------
        eq: str
            A text or LaTeX equation
        Returns
        -------
        A tuple (See cartesianEqToStrMatrix above, See planeToBool above)
        """
        eq, mode = self.strToExpr(eq)
        plane = self.calculateCartesianPlane(eq)
        currentMatrix = self.cartesianPlaneToStrMatrix(plane, mode)
        canvasMask = self.planeToBool(plane, mode)
        return currentMatrix, canvasMask

    def cartesianAxes(self):
        xAxis = self.compileCartesianEquation("y=0")[0]
        yAxis = self.compileCartesianEquation("x=0")[0]
        arrows = {
            "<": (np.real(self.X())[0][1], 0),
            ">": (self.bounds[0][1], 0),
            "v": (0, np.real(self.Y())[-2][0]),
            "^": (0, self.bounds[1][1]),
        }
        arrowsMatrix = self.newCanvasMatrix()
        for arrow, position in arrows.items():
            arrowsMatrix = self.overlayStrMatrices(
                [
                    arrowsMatrix,
                    self.cartesianPointsToStrMatrix([position], arrow),
                ],
            )
        axesMatrix = self.overlayStrMatrices(
            [yAxis, xAxis], intersect="+", contourOnTop=False
        )
        return self.overlayStrMatrices(
            [axesMatrix, arrowsMatrix], contourOnTop=False, intersect=False
        )

    def cartesianEqToStrMatrix(
        self,
        eqs: [str],
        drawAxes: bool = True,
        system: bool = False,
        intersect: str = "x",
        contourOnTop=True,
    ) -> str:
        """
        Description
        -----------
        Magic

        Parameters
        ----------
        eqs: [str]
            A list of text or LaTeX equations to be drawn on one coordinate plane
        drawAxes: bool = True
            Whether coordinate axes y and x should be drawn behind the equations
        system: bool = False
            Whether to only show the graph of points that satisfy all given equations.
            If set to False, the equations will only be overlayed instead
        intersect: str = "x"
            See intersect from overlayStrMatrices above
        contourOnTop=True
            See contourOnTop from overlayStrMatrices above

        Returns
        -------
        See overlayStrMatrices above
        """
        strMatrix = self.newCanvasMatrix()
        boolMask = strMatrix == " "
        for i in np.arange(len(eqs)):
            eq = eqs[i]
            currentMatrix, currentMask = self.compileCartesianEquation(eq)
            if system:
                boolMask &= currentMask
            strMatrix = self.overlayStrMatrices(
                [strMatrix, currentMatrix],
                intersect=intersect,
                contourOnTop=contourOnTop,
            )

        strMatrix = self.maskStrMatrix(strMatrix, boolMask)
        if drawAxes:
            strMatrix = self.overlayStrMatrices(
                [self.cartesianAxes(), strMatrix],
                intersect=None,
                contourOnTop=False,
            )

        return strMatrix

    def plotCartesianAsciiEquations(
        self,
        eqs: [str],
        drawAxes: bool = True,
        system: bool = False,
        intersect: str = "x",
        contourOnTop=True,
    ) -> str:
        """
        Description
        -----------
        See cartesianEqToStrMatrix above

        Returns
        -------
        str
        """
        plot = self.strMatrixToStr(
            self.cartesianEqToStrMatrix(
                eqs, drawAxes, system, intersect, contourOnTop
            )
        )

        return plot

    def cartesianPointsToStrMatrix(
        self,
        positions: [(int, int)],
        character: str = "o",
    ):
        """
        Description
        -----------
        Generates a strMatrix with a character corresponding to the given position on the coordinate plane.
        Uses overlayStrMatrices(cartesianEqToStrMatrix(...),cartesianEqToStrMatrix(...))

        Parameters
        ----------
        positions: [(int,int)]
            The a list of the (x, y) tuples defining the positions of the points.
        character:str
            The character to be assigned to the corresponding position in the strMatrix

        Returns
        -------
            See overlayStrMatrices above
        """
        strMatrix = self.newCanvasMatrix()
        for pos in positions:
            pointStrMatrix = self.cartesianEqToStrMatrix(
                [f"x={pos[0]}", f"y={pos[1]}"],
                contourOnTop=True,
                intersect=character,
                drawAxes=False,
                system=True,
            )
            strMatrix = self.overlayStrMatrices(
                [strMatrix, pointStrMatrix], intersect=character
            )
        return strMatrix

    def plotCartesianAsciiPoints(
        self,
        positions: [(int, int)],
        character: str = "o",
    ):
        """
        Description
        -----------
        See cartesianPointsToStrMatrix above

        Returns
        -------
            See overlayStrMatrices above
        """
        strMatrix = self.cartesianPointsToStrMatrix(positions, character)
        return self.strMatrixToStr(strMatrix)


def test():
    """
    Description
    -----------
    Observe
    """
    plotter = AsciiPlotter()  # (canvasSize=(9, 19))

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

    # individual===============================================
    for eq in eqs:
        print("", tex2py(eq))
        print(plotter.plotCartesianAsciiEquations([eq]))

    # overlay===============================================
    plot = plotter.plotCartesianAsciiEquations(
        eqs, system=False, contourOnTop=False, intersect=None
    )
    for eq in eqs:
        print("[", tex2py(eq))
    print(plot)

    # overlay + contour + intersect===============================================
    plot = plotter.plotCartesianAsciiEquations(eqs)
    for eq in eqs:
        print(":", tex2py(eq))
    print(plot)

    # system===============================================
    plot = plotter.plotCartesianAsciiEquations(eqs, system=True)
    for eq in eqs:
        print("|", tex2py(eq))
    print(plot)

    # print(plotter.Y())


if __name__ == "__main__":
    test()