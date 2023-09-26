import numpy as np
import random
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


class ConsoleColors:  # the available console colors to use for coloring the string matrices
    END = b"\33[0m"
    BOLD = b"\33[1m"
    ITALIC = b"\33[3m"
    URL = b"\33[4m"
    BLINK = b"\33[5m"
    BLINK2 = b"\33[6m"
    SELECTED = b"\33[7m"
    BLACK = b"\33[30m"
    RED = b"\33[31m"
    GREEN = b"\33[32m"
    YELLOW = b"\33[33m"
    BLUE = b"\33[34m"
    VIOLET = b"\33[35m"
    BEIGE = b"\33[36m"
    WHITE = b"\33[37m"
    GREY = b"\33[90m"

    RED2 = b"\33[91m"
    GREEN2 = b"\33[92m"
    YELLOW2 = b"\33[93m"
    BLUE2 = b"\33[94m"
    VIOLET2 = b"\33[95m"
    BEIGE2 = b"\33[96m"
    WHITE2 = b"\33[97m"

    BLACKBG = b"\33[40m"
    REDBG = b"\33[41m"
    GREENBG = b"\33[42m"
    YELLOWBG = b"\33[43m"
    BLUEBG = b"\33[44m"
    VIOLETBG = b"\33[45m"
    BEIGEBG = b"\33[46m"
    WHITEBG = b"\33[47m"
    GREYBG = b"\33[100m"

    REDBG2 = b"\33[101m"
    GREENBG2 = b"\33[102m"
    YELLOWBG2 = b"\33[103m"
    BLUEBG2 = b"\33[104m"
    VIOLETBG2 = b"\33[105m"
    BEIGEBG2 = b"\33[106m"
    WHITEBG2 = b"\33[107m"


np.seterr(divide="ignore")  # remove div by zero because polar plots
np.seterr(invalid="ignore")  # remove div by zero because polar plots


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
        return self.coordLineTile(np.int_(self.canvasSize[::-1]) + 1, self.bounds[1]).T[
            ::-1
        ]

    def Theta(self):
        theta = (
            np.arctan(np.real(self.Y() / self.X()))
            + np.float_(self.X() < 0) * np.pi
            + np.float_((self.X() >= 0) * (self.Y() < 0)) * 2 * np.pi
        )
        theta[np.isnan(theta)] = 0
        return theta

    def Radius(self):
        return np.sqrt(np.real(self.X() ** 2 + self.Y() ** 2))

    def __init__(
        self,
        canvasSize: (int, int) = (45, 21),
        bounds: ((float, float), (float, float)) = ((-12, 12), (-12, 12)),
    ):
        """
        Parameters
        ----------
        canvasSize: (int, int)
            The number of (columns, lines) of the working space.
            It governs the size of the numpy arrays for the calculations
            and the output arrays and strings from functions. Floors to closest odd pair to avoid visual mismatch.
        bounds: ((float, float), (float, float))
            Governs the extent of the coordinate plane used for the calculations.
            format: ((min_x, max_x),(min_y, max_y))
        """
        _canvasSize = np.array(canvasSize)
        if _canvasSize[0] % 2 == 0:
            _canvasSize[0] -= 1
        if _canvasSize[1] % 2 == 0:
            _canvasSize[1] -= 1
        canvasSize = tuple(_canvasSize)
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
        self.x = self.X()
        self.y = self.Y()
        self.r = self.Radius()
        self.t = self.Theta()
        self.functions.update({"x": self.x})
        self.functions.update({"y": self.y})
        self.functions.update({"r": self.r})
        self.functions.update({"t": self.t})
        self.marchingSqMask = (
            {  # used for bitwise operations with charsetMode (see below)
                np.bool_([[1, 1], [1, 1]]).tobytes(): 0b0000000000000001,
                np.bool_([[1, 1], [0, 1]]).tobytes(): 0b0000000000000010,
                np.bool_([[1, 1], [1, 0]]).tobytes(): 0b0000000000000100,
                np.bool_([[1, 1], [0, 0]]).tobytes(): 0b0000000000001000,
                np.bool_([[1, 0], [1, 1]]).tobytes(): 0b0000000000010000,
                np.bool_([[1, 0], [0, 1]]).tobytes(): 0b0000000000100000,
                np.bool_([[1, 0], [1, 0]]).tobytes(): 0b0000000001000000,
                np.bool_([[1, 0], [0, 0]]).tobytes(): 0b0000000010000000,
                np.bool_([[0, 1], [1, 1]]).tobytes(): 0b0000000100000000,
                np.bool_([[0, 1], [0, 1]]).tobytes(): 0b0000001000000000,
                np.bool_([[0, 1], [1, 0]]).tobytes(): 0b0000010000000000,
                np.bool_([[0, 1], [0, 0]]).tobytes(): 0b0000100000000000,
                np.bool_([[0, 0], [1, 1]]).tobytes(): 0b0001000000000000,
                np.bool_([[0, 0], [0, 1]]).tobytes(): 0b0010000000000000,
                np.bool_([[0, 0], [1, 0]]).tobytes(): 0b0100000000000000,
                np.bool_([[0, 0], [0, 0]]).tobytes(): 0b1000000000000000,
            }
        )
        self.tmp = np.array2string(np.bool_([[0, 0], [0, 0]]))
        self.charset = [  # the characters used to represent different marching square states as a block of text in the console
            b"+",
            b"\\",
            b"/",
            b"-",
            b"\\",
            b"\\",
            b"|",
            b"/",
            b"/",
            b"|",
            b"/",
            b"\\",
            b"-",
            b"/",
            b"\\",
            b":",
        ]
        self.charsetMode = {  # different modes select relevant marching squares and replace them with a symbol from the charset having the same index as the multiplexed value of the square frommarchingSqMask
            "<=": 0b1111111111111110,
            "!=": 0b1000000000000001,
            ">=": 0b0111111111111111,
            "=": 0b0111111111111110,
            "<": 0b1000000000000000,
            ">": 0b0000000000000001,
        }
        self.colors = ConsoleColors
        self.colorCodes = [
            getattr(self.colors, attr)
            for attr in dir(self.colors)
            if not callable(getattr(self.colors, attr)) and not attr.startswith("__")
        ]

    def newCanvasMatrix(
        self, canvasSize: (int, int) = (-1, -1), character: str = " "
    ) -> np.string_:
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
        character: str
            The char to be used to fill the space
        Returns
        -------
        A np.ndarray of shape canvasSize, full of space strings (' ')
        """
        canvasSize = list(canvasSize)
        if canvasSize[0] <= 0:
            canvasSize[0] = self.canvasSize[0]
        if canvasSize[1] <= 0:
            canvasSize[1] = self.canvasSize[1]

        canvasMatrix = np.zeros((canvasSize[1], canvasSize[0]), dtype=np.dtype("<S13"))
        canvasMatrix[canvasMatrix == b""] = character
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
        # if not "x" in s and not "y" in s:
        #     s = f"y={split(s,[charsetModeString for charsetModeString, charsetModeMask in self.charsetMode.items()])[0]}"

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
                (np.sign(plane) + (1 if returnType == np.bool_ else 0) * bias) * bias
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
                strSquare = square.tobytes()
                charCandidate = self.marchingSqMask[strSquare]
                boolMatrix[i][j] = charCandidate & charsetMask > 0

        return boolMatrix

    def cartesianPlaneToStrMatrix(
        self,
        plane: np.ndarray,
        charsetMask: int = 0b0111111111111110,
        color: str = None,
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
        color: str = None
            The color in which the plot of this equation is drawn in the
            console (or in the string using ANSI color coding)
        Returns
        -------
        [[str,str,...],[str,str,...],...] a 2D array of characters that make up the ascii graph of the given plane.
        """
        canvasMatrix = self.newCanvasMatrix()
        for i in np.arange(len(plane) - 1):
            for j in np.arange(len(plane[0]) - 1):
                square = plane[i : i + 2, j : j + 2]
                strSquare = square.tobytes()
                char = " "
                charCandidate = self.marchingSqMask[strSquare]
                if charCandidate & charsetMask > 0:
                    char = self.charset[mux(charCandidate)]
                    canvasMatrix[i][j] = char

        if color != None:
            canvasMatrix = self.colorizeStrMatrix(color)

        return canvasMatrix

    def strMatrixToStr(self, strMatrix: np.ndarray) -> str:
        """
        Description
        -----------
        Stitches the rectangular array of characters to a block of string.

        Parameters
        ----------
        strMatrix: np.ndarray
            See cartesianEqsToStrMatrix above
        """
        canvasRows = [b"".join(char) for char in strMatrix]
        outputCanvas = b"\n".join(canvasRows)
        return outputCanvas.decode()

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
            See cartesianEqsToStrMatrix above
        strMatrix2: str
            See cartesianEqsToStrMatrix above

        contourOnTop: bool
            If set to True, the contour characters of the first matrix will be drawn on top of the second.

        intersect: str
            if contours from the two matrices occupy the same spot, it will be filled with the specified character

        Returns
        -------
        See cartesianEqsToStrMatrix above

        """
        newStrMatrix = self.newCanvasMatrix()
        decolorizedStrMatrices = [self.decolorizeStrMatrix(m) for m in strMatrices]
        for k in np.arange(len(strMatrices) - 1):
            strMatrix1 = strMatrices[k]
            strMatrix2 = strMatrices[k + 1]
            decolorizedStrMatrix1 = decolorizedStrMatrices[k]
            decolorizedStrMatrix2 = decolorizedStrMatrices[k + 1]

            for i in np.arange(self.canvasSize[1]):
                for j in np.arange(self.canvasSize[0]):
                    if decolorizedStrMatrix2[i][j] == b" ":
                        newStrMatrix[i][j] = strMatrix1[i][j]
                    else:
                        if intersect != None or contourOnTop:
                            contourChars = self.charset[1:-1] + (
                                [intersect] if intersect != None else []
                            )
                            if decolorizedStrMatrix1[i][j] in contourChars:
                                if (
                                    decolorizedStrMatrix2[i][j] in contourChars
                                    and intersect != None
                                ):
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

    def maskStrMatrix(self, strMatrix: np.ndarray, booleanMask: np.bool_) -> np.ndarray:
        """
        Description
        -----------
        Uses the booleanMask to remove characters from the strMatrix where the values at a specific index are False

        Parameters
        ----------
        strMatrix: numpy.ndarray
            See cartesianEqsToStrMatrix above
        booleanMask: np.bool_
            See planeToBool above

        Returns
        -------
        See cartesianEqsToStrMatrix above
        """
        newStrMatrix = self.newCanvasMatrix()
        for i in np.arange(self.canvasSize[1]):
            for j in np.arange(self.canvasSize[0]):
                if booleanMask[i][j]:
                    newStrMatrix[i][j] = strMatrix[i][j]
                else:
                    newStrMatrix[i][j] = b" "
        return newStrMatrix

    def compileCartesianEquation(self, eq: str) -> str:
        """
        Parameters
        ----------
        eq: str
            A text or LaTeX equation
        Returns
        -------
        A tuple (See cartesianEqsToStrMatrix above, See planeToBool above)
        """
        eq, mode = self.strToExpr(eq)
        plane = self.calculateCartesianPlane(eq)
        currentMatrix = self.cartesianPlaneToStrMatrix(plane, mode)
        canvasMask = self.planeToBool(plane, mode)
        return currentMatrix, canvasMask

    def cartesianAxes(self, color=None):  # defaults to grey
        if color == None:
            color = self.colors.GREY
        xAxis = self.compileCartesianEquation("y=0")[0]
        yAxis = self.compileCartesianEquation("x=0")[0]
        arrows = {
            "<": (self.bounds[0][0], 0),
            ">": (self.bounds[0][1], 0),
            "v": (0, self.bounds[1][0]),
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
        axesMatrix = self.overlayStrMatrices(
            [axesMatrix, arrowsMatrix], contourOnTop=False, intersect=False
        )
        axesMatrix = self.colorizeStrMatrix(axesMatrix, color)
        return axesMatrix

    def cartesianEqsToStrMatrix(
        self,
        eqs: [str],
        drawAxes: bool = True,
        system: bool = False,
        intersect: str = "x",
        contourOnTop=True,
        colors: [str] = [None],
    ) -> np.ndarray:
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
        colors: str = None
            The array of colors in which the graphs of the equations is drawn in the
            console (or in the string using ANSI color coding), ordered respectively.
            If len(colors) < len(eqs), the colors start repeating from the start
        Returns
        -------
        See overlayStrMatrices above
        """
        strMatrix = self.newCanvasMatrix()
        boolMask = strMatrix == b" "
        for i in np.arange(len(eqs)):
            eq = eqs[i]
            color = colors[i % len(colors)]
            currentMatrix, currentMask = self.compileCartesianEquation(eq)
            if system:
                boolMask &= currentMask
                strMatrix = self.maskStrMatrix(strMatrix, currentMask)
                currentMatrix = self.maskStrMatrix(currentMatrix, boolMask)
            if color != None:
                currentMatrix = self.colorizeStrMatrix(currentMatrix, color)
            strMatrix = self.overlayStrMatrices(
                [strMatrix, currentMatrix],
                intersect=intersect,
                contourOnTop=contourOnTop,
            )
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
        colors=[None],
    ) -> str:
        """
        Description
        -----------
        See cartesianEqsToStrMatrix above

        Returns
        -------
        str
        """
        plot = self.strMatrixToStr(
            self.cartesianEqsToStrMatrix(
                eqs, drawAxes, system, intersect, contourOnTop, colors
            )
        )

        return plot

    def cartesianPointsToStrMatrix(
        self, positions: [(int, int)], characters: [str] = ["o"], colors: [str] = [None]
    ):
        """
        Description
        -----------
        Generates a strMatrix with a characters corresponding to the given position on the coordinate plane.

        Parameters
        ----------
        positions: [(int,int)]
            The a list of the (x, y) tuples defining the positions of the points.
        characters: [str]
            The characters to be assigned to the corresponding position in the strMatrix. If len(characters) is smaller than len(positions), the chars start repeating from the start
        colors: str = None
            The array of colors in which the graphs of the equations is drawn in the
            console (or in the string using ANSI color coding), ordered respectively.
            If len(colors) < len(eqs), the colors start repeating from the start
        Returns
        -------
            See overlayStrMatrices above
        """
        strMatrix = self.newCanvasMatrix()
        xAxis = self.coordLine(self.canvasSize[0], self.bounds[0])
        yAxis = self.coordLine(self.canvasSize[1], self.bounds[1])[::-1]
        for i in np.arange(len(positions)):
            pos = positions[i]
            char = str.encode(characters[i % len(characters)])
            color = colors[i % len(colors)]
            if not (pos[0] >= xAxis[0] and pos[0] <= xAxis[-1]) or not (
                pos[1] <= yAxis[0] and pos[1] >= yAxis[-1]
            ):
                continue
            xDistance = np.abs(np.real(xAxis) - pos[0])
            xIndex = np.where(xDistance == xDistance.min())[0][-1]
            yDistance = np.abs(np.real(yAxis) - pos[1])
            yIndex = np.where(yDistance == yDistance.min())[0][-1]

            if color != None:
                char = color + char + self.colors.END

            strMatrix[yIndex][xIndex] = char

        return strMatrix

    def plotCartesianAsciiPoints(
        self,
        positions: [(int, int)],
        character: str = "o",
        drawAxes: bool = True,
        colors: [str] = [None],
    ):
        """
        Description
        -----------
        See cartesianPointsToStrMatrix above

        Returns
        -------
            See overlayStrMatrices above
        """
        strMatrix = self.cartesianPointsToStrMatrix(positions, character, colors=colors)
        if drawAxes:
            strMatrix = self.overlayStrMatrices([self.cartesianAxes(), strMatrix])
        return self.strMatrixToStr(strMatrix)

    def polarAxes(self, radii: (float) = (0.75,), color=None):
        if color == None:
            color = self.colors.GREY
        minBound = min(
            abs(self.bounds[0][1] - self.bounds[0][0]),
            abs(self.bounds[1][1] - self.bounds[1][0]),
        )
        strMatrix = self.cartesianEqsToStrMatrix(
            [f"x^2+y^2-{minBound/2*radius}^2=0" for radius in radii],
            intersect=None,
            contourOnTop=False,
            system=False,
            colors=[color],
        )
        return strMatrix

    def polarEqsToStrMatrix(
        self,
        eqs: [str],
        drawAxes: bool = True,
        system: bool = False,
        intersect: str = None,
        contourOnTop=False,
        axesRadii=(0.75,),
        colors: [str] = [None],
        bounds: [(float, float)] = [(-np.pi, np.pi)],
    ) -> np.ndarray:
        """
        Description
        -----------
        Magic

        Parameters
        ----------
        eqs: [str]
            A list of text or LaTeX equations to be drawn on one polar coordinate plane
        drawAxes: bool = True
            Whether coordinate axes y and x and a circle at different radii should be drawn behind the equations
        system: bool = False
            Whether to only show the graph of points that satisfy all given equations.
            If set to False, the equations will only be overlayed instead
        intersect: str = None
            See intersect from overlayStrMatrices above
        contourOnTop=True
            See contourOnTop from overlayStrMatrices above
        axesRadii: (float,) = (.75,)
            The radii of the circles that are part of the coordinate axes.
            The range between 0 and 1 is mapped between 0 an the smaller bound of the coordinate plane

        Returns
        -------
        See overlayStrMatrices above
        """
        strMatrix = self.cartesianEqsToStrMatrix(
            eqs=eqs,
            drawAxes=False,
            system=system,
            intersect=intersect,
            contourOnTop=contourOnTop,
            colors=colors,
        )
        if drawAxes:
            strMatrix = self.overlayStrMatrices(
                [
                    self.polarAxes(axesRadii),
                    strMatrix,
                ],
                contourOnTop=False,
                intersect=None,
            )

        return strMatrix

    def plotPolarAsciiEquations(
        self,
        eqs: [str],
        drawAxes: bool = True,
        system: bool = False,
        intersect: str = None,
        contourOnTop=True,
        axesRadii=(0.75,),
        colors: [str] = [None],
    ) -> str:
        """
        Description
        -----------
        See polarEqsToStrMatrix above

        Returns
        -------
        str
        """
        plot = self.strMatrixToStr(
            self.polarEqsToStrMatrix(
                eqs,
                drawAxes,
                system,
                intersect,
                contourOnTop,
                axesRadii=axesRadii,
                colors=colors,
            )
        )
        return plot

    def polarPointsToStrMatrix(
        self, positions: [(int, int)], characters: [str] = ["o"], colors: [str] = [None]
    ):
        """
        Description
        -----------
        Generates a strMatrix with a characters corresponding to the given position on the polar coordinate plane.

        Parameters
        ----------
        positions: [(int,int)]
            The a list of the (radius, theta) tuples defining the positions of the points.
            The angle is used in degrees.
        characters: [str]
            The characters to be assigned to the corresponding position in the strMatrix.
            If len(characters) is smaller than len(positions), the chars start repeating from the start

        Returns
        -------
            See overlayStrMatrices above
        """
        r, t = np.hsplit(np.array(positions), 2)
        t = np.deg2rad(t)
        xNorm = np.cos(t)
        yNorm = np.sin(t)
        pointsNorm = np.hstack((xNorm, yNorm))
        points = np.ndarray.tolist(pointsNorm * r)

        strMatrix = self.cartesianPointsToStrMatrix(points, characters, colors=colors)

        return strMatrix

    def plotPolarAsciiPoints(
        self,
        positions: [(int, int)],
        character: str = "o",
        drawAxes: bool = True,
        colors: [str] = [None],
    ):
        """
        Description
        -----------
        See polarPointsToStrMatrix above

        Returns
        -------
            See overlayStrMatrices above
        """
        strMatrix = self.polarPointsToStrMatrix(positions, character, colors=colors)
        if drawAxes:
            strMatrix = self.overlayStrMatrices([self.polarAxes(), strMatrix])
        return self.strMatrixToStr(strMatrix)

    def decolorizeStrMatrix(self, strMatrix):
        strMatrix = np.array(
            [[chr(val[-5 % len(val)]) for val in row] for row in strMatrix],
            dtype=np.dtype("<S11"),
        )
        return strMatrix

    def colorizeStrMatrix(self, strMatrix: np.ndarray, color: str) -> np.ndarray:
        strMatrix = self.decolorizeStrMatrix(strMatrix)
        prefixMatrix = self.newCanvasMatrix(character=color)
        suffixMatrix = self.newCanvasMatrix(character=self.colors.END)
        strMatrix = np.char.add(prefixMatrix, strMatrix)
        strMatrix = np.char.add(strMatrix, suffixMatrix)
        return strMatrix
