"""
Robby the Robot simulator
version 2.1.2

Written by Jim Marshall
Sarah Lawrence College, spring 2015
https://science.slc.edu/jmarshall

- Updated for Python 3, spring 2019
- Fixed some animation problems and other bugs, spring 2023
- Added better docstrings, spring 2026

Based on Chapter 9 of the book "Complexity: A Guided Tour"
by Melanie Mitchell, Oxford University Press, 2009.

Commands
--------
rw = World(10, 10)

rw.get_current_position()
rw.get_percept()
rw.get_percept_code()
rw.distribute_cans(density=0.50)
rw.goto(row, col)
rw.perform_action(action)

rw.north()
rw.south()
rw.east()
rw.west()
rw.stay()
rw.grab()
rw.move()
rw.look() -- same as rw.get_percept()

rw.show()
rw.graphics_off(message="")
rw.graphics_on()
rw.demo(strategy, steps=200, init=0.50)

rw.load(filename)
rw.save(filename)

rw.strategyM -- Melanie Mitchell's hand-designed strategy string

"""

from .graphics import *
from .graphics import update as module_update
import random, time, os

POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]

# parameters for demo method
PAUSE = 0.08
CYCLE_LIMIT = 3
FAST_STEPS = 100  # number of steps to run at high speed after detecting a cycle

UPDATE_RATE = 100  # seems to give reliable performance

class GridCell:
    def __init__(self, world, row, col):
        self.world = world
        self.row = row
        self.col = col
        # center of grid cell
        x = (col+1)*world.cellw + world.cellw//2
        y = (row+1)*world.cellh + world.cellh//2
        self.icons = {}
        iconNames = ["coke_can", "robby", "robby_can"]
        if row == world.topRow:
            iconNames.extend(["crash_n", "crash_can_n"])
        if row == world.bottomRow:
            iconNames.extend(["crash_s", "crash_can_s"])
        if col == world.leftCol:
            iconNames.extend(["crash_w", "crash_can_w"])
        if col == world.rightCol:
            iconNames.extend(["crash_e", "crash_can_e"])
        dirname = "robby" + os.sep
        for name in iconNames:
            self.icons[name] = Image(Point(x, y), dirname+name+".gif")
        if row == world.topRow:
            self.icons["ow_n"] = Image(Point(x, y-world.cellh), dirname+"ow_n.gif")
        if row == world.bottomRow:
            self.icons["ow_s"] = Image(Point(x, y+world.cellh), dirname+"ow_s.gif")
        if col == world.leftCol:
            self.icons["ow_w"] = Image(Point(x-world.cellw, y), dirname+"ow_w.gif")
        if col == world.rightCol:
            self.icons["ow_e"] = Image(Point(x+world.cellw, y), dirname+"ow_e.gif")
        self.contents = "E"
        self.icon = None
        self.owIcon = None
        if self.robbyIsHere():
            self.icon = self.icons["robby"]
            self.icon.draw(world)

    def robbyIsHere(self):
        return self.row == self.world.robbyRow and self.col == self.world.robbyCol

    def setContents(self, newContents):
        assert newContents in ["E", "C"]
        self.contents = newContents
        self.updateGraphics()

    def clearOwIcon(self):
        if self.owIcon is not None:
            self.owIcon.undraw()
            self.owIcon = None

    def updateGraphics(self, immediately=True):
        if not self.world.graphicsEnabled:
            return
        self.clearOwIcon()
        if self.robbyIsHere():
            newIcon = self.icons["robby_can" if self.contents == "C" else "robby"]
        else:
            newIcon = self.icons["coke_can"] if self.contents == "C" else None
        if newIcon is not self.icon:
            if newIcon is not None:
                newIcon.draw(self.world)
            if self.icon is not None:
                self.icon.undraw()
            self.icon = newIcon
        if immediately:
            module_update(UPDATE_RATE)

    def undrawRobby(self):
        if not self.world.graphicsEnabled:
            return
        self.clearOwIcon()
        if self.contents == "C":
            newIcon = self.icons["coke_can"]
            newIcon.draw(self.world)
            self.icon.undraw()
            self.icon = newIcon
        else:
            self.icon.undraw()
            self.icon = None
        module_update(UPDATE_RATE)

    def crashIntoWall(self, action):
        if not self.world.graphicsEnabled:
            return
        if action == "MoveNorth":
            direction = "_n"
        elif action == "MoveSouth":
            direction = "_s"
        elif action == "MoveEast":
            direction = "_e"
        elif action == "MoveWest":
            direction = "_w"
        else:
            raise Exception("bad crash action: %s" % action)
        crashIconName = "crash_can" if self.contents == "C" else "crash"
        crashIcon = self.icons[crashIconName + direction]
        owIcon = self.icons["ow" + direction]
        if self.icon is not None:
            self.icon.undraw()
        if self.owIcon is not None:
            self.owIcon.undraw()
        crashIcon.draw(self.world)
        owIcon.draw(self.world)
        self.icon = crashIcon
        self.owIcon = owIcon
        module_update(UPDATE_RATE)


class World(GraphWin):

    # Melanie Mitchell's hand-designed strategy
    strategyM = "656353656252353252656353656151353151252353252151353151656353656252353252656353656050353050252353252050353050151353151252353252151353151050353050252353252050353050656353656252353252656353656151353151252353252151353151656353656252353252656353454"

    def __init__(self, numRows=10, numCols=10):
        """Creates a new numRows-by-numCols grid world (default size is 10x10)."""
        assert numRows >= 1 and numCols >= 1, "invalid world dimensions"
        # create the grid
        iconSize = 40
        spacing = 3
        windowWidth = (iconSize + 2*spacing) * (numCols + 2)
        windowHeight = (iconSize + 2*spacing) * (numRows + 2)
        GraphWin.__init__(self, "Robby the Robot", windowWidth, windowHeight)
        self.setBackground("white")
        self.graphicsEnabled = True
        self.blank = Rectangle(Point(0,0), Point(windowWidth,windowHeight))
        self.blank.setFill("gray")
        self.blank.setOutline("gray")
        self.text = Text(Point(windowWidth//2, windowHeight//2), "")
        self.text.setStyle("italic")
        self.text.setSize(36)
        self.cellw = iconSize + 2*spacing
        self.cellh = iconSize + 2*spacing
        x1, y1 = self.cellw, self.cellh
        x2, y2 = self.cellw*(numCols+1), self.cellh*(numRows+1)
        x, y = x1, y1
        for r in range(max(numRows,numCols)+1):
            Line(Point(x1, y), Point(x2, y)).draw(self)
            Line(Point(x, y1), Point(x, y2)).draw(self)
            x += self.cellw
            y += self.cellh
        self.numRows = numRows
        self.numCols = numCols
        self.topRow = 0
        self.bottomRow = numRows-1
        self.leftCol = 0
        self.rightCol = numCols-1
        # current position of robby
        self.robbyRow = 0
        self.robbyCol = 0
        # create the cells
        self.grid = [[GridCell(self,r,c) for c in range(numCols)] for r in range(numRows)]
        module_update(UPDATE_RATE)
        
    def __repr__(self):
        return "RobbyWorld(%d x %d)" % (self.numRows, self.numCols)

    # just for the user interface if needed
    def update(self, rate=None):
        if rate is None:
            module_update()
        else:
            module_update(rate)

    # graphics_off should NOT call module_update if the graphics
    # are already off (and vice versa for graphics_on) because if
    # the fitness function includes a call to graphics_off, this
    # will slow down the GA enormously

    def graphics_off(self, message=""):
        """Turns off the graphics.  This is useful for simulating many
        actions at high speed when evaluating the fitness of a strategy.
        The optional message string will be displayed while the graphics
        are turned off."""
        if self.graphicsEnabled:
            self.blank.draw(self)
            self.text.setText(message)
            self.text.draw(self)
            self.graphicsEnabled = False
            module_update(UPDATE_RATE)

    def graphics_on(self):
        """Turns the graphics back on, and updates the grid to reflect
        the current state of the world."""
        if not self.graphicsEnabled:
            self.text.undraw()
            self.blank.undraw()
            self.graphicsEnabled = True
            self._updateGrid()

    def _updateGrid(self):
        for r in range(self.numRows):
            for c in range(self.numCols):
                self.grid[r][c].updateGraphics(immediately=False)
        module_update(UPDATE_RATE)

    def distribute_cans(self, density=0.50):
        """Randomly distributes soda cans throughout the world, with
        0 <= density <= 1 specifying the probability of a can occupying
        a grid cell.  The default value is 0.50."""
        for r in range(self.numRows):
            for c in range(self.numCols):
                self.grid[r][c].contents = "C" if random.uniform(0, 1) < density else "E"
        if self.graphicsEnabled:
            self._updateGrid()

    def demo(self, strategy, steps=200, init=0.50):
        """Turns on the graphics and demos a strategy for the specified number
        of simulation steps (default 200) with a randomized soda can density of
        init (default 0.50).  Optionally, init can be a string specifying the
        name of a grid world configuration file created with the save command.
        Note: this method cannot be used to calculate the fitness of a strategy;
        it simply shows the strategy in action."""
        if type(strategy) is not str or len(strategy) != 243:
            raise Exception("strategy is not a string of length 243")
        for char in strategy:
            if char not in "0123456":
                raise Exception("strategy contains a bad character: '%s'" % char)
        if type(steps) is not int or steps < 1:
            raise Exception("steps must be an integer > 0")
        if type(init) is str:
            # init is a config file
            self.load(init)
        elif type(init) in [int, float] and 0 <= init <= 1:
            # init is a can density
            self.goto(0, 0)
            self.distribute_cans(init)
        else:
            raise Exception("invalid initial configuration")
        history = []
        cycleDetected = False
        cycleStart = 0
        self.graphics_on()
        time.sleep(1)
        for i in range(steps):
            p = self.get_percept_code()
            action = POSSIBLE_ACTIONS[int(strategy[p])]
            self.perform_action(action)
            module_update(UPDATE_RATE)
            # This section checks to see if Robby is going in circles,
            # and speeds up the simulation if a repetitive pattern of
            # behavior is detected. It works reasonably well most of
            # the time, but there is still some room for improvement.
            if not cycleDetected:
                # skip after having detected a cycle
                time.sleep(PAUSE)
                state = [action, self.robbyRow, self.robbyCol, self._gridContents()]
                if action != "MoveRandom":
                    period = self._checkForCycle(state, history, CYCLE_LIMIT)
                    if period > 0:
                        #print("cycle of period %d detected" % period)
                        cycleDetected = True
                        if period == 1:
                            runFastUntil = i + FAST_STEPS//2
                        else:
                            runFastUntil = i + FAST_STEPS
                history.append(state)
            elif self.graphicsEnabled and i > runFastUntil:
                # disable graphics after running for FAST_STEPS
                self.graphicsEnabled = False
        if not self.graphicsEnabled:
            self.graphicsEnabled = True
            self._updateGrid()
        return None

    def _checkForCycle(self, state, history, limit):
        for period in range(1, len(history)//limit + 1):
            if self._checkForCycleOfPeriod(period, state, history, limit):
                # cycle found
                return period
        return -1

    def _checkForCycleOfPeriod(self, period, state, history, limit):
        if len(history) < period * limit:
            # not enough history yet
            return False
        for i in range(1, limit+1):
            k = len(history) - i * period
            if history[k] != state:
                return False
        return True

    def _gridContents(self):
        return "".join([self.grid[r][c].contents for r in range(self.numRows) for c in range(self.numCols)])

    def perform_action(self, action):
        """Causes Robby to perform the specified action, where action
        is one of the strings "MoveNorth", "MoveSouth", "MoveEast",
        "MoveWest", "StayPut", "PickUpCan", or "MoveRandom". The value
        returned is the amount of reward or punishment received for
        the action."""
        if action not in POSSIBLE_ACTIONS:
            print("ERROR -- possible actions are:\n%s" % POSSIBLE_ACTIONS)
            return
        if action == "MoveNorth" and self.robbyRow == self.topRow or \
                action == "MoveSouth" and self.robbyRow == self.bottomRow or \
                action == "MoveEast" and self.robbyCol == self.rightCol or \
                action == "MoveWest" and self.robbyCol == self.leftCol:
            self.grid[self.robbyRow][self.robbyCol].crashIntoWall(action)
            return -5
        elif action == "MoveNorth":
            self.grid[self.robbyRow][self.robbyCol].undrawRobby()
            self.robbyRow -= 1
            self.grid[self.robbyRow][self.robbyCol].updateGraphics()
            return 0
        elif action == "MoveSouth":
            self.grid[self.robbyRow][self.robbyCol].undrawRobby()
            self.robbyRow += 1
            self.grid[self.robbyRow][self.robbyCol].updateGraphics()
            return 0
        elif action == "MoveEast":
            self.grid[self.robbyRow][self.robbyCol].undrawRobby()
            self.robbyCol += 1
            self.grid[self.robbyRow][self.robbyCol].updateGraphics()
            return 0
        elif action == "MoveWest":
            self.grid[self.robbyRow][self.robbyCol].undrawRobby()
            self.robbyCol -= 1
            self.grid[self.robbyRow][self.robbyCol].updateGraphics()
            return 0
        elif action == "MoveRandom":
            return self.perform_action(random.choice(["MoveNorth", "MoveSouth", "MoveEast", "MoveWest"]))
        elif action == "StayPut":
            self.grid[self.robbyRow][self.robbyCol].updateGraphics()
            return 0
        elif action == "PickUpCan":
            reward = +10 if self.grid[self.robbyRow][self.robbyCol].contents == "C" else -1
            self.grid[self.robbyRow][self.robbyCol].setContents("E")
            return reward
        else:
            # should never happen
            raise Exception("bad action: %s" % action)
            
    # abbreviations
    def north(self):
        """Same as perform_action("MoveNorth")."""
        return self.perform_action("MoveNorth")
    def south(self):
        """Same as perform_action("MoveSouth")."""
        return self.perform_action("MoveSouth")
    def east(self):
        """Same as perform_action("MoveEast")."""
        return self.perform_action("MoveEast")
    def west(self):
        """Same as perform_action("MoveWest")."""
        return self.perform_action("MoveWest")
    def stay(self):
        """Same as perform_action("StayPut")."""
        return self.perform_action("StayPut")
    def grab(self):
        """Same as perform_action("PickUpCan")."""
        return self.perform_action("PickUpCan")
    def move(self):
        """Same as perform_action("MoveRandom")."""
        return self.perform_action("MoveRandom")
    def look(self):
        """Same as get_percept()."""
        return self.get_percept()

    def get_percept(self, codeNum=None):
        """Returns the percept string specifying what Robby currently sees to
        the North, South, East, West, and Here (in that order).  If codeNum
        is specified, returns the percept string for that code number."""
        if codeNum != None:
            assert type(codeNum) == int and 0 <= codeNum < 243, "bad percept code number: %s" % codeNum
            s = ''.join([str(codeNum//(3**n) % 3) for n in (4,3,2,1,0)])  # convert to base 3
            return s.replace("0", "E").replace("1", "C").replace("2", "W")
        s = ""
        s += "W" if self.robbyRow == self.topRow else self.grid[self.robbyRow-1][self.robbyCol].contents
        s += "W" if self.robbyRow == self.bottomRow else self.grid[self.robbyRow+1][self.robbyCol].contents
        s += "W" if self.robbyCol == self.rightCol else self.grid[self.robbyRow][self.robbyCol+1].contents
        s += "W" if self.robbyCol == self.leftCol else self.grid[self.robbyRow][self.robbyCol-1].contents
        s += self.grid[self.robbyRow][self.robbyCol].contents
        return s

    def get_percept_code(self, percept=None):
        """Returns the code number of the specified percept string as an integer
        in the range 0-242, or Robby's current percept string if percept=None.
        Example: returns 19 for "EEWEC"."""
        if percept == None:
            percept = self.get_percept()
        else:
            assert type(percept) == str and len(percept) == 5, "bad percept string: %s" % percept
            badChars = ''
            for ch in percept:
                if ch not in 'ecwECW':
                    badChars += ch
            assert badChars == '', "bad characters in percept string: %s" % badChars
            percept = percept.upper()
        return int(percept.replace("E", "0").replace("C", "1").replace("W", "2"), 3)

    def get_current_position(self):
        """Returns the current 0-based row, column position of Robby in the grid"""
        return self.robbyRow, self.robbyCol

    def goto(self, newRow, newCol):
        """Moves Robby to the specified grid location."""
        assert 0 <= newRow < self.numRows and 0 <= newCol < self.numCols
        self.grid[self.robbyRow][self.robbyCol].undrawRobby()
        self.robbyRow = newRow
        self.robbyCol = newCol
        self.grid[self.robbyRow][self.robbyCol].updateGraphics()

    def show(self):
        """Prints out a non-graphical representation of the current state of the world."""
        s = ""
        for r in range(self.numRows):
            for c in range(self.numCols):
                if r == self.robbyRow and c == self.robbyCol:
                    s += "CR " if self.grid[r][c].contents == "C" else "R  "
                else:
                    s += "C  " if self.grid[r][c].contents == "C" else ".  "
            s += "\n"
        print(s.strip())

    def load(self, filename):
        """Loads a grid world configuration from a text file,
        where filename is a string."""
        f = open(filename)
        lines = [line.strip() for line in f]
        f.close()
        if len(lines) != self.numRows + 2 or len(lines[0]) != self.numCols:
            print("ERROR -- invalid grid format in file %s" % filename)
            return
        self.goto(int(lines[-2]), int(lines[-1]))
        for r in range(self.numRows):
            for c in range(self.numCols):
                self.grid[r][c].contents = "E" if lines[r][c] == "." else "C"
        self._updateGrid()

    def save(self, filename):
        """Saves the current grid world configuration as a text file,
        where filename is a string."""
        f = open(filename, 'w')
        for r in range(self.numRows):
            for c in range(self.numCols):
                f.write("." if self.grid[r][c].contents == "E" else "C")
            f.write("\n")
        f.write("%d\n" % self.robbyRow)
        f.write("%d\n" % self.robbyCol)
        f.close()
        print("Configuration saved in file %s" % filename)

