
class Globals:
    fig = None
    selected_axis = None
    height = 0.5
    plane = None
    center_point = None
    SA_AXIS = None


glvars = Globals()


class Lines:
    lines = dict()
    '''
    {
        "name":{
            "A": Point3D,
            "B": Point3D,
            "rotations": [(x,y,z),...],
            
        }
    }
    '''

    def setPoints(self, name, A, B):
        if name not in self.lines:
            self.lines[name] = {}
        self.lines[name]["A"] = A
        self.lines[name]["B"] = B

    def addRotation(self, name, x, y, z):
        if name not in self.lines:
            self.lines[name] = {}
        self.lines[name]["rotations"].append((x, y, z))

    def clearRotation(self, name):
        if name not in self.lines:
            self.lines[name] = {}
        self.lines[name]["rotations"] = []


gllines = Lines()
