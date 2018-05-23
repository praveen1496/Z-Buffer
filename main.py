from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from random import randint
import numpy as np
import os

lines = ""
vertexes = ""

GRAPH_OFFSET = (1,   0,   1)
OFFSET = 3
SCREEN_SIZE = 600
SCREEN_SCOPE = SCREEN_SIZE*(1+OFFSET)


#Reading the text file with coordinates
def ReadFile(path=None):
    if not path:
        path = "C:/Users/prave/PycharmProjects/zbuffer/data/soccer.d.txt"       #Change the path to the required file
    _sum = []
    vertexes = []
    lines = []
    _first_line = 1
    _vertex_line = 0
    with open(path) as f:
        for data in f.readlines():
            data = data.strip('\n')
            # nums for one line
            nums = data.split(" ")
            # read fist line
            if _first_line:
                for re in nums:
                    if re:
                        _sum.append(re)
                _first_line = 0
                continue
            _vertex_line = int(_sum[1])

    # reading vertices and lines
    with open(path) as f:
        i = 1
        for data in f.readlines():
            data = data.strip('\n')
            nums = data.split(" ")
            if i:
                i = 0
                continue
            if _vertex_line:
                _vertex = []
                #storing each x,y,z coordinates into vertex
                for re in nums:
                    if re:
                        _vertex.append(float(re))
                # storing all the vertexes coordinates into vertexes
                vertexes.append(_vertex)
                _vertex_line -= 1
                continue

            # lines
            _line = []
            for re in nums:
                if re:
                    _line.append(int(re)-1)    #Subtracting each coordinate by 1, since indexing starts from 0
            lines.append(_line[1:])
    return vertexes, lines


#funtion to perform cross product
def vertex_multiply(a, b):
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return [cx, cy, cz]


#function to perform dot product
def vertex_dot_multiply(a, b):
    ax = a[0]
    ay = a[1]
    az = a[2]
    bx = b[0]
    by = b[1]
    bz = b[2]
    return ax*bx + ay*by + az*bz


#Function to find M perspective and M view
def FindPersView(c, p, v_prime, d=1.0, f=500.0, h=15.0):
    C = np.mat(c)
    P = np.mat(p)
    #To calculate this, N = 1/math.sqrt((P-C)*(P-C).T) * (P-C)
    N = (P-C) / np.linalg.norm(P - C)  # Z-axis
    N = N.tolist()[0]
    U = vertex_multiply(N, v_prime) / np.linalg.norm(vertex_multiply(N, v_prime))  # X-axis
    V = vertex_multiply(U, N)  # Y-axis
    U = U.tolist()
    U_val = U + [0]
    V_val = V + [0]
    N_val = N + [0]
    R = np.mat([U_val, V_val, N_val, [0, 0, 0, 1]])
    T = np.mat([[1, 0, 0, -c[0]], [0, 1, 0, -c[1]], [0, 0, 1, -c[2]], [0, 0, 0, 1]])
    #Calculating M view
    M_view = R * T
    #Calculating M perspective
    M_pers = np.mat([[d/h, 0, 0, 0], [0, d/h, 0, 0], [0, 0, f/(f-d), -d*f/(f-d)], [0, 0, 1, 0]])
    return M_view, M_pers


#Function to perform the transformation
def transformation(c, p, v_prime, d=1.0, f=500.0, h=15.0):
    M_view, M_pers = FindPersView(c, p, v_prime, d, f, h)
    vertexes, lines = ReadFile()
    # Mview * points
    view_vs = []
    for v in vertexes:
        v = v + [1]
        v = np.mat(v).T     #Self Transpose
        view_v = M_view * v
        view_v = view_v.T.tolist()[0]
        view_vs.append(view_v)
    #To find normal of the surface
    visible_surfaces = []
    for face in lines:
        ves = []
        for v in face:
            ves.append(view_vs[v])
        if not ves:
            continue
        if len(ves) < 3:
            visible_surfaces.append(face)
            continue
        dot1 = np.mat(ves[0])
        dot2 = np.mat(ves[1])
        dot3 = np.mat(ves[2])
        line1 = dot2 - dot1
        line2 = dot3 - dot2
        line_of_sight = (np.mat([0, 0, 0]) - np.mat(ves[0][:-1])).tolist()[0]
        vertex1 = line1.tolist()[0][0:-1]
        vertex2 = line2.tolist()[0][0:-1]
        normal = vertex_multiply(vertex1, vertex2)
        visible = vertex_dot_multiply(normal, line_of_sight)
        if visible > 0:
            visible_surfaces.append(face)
    new_vs = []
    for v in view_vs:
        v = v
        v = np.mat(v).T
        new_v = M_pers * v  # Mp*Mv*V.T
        new_v = new_v.T.tolist()[0]   # lists
        # dividing x,y,z by W
        new_v[0] = int((new_v[0] / new_v[-1] + OFFSET) * SCREEN_SIZE)
        new_v[1] = int((new_v[1] / new_v[-1] + OFFSET) * SCREEN_SIZE)
        new_v[2] = abs((new_v[2] / new_v[-1]))
        new_vs.append(new_v)
    return new_vs, visible_surfaces


#Function for Z Buffer
def zbuffer():
    depth_buffer, frame_buffer = Find_Depth_Frame_Buffer()
    ScanLine(depth_buffer, frame_buffer)    #Perform Scan line
    glBegin(GL_POINTS)
    #Coloring
    for x in range(len(frame_buffer)):
        for y in range(len(frame_buffer)):
            color = frame_buffer[x][y]
            glColor3f(color[0], color[1], color[2])
            glVertex2i(x, y)
    glEnd()
    glFlush()

#Function to implement Scan line
def ScanLine(depth_buffer, frame_buffer):
    # init data
    d = 3.8
    f = 1
    h = 0.5
    V_prime = [0, 1, 0]  # Y-direction of camera
    vertexes, lines = transformation(C, P, V_prime, d, f, h)

    surfaces = []
    for line in lines:
        surface = []
        for v in line:
            vertex = vertexes[v][:-1]
            surface.append(vertex)
        surfaces.append(surface)

    for s in surfaces:
        r = randint(1, 255)/255.0
        g = randint(1, 255)/255.0
        b = randint(1, 255)/255.0
        color = (r, g, b)
        smax, smin = ScanlineScope(s)

        EdgeTable = Edge_Table(smax, smin)
        AET = EdgeModel()

        v_num = len(s)
        for i in range(v_num):
            x0 = s[(i-1+v_num) % v_num][0]
            x1 = s[i][0]
            z1 = s[i][2]
            x2 = s[(i+1) % v_num][0]
            z2 = s[(i+1) % v_num][2]
            x3 = s[(i+2) % v_num][0]
            y0 = s[(i-1 + v_num) % v_num][1]
            y1 = s[i][1]
            y2 = s[(i + 1) % v_num][1]
            y3 = s[(i + 2) % v_num][1]

            if y1 == y2:
                continue

            #Finding Xmin, Xmax, Ymin, Ymax
            ymin = y1 if y1 < y2 else y2
            ymax = y1 if y1 > y2 else y2
            xmin = x1 if y1 < y2 else x2
            dx = (x1-x2) * 1.0 / (y1-y2)

            if (y2 > y1 > y0) or (y1 > y2 > y3):
                ymin += 1
                xmin += dx

            edge = EdgeModel()
            edge.ymax = ymax
            edge.xmin = xmin
            edge.dx = dx
            edge.edge_next = EdgeTable[ymin]
            edge.edge_vertex1 = [x1, y1, z1]
            edge.edge_vertex2 = [x2, y2, z2]
            EdgeTable[ymin] = edge
        Scanning(smin, smax, EdgeTable, AET, color, depth_buffer, frame_buffer)


def Scanning(ymin, ymax, edge_table, AET, color, depth_buffer, frame_buffer):

    for y in range(ymin, ymax + 1):

        while edge_table[y]:
            edge_insert = edge_table[y]

            edge_in_AET = AET
            while edge_in_AET.edge_next:
                if edge_insert.xmin > edge_in_AET.edge_next.xmin:
                    edge_in_AET = edge_in_AET.edge_next
                    continue

                if edge_insert.xmin == edge_in_AET.edge_next.xmin and edge_insert.dx > edge_in_AET.edge_next.dx:
                    edge_in_AET = edge_in_AET.edge_next
                    continue
                break

            edge_table[y] = edge_insert.edge_next
            edge_insert.edge_next = edge_in_AET.edge_next
            edge_in_AET.edge_next = edge_insert

        p = AET
        while p.edge_next and p.edge_next.edge_next:
            v1 = p.edge_next.edge_vertex1
            v2 = p.edge_next.edge_vertex2
            v3 = p.edge_next.edge_next.edge_vertex1
            v4 = p.edge_next.edge_next.edge_vertex2
            Za = v1[2] - (v1[2]-v2[2]) * (v1[1]-y)*1.0 / (v1[1] - v2[1])
            Zb = v3[2] - (v3[2]-v4[2]) * (v3[1]-y)*1.0 / (v3[1] - v4[1])
            for x in range(int(p.edge_next.xmin), int(p.edge_next.edge_next.xmin)):
                # TODO X, Y, Z, COLOR
                xa = int(p.edge_next.xmin)
                xb = int(p.edge_next.edge_next.xmin)
                xp = x
                Zp = Zb - (Zb - Za) * (xb - xp)*1.0 / (xb - xa)

                if x >= SCREEN_SCOPE or y >= SCREEN_SCOPE:
                    continue

                if Zp < depth_buffer[x][y]:
                    depth_buffer[x][y] = Zp
                    frame_buffer[x][y] = color
                # glVertex2i(x, y)

            p = p.edge_next.edge_next
        p = AET
        while p.edge_next:
            if p.edge_next.ymax == y:
                p.edge_next = p.edge_next.edge_next
            else:
                p = p.edge_next

        p = AET
        while p.edge_next:
            p.edge_next.xmin += p.edge_next.dx
            p = p.edge_next


#Finding Scanline scope
def ScanlineScope(vertexes):
    ymax = 0
    ymin = SCREEN_SCOPE
    for v in vertexes:
        if v[1] > ymax:
            ymax = v[1]
        if v[1] < ymin:
            ymin = v[1]
    return ymax, ymin

#Creating edge table
def Edge_Table(ymax, ymin):
    edge_table = {}
    for i in range(ymin, ymax+1):
        edge_table[i] = None
    return edge_table

#Edge model class
class EdgeModel:
    def __init__(self):
        self.ymax = None
        self.xmin = None
        self.dx = None
        self.edge_next = None
        self.edge_vertex1 = None
        self.edge_vertex2 = None


#Function to get frame buffer and depth buffer
def Find_Depth_Frame_Buffer():
    # init depth_buffer
    depth_buffer = [[1 for col in range(SCREEN_SCOPE)] for row in range(SCREEN_SCOPE)]
    frame_buffer = [[(0, 0, 0) for col in range(SCREEN_SCOPE)] for row in range(SCREEN_SCOPE)]

    return depth_buffer, frame_buffer


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    gluOrtho2D(0, SCREEN_SCOPE, 0, SCREEN_SCOPE)


#for the output window
def Output():
    global lines
    global vertexes
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
    glutInitWindowSize(SCREEN_SIZE, SCREEN_SIZE)
    glutCreateWindow("Output")
    glutDisplayFunc(zbuffer)
    init()
    glutMainLoop()



if __name__ == '__main__':
    C = [4.0, 5.0, 7.0]  # Camera position
    P = [0.0, 0.0, 0.0]
    V_prime = [0, 1, 0]  # V' co-ordinates, Y-direction of Camera
    a = [2, 3, 5]
    b = [0, 1, 0]
    e1 = EdgeModel()
    e2 = EdgeModel()
    e3 = EdgeModel()
    e_in = EdgeModel()
    e1.dx = 1
    e2.dx = 2
    e3.dx = 3
    e_in.dx = 2.5
    e1.edge_next = e2
    e2.edge_next = e3
    e3.edge_next = None
    p = e1
    while p.edge_next:
        if e_in.dx > p.edge_next.dx:
            p = p.edge_next
            continue
        break
    e_in.edge_next = p.edge_next
    p.edge_next = e_in
    Output()
