import numpy as np
from PIL import Image
import math
import random
from obj_parser import Obj_Parser


class image:

    def __init__(self, h, w, color=[255, 255, 255]):

        def CreateImage(h, w, color):
            '''создает  из-е размером h*w цвета color (по умолчанию бое)'''
            mas = np.empty((h, w, 3), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    mas[i][j] = color
            return mas
        self.h = h
        self.w = w
        self.__color = color
        self.__mas = CreateImage(self.h, self.w, self.__color)

    def size(self):
        print(self.__h, ' x ', self.__w)

    def draw_point(self, x, y, color=[0, 0, 0]):
        r, g, b = color[0], color[1], color[2]
        # т.к. [номер строки][номер столба], y,x - помняли местами
        self.__mas[y][x][0] = r
        self.__mas[y][x][1] = g
        self.__mas[y][x][2] = b

    def draw_line(self, x0, y0, x1, y1, color=[0, 0, 255]):
        t = 0
        while t < 1:
            x = round(x0*(1-t) + x1*t)
            y = round(y0*(1-t) + y1*t)
            self.draw_point(x, y, color)
            t += 0.01

    def draw_brezenkhem_line(self, x0, y0, x1, y1, color=[0, 0, 0]):
        '''Алгоритм Брезенхема 
        '''
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0
        derror = abs(dy*2)
        error = 0
        y = y0
        x = x0

        while x <= x1:
            if steep:
                self.draw_point(y, x, color)
            else:
                self.draw_point(x, y, color)
            error += derror
            if error > abs(dx):
                if y1 > y0:
                    y += 1
                else:
                    y -= 1
                error -= 2*abs(dx)
            x += 1

    def rotate_point(self, Point, phi, CenterPt, color=[0, 0, 0]):
        '''на массиве Img повернуть точку point на угол phi относительно точки CenterPt(t1,t2)
                цвет по умолчанию - черный '''

        rotate_matrix = np.array([[math.cos(phi), math.sin(phi)],
                                  [-math.sin(phi), math.cos(phi)]])
        tx, ty = CenterPt[0], CenterPt[1]  # координаты центра вращения
        x, y = Point[0]-tx, Point[1]-ty

        coordinate = np.array([x, y])
        # домножение на матрицу поворота
        new_coordinate = rotate_matrix.dot(coordinate)
        x_new, y_new = round(new_coordinate[0])+tx, round(new_coordinate[1])+ty

        # x_gran, y_gran = Img.shape[0],Img.shape[1] # границы массива (изображения)
        # if ((x_new < 0) or (x_new > x_gran) or (y_new < 0) or (y_new > y_gran)): return
        # print(x_new,y_new)
        self.draw_point(x_new, y_new, [0, 0, 0])  # новую точку в черный цвет
        # self.draw_point(Point[0], Point[1], self.__color) #старую точку в цвет изображения

    def rotate_point_3D(self, x, y, z, alpha, beta, gama):
        '''поворачивает точку x,y,z на углы альфа, бета, гамма в пространстве '''
        rotate_matrix1 = np.array([[1, 0, 0], [0, math.cos(alpha), math.sin(alpha)], [
                                  0, -math.sin(alpha), math.cos(alpha)]])
        rotate_matrix2 = np.array([[math.cos(beta), 0, math.sin(beta)], [
                                  0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
        rotate_matrix3 = np.array([[math.cos(gama), math.sin(
            gama), 0], [-math.sin(gama), math.cos(gama), 0], [0, 0, 1]])
        dot1 = rotate_matrix1.dot(rotate_matrix2)
        rotate_matrix = dot1.dot(rotate_matrix3)  # матрица поворота 3D

        coordinates = np.array([x, y, z])  # старые координаты
        new_coordinates = coordinates.dot(rotate_matrix)  # новые координаты
        x_new = round(new_coordinates[0])
        y_new = round(new_coordinates[1])
        z_new = round(new_coordinates[2])

        self.draw_point(x_new, y_new, [0, 0, 0])  # новую точку в черный цвет

    def show(self):
        '''вывод картинки'''
        Image.fromarray(self.__mas).show()

    def save(self, name_of_file):
        Image.fromarray(self.__mas).save(name_of_file)

    def draw_bar(self, Point, A0, A1, A2, Color):
        '''принимает точку P=(x,y), вершины Ai=(xi,yi), Color = (r,g,b)
         если ее барицентрические координаты >= 0 - закрашивает'''

        def get_bar_coordinate(P, A0, A1, A2):
            ''' принимает координаты точки P(x,y), Координаты трех вершин  Ai(xi,yi)
                    возвращает l0,l1,l2 - барицентрические координаты точки P(x,y)'''
            x, y = P[0], P[1]
            x0, y0 = A0[0], A0[1]
            x1, y1 = A1[0], A1[1]
            x2, y2 = A2[0], A2[1]
            l0 = ((y-y2)*(x1-x2)-(x-x2)*(y1-y2)) / \
                ((y0-y2)*(x1-x2)-(x0-x2)*(y1-y2))
            l1 = ((y-y0)*(x2-x0)-(x-x0)*(y2-y0)) / \
                ((y1-y0)*(x2-x0)-(x1-x0)*(y2-y0))
            l2 = ((y-y1)*(x0-x1)-(x-x1)*(y0-y1)) / \
                ((y2-y1)*(x0-x1)-(x2-x1)*(y0-y1))
            return l0, l1, l2

        l0, l1, l2 = get_bar_coordinate(Point, A0, A1, A2)
        if (l0 >= 0 and l1 >= 0 and l2 >= 0):
            self.draw_point(Point[0], Point[1], Color)
        else:
            return

    def draw_treugolnik(self, A0, A1, A2, Color):
        ''' на Img закрашивает треугольник с вершинами A0,A1,A2 в цвет Color(r,g,b)'''
        # границы треугольника
        x_min = min(A0[0], A1[0], A2[0])
        x_max = max(A0[0], A1[0], A2[0])
        y_min = min(A0[1], A1[1], A2[1])
        y_max = max(A0[1], A1[1], A2[1])

        x_gran, y_gran = self.w, self.h  # границы массива (изображения)
        x_min, x_max, y_min, y_max = int(np.floor(x_min)), int(
            np.ceil(x_max)), int(np.floor(y_min)), int(np.ceil(y_max))

        if x_min < 0:
            x_min = 0
        if x_max > x_gran:
            x_max = x_gran
        if y_min < 0:
            y_min = 0
        if y_max > y_gran:
            y_max = y_gran

        for i in range(x_min, x_max):  # от мин до макс по Х
            for j in range(y_min, y_max):  # от мин до макс по y
                x, y = i, j
                self.draw_bar((x, y), A0, A1, A2, Color)


# -----------------------------------------------------------------------------
def rotate_point_3D(x, y, z, alpha, beta, gama):
    ''' поворот точки x,y,z вокруг осей Ox, Oy, Oz на углы alpha, beta, gama'''
    rotate_matrix1 = np.array([[1, 0, 0], [0, math.cos(alpha), math.sin(alpha)], [
                              0, -math.sin(alpha), math.cos(alpha)]])
    rotate_matrix2 = np.array([[math.cos(beta), 0, math.sin(beta)], [
                              0, 1, 0], [-math.sin(beta), 0, math.cos(beta)]])
    rotate_matrix3 = np.array([[math.cos(gama), math.sin(
        gama), 0], [-math.sin(gama), math.cos(gama), 0], [0, 0, 1]])
    dot1 = rotate_matrix1.dot(rotate_matrix2)

    rotate_matrix = dot1.dot(rotate_matrix3)  # матрица поворота 3D
    coordinates = np.array([x, y, z])
    new_coordinates = coordinates.dot(rotate_matrix)
    return new_coordinates


def normalForPolygon(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    '''вычисляет косинус между нормалью и вектором падения света для полигона '''
    vec_1 = np.array([x1-x0, y1-y0, z1-z0])
    vec_2 = np.array([x1-x2, y1-y2, z1-z2])

    normal = np.cross(vec_1, vec_2)  # нормаль к полигону
    l = np.array([0, 0, 1])  # направление падения света
    norm_normal = np.linalg.norm(normal)
    norm_l = np.linalg.norm(l)
    # косинус угла между нормалью и направлением падения света
    cos = normal.dot(l)/(norm_normal*norm_l)
    return cos


def get_bar_coordinate(P, A0, A1, A2):
    ''' принимает координаты точки P(x,y), Координаты трех вершин  Ai(xi,yi)
            возвращает l0,l1,l2 - барицентрические координаты точки P(x,y)'''
    x, y = P[0], P[1]
    x0, y0 = A0[0], A0[1]
    x1, y1 = A1[0], A1[1]
    x2, y2 = A2[0], A2[1]
    l0 = ((y-y2)*(x1-x2)-(x-x2)*(y1-y2))/((y0-y2)*(x1-x2)-(x0-x2)*(y1-y2))
    l1 = ((y-y0)*(x2-x0)-(x-x0)*(y2-y0))/((y1-y0)*(x2-x0)-(x1-x0)*(y2-y0))
    l2 = ((y-y1)*(x0-x1)-(x-x1)*(y0-y1))/((y2-y1)*(x0-x1)-(x2-x1)*(y0-y1))
    return l0, l1, l2


def proective_preobr(x, y, z, mashtab, Centerx, Centery, tz):
    '''проективное преобразование координат'''
    proect_matrix = np.array(
        [[mashtab, 0, Centerx], [0, -mashtab, Centery], [0, 0, 1]])
    vec = np.array([x, y, z])
    vect = np.array([0, 0, tz])
    vec1 = proect_matrix.dot(vec+vect)
    x1, y1, z1 = vec1[0], vec1[1], vec1[2]
    return x1, y1, z1


def z_buffer(im, x0, y0, z0, x1, y1, z1, x2, y2, z2, Z_matrix, Color):
    '''реализация алгоритма z буфера '''

    # границы полигона
    x_min = min(x0, x1, x2)
    x_max = max(x0, x1, x2)
    y_min = min(y0, y1, y2)
    y_max = max(y0, y1, y2)
    x_min, x_max, y_min, y_max = int(np.floor(x_min)), int(
        np.ceil(x_max)), int(np.floor(y_min)), int(np.ceil(y_max))

    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            y, x = i, j
            l0, l1, l2 = get_bar_coordinate(
                (j, i), (x0, y0), (x1, y1), (x2, y2))
            z_new = z0*l0+z1*l1+z2*l2
            if z_new <= Z_matrix[i][j]:
                if (l0 >= 0) and (l1 >= 0) and (l2 >= 0):
                    im.draw_point(j, i, color=Color)
                    Z_matrix[i][j] = z_new


def rotate_object(image, file_name, alpha, beta, gamma):
    t = 300
    obj_file = Obj_Parser(file_name)
    polygons_list = obj_file.get_polygons_list()
    v_list = obj_file.get_vertex_list()

    Z_matrix = np.empty((1000, 1000))
    for i in range(0, 1000):
        Z_matrix[i] = 1000000

    for string_with_idx in polygons_list:

        idx_A = string_with_idx[0]
        idx_B = string_with_idx[1]
        idx_C = string_with_idx[2]

        x, y, z = v_list[idx_A]
        x0 = x
        y0 = y
        z0 = z

        x, y, z = v_list[idx_B]
        x1 = x
        y1 = y
        z1 = z

        x, y, z = v_list[idx_C]
        x2 = x
        y2 = y
        z2 = z

        # поворот модели
        x0, y0, z0 = rotate_point_3D(x0, y0, z0, alpha, beta, gamma)
        x1, y1, z1 = rotate_point_3D(x1, y1, z1, alpha, beta, gamma)
        x2, y2, z2 = rotate_point_3D(x2, y2, z2, alpha, beta, gamma)

        # вычисление нормали к полигону и определение лицевой стороны
        cos = normalForPolygon(x0, y0, z0, x1, y1, z1, x2, y2, z2)

        if cos >= 0:

            r = 255*cos
            g = 255*cos
            b = 255*cos
            color = r, g, b

            # z буфер
            z_buffer(im, (t*x0)+500, (t*y0)+500, z0, t*x1+500, t*y1+500,
                     z1, t*x2+500, t*y2+500, z2, Z_matrix, color)

# ----------------------------------------------------------------------------------
# Поворот картинки с Z-буффером


im = image(1000, 1000, color=[0, 0, 0])
rotate_object(im, 'african_head.obj', alpha=0, beta=np.pi/3, gamma=np.pi)

# im.save('голова.png')
im.show()
