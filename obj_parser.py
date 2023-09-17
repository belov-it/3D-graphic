

class Obj_Parser:
    '''класс для работы с файлом obj'''

    def __init__(self, name_of_file):
        self.__list_obj = []  # список всех элементов файла
        self.__vertex_list = []  # список для вершин файла
        # список полигонов (список с индексами вершин)
        self.__polygons_list = []
        self.__vn_list = []
        with open(name_of_file) as f:
            for line in f:
                # массив со строками obj файла без '\n'
                self.__list_obj.append(line.strip('\n'))

    def get_obj_list(self):
        ''' возвращает список со всеми элементами obj файла'''
        return self.__list_obj

    def parsing_vertex(self, obj_list):

        def every_string_v_parse(string):
            '''принимает строку "v  x y z ",
             возвращает список [x,y,z] - координаты
            '''
            coordinates = []
            coordinate = ''
            for char in string[2:]:
                coordinate += char
                if char == ' ' and coordinate != ' ':  # переход к следующей координате
                    coordinates.append(float(coordinate))
                    coordinate = ''
            coordinates.append(float(coordinate))  # последняя координата
            return coordinates

        for line in obj_list:
            if line[0:2] == 'v ':
                self.__vertex_list.append(every_string_v_parse(line))
        return self.__vertex_list

    def get_vertex_list(self):
        ''' возвращает список со всеми вершинами obj файла'''

        self.parsing_vertex(self.__list_obj)
        return self.__vertex_list

    def parsing_polygons(self, obj_list):

        def troiki_in_f(line_in_f):
            '''принимает строчку f
                    возвращает список с тройкой элементов ['a/a/a','a/a/a','a/a/a']
            '''
            troiki = []  # список для хранения хранения ['a/a/a','a/a/a','a/a/a']
            troika = ''  # для хранения 'a/a/a'
            for char in line_in_f[2:]:
                troika += char
                if char == ' ' and troika != ' ':
                    troiki.append(troika)
                    troika = ' '
            if troika != ' ':  # на конце строки f возможен ' ' , его не добавлять
                troiki.append(troika)
            return troiki

        def first_value_in_troika(troika):
            '''принимает ['a1/a2/a3'']
            возвращает число a1 до первой черты'''
            point = ''
            for char in troika:
                if char == '/':
                    return int(point)
                point += char

        def every_string_f_parse(string):
            '''принимает строку 'f a1/a/a b1/b/b c1/c/c'
                    возвращает список [a1,b1,c1]'''
            index_points = []  # список для [a1,b1,c1]
            troiki = troiki_in_f(string)
            for elem in troiki:
                # индексы в массиве с вершинами начинаются с 0 ,поэтому -1
                index_points.append(first_value_in_troika(elem)-1)
            return index_points

        for line in obj_list:
            if line[0:2] == 'f ':
                self.__polygons_list.append(every_string_f_parse(line))
        return self.__polygons_list

    def get_polygons_list(self):
        ''' возвращает список полигонов obj файла'''
        self.parsing_polygons(self.__list_obj)
        return self.__polygons_list

    def parsing_vn(self, obj_list):
        ''' принимает список с элементами obj файла
                возвращает список с vn'''

        def every_string_vn_parse(string):
            ''' принимает строку vn aaa bbb ccc 
                    возвращает список [aaa,bbb,ccc]
                    '''
            elements_in_vn_string = []  # список для хранения элементов строки vn
            element = ''
            for char in string[2:]:
                element += char
                if char == ' ' and element != ' ':
                    elements_in_vn_string.append(element)
                    element = ''
            elements_in_vn_string.append(element)  # последняя координата
            return elements_in_vn_string

        for line in obj_list:
            if line[0:2] == 'vn':
                self.__vn_list.append(every_string_vn_parse(line))
        return self.__vn_list

    def get_vn_list(self):

        self.parsing_vn(self.__list_obj)
        return self.__vn_list


# ------------------------------------------------
# Test
# krolik = Obj_Parser('Test.obj')
# kr_vn = krolik.get_vn_list()
# print(kr_vn[-1])
# kr_v = krolik.get_vertex_list()
# print(kr_v[-1])
# print(kr_f[-1])

# krolik = Obj_Parser('Test.obj')
# krolik_v = krolik.get_vertex_list()
