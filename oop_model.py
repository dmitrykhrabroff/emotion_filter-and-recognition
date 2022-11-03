import cv2
import numpy as np
import math
import base_utils as utils
from _datetime import datetime
class BaseFilter:
    faces_dict = {}
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    CHIN_CONTOUR = [177, 215, 138, 135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367, 435, 401]
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40,
            39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
    # Left eyes indices
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    FONTS = cv2.FONT_HERSHEY_COMPLEX
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_IRIS = [473, 474, 475, 476]

    def __init__(self,face_count):
        '''face_count - переменная-счетчик обнаруженных диц
        '''
        self.face_count = face_count
        self.filters = {}

    @staticmethod
    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance

    @staticmethod
    def resized_filter(filter_image, width=None, height=None):
        '''На вход подается фотография(np.array), и параметр width (ширину) и/или height(высоту) до значения которого нужно изменить размер фото'''
        filter_img_height, filter_img_width, _ = filter_image.shape
        if height and width:
            resized_filter_img = cv2.resize(filter_image, (width, height))
        elif height:
            resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (height / filter_img_height)), height))
        elif width:
            resized_filter_img = cv2.resize(filter_image, (width, int(filter_img_height *
                                                           (width / filter_img_width))))
        else:
            raise ValueError('Должны быть указаны величина длины и/или ширины до которой изменяем размер')
        return resized_filter_img

    @classmethod
    def mask_filter(cls, filter_image: np.array, colors_bounds: tuple):
        '''В данном методе на вход подается filter_image(np.array), и colors_bounds  - границы диапазонов по которым проводится фильтрация фона
        (кортеж состоящий из начального цвета диапазона (np.array), конечного цвета диапазона  (np.array))'''
        hsv = cv2.cvtColor(filter_image, cv2.COLOR_BGR2HSV)
        filter_img_mask = cv2.inRange(hsv,  colors_bounds[0], colors_bounds[1])
        filter_img_mask = cv2.bitwise_not(filter_img_mask)
        res = cv2.bitwise_and(filter_image, filter_image, mask=filter_img_mask) # получаем изображение нашего объекта на черном фоне
        coords = np.argwhere(filter_img_mask > 0)  # определяем координаты рамки в которых расположен нащ объект
        filter_img_mask = cv2.bitwise_not(filter_img_mask) # преобразуем маску объекта, на маску фона, нужно для реализации функции
        try:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            res, filter_img_mask = res[x_min:x_max + 1, y_min: y_max + 1], filter_img_mask[x_min:x_max + 1,
                                                                                     y_min: y_max + 1]
            return res, filter_img_mask
        except:
            print('except crop and mask')
        return res, filter_img_mask

    @staticmethod
    def _crop_filter_and_mask(filter_img: np.array, filter_mask: np.array, location: tuple, direction_anchor_point: str, img_height: int, img_width: int):
        ''' Данный метод необходим для проверки не выходят ли размеры фильтра за размеры изображения, на которое будет наложен наш фильтр.
        На вход подается filter_image - изображение объекта на черном фоне, filter_mask -  маска объекта, location - координаты х,у точки к которой привязываем объек-фильтр,
       direction_anchor_point - направление привязки один из вариантов (center, left_upper, left_bottom, right_bottom, right_upper), img_height,img_width: размеры исходного изображения
       Метод возвращает обрезанные изображения фильтра и маски'''
        filter_img_height, filter_img_width, _ = filter_img.shape
        if 'center' in direction_anchor_point:
            required_height = min(filter_img_height, min(location[1], img_height - location[1])*2)
            required_width = min(filter_img_width, min(location[0], img_width - location[0])*2)
            filter_img = filter_img[-required_height:-filter_img_height + required_height-1, -required_width: -filter_img_width + required_width-1]
            filter_mask = filter_mask[-required_height:-filter_img_height + required_height-1, -required_width: -filter_img_width + required_width-1]
            return filter_img, filter_mask
        if'left' in direction_anchor_point:
            required_width = min(filter_img_width, int(img_width - location[0]))
            filter_img = filter_img[:, :required_width]
            filter_mask = filter_mask[:, :required_width]
        else:
            required_width = min(filter_img_width, location[0])
            filter_img = filter_img[:, -required_width:]
            filter_mask = filter_mask[:, -required_width:]
        if 'bottom' in direction_anchor_point:
            required_height = min(filter_img_height, location[1])
            filter_img = filter_img[-required_height:, :]
            filter_mask = filter_mask[-required_height:, :]
        else:
            required_height = min(filter_img_height, img_height - location[1])
            filter_img = filter_img[:required_height, :]
            filter_mask = filter_mask[:required_height, :]

        return filter_img, filter_mask


    @staticmethod
    def filter_overlay(image: np.array, filter_img: np.array, filter_img_mask: np.array, location: tuple,direction_anchor_point = 'left_upper',  postprocess=False):
        '''Накладываем фильтр на исходное изображение.На вход подается filter_image - изображение объекта на черном фоне, filter_mask -  маска объекта,
        location - координаты х,у точки к которой привязываем объект-фильтр,
       direction_anchor_point - направление привязки один из вариантов (center, left_upper, left_bottom, right_bottom, right_upper), postprocess - постобработка после вставки фильтра
       Метод возвращает изображение после накладывания на него фильтра'''
        annotated_image = image.copy()
        directions = { 'left_upper' : ((0, 1), (0, 1)), 'left_bottom' : ((1, 0), (0, 1)),
                       'right_bottom' : ((1, 0), (1, 0)), 'right_upper' : ((0,1), (1,0))}
        filter_img_height, filter_img_width, _ = filter_img.shape
        direction = directions[direction_anchor_point]
        ROI = image[location[1] - direction[0][0]*filter_img_height: location[1] + direction[0][1]*filter_img_height,           # Делаем срез с исходного изображения размером равным размеру фильтра
                         location[0] - direction[1][0]*filter_img_width: location[0] +  direction[1][1]*filter_img_width]
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask) # c помощью обратной маски на срезе изображения в тех местах где будет распологаться накладываемый фильтр устанавливаем значение 0
        try:
            resultant_image = cv2.add(resultant_image, filter_img) # сложение матриц фильтра (объект на черном фоне) и среза изображения с 0 на месте расположения объекта. Получаем нужную картинку.
            annotated_image[location[1] - direction[0][0]*filter_img_height: location[1] + direction[0][1]*filter_img_height,           # Делаем срез с исходного изображения размером равным размеру фильтра
                         location[0] - direction[1][0]*filter_img_width: location[0] +  direction[1][1]*filter_img_width] = resultant_image
            if postprocess:
                k = 3
                annotated_image[location[1]-k*filter_img_height: location[1] + k*filter_img_height,
                  location[0] - k*filter_img_width: location[0] + k*filter_img_width] = cv2.medianBlur(
                    annotated_image[location[1]-k*filter_img_height: location[1] + k*filter_img_height,
                  location[0] - k*filter_img_width: location[0] + k*filter_img_width], 3, 0)
        except Exception as e:
            print('except filteroverlay')
            print(e)

        return annotated_image

    def get_filter_video(self, filter_name):
        '''Метод принимает имя(ключ) и по данному ключу из словаря-атрибута класса self.filters,
        где каждому ключу соответствует список [объект захвата видео(cv2.videocapture()), счетчик прочитанных фреймов для видео,
        начальный цвет диапазона (np.array) и конечный цвет диапазона  (np.array), по которым должна производится фильтрация
        возвращает ret (True - успешно ли открыт файл), frame - фрагмент изображения фильтра,  '''
        color_bounder = self.filters[filter_name][2:]
        self.filters[filter_name][1] += 1
        if self.filters[filter_name][1] == self.filters[filter_name][0].get(cv2.CAP_PROP_FRAME_COUNT):
            self.filters[filter_name][0].set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.filters[filter_name][1] = 0
        ret, frame = self.filters[filter_name][0].read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, frame, color_bounder

    @staticmethod
    def check_status(ratio, face_part_status, threshold=5, delay=5):
        '''Вспомогательный метод принимает ratio(вычисленный коэффициент показывающий открыта ли часть тела),  face_part_status - аттрибут класса для каждой части тела,
         который представляет собой список [статус части тела на прошлой итерации(OPEN, CLOSE), счетчик показывающий сколько итераций прошло с изменения статуса тела (не больше delay),
          threshold - порог по которому определют открыта ли часть тела, delay - после какого количества итераций меняется статус части тела,
        и возвращает face_part_status '''
        if ratio < threshold: # сравнение вычисленного коэффициента с порогом
            if face_part_status[0] == 'OPEN':
                face_part_status[1] = 0 # если статус такой же как предыдущий то счетчик итераций обнуляется
            else:
                face_part_status[1] += 1 # если статус изменился то счетчик итераций увеличивается на 1,
                if face_part_status[1] > delay:  # и сравнивают с порогом задержки, и меняем только если оно становится больше порога
                    face_part_status[0] = 'OPEN'
        else:
            if face_part_status[0] == 'CLOSED': # все аналогично открытому состоянию
                face_part_status[1] = 0
            else:
                face_part_status[1] += 1
                if face_part_status[1] > delay:
                    face_part_status[0] = 'CLOSED'
        return face_part_status

    def facepartExtractor(self, frame, coords, scale_x=1, scale_y=2, direction_anchor_point = 'left_upper', postprocess = False):
        '''Изменение размера выбранной части тела.
        На вход подается frame - исходное изображение для обработки, coords -  координаты части тела по которым нужно вырезать объект,
        scale_x, scale_y - изменение размера части телапо х и у, direction_anchor_point - направление привязки один из вариантов (center, left_upper),
        postprocess - постобработка после вставки фильтра
       Метод возвращает изображение после накладывания на него фильтра'''
        img_height, img_width, _ = frame.shape
        mask = np.zeros((img_height, img_width), dtype=np.uint8) # создаем черный экран размером с наше изображение в качестве маски
        cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 255) # заполняем  площадь внутри наших координат белым цветом,
        part_face = cv2.bitwise_and(frame, frame, mask=mask)  #  вырезаем с исходного изображения по средством маски те части которые входит в область наших координат. Получаем часть тела на черном фоне
        max_x, min_x = (max(coords, key=lambda item: item[0]))[0], (min(coords, key=lambda item: item[0]))[0] # определяем крайние границы наших координат
        max_y, min_y = (max(coords, key=lambda item: item[1]))[1], (min(coords, key=lambda item: item[1]))[1]
        cropped_img = part_face[min_y: max_y, min_x: max_x] # вырезаем из преобразованного изображения на черном фоне прямоугольник содержащий наши координаты
        cropped_img = cv2.resize(cropped_img, None, fx=scale_x, fy=scale_y, interpolation = cv2.INTER_CUBIC)  # изменяем размер обрезанного изображения по заданным масштабам
        filter_img_height, filter_img_width, _ = cropped_img.shape
        if 'center' in direction_anchor_point:
            location = (min_x - (filter_img_width - (max_x - min_x)) // 2, min_y - (filter_img_height - (max_y - min_y)) // 2)  # Выбираем точку прикрепления вырезанного объекта относительно центра
        elif 'left' in direction_anchor_point:
            location = (min_x, min_y) # Выбираем точку прикрепления вырезанного объекта относительно левого верхнего угла
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY), # СОздаем маску по вырезанному изображению
                                           25, 255, cv2.THRESH_BINARY_INV)
        cropped_img,filter_img_mask = self._crop_filter_and_mask(cropped_img, filter_img_mask, location, direction_anchor_point, img_height, img_width)
        image = self.filter_overlay(frame, cropped_img, filter_img_mask, location, postprocess=postprocess)

        return image


class AngryFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        iris_filter = cv2.imread('media/angry_iris_fire.png')
        self.iris_filter = cv2.cvtColor(iris_filter, cv2.COLOR_BGR2RGB)
        hat_filter = cv2.imread('media/angry_evil_horn.png')
        self.hat_filter = cv2.cvtColor(hat_filter, cv2.COLOR_BGR2RGB)
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        '''self.mouth_status - список состоящий из строки-индикатора открыт ли рот, и счетчика сколько итераций подряд изменился индикатор '''
        super().__init__(face_count)
        self.filters = {'mouth': [cv2.VideoCapture('media/angry_mouth_fire.mp4'), 0, np.array([34, 32, 0], np.uint8), np.array([80, 255, 255], dtype=np.uint8)],
                        'nose': [cv2.VideoCapture('media/angry_nose_smoke.mp4'), 0, np.array([36, 200, 63], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]}

    def mouth_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('mouth')
        p_right,p_left, p_top, p_bottom = mesh_coord[308], mesh_coord[78], mesh_coord[12], mesh_coord[15]
        center = (p_bottom[0], int(p_bottom[1] / 2 + p_top[1] / 2))
        vDistance = self.euclaideanDistance(p_top, p_bottom) + 1
        required_height = int(vDistance * 10)
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width / 2), int(center[1]))
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location)
        return frame

    def noses_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('nose')
        mouth_filter = cv2.cvtColor(mouth_filter, cv2.COLOR_RGB2BGR)
        p_right_nose_top, p_right_nose_bottom = mesh_coord[309],  mesh_coord[290]
        p_left_nose_top, p_left_nose_bottom = mesh_coord[79], mesh_coord[60]
        p_bottom = mesh_coord[18]
        required_height = p_bottom[1] - p_left_nose_bottom[1]
        right_center = (p_right_nose_top[0], int(p_right_nose_top[1] / 2 + p_right_nose_bottom[1] / 2))
        left_center = (p_left_nose_top[0], int(p_left_nose_top[1] / 2 + p_left_nose_bottom[1] / 2))
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1]))
        left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1]))
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, right_location,
                                                                     'left_upper', img_height, img_width)
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, left_location,
                                                                     'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask, right_location)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask, left_location)
        return frame

    def leftIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), l_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.LEFT_IRIS], dtype = np.int32))
        left_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6*l_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, left_location)
        return frame

    def rightIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), r_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.RIGHT_IRIS], dtype = np.int32))
        right_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6 * r_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, right_location)
        return frame

    def hatOverlay(self, frame, filter_image, mesh_coords):
        img_height, img_width, _ = frame.shape
        center = int((mesh_coords[109][0] + mesh_coords[338][0]) / 2), int(
            (mesh_coords[109][1] + mesh_coords[338][1]) / 2)
        required_width = mesh_coords[284][0] - mesh_coords[54][0]
        filter_img_height, filter_img_width, _ = filter_image.shape
        required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
        resized_filter_img = cv2.resize(filter_image,
                                        (required_width, int(filter_img_height * (required_width / filter_img_width))))
        resized_filter_img = resized_filter_img[-required_height:, :]

        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width / 2), center[1])
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, location,
                                    direction_anchor_point='left_bottom')
        return frame

    def forward(self, frame, mesh_coords):
        mouth_ratio = utils.isOpen(mesh_coords[308], mesh_coords[78], mesh_coords[12], mesh_coords[15])  # is mouth open
        right_eye_ratio = utils.isOpen(mesh_coords[133],  mesh_coords[33], mesh_coords[145], mesh_coords[159])  # is mouth open
        left_eye_ratio = utils.isOpen(mesh_coords[263],mesh_coords[362], mesh_coords[374],  mesh_coords[386])  # is mouth open
        condition = self.left_eye_status[0]
        self.left_eye_status = self.check_status(left_eye_ratio, self.left_eye_status,4.5)
        if condition == 'OPEN':
            frame = self.leftIrisOverlay(frame, self.iris_filter, mesh_coords)
            # frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYE], 1, 2, 'center')
        self.right_eye_status = self.check_status(right_eye_ratio, self.right_eye_status, 4.5)
        condition = self.right_eye_status[0]
        if condition == 'OPEN':
            frame = self.rightIrisOverlay(frame, self.iris_filter, mesh_coords)
            # frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYE], 1, 2, 'center')
        mouth_status = self.check_status(mouth_ratio, self.mouth_status)
        condition = mouth_status[0]
        if condition == 'OPEN':
            frame = self.mouth_overlay(frame, mesh_coords)
        else:
            # frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LIPS], 1.2, 1.6, postprocess=True, direction_anchor_point='center')
            frame = self.noses_overlay(frame, mesh_coords)
        frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYEBROW], 1, 3, postprocess=True,
                                       direction_anchor_point='center')
        frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYEBROW], 1, 3, postprocess=True,
                                       direction_anchor_point='center')
        frame = self.hatOverlay(frame,self.hat_filter,  mesh_coords)
        frame = cv2.medianBlur(frame, 3, 0)
        return frame

class FearFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        self.iris_filter = cv2.imread('media/fear_iris_fire.png')
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        '''self.mouth_status - список состоящий из строки-индикатора открыт ли рот, и счетчика сколько итераций подряд изменился индикатор '''
        super().__init__(face_count)
        self.filters = {'mouth': [cv2.VideoCapture('media/fear_spider.mp4'), 0, np.array([0, 0, 103], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]}

    def mouth_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('mouth')
        p_right,p_left, p_top, p_bottom = mesh_coord[308], mesh_coord[78], mesh_coord[12], mesh_coord[15]
        center = (p_right[0], int(p_bottom[1] / 2 + p_top[1] / 2))
        hDistance = self.euclaideanDistance(p_right, p_left) + 1
        required_height = int(hDistance*2)
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width/2), int(center[1] - filter_img_height / 2 ))
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location)
        return frame

    def leftIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), l_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.LEFT_IRIS], dtype = np.int32))
        left_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6*l_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, left_location)
        return frame

    def rightIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), r_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.RIGHT_IRIS], dtype = np.int32))
        right_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6 * r_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, right_location)
        return frame


    def forward(self, frame, mesh_coords):
        mouth_ratio = utils.isOpen(mesh_coords[308], mesh_coords[78], mesh_coords[12], mesh_coords[15])  # is mouth open
        right_eye_ratio = utils.isOpen(mesh_coords[133],  mesh_coords[33], mesh_coords[145], mesh_coords[159])  # is mouth open
        left_eye_ratio = utils.isOpen(mesh_coords[263],mesh_coords[362], mesh_coords[374],  mesh_coords[386])  # is mouth open
        condition = self.left_eye_status[0]
        self.left_eye_status = self.check_status(left_eye_ratio, self.left_eye_status,4.5)
        if condition == 'OPEN':
            frame = self.leftIrisOverlay(frame, self.iris_filter, mesh_coords)
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYE], 1, 2, 'center')
        self.right_eye_status = self.check_status(right_eye_ratio, self.right_eye_status, 4.5)
        condition = self.right_eye_status[0]
        if condition == 'OPEN':
            frame = self.rightIrisOverlay(frame, self.iris_filter, mesh_coords)
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYE], 1, 2, 'center')
        mouth_status = self.check_status(mouth_ratio, self.mouth_status)
        condition = mouth_status[0]
        if condition == 'OPEN':
            frame = self.mouth_overlay(frame, mesh_coords)
        else:
            self.filters['mouth'][0].set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = cv2.medianBlur(frame, 3, 0)
        return frame


class SurpriseFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        self.random_seed = np.random.randint(3,10)
        self.random_count = datetime.now()
        '''self.mouth_status - список состоящий из строки-индикатора открыт ли рот, и счетчика сколько итераций подряд изменился индикатор '''
        super().__init__(face_count)
        self.filters = {'mouth': [cv2.VideoCapture('media/surprise_track.mp4'), 0, np.array([43, 42, 103], np.uint8), np.array([84, 255, 255], dtype=np.uint8)],
                        'head': [cv2.VideoCapture('media/surprise_bomb.mp4'), 0, np.array([43, 42, 103], np.uint8), np.array([84, 255, 255], dtype=np.uint8)]}

    def mouth_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('mouth')
        p_right,p_left, p_top, p_bottom = mesh_coord[308], mesh_coord[78], mesh_coord[12], mesh_coord[15]
        center = (p_bottom[0], p_top[1])
        hDistance = self.euclaideanDistance(p_right, p_left) + 1
        required_height = int(hDistance*1)
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height / 2 ))
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, center, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,center)
        return frame

    def head_overlay(self, frame, mesh_coords):
        ret, head_filter, color_bounder = self.get_filter_video('head')
        if ret:
            img_height, img_width, _ = frame.shape
            center = int((mesh_coords[109][0] + mesh_coords[338][0]) / 2), int(
                (mesh_coords[109][1] + mesh_coords[338][1]) / 2)
            required_width = mesh_coords[284][0] - mesh_coords[54][0]
            filter_img_height, filter_img_width, _ = head_filter.shape
            required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
            resized_filter_img = self.resized_filter(filter_image=head_filter, height=required_height)
            resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
            filter_img_height, filter_img_width, _ = resized_filter_img.shape
            img_height, img_width, _ = frame.shape
            location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height))
            resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'left_upper', img_height, img_width)
            frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location)
        return frame


    def forward(self, frame, mesh_coords):
        mouth_ratio = utils.isOpen(mesh_coords[308], mesh_coords[78], mesh_coords[12], mesh_coords[15])  # is mouth open
        right_eye_ratio = utils.isOpen(mesh_coords[133],  mesh_coords[33], mesh_coords[145], mesh_coords[159])  # is mouth open
        left_eye_ratio = utils.isOpen(mesh_coords[263],mesh_coords[362], mesh_coords[374],  mesh_coords[386])  # is mouth open
        condition = self.left_eye_status[0]
        self.left_eye_status = self.check_status(left_eye_ratio, self.left_eye_status,4.5)
        if condition == 'OPEN':
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYE], 1, 2, 'center')
        self.right_eye_status = self.check_status(right_eye_ratio, self.right_eye_status, 4.5)
        condition = self.right_eye_status[0]
        if condition == 'OPEN':
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYE], 1, 2, 'center')
        mouth_status = self.check_status(mouth_ratio, self.mouth_status,6)
        condition = mouth_status[0]
        if condition == 'OPEN':
            frame = self.mouth_overlay(frame, mesh_coords)
        else:
            self.filters['mouth'][0].set(cv2.CAP_PROP_POS_FRAMES, 0)
        if (datetime.now() - self.random_count).total_seconds() >= self.random_seed:
            frame = self.head_overlay(frame, mesh_coords)
            if self.filters['head'][1] == self.filters['head'][0].get(cv2.CAP_PROP_FRAME_COUNT):
                self.filters['head'][0].set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.filters['head'][1] = 0
                self.random_count = datetime.now()
                self.random_seed = np.random.randint(3, 10)
        frame = cv2.medianBlur(frame, 3, 0)
        return frame


class HappyFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        iris_filter = cv2.imread('media/happy_smile.png')
        self.iris_filter = cv2.cvtColor(iris_filter, cv2.COLOR_BGR2RGB)
        hat_filter = cv2.imread('media/happy_party_hat.png')
        self.hat_filter = cv2.cvtColor(hat_filter, cv2.COLOR_BGR2RGB)
        self.__initialized = True
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        '''self.mouth_status - список состоящий из строки-индикатора открыт ли рот, и счетчика сколько итераций подряд изменился индикатор '''
        super().__init__(face_count)
        self.filters = {'head': [cv2.VideoCapture('media/happy_head.mp4'), 0, np.array([55, 109, 32], np.uint8), np.array([78, 255, 255], dtype=np.uint8)],
                        'mouth': [cv2.VideoCapture('media/happy_mouth_horn.mp4'),
                                  0, np.array([0, 202, 255], np.uint8), np.array([255, 255, 255], dtype=np.uint8)]}

    def mouth_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('mouth')
        p_right,p_left, p_top, p_bottom = mesh_coord[308], mesh_coord[78], mesh_coord[12], mesh_coord[15]
        center = (p_top[0], int(p_top[1]))
        required_height = (mesh_coord[152][1] -  mesh_coord[10][1])
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, center, 'right_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,center, 'right_upper')
        return frame

    def head_overlay(self, frame, mesh_coords):
        ret, head_filter, color_bounder = self.get_filter_video('head')
        img_height, img_width, _ = frame.shape
        center = int((mesh_coords[107][0] + mesh_coords[336][0]) / 2), int(
            (mesh_coords[107][1] + mesh_coords[336][1]) / 2)
        required_width = mesh_coords[284][0] - mesh_coords[54][0]
        filter_img_height, filter_img_width, _ = head_filter.shape
        required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
        resized_filter_img = self.resized_filter(filter_image=head_filter, height=required_height)
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        img_height, img_width, _ = frame.shape
        location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height))
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location)
        return frame

    def leftIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), l_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.LEFT_IRIS], dtype = np.int32))
        left_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6*l_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, left_location)
        return frame

    def rightIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), r_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.RIGHT_IRIS], dtype = np.int32))
        right_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6 * r_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, right_location)
        return frame

    def hatOverlay(self, frame, filter_image, mesh_coords):
        img_height, img_width, _ = frame.shape
        center = (mesh_coords[284][0] , mesh_coords[284][1] - (mesh_coords[151][1] - mesh_coords[10][1]))
        required_width = int((mesh_coords[284][0] - mesh_coords[54][0])*1.3)
        filter_img_height, filter_img_width, _ = filter_image.shape
        required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
        resized_filter_img = cv2.resize(filter_image,
                                        (required_width, int(filter_img_height * (required_width / filter_img_width))))
        resized_filter_img = resized_filter_img[-required_height:, :]

        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width / 2), center[1])
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, location, direction_anchor_point='left_bottom')
        return frame

    def forward(self, frame, mesh_coords):
        mouth_ratio = utils.isOpen(mesh_coords[308], mesh_coords[78], mesh_coords[12], mesh_coords[15])  # is mouth open
        right_eye_ratio = utils.isOpen(mesh_coords[133],  mesh_coords[33], mesh_coords[145], mesh_coords[159])  # is mouth open
        left_eye_ratio = utils.isOpen(mesh_coords[263],mesh_coords[362], mesh_coords[374],  mesh_coords[386])  # is mouth open
        condition = self.left_eye_status[0]
        frame = self.hatOverlay(frame,self.hat_filter,  mesh_coords)
        self.left_eye_status = self.check_status(left_eye_ratio, self.left_eye_status,4.5)
        if condition == 'OPEN':
            frame = self.leftIrisOverlay(frame, self.iris_filter, mesh_coords)
        self.right_eye_status = self.check_status(right_eye_ratio, self.right_eye_status, 4.5)
        condition = self.right_eye_status[0]
        if condition == 'OPEN':
            frame = self.rightIrisOverlay(frame, self.iris_filter, mesh_coords)
        mouth_status = self.check_status(mouth_ratio, self.mouth_status,3)
        condition = mouth_status[0]
        frame = self.head_overlay(frame, mesh_coords)
        if condition == 'OPEN':
            self.filters['mouth'][0].set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            frame = self.mouth_overlay(frame, mesh_coords)
        # frame = cv2.medianBlur(frame, 3, 0)
        return frame

class NeutralFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance

    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        super().__init__(face_count)
        self.filters = {'head': [cv2.VideoCapture('media/neutral_monkey.mp4'), 0, np.array([55, 102, 51], np.uint8), np.array([86, 255, 255], dtype=np.uint8)]}


    def head_overlay(self, frame, mesh_coords):
        ret, head_filter, color_bounder = self.get_filter_video('head')
        head_filter = cv2.cvtColor(head_filter, cv2.COLOR_RGB2BGR)
        img_height, img_width, _ = frame.shape
        center = int((mesh_coords[107][0] + mesh_coords[336][0]) / 2), int(
            (mesh_coords[107][1] + mesh_coords[336][1]) / 2)
        required_width = mesh_coords[284][0] - mesh_coords[54][0]
        filter_img_height, filter_img_width, _ = head_filter.shape
        required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
        resized_filter_img = self.resized_filter(filter_image=head_filter, height=required_height)
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        img_height, img_width, _ = frame.shape
        location = (int(center[0] - filter_img_width / 2), int(center[1] - filter_img_height))
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location,
                                                                     'left_upper', img_height, img_width)
        resized_filter_img = cv2.cvtColor(resized_filter_img, cv2.COLOR_RGB2BGR)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask, location)
        return frame

    def forward(self, frame, mesh_coords):
        frame = self.head_overlay(frame, mesh_coords)
        frame = cv2.medianBlur(frame, 3, 0)
        return frame

class SadnessFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]
        hat_filter = cv2.imread('media\sadness_hat.png')
        self.hat_filter = cv2.cvtColor(hat_filter, cv2.COLOR_BGR2RGB)
        '''self.mouth_status - список состоящий из строки-индикатора открыт ли рот, и счетчика сколько итераций подряд изменился индикатор '''
        super().__init__(face_count)
        self.filters = {'mouth': [cv2.VideoCapture('media\sadness_mouth_cigarette.mp4'), 0, np.array([55, 109, 32], np.uint8), np.array([78, 255, 255], dtype=np.uint8)],
                        'left_eyes' : [cv2.VideoCapture('media/sadness_left_eye_tears.mp4'), 0, np.array([32, 93, 158], np.uint8), np.array([87, 204, 229], dtype=np.uint8)],
                        'right_eyes' : [cv2.VideoCapture('media/sadness_right_eye_tears.mp4'), 0, np.array([32, 93, 158], np.uint8), np.array([87, 204, 229], dtype=np.uint8)]}

    def mouth_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('mouth')
        p_right,p_left, p_top, p_bottom = mesh_coord[308], mesh_coord[78], mesh_coord[12], mesh_coord[15]
        center = (p_top[0], int(p_top[1]))
        required_height = (mesh_coord[152][1] -  mesh_coord[10][1])
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, center, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,center, 'left_upper')
        return frame

    def right_eye_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('right_eyes')
        mouth_filter = utils.rotateImage(mouth_filter, 180)
        location = mesh_coord[33]
        hDistance = self.euclaideanDistance(mesh_coord[33], mesh_coord[133]) + 1
        required_height = int(hDistance*5)
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'right_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location, 'right_upper')
        return frame

    def left_eye_overlay(self, frame, mesh_coord):
        ret, mouth_filter, color_bounder = self.get_filter_video('left_eyes')
        mouth_filter = utils.rotateImage(mouth_filter, 180)
        location = mesh_coord[263]
        hDistance = self.euclaideanDistance(mesh_coord[362], mesh_coord[263]) + 1
        required_height = int(hDistance*5)
        resized_filter_img = self.resized_filter(filter_image=mouth_filter, height=required_height)
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        resized_filter_img, filter_mask = self.mask_filter(resized_filter_img, color_bounder)
        img_height, img_width, _ = frame.shape
        resized_filter_img, filter_mask = self._crop_filter_and_mask(resized_filter_img, filter_mask, location, 'left_upper', img_height, img_width)
        frame = self.filter_overlay(frame, resized_filter_img, filter_mask,location, 'left_upper')
        return frame

    def leftIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), l_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.LEFT_IRIS], dtype = np.int32))
        left_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6*l_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        left_location = (int(left_center[0] - filter_img_width / 2), int(left_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, left_location)
        return frame

    def rightIrisOverlay(self, frame, filter_image, mesh_coords):
        (cx, cy), r_radius = cv2.minEnclosingCircle(np.array([mesh_coords[x] for x in self.RIGHT_IRIS], dtype = np.int32))
        right_center = np.array([cx, cy], dtype=np.int32)
        required_height = int(1.6 * r_radius)
        filter_img_height, filter_img_width, _ = filter_image.shape
        resized_filter_img = cv2.resize(filter_image, (int(filter_img_width *
                                                           (required_height / filter_img_height)),
                                                       required_height))
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        right_location = (int(right_center[0] - filter_img_width / 2), int(right_center[1] - filter_img_height / 2))
        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)
        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, right_location)
        return frame

    def hatOverlay(self, frame, filter_image, mesh_coords):
        img_height, img_width, _ = frame.shape
        center = int((mesh_coords[193][0] + mesh_coords[417][0]) / 2), int(
            (mesh_coords[193][1] + mesh_coords[417][1]) / 2)
        required_width = int((mesh_coords[284][0] - mesh_coords[54][0])*1.4)
        filter_img_height, filter_img_width, _ = filter_image.shape
        required_height = min(int(filter_img_height * (required_width / filter_img_width)), center[1])
        resized_filter_img = cv2.resize(filter_image,
                                        (required_width, int(filter_img_height * (required_width / filter_img_width))))
        resized_filter_img = resized_filter_img[-required_height:, :]
        filter_img_height, filter_img_width, _ = resized_filter_img.shape
        location = (int(center[0] - filter_img_width / 2), center[1])

        _, filter_img_mask = cv2.threshold(cv2.cvtColor(resized_filter_img, cv2.COLOR_BGR2GRAY),
                                           25, 255, cv2.THRESH_BINARY_INV)

        frame = self.filter_overlay(frame, resized_filter_img, filter_img_mask, location, direction_anchor_point='left_bottom')
        return frame

    def forward(self, frame, mesh_coords):
        mouth_ratio = utils.isOpen(mesh_coords[308], mesh_coords[78], mesh_coords[12], mesh_coords[15])  # is mouth open
        right_eye_ratio = utils.isOpen(mesh_coords[133],  mesh_coords[33], mesh_coords[145], mesh_coords[159])  # is mouth open
        left_eye_ratio = utils.isOpen(mesh_coords[263],mesh_coords[362], mesh_coords[374],  mesh_coords[386])  # is mouth open
        condition = self.left_eye_status[0]
        self.left_eye_status = self.check_status(left_eye_ratio, self.left_eye_status,4.5)
        if condition == 'OPEN':
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYE], 1, 2, 'center')
        self.right_eye_status = self.check_status(right_eye_ratio, self.right_eye_status, 4.5)
        condition = self.right_eye_status[0]
        if condition == 'OPEN':
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYE], 1, 2, 'center')
        mouth_status = self.check_status(mouth_ratio, self.mouth_status,3.5)
        condition = mouth_status[0]
        if condition == 'OPEN':
            self.filters['mouth'][0].set(cv2.CAP_PROP_POS_FRAMES, 0)

        else:
            frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LIPS], 1.2, 1.6, postprocess=True,
                                           direction_anchor_point='center')
            frame = self.mouth_overlay(frame, mesh_coords)

        frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.RIGHT_EYEBROW], 1, 1.5, postprocess=True,
                                       direction_anchor_point='center')
        frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.LEFT_EYEBROW], 1, 1.5, postprocess=True,
                                       direction_anchor_point='center')
        frame = self.hatOverlay(frame,self.hat_filter,  mesh_coords)
        frame = self.right_eye_overlay(frame, mesh_coords)
        frame = self.left_eye_overlay(frame, mesh_coords)
        frame = cv2.medianBlur(frame, 3, 0)
        return frame

class DisgustFilter(BaseFilter):
    __faces_dict = {}
    def __new__(cls, *args, **kwargs):
        '''Для каждого обнаруженного объекта (в инициализатор передается переменная facecount) создается только один экзмепляр класса'''
        if args[0] not in cls.__faces_dict:
            instance = super().__new__(cls)
            cls.__faces_dict.update({args[0]: instance})
            instance.__initialized = False
        else:
            instance = cls.__faces_dict[args[0]]
        return instance


    def __init__(self, face_count):
        if self.__initialized:
            return
        self.__initialized = True
        self.mouth_status = ['CLOSED', 0]
        self.left_eye_status = ['CLOSED', 0]
        self.right_eye_status = ['CLOSED', 0]

        super().__init__(face_count)
        self.filters = {}


    def forward(self, frame, mesh_coords):
        frame = self.facepartExtractor(frame, [mesh_coords[x] for x in self.FACE_OVAL], 1, 1.5, postprocess=True,
                                        direction_anchor_point='left_upper')
        frame = cv2.medianBlur(frame, 3, 0)
        return frame