# coding:utf8

import cPickle as pickle
import json
#from sympy.geometry import Point,Polygon
from sympy import Point, Polygon, sqrt, pi
import itertools

class Room(object):

    def __init__(self,room_type,intype_index,room_tag,*points):

        self.polygon = Polygon(*points)
        self.room_type = room_type
        self.tag = room_tag
        self.intype_index = intype_index
        self.reflex_sides = []

    @property
    def feature_area(self):
        return int(abs(self.polygon.area))

    @property
    def feature_perimeter(self):
        return int(self.polygon.perimeter)

    @property
    def feature_sides_variance(self):
        sides_length = [x.length for x in self.polygon.sides]
        sides_length.sort(reverse=True)
        effective_sides = sides_length[:4] if len(sides_length) >= 4 else sides_length

        mean_value = sum(effective_sides)/ len(effective_sides)

        total = 0
        for side_len in effective_sides:
            total += (side_len - mean_value) ** 2
        return int(round(sqrt(total/len(effective_sides))))

    @property
    def all_feature_sides_variance(self):
        sides_length = [x.length for x in self.polygon.sides]
        #sides_length.sort(reverse=True)
        #effective_sides = sides_length[:4] if len(sides_length) >= 4 else sides_length

        mean_value = sum(sides_length)/ len(sides_length)

        total = 0
        for side_len in sides_length:
            total += (side_len - mean_value) ** 2
        return int(round(sqrt(total/len(sides_length))))

    @property
    def num_of_sides(self):
        return len(self.polygon.sides)

    @property
    def num_of_reflex_angle(self):
        return len([each for each in self.polygon.angles.values() if each > pi])

    @property
    def door_direction(self):
        pass

    def _calc_reflex_sides(self):
        if not self.reflex_sides:
            for point, angle in self.polygon.angles.items():
                if angle > pi:
                    self.reflex_sides.append(tuple([side for side in self.polygon.sides if point in side.points]))
        self.reflex_sides = set(itertools.chain(*self.reflex_sides))
    
    
    @property
    def reflex_sides_average_len(self):
        if self.num_of_reflex_angle == 0:
            return 0
        self._calc_reflex_sides()
        
            
        return


    @property
    def reflex_sides_average_var(self):
        self._calc_reflex_sides()
        return


    def extract(self):
        featurename = ['feature_area', 'feature_perimeter' ,'num_of_reflex_angle','num_of_sides','feature_sides_variance','room_type','intype_index','tag']
        features = []
        for each in featurename:
            features.append(getattr(self ,each))

        return features



if __name__ == '__main__':
    f = open('taged_sample.pkl', 'rb')
    rooms = pickle.load(f)
    f.close()
    #for each in rooms[1]['points']:
    #    print each
    points = rooms[236]['room_points']
    points = map(Point, points)
    print points
    pp = Room(rooms[236]['room_type'],rooms[236]['intype_index'],rooms[236]['tag'], *points)
    print pp.feature_area
    print pp.feature_perimeter
    print pp.feature_sides_variance
    print pp.num_of_sides
    print pp.num_of_reflex_angle
    print pp.reflex_sides_average_len
    print pp.reflex_sides
    print pp.extract()
