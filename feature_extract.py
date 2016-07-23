# coding:utf8

import cPickle as pickle
import json
#from sympy.geometry import Point,Polygon
from sympy import Point, Polygon, sqrt, pi


class Room(object):
    
    def __init__(self,room_type,intype_index,room_tag,*points):
        
        self.polygon = Polygon(*points)
        self.room_type = room_type
        self.tag = room_tag
        self.intype_index = intype_index
    
    @property
    def feature_area(self):
        return int(sqrt(abs(self.polygon.area)))
        
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
        
        
    def extract(self):
        featurename = ['feature_area', 'feature_perimeter' ,'num_of_reflex_angle','num_of_sides','feature_sides_variance','room_type','intype_index','tag']
        features = []
        for each in featurename:
            features.append(getattr(self ,each))
            
        return features
        
            
        #mean = sum/len()
#def con_poly(*points):

 #   points = map(Point, points)
 #   print points
 #   return Polygon(*points)


if __name__ == '__main__':
    f = open('taged_sample.pkl', 'rb')
    rooms = pickle.load(f)
    f.close()
    #for each in rooms[1]['points']:
    #    print each
    points = rooms[551]['room_points']
    points = map(Point, points)
    print points
    pp = Room(rooms[551]['room_type'],rooms[551]['intype_index'],rooms[551]['tag'], *points)
    print pp.feature_area_sqrt
    print pp.feature_perimeter
    print pp.feature_sides_variance
    print pp.num_of_sides
    print pp.num_of_reflex_angle
    print pp.extract()