# coding:utf8
import json
import cPickle as pickle


def add_tag():
    f = open('sample.pkl')
    rooms = pickle.load(f)
    f.close()

    roomtype_index = {}
    for i, room in enumerate(rooms):
        
        room['tag'] = i
        
        for each in room['room_points']:
            each[0] = int(round(each[0], 0))
            each[1] = int(round(each[1], 0))
        
        if roomtype_index.get(room['room_type'], -1) == -1:
            roomtype_index[room['room_type']] = 0
        room['intype_index'] = roomtype_index[room['room_type']]
        roomtype_index[room['room_type']] += 1

    f = open('taged_sample.pkl', 'wb')
    pickle.dump(rooms, f)
    f.close()


if __name__ == '__main__':
    add_tag()
