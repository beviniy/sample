# coding:utf8

import cPickle as pickle
from PIL import Image, ImageDraw, ImageFont

from room_type import ROOM_TYPE_CHOICES

def draw_room(fname,room_type,room_index, *points):

    Xs, Ys = zip(*points)
    minX = min(Xs)
    minY = min(Ys)
    width = max(Xs) - min(Xs)
    height = max(Ys) - min(Ys)

    pp = 10  # padding_pixels
    scale = 1
    gpp = pp * 2 + 1
    con = lambda x: int(round(x))
    im = Image.new('RGBA', (con(width * scale) + gpp, con(height * scale) + gpp + gpp + gpp), (255, 255, 255))
    dim = ImageDraw.Draw(im)
    new_points = map(lambda p: (con(scale * (p[0] - minX) + pp), con(height * scale) - con(scale * (p[1] - minY)) + pp), points)
    new_points.append(new_points[0])
    dim.line(new_points, (0, 0, 0), width=10)
    font = ImageFont.truetype('simsun.ttc', 24)
    dim.text((con(height * scale) / 2, con(height * scale) + gpp), fname, fill=(0, 0, 0), font=font)

    # print new_points
    # print (con(width*scale)+gpp, con(height*scale)+gpp)
    # im.show()
    # print 'pics/%s.png',fname
    #im.save('samples_pics/%s%s.png' % (ROOM_TYPE_CHOICES[room_type][1],fname))
    #im.save('samples_pics/all/%s.png' % fname)
    im.save('samples_pics/sub/%s_%s_%s.png' % (ROOM_TYPE_CHOICES[room_type][1],room_index,fname))
    #im.save('samples_pics/%s_%s_%s.png' % (ROOM_TYPE_CHOICES[room_type][1],room_index,fname))


def draw_rooms():

    f = open('taged_sample.pkl', 'rb')
    rooms = pickle.load(f)
    f.close()

    #draw_room(rooms[0]['tag'], *rooms[0]['points'])
    for i,room in enumerate(rooms):
        draw_room(room['tag'],room['room_type'],room['intype_index'], *room['room_points'])
        #if i == 551:
        #    print room['room_points']

if __name__ == '__main__':
    draw_rooms()
