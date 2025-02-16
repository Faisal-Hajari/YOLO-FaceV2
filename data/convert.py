# -*- coding: utf-8 -*-

import shutil
import random
import os
import string
from skimage import io

headstr = """\
<annotation>
    <folder>VOC2012</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2012</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
        <kpt>
            <x1>%d</x1>
            <y1>%d</y1>
            <x2>%d</x2>
            <y2>%d</y2>
            <x3>%d</x3>
            <y3>%d</y3>
            <x4>%d</x4>
            <y4>%d</y4>
            <x5>%d</x5>
            <y5>%d</y5>
        </kpt>
    </object>
"""

tailstr = '''\
</annotation>
'''




def writexml(idx, head, bbxes, tail):
    filename = ("Annotations/%06d.xml" % (idx))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3],
                          #5 keypoints
                          bbx[4], bbx[5], bbx[6], bbx[7], 
                          bbx[8], bbx[9], bbx[10], 
                          bbx[11], bbx[12], bbx[13]))
    f.write(tail)
    f.close()


def clear_dir():
    if shutil.os.path.exists(('Annotations')):
        shutil.rmtree(('Annotations'))
    if shutil.os.path.exists(('ImageSets')):
        shutil.rmtree(('ImageSets'))
    # if shutil.os.path.exists(('JPEGImages')):
    #     shutil.rmtree(('JPEGImages'))
    if shutil.os.path.exists(('images')):
        shutil.rmtree(('images'))

    shutil.os.mkdir(('Annotations'))
    shutil.os.makedirs(('ImageSets/Main'))
    # shutil.os.mkdir(('JPEGImages'))
    shutil.os.mkdir(('images'))


def get_count(file): 
    original_position = file.tell()
    count = 0
    for line in file:
        if '#' in line:
            break
        count += 1

    file.seek(original_position)
    return count


def excute_datasets_kpt(idx, datatype):
    f = open(('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open("/home/temp/YOLO-FaceV2/WIDER_FACE/train/label.txt", 'r')
    
    while True: 
        filename = f_bbx.readline().strip('\n')
        
        if not filename: 
            break 
        
        filename = filename.replace("# ", "")
        im = io.imread(('WIDER_' + datatype + '/images/' + filename))
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        nums = get_count(f_bbx)
        bbxes = []
        if nums == 0: 
            bbx_info= f_bbx.readline()
            continue
        
        # print(int(nums))
        for ind in range(int(nums)):
            bbx_info = f_bbx.readline().strip(' \n').split(' ')[:-1]
            # bbx = [int(float(bbx_info[i])) for i in range(len(bbx_info))]
            bbx = [int(float(bbx_info[i])) for i in range(len(bbx_info)) if i < 4 or (i >= 4 and (i-4) % 3 < 2)]
            
            # # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            # if bbx[7] == 0:
            #     bbxes.append(bbx)
            bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        # shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('JPEGImages/%06d.jpg' % (idx)))
        shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('images/%06d.jpg' % (idx)))
        f.write('%06d\n' % (idx))
        idx += 1
    f.close()
    f_bbx.close()
    return idx

def excute_datasets(idx, datatype):
    if datatype == 'train': 
        return excute_datasets_kpt(idx, datatype)
    f = open(('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(('wider_face_split/wider_face_' + datatype + '_bbx_gt.txt'), 'r')

    while True:
        filename = f_bbx.readline().strip('\n')

        if not filename:
            break


        im = io.imread(('WIDER_' + datatype + '/images/' + filename))
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        nums = f_bbx.readline().strip('\n')
        bbxes = []
        if nums=='0':
            bbx_info= f_bbx.readline()
            continue
        for ind in range(int(nums)):
            bbx_info = f_bbx.readline().strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info))]
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7] == 0:
                bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        # shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('JPEGImages/%06d.jpg' % (idx)))
        shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('images/%06d.jpg' % (idx)))
        f.write('%06d\n' % (idx))
        idx += 1
    f.close()
    f_bbx.close()
    return idx


if __name__ == '__main__':
    clear_dir()
    idx = 1
    idx = excute_datasets(idx, 'train')
    # idx = excute_datasets(idx, 'val')
    print('Complete...')
