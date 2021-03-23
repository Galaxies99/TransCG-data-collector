from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
import xml.dom.minidom
from transforms3d.quaternions import mat2quat, quat2axangle
from transforms3d.euler import quat2euler
import numpy as np
from trans3d import get_mat, pos_quat_to_pose_4x4, get_pose
import os
import json


class xmlWriter():
    def __init__(self, topfromreader=None):
        self.topfromreader = topfromreader
        self.poselist = []
        self.objnamelist = []
        self.objpathlist = []
        self.objidlist = []

    def addobject(self, pose, objname, objpath, objid):
        # pose is the 4x4 matrix representation of 6d pose
        self.poselist.append(pose)
        self.objnamelist.append(objname)
        self.objpathlist.append(objpath)
        self.objidlist.append(objid)

    def objectlistfromposevectorlist(self, posevectorlist, objdir, objnamelist, objidlist):
        self.poselist = []
        self.objnamelist = []
        self.objidlist = []
        self.objpathlist = []
        for i in range(len(posevectorlist)):
            id, x, y, z, alpha, beta, gamma = posevectorlist[i]
            objname = objnamelist[objidlist[i]]
            print(objname)
            self.addobject(get_mat(x, y, z, alpha, beta, gamma),
                           objname, os.path.join(objdir, objname), id)

    def writexml(self, xmlfilename='scene.xml'):
        if self.topfromreader is not None:
            self.top = self.topfromreader
        else:
            self.top = Element('scene')
        for i in range(len(self.poselist)):
            obj_entry = SubElement(self.top, 'obj')

            obj_name = SubElement(obj_entry, 'obj_id')
            obj_name.text = str(self.objidlist[i])

            obj_name = SubElement(obj_entry, 'obj_name')
            obj_name.text = self.objnamelist[i]

            obj_path = SubElement(obj_entry, 'obj_path')
            obj_path.text = self.objpathlist[i]
            pose = self.poselist[i]
            pose_in_world = SubElement(obj_entry, 'pos_in_world')
            pose_in_world.text = '{:.4f} {:.4f} {:.4f}'.format(
                pose[0, 3], pose[1, 3], pose[2, 3])

            rotationMatrix = pose[0:3, 0:3]
            quat = mat2quat(rotationMatrix)

            ori_in_world = SubElement(obj_entry, 'ori_in_world')
            ori_in_world.text = '{:.4f} {:.4f} {:.4f} {:.4f}'.format(
                quat[0], quat[1], quat[2], quat[3])
            
        xmlstr = xml.dom.minidom.parseString(
            tostring(self.top)).toprettyxml(indent='    ')
        # remove blank line
        xmlstr = "".join([s for s in xmlstr.splitlines(True) if s.strip()])
        with open(xmlfilename, 'w') as f:
            f.write(xmlstr)
            print('log:write annotation file '+xmlfilename)


class xmlReader():
    def __init__(self, xmlfilename):
        self.xmlfilename = xmlfilename
        etree = ET.parse(self.xmlfilename)
        self.top = etree.getroot()

    def showinfo(self):
        print('Resumed object(s) already stored in '+self.xmlfilename+':')
        for i in range(len(self.top)):
            print(self.top[i][1].text)

    def gettop(self):
        return self.top

    def getposevectorlist(self):
        # posevector foramat: [objectid,x,y,z,alpha,beta,gamma]
        posevectorlist = []
        for i in range(len(self.top)):
            objectid = int(self.top[i][0].text)
            objectname = self.top[i][1].text
            objectpath = self.top[i][2].text
            translationtext = self.top[i][3].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][4].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            alpha, beta, gamma = quat2euler(quat)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([objectid, x, y, z, alpha, beta, gamma])
        return posevectorlist


def empty_pose_vector(objectid):
	return [objectid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def get_pose_vector(objectid, times, objectfilenamelist):
    transformation_file = 'results/transformation/{}.npy'.format(objectid)
    if not os.path.exists(transformation_file):
        return empty_pose_vector(objectid)
    
    with open(transformation_file, 'rb') as f:
        T = np.load(f)
    
    json_file = 'results/{}-{}.json'.format(objectid, times)
    assert os.path.exists(json_file)

    with open(json_file, 'r') as f:
        js = json.load(f)

    obj_name, _ = os.path.splitext(objectfilenamelist[objectid])

    js = js['TrackerData']['TargetPoses']
    obj_found = False
    for sub_js in js:
        if sub_js['TargetPose']['name'] == obj_name:
            T_tracker_marker = np.array(sub_js['TargetPose']['TransformationMatrix']).reshape(4, 4)
            obj_found = True
    print(T)
    T_initial = T.dot(T_tracker_marker)
    x, y, z, alpha, beta, gamma = get_pose(T_initial)
    return [objectid, x, y, z, alpha, beta, gamma]