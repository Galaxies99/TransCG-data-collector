from xmlhandler import xmlWriter
from trans3d import get_mat,pos_quat_to_pose_4x4,pose_4x4_to_pos_quat
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
import numpy as np
import os
from icp import refine_6d_pose
import open3d as o3d
class frameTransformer():
    def __init__(self,cameraPosesNpyFile,baseNumFrame,framePoseVectorList,objnamelist,objidlist,cam,models_ply,existed_object_file_name_list,rgb_image_dir,depth_image_dir,is_icp):
        self.camera_poses = self.loadCameraPose(cameraPosesNpyFile)
        self.baseNumFrame = baseNumFrame
        self.totalFrameNumber = len(framePoseVectorList)
        self.framePoseVectorList = framePoseVectorList
        self.objectNameList = objnamelist
        self.objectIdList = objidlist
        self.existed_object_file_name_list = existed_object_file_name_list
        self.cam = cam
        self.T = self.getT() # T is a list of matrix, each matrix is for one object
        self.models_ply = models_ply
        self.rgb_image_dir = rgb_image_dir
        self.depth_image_dir = depth_image_dir
        self.is_icp = is_icp
    

    def loadCameraPose(self,npyfile):
        cameraposes = np.load(npyfile)
        # because there are two versions of camera poses npy files.
        # one is the (:,7) shape as [pos(3),quat(4)]
        # the other is the (:,4,4) shape that specifies the 4x4 pose matrix
        if cameraposes.shape[1] == 4:
            list_cameraposes=[]
            # check if it is 4x4 shape
            assert cameraposes.shape[2] == 4
            for i in range(len(cameraposes)):
                pos,quat = pose_4x4_to_pos_quat(cameraposes[i])
                list_cameraposes.append([pos[0],pos[1],pos[2],quat[0],quat[1],quat[2],quat[3]])
            return np.array(list_cameraposes)
        # the other shape
        elif cameraposes.shape[1] == 7:
            return cameraposes
        else:
            # other shape is not accepted
            raise ValueError('Shape for camerapose should be either (:,7) or (:,4,4)')

        
    # T is the 4x4 transformation matrix of the object with respect to the fixed base of the arm.
    # The dtype of T is np.matrix
    # T = T0 * T0_
    def getT(self):
        # posevector foramat: [objectid,x,y,z,alpha,beta,gamma] 
        baseCameraPose = self.camera_poses[self.baseNumFrame]
        # camera pose format: [translation, quat]
        translation0 = baseCameraPose[0:3]
        quat0 = baseCameraPose[3:7]
        # T0 is the transformation matrix of camera to arm base
        T0 = np.matrix(pos_quat_to_pose_4x4(translation0,quat0))
        # T is a list of matrix, each matrix describe the pose of one object
        T = []
        for baseFramePoseVector in self.framePoseVectorList[self.baseNumFrame]:
            translation0_ = np.array(baseFramePoseVector[1:4])
            alpha,beta,gamma=baseFramePoseVector[4:7]
            # convert the unit of alpha beta and gamma into radian 
            euler0_ = np.array([alpha, beta, gamma]) / 180.0 * np.pi
            quat0_ = np.array(euler2quat(euler0_[0],euler0_[1],euler0_[2]))
            T0_ =  np.matrix(pos_quat_to_pose_4x4(translation0_,quat0_))
            # T0_ is the transformation matrix of object to camera
            T.append(np.matmul(T0,T0_))
        return T

    def getPoseVectorList(self,frameNumber):
        # the key equation is T = T0 * T0_ = Tn * Tn_
        print('log: converting frame:%d' % frameNumber)
        cameraPose = self.camera_poses[frameNumber]
        quat = cameraPose[3:7]
        translation = cameraPose[0:3]
        Tn = np.matrix(pos_quat_to_pose_4x4(translation,quat))
        TnInverse = np.linalg.inv(Tn)
        PoseVectorList = []
        for i in range(len(self.T)):
            Tn_ = np.matmul(TnInverse,self.T[i])
            rotationMatrixn_ = Tn_[0:3,0:3]
            translationn_ = np.array(Tn_[0:3,3].T)[0]
            quatn_ = mat2quat(rotationMatrixn_)
            eulern_ = quat2euler(quatn_)
            x,y,z = translationn_
            alpha,beta,gamma = np.array(eulern_) / np.pi * 180.0
            if self.models_ply[i] is not None and self.is_icp:
                x, y, z, alpha, beta, gamma = refine_6d_pose(os.path.join(self.rgb_image_dir,'%04d.png' % frameNumber),os.path.join(self.depth_image_dir,'%04d.png' % frameNumber),self.models_ply[i], self.cam, x, y, z, alpha, beta, gamma)
            poseVector = [self.objectIdList[i],x,y,z,alpha,beta,gamma]
            PoseVectorList.append(poseVector)
        return PoseVectorList

    def transform(self):
        for numFrame in range(self.totalFrameNumber):
            self.framePoseVectorList[numFrame] = self.getPoseVectorList(numFrame)

    # def writeFramexml(self):
    #     for numFrame in range(self.totalFrameNumber):
    #         self.getFramePoseVectorList(numFrame)
    #         if numFrame == 0:
    #             # write an xml especially for the first frame 
    #             self.xmlWriter.writexml(xmlfilename=self.FLAGS.output_xml_filename)
    #         self.xmlWriter.writexml(xmlfilename = os.path.join(self.FLAGS.output_xml_dir,self.FLAGS.output_xml_filename.replace('.xml','-Frame-'+str(numFrame)+'.xml')))
