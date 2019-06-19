import cv2
import cv2.aruco as aruco
import numpy as np
import math

def detect_show_marker(img, gray, aruco_dict, parameters, cameraMatrix, 
                       distCoeffs):
    detected_1, detected_2 = False, False
    id_aruco = 6  # Id of target aruco.
    distance = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, 
                                                          parameters=parameters)
    img = aruco.drawDetectedMarkers(img, corners, ids)
    if ids is not None:
        for k in range(0, len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[k], 
                                                                       0.045, 
                                                                       cameraMatrix, 
                                                                       distCoeffs)
            if  ids[k] == id_aruco:
                img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.05)
                distance = tvec[0][0][2]
                rmat = cv2.Rodrigues(rvec)[0]
                
                # Orientation aruco regarding camera.
                angles1 = rotmtx_to_euler_angles(rmat) 
                
                # Camera position regarding aruco.
                pos_cam_to_aruco = -np.matrix(rmat).T * np.matrix(tvec).T 
                cam_rotmtx =  np.matrix(rmat).T
                
                # Reverse orientation camera to the aruco.
                angles2 = rotmtx_to_euler_angles(cam_rotmtx) 
    if (distance is not None):
        cv2.putText(img, 'Id' + str(id_aruco) + ' %.2fsm' % (distance * 100), (0, 64), font, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.imshow('frame', img) # Final img.


def rotmtx_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6 
    if  not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else :
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])
