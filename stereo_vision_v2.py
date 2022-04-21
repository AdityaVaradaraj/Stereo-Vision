import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def getParams(dataset):
    if dataset == 'curule':
        cam0=np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        cam1=np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15] ,[0, 0, 1]])
        doffs=0
        baseline=88.39
        width=1920
        height=1080
        ndisp=220
        vmin=55
        vmax=195
        window_size = 15
        stride = 35
    elif dataset == 'octagon':
        cam0=np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        cam1=np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        doffs=0
        baseline=221.76
        width=1920
        height=1080
        ndisp=100
        vmin=29
        vmax=61
        window_size = 20
        stride= 35
    elif dataset == 'pendulum':
        cam0=np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        cam1=np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        doffs=0
        baseline=537.75
        width=1920
        height=1080
        ndisp=180
        vmin=25
        vmax=150
        window_size = 20
        stride=100
    K1 = cam0
    K2 = cam1
    return K1, K2, baseline, window_size, stride

def compute_Fundamental(pts1, pts2):
    A = np.zeros((len(pts2), 9))
    
    for i in range(len(pts2)):
        A[i, :] = [pts1[i,0]*pts2[i,0], pts1[i,0]*pts1[i,1], pts1[i,0], pts1[i,1]*pts2[i,0], pts1[i,1]*pts1[i,1], pts1[i,1], pts2[i,0], pts1[i,1], 1]
    
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1,:].T
    F = np.reshape(F, (3,3))
    U ,sigma , vt = np.linalg.svd(F)
    sigma = np.diag(sigma)
    sigma[2,2] = 0
    F_ = np.dot(U , np.dot(sigma , vt))
    return F_

def epipolarError(pt1 , pt2 , F):
    pt1 = np.array([pt1[0] , pt1[1] , 1])
    pt2 = np.array([pt2[0] , pt2[1] , 1]).T
    error = np.dot(pt2 , np.dot(F,pt1))

    return abs(error)

def ransac_Fundamental(feat_1,feat_2):
    eps =0.02
    max_inliers = 0
    N = 100
    n_rows = feat_1.shape[0]
    for i in range(N):
        indices = []
        random_indices = np.random.choice(n_rows , size = 8)
        points1_8 = feat_1[random_indices]
        points2_8 = feat_2[random_indices]
        F = compute_Fundamental(points1_8 , points2_8)
        # print(F)
        for j in range(n_rows):
            error = epipolarError(feat_1[j] , feat_2[j] , F)
            if error < eps:
                indices.append(j)
        if len(indices) > max_inliers:
            max_inliers = len(indices)
            inliers = indices
            F_final = F
    pts1_inliers , pts2_inliers = feat_1[inliers] , feat_2[inliers]
    return F_final , pts1_inliers , pts2_inliers

def SIFT_feature_detection(image):
    sift = cv2.SIFT_create()    
    (kpoints, features) = sift.detectAndCompute(image, None)
    return (kpoints, features)

def matchPoints(kpsA,featuresA, kpsB, featuresB):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(featuresA,featuresB, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)
    kp1_lst = []
    kp2_lst = []
    for match in good_matches:
        ImgA_idx = match.queryIdx
        ImgB_idx = match.trainIdx
        (x1,y1) = kpsA[ImgA_idx].pt
        (x2,y2) = kpsB[ImgB_idx].pt
        kp1_lst.append((x1,y1))
        kp2_lst.append((x2,y2)) 
    return np.array(kp1_lst), np.array(kp2_lst)

def compute_Essential(K1, K2,F):
    E = np.matmul(K2.T, np.matmul(F, K1))
    u,s,v = np.linalg.svd(E)
    s = [1,1,0]
    E_fin = np.dot(u, np.dot(np.diag(s),v))
    return E_fin

def decompose_E(E):
    u,s,vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
    c1 = u[:,2]
    r1 = np.dot(u , np.dot(W , vt))
    c2 = -u[:,2]
    r2 = np.dot(u , np.dot(W , vt))
    c3 = u[:,2]
    r3 = np.dot(u , np.dot(W.T , vt))
    c4 = -u[:,2]
    r4 = np.dot(u , np.dot(W.T , vt))
    if (np.linalg.det(r1) < 0):
        r1 = -r1
        c1 = -c1
    if (np.linalg.det(r2) < 0):
        r2 = -r2
        c2 = -c2
    if (np.linalg.det(r3) < 0):
        r3 = -r3
        c3 = -c3
    if (np.linalg.det(r4) < 0):
        r4 = -r4
        c4 = -c4
    c1 = c1.reshape((3,1))
    c2 = c2.reshape((3,1))
    c3 = c3.reshape((3,1))
    c4 = c4.reshape((3,1))
    
    return [r1,r2,r3,r4] , [c1,c2,c3,c4]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    row,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def SSD(win1 , win2):
    if win1.shape != win2.shape:
        return -1
    ssd = np.sum(np.dot((win1 - win2).T, (win1 - win2)))
    return ssd

def disparityMap(gray1, gray2, window_size, search_stride):
    height , width = gray1.shape
    dis_map = np.zeros_like(gray1, dtype=np.float64)
    gray1 , gray2 = gray1.astype(np.float64) , gray2.astype(np.float64)
    for y in tqdm(range(window_size, (height-window_size))):
        for x in range(window_size, (width - window_size)):
            window = gray1[y-(window_size//2):y+(window_size//2) , x-(window_size//2):x+(window_size//2)]
            x1 = blockMatching(y,x,window,gray2 , window_size , search_stride)
            dis_map[y,x] = (x1 - x)

    # Shift of Origin to make most Negative value zero
    dis_map = dis_map + np.abs(np.min(dis_map))
    dis_map_2= (dis_map/np.max(dis_map))*255
            
    return dis_map_2.astype(np.uint8), dis_map

def blockMatching(y,x,window,gray2 , window_size , searchRange):
    height1 , width1 = gray2.shape
    x_start = max(0, x - searchRange)
    x_end = min(width1 , x + searchRange)
    min_x = np.inf
    min_ssd  = np.inf
    
    for x in range(x_start , x_end, window_size):
        window2 = gray2[y-(window_size//2):y+(window_size//2) , x-(window_size//2):x+(window_size//2)]
        ssd = SSD(window , window2)
        if ssd < min_ssd:
            min_ssd = ssd 
            min_x = x
    return  min_x

def Compute_Depth(disparity_map , baseline , f):
    depth_map = (baseline*f)/(disparity_map + 1e-10)
    depth_map = np.uint8(depth_map *255 / np.max(depth_map))
    return depth_map


if __name__ == '__main__':

    dataInput = input('Enter dataset name (curule/pendulum/octagon) : ')
    imageFold = './data/'

    dataPath0 = imageFold + dataInput + '/im0.png'
    dataPath1 = imageFold + dataInput + '/im1.png'

    ImgA_c = cv2.imread(dataPath0)
    ImgB_c = cv2.imread(dataPath1)
    ImgA = cv2.cvtColor(ImgA_c, cv2.COLOR_BGR2GRAY)
    ImgB = cv2.cvtColor(ImgB_c, cv2.COLOR_BGR2GRAY)
    ha,wa = ImgA.shape[:2]
    hb,wb = ImgB.shape[:2]
    kpsA, featuresA = SIFT_feature_detection(ImgA)
    kpsB, featuresB = SIFT_feature_detection(ImgB)
    print("Features detected")

    kp1_lst, kp2_lst = matchPoints(kpsA,featuresA,kpsB,featuresB)
    kp1_lst, kp2_lst = np.array(kp1_lst), np.array(kp2_lst)

    print("calculating fm")
    F , pts1_inliers , pts2_inliers = ransac_Fundamental(np.int32(kp1_lst),np.int32(kp2_lst))
    print("Fundamental Matrix:")
    print(F)

    K1, K2, baseline, window_size, stride = getParams(dataInput)


    print("calculating em")
    E = compute_Essential(K1, K2,F)
    print("Essential Matrix:")
    print(E)
    print("calculated em")
    print("calculating R&C")
    R,C = decompose_E(E)
    print("calculated R&C")


    lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(ImgA, ImgB, lines1, pts1_inliers, pts2_inliers)
    lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(ImgB, ImgA, lines2, pts2_inliers, pts1_inliers)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines in both images")
    plt.savefig("Epilines_with_features.png")
    plt.show()

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1_inliers), np.float32(pts2_inliers), F, imgSize=(wa, ha))
    print("H1:")
    print(H1)
    print("H2:")
    print(H2)
    img1_rectified = cv2.warpPerspective(ImgA, H1, (wa, ha))
    img2_rectified = cv2.warpPerspective(ImgB, H2, (wb, hb))
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1_rectified, cmap="gray")
    axes[1].imshow(img2_rectified, cmap="gray")
    axes[0].axhline(250)
    axes[1].axhline(250)
    axes[0].axhline(450)
    axes[1].axhline(450)
    plt.suptitle("Rectified images")
    plt.savefig("rectified_images.png")
    plt.show()
    print("Window Size: ", window_size)
    print("Stride: ", stride)
    disparity, disparity_original = disparityMap(img2_rectified, img1_rectified, window_size, stride)
    plt.imshow(disparity , cmap = 'gray')
    plt.savefig("disparity_map.png")
    plt.show()
    print(disparity)
    f = K1[0,0]
    Depth_Value = np.zeros((ha,wa), dtype=np.uint8)
    depthMap = np.zeros((ha,wa), dtype=np.uint8)
    heatmap = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow('disparity heatmap', heatmap)
    cv2.imwrite("disparity_heatmap.png", heatmap)
    
    depthMap = Compute_Depth(disparity_original, baseline, f)
    depth_heatmap = cv2.applyColorMap(depthMap, cv2.COLORMAP_JET)
    cv2.imshow('depth heatmap', depth_heatmap)
    cv2.imwrite("depth_heatmap.png", depth_heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.imshow(depthMap , cmap = 'gray')
    plt.savefig("depth_map.png")
    plt.show()
    

