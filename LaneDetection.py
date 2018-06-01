import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from sklearn import linear_model
import timeit

#def on_mouse(event, x, y, flags, param):
#    if event == cv.EVENT_MOUSEMOVE:
#        print(x, y)

def main(video_path):
    start = timeit.default_timer()

# Adding data path and capture
    captured = cv.VideoCapture(video_path)
    videolength = int(captured.get(cv.CAP_PROP_FRAME_COUNT))
    #print(videolength)
    # get captured property
    width = int(captured.get(3))   # float
    height = int(captured.get(4)) # float
# Creating masking polygon
    #cv.namedWindow("output")
    #cv.setMouseCallback("output", on_mouse, 0)
    mask = np.zeros((height, width, 1), np.uint8)

    # masking region for each video
    if width ==1280:
        maskPoints = np.array([[1158, 649], [196, 649], [196, 649],[535 , 461],[ 740, 461], [1158, 649]])#udacity
    else:
        maskPoints = np.array([[639, 270], [1, 270], [1, 180], [250, 90], [450, 90], [639, 180]])#3dk

    mask = cv.fillConvexPoly(mask, maskPoints, 255)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output = cv.VideoWriter('output.mkv', fourcc, 20, (width,height))

    frame_no = 0
    previous_frames=np.zeros((2,height, width,3))

# previous_curves matrix will be used for prediction after polynom fitting
    previous_curves = np.zeros((3, 2, 3))

# Read frame-by-frame in loop
    while captured.isOpened():
        frame_no = frame_no + 1
        print(frame_no)
        if frame_no == videolength:
            break
        ret, frame = captured.read()
        if frame_no < 3:
            previous_frames[0] = frame
            previous_frames[1] = frame

        average_frame = (previous_frames[0]+previous_frames[1]+frame)/3
        average_frame = average_frame.astype(np.uint8)
        previous_frames[0] = previous_frames[1]
        previous_frames[1] = frame
# Filtering
        #blurred = cv.medianBlur(frame, 3)
        #blurred = cv.blur(frame, (3, 3))
        blurred = cv.GaussianBlur(average_frame, (3, 3), 3)
        blurred = cv.bilateralFilter(blurred, 3, 25, 25)
        laplacian64f = cv.Laplacian(blurred, cv.CV_64F)
        laplacian64uf = np.absolute(laplacian64f)
        laplacian8ui = np.uint8(laplacian64uf)
        ret, thresholded_binary = cv.threshold(laplacian8ui, 8, 255, cv.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated_frame = cv.dilate(thresholded_binary, kernel)
        eroded_frame = cv.erode(dilated_frame, kernel)
        thresholded_filtered_binary = cv.cvtColor(eroded_frame, cv.COLOR_BGR2GRAY)
        thresholded_filtered_binary = cv.bitwise_and(thresholded_filtered_binary, mask)

## end of FT's work

## Taha & Ercan's work

# Set parameters and apply Hough Transform
        threshold = 10
        minLineLength = 1
        maxLineGap = 5
        lines = cv.HoughLinesP(thresholded_filtered_binary, 1, np.pi / 180, threshold, 0, minLineLength, maxLineGap)
# Eliminate lines which has a slope lower than a determined threshold
        slope_threshold=0.1
        slope = np.divide(lines[:,0,3]-lines[:,0,1],0.01+lines[:,0,2]-lines[:,0,0])
        #thresholded_lines = lines[np.where(abs(slope)>slope_threshold)]
        thresholded_lines=lines[np.where((slope > 0) ^ (lines[:,0,2] < (width/2)))]
# Collect start and end points of lines in a single matrix
        points = np.concatenate((thresholded_lines[:, :, 0:2], thresholded_lines[:, :, 2:4]), axis=0)
# Cluster points to two classes, meaning left and right lines of the lane
        points = points[:,0]

        '''
    # Clustering with a strait column    
        cluster_1= points[np.where(points[:,0]<width/2),:]
        cluster_2= points[np.where(points[:,0]>width/2),:]
        '''
    # Perform Kmeans for the first frame
        if frame_no == 1:
            kmeans = KMeans(init='k-means++', n_clusters=2)
            kmeans = kmeans.fit(points)
            c = kmeans.cluster_centers_
    # Perform KMeans with final cluster centers of previous iteration as initial cluster centers
        kmeans = KMeans(n_clusters=2, n_init=1, init=c)
        kmeans = kmeans.fit(points)
        labels = kmeans.predict(points)
        c = kmeans.cluster_centers_

# Fill clusters by looking at labels
        cluster_1 = points[np.where(labels == 0),:]
        cluster_2 = points[np.where(labels == 1),:]


        if cluster_1.shape[1]<6 or cluster_2.shape[1]<6:
            X1 = np.polyval(5 * previous_curves[2][0] / 2 - 2 * previous_curves[1][0] + previous_curves[0][0] / 2, Y)
            X2 = np.polyval(5 * previous_curves[2][1] / 2 - 2 * previous_curves[1][1] + previous_curves[0][1] / 2, Y)
            continue


# Eliminate outliers by Random Sample Consensus
        ransac = linear_model.RANSACRegressor()

        # Cluster 1
        if cluster_1.shape[1]>10:
            ransac.fit(cluster_1[:, :, 0].T, cluster_1[:, :, 1].T)
            inlier_mask_1 = ransac.inlier_mask_
            filtered_cluster_1 = cluster_1[0, np.where(inlier_mask_1 == True)]
        else:
            filtered_cluster_1 = cluster_1

        # Cluster 2
        if cluster_2.shape[1] > 10:
            ransac.fit(cluster_2[:, :, 0].T, cluster_2[:,:,1].T)
            inlier_mask_2 = ransac.inlier_mask_
            filtered_cluster_2 = cluster_2[0, np.where(inlier_mask_2 == True)]
        else:
            filtered_cluster_2 = cluster_2

# Fit clusters to second order polynomials / x=a*y^2+b*y+c
        curve_1 = np.polyfit(filtered_cluster_1[0, :, 1], filtered_cluster_1[0, :, 0], 2)
        curve_2 = np.polyfit(filtered_cluster_2[0, :, 1], filtered_cluster_2[0, :, 0], 2)

        #print(curve_1)
        #print(curve_2)

# Create point spaces for curvy lines
        y_end = min(np.amin(cluster_1[0,:,1]),np.amin(cluster_2[0,:,1]))
        y_start = max(np.amax(cluster_1[0,:,1]),np.amax(cluster_2[0,:,1]))
        num_of_points=200
        Y = np.linspace(y_end, y_start, num_of_points)
        X1 = np.polyval(curve_1, Y)
        X2 = np.polyval(curve_2, Y)

# a quadratic prediction for bad curves
        if frame_no>3:
            if abs(X1[num_of_points-1] - X2[num_of_points-1]) < 500:
                X1 = np.polyval(5*previous_curves[2][0]/2 - 2*previous_curves[1][0] + previous_curves[0][0]/2 , Y )
                X2 = np.polyval(5*previous_curves[2][1]/2 - 2*previous_curves[1][1] + previous_curves[0][1]/2 , Y)
            if abs(curve_1[0]) > 0.005:
                X1 =  np.polyval(5 * previous_curves[2][0] / 2 - 2 * previous_curves[1][0] + previous_curves[0][0] / 2, Y)
            if abs(curve_2[0]) > 0.005:
                X2 =  np.polyval(5 * previous_curves[2][1] / 2 - 2 * previous_curves[1][1] + previous_curves[0][1] / 2, Y)

        previous_curves[0][0] = previous_curves[1][0]
        previous_curves[0][1] = previous_curves[1][1]
        previous_curves[1][0] = previous_curves[2][0]
        previous_curves[1][1] = previous_curves[2][1]
        previous_curves[2][0] = curve_1
        previous_curves[2][1] = curve_2


        #print('curve1:')
        #print(curve_1)
        #print('curve2:')
        #print(curve_2)



# Visualize the lanes
        '''
    # Filtered Image
        cv.imshow("output" , thresholded_filtered_binary)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.waitKey()
        '''
        '''
    # Hough Lines
        if lines is not None:
            for i in range(len(lines)):
                cv.line(frame, (lines[i, 0, 0], lines[i, 0, 1]), (lines[i, 0, 2], lines[i, 0, 3]), (0, 0, 255), 3, cv.LINE_AA)
        '''
        '''
    # Thresholded Hough Lines
        if thresholded_lines is not None:
            for i in range(len(thresholded_lines)):
                cv.line(frame, (thresholded_lines[i, 0, 0], thresholded_lines[i, 0, 1]), (thresholded_lines[i, 0, 2], thresholded_lines[i, 0, 3]), (0, 255, 0), 3, cv.LINE_AA)
        '''
        '''
    # Cluster 1
        if cluster_1 is not None:
            for i in range(cluster_1.shape[1]):
                cv.line(frame, (cluster_1[0, i, 0], cluster_1[0, i, 1]), (cluster_1[0, i, 0], cluster_1[0, i, 1]), (0, 0, 0), 3, cv.LINE_AA)
        '''
        '''
    # Cluster 2
        if cluster_2 is not None:
            for i in range(cluster_2.shape[1]):
                cv.line(frame, (cluster_2[0, i, 0], cluster_2[0, i, 1]),(cluster_2[0, i, 0], cluster_2[0, i, 1]), (0, 0, 0), 3, cv.LINE_AA)
        '''
        '''
    # Cluster 1 inliers
        if filtered_cluster_1 is not None:
            for i in range(filtered_cluster_1.shape[1]):
                cv.line(frame, (filtered_cluster_1[0, i, 0], filtered_cluster_1[0, i, 1]), (filtered_cluster_1[0, i, 0], filtered_cluster_1[0, i, 1]), (0, 255, 255), 3, cv.LINE_AA)
        '''
        '''
    # Cluster 2 inliers
        if filtered_cluster_2 is not None:
            for i in range(filtered_cluster_2.shape[1]):
                cv.line(frame, (filtered_cluster_2[0, i, 0], filtered_cluster_2[0, i, 1]),(filtered_cluster_2[0, i, 0], filtered_cluster_2[0, i, 1]), (255, 255, 0), 3, cv.LINE_AA)
        '''
    # Estimated road lines
        for i in range(num_of_points-2):
            cv.line(frame, (int(X1[i]), int(Y[i])), (int(X2[i]), int(Y[i])), (0, 0, 255), 3, cv.LINE_AA)

    # Video
        #'''
        output.write(frame)
        cv.imshow("output", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        #'''
        #cv.waitKey()
    stop = timeit.default_timer()
    print('runtime: ' + str(stop - start))

# add video path and run
main('input_video_50_sec.mkv')