import cv2 as cv
import timeit
#import numpy as np


def compute_jaccard_index(set_1, set_2):
    return len(set_1.intersection(set_2)) / float(len(set_1.union(set_2)))


def compute_jaccard_index_2(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print("mouse", "x=" + str(x), "y=" + str(y))
        print("frame", frame[y, x])


# Adding data path and capture
v2_out_video_path = '2minutes-output.mkv'
v50_uda_out_video_path = '50_sec_out_UDACITY_CODE.mp4'
v50_uda_red_out_video_path='50_sec_out_RED_UDACITY_CODE.mp4'
v50_out_video_path = '50second-output.mkv'
v2_gro_video_path = 'groundtruth_2_min.mkv'
#v50_gro_video_path = 'groundtruth_50_sec.mkv'
v50_gro_1280_video_path='groundtruth_50_sec_1280_720.mkv'
v2_in_video_path = 'input_video_2_min.mkv'
v50_in_video_path = 'input_video_50_sec.mkv'

frame_acc = []

captured_out = cv.VideoCapture(v2_out_video_path)
captured_out2_ground = cv.VideoCapture(v2_gro_video_path)



y_alt=[253, 254, 254, 250, 249, 250, 257, 242, 258, 258, 254, 253, 246, 246, 245, 246, 246, 250, 254, 256, 259, 257, 256, 248, 250, 254, 252, 254, 254, 254, 258, 258, 254, 254, 240, 229, 239, 232, 226, 232, 241, 244, 242, 250, 255, 258, 259, 258, 255, 259, 261, 255, 258, 254, 254, 246, 239, 250, 246, 250, 252, 256, 258, 252, 245, 242, 241, 241, 244, 243, 246, 254, 253, 236, 235, 240, 247, 247, 240, 246, 242, 239, 230, 246, 246, 254, 250, 246, 246, 254, 215, 246, 246, 246, 246, 238, 239, 242, 240, 251, 255, 245, 240, 252, 256, 249, 241, 223, 214, 200, 190, 262, 256, 254, 270, 269, 267, 170, 174, 182, 191, 194, 199, 208, 214, 217, 218, 222, 219, 221, 217, 222, 215, 217, 215, 217, 220, 218, 222, 223, 226, 226, 219, 219, 215, 214, 217, 217, 218, 219, 221, 220, 218, 221, 218, 218, 216, 215, 214, 210, 210, 208, 209, 211, 213, 216, 210, 211, 202, 196, 182, 177, 168, 270, 269, 269, 192, 197, 226, 226, 236, 250, 250, 244, 246, 260, 236, 235, 236, 246, 246, 249, 244, 246, 240, 241, 240, 224, 246, 242, 242, 244, 234, 225, 245, 245, 247, 249, 249, 253, 255, 258, 252, 258, 258, 230, 241, 254, 250, 250, 250, 245, 245, 246, 247, 249, 254, 255, 254, 259, 259, 249, 258, 258, 254, 258, 246, 248, 254, 254, 260, 258, 236, 242, 254, 254, 241, 227, 211, 190, 269, 167, 177, 187, 210, 230, 249, 246, 214, 233, 244, 248, 239, 250, 242, 242, 242, 237, 238, 236, 235, 234, 232, 228, 224, 219, 216, 222, 223, 226, 221, 218, 222, 210, 228, 230, 228, 231, 232, 227, 222, 231, 226, 231, 230, 232, 239, 243, 242, 242, 243, 241, 238, 232, 232, 234, 236, 232, 239, 233, 239, 239, 238, 235, 234, 232, 235, 234, 237, 238, 235, 230, 243, 242, 239, 235, 233, 238, 232, 222, 231, 230, 229, 227, 233, 229, 230, 235, 224, 227, 230, 232, 227, 229, 236]
# print("length_y_alt",len(y_alt))
y_ust=[93, 90, 89, 97, 92, 91, 91, 99, 97, 95, 91, 91, 98, 95, 93, 93, 91, 97, 96, 95, 90, 101, 94, 98, 95, 90, 91, 99, 95, 91, 91, 102, 98, 96, 95, 108, 90, 106, 98, 98, 97, 92, 97, 94, 91, 101, 90, 95, 94, 91, 90, 98, 96, 91, 91, 92, 94, 97, 91, 90, 100, 96, 93, 90, 96, 97, 93, 90, 91, 95, 90, 90, 90, 95, 97, 90, 91, 98, 98, 91, 91, 98, 97, 93, 90, 93, 92, 91, 90, 90, 95, 91, 90, 98, 94, 92, 102, 99, 95, 94, 98, 96, 94, 89, 89, 95, 91, 102, 114, 89, 91, 90, 89, 120, 92, 94, 88, 89, 90, 86, 89, 89, 94, 94, 94, 128, 91, 92, 93, 92, 91, 91, 94, 94, 91, 93, 94, 94, 94, 92, 97, 98, 92, 90, 98, 98, 90, 90, 90, 90, 92, 98, 90, 92, 92, 98, 93, 94, 91, 96, 93, 90, 91, 98, 95, 90, 93, 91, 90, 89, 98, 105, 122, 90, 88, 89, 113, 97, 91, 91, 89, 95, 95, 91, 102, 98, 97, 99, 99, 94, 93, 96, 94, 93, 89, 89, 90, 90, 90, 94, 96, 94, 90, 91, 96, 90, 92, 95, 96, 92, 94, 92, 98, 95, 92, 91, 98, 97, 95, 93, 92, 91, 90, 91, 90, 91, 93, 93, 91, 95, 94, 94, 89, 90, 90, 90, 98, 98, 89, 90, 97, 93, 94, 102, 100, 94, 97, 93, 122, 98, 90, 93, 90, 91, 92, 98, 99, 90, 91, 99, 92, 91, 98, 94, 95, 98, 91, 95, 90, 94, 93, 93, 91, 97, 94, 94, 93, 93, 100, 94, 97, 98, 94, 90, 94, 98, 91, 92, 98, 92, 93, 92, 91, 93, 94, 95, 93, 94, 92, 91, 91, 92, 97, 97, 94, 92, 96, 94, 91, 93, 92, 98, 96, 97, 92, 92, 91, 94, 99, 91, 92, 94, 92, 94, 93, 92, 98, 96, 94, 93, 92, 93, 92, 98, 91, 97, 98, 95, 91, 92, 98, 98, 99, 99, 99]
# print("length_y_ust",len(y_ust))



# Read frame-by-frame in loop
frame_no = 0

start = timeit.default_timer()

while captured_out.isOpened():
    frame_no = frame_no + 1

    # captured_out2_ground.set(cv.CAP_PROP_FRAME_WIDTH,captured_out.get(cv.CAP_PROP_FRAME_WIDTH))
    # captured_out2_ground.set(cv.CAP_PROP_FRAME_HEIGHT,captured_out.get(cv.CAP_PROP_FRAME_HEIGHT))

    ret, frame = captured_out.read()
    ret2, frame2 = captured_out2_ground.read()

    pixels_frame = set()
    pixels_frame2 = set()

    ''' BGR
    0 blue
    1 green
    2 red
    '''

    videolength = int(captured_out.get(cv.CAP_PROP_FRAME_COUNT))
    curr_frame_no = int(captured_out.get(cv.CAP_PROP_POS_FRAMES))
    # videolength2 = int(captured_out2_ground.get(cv.CAP_PROP_FRAME_COUNT))
    # curr_frame_no2 = int(captured_out2_ground.get(cv.CAP_PROP_POS_FRAMES))
    v_width = int(captured_out.get(cv.CAP_PROP_FRAME_WIDTH))
    v_height = int(captured_out.get(cv.CAP_PROP_FRAME_HEIGHT))
    v_width2 = int(captured_out2_ground.get(cv.CAP_PROP_FRAME_WIDTH))
    # v_height2 = int(captured_out2_ground.get(cv.CAP_PROP_FRAME_HEIGHT))

    print("frame_no:", frame_no, "curr_frame_no:", curr_frame_no)
    print("frame count:", videolength)
    print("w=", v_width, "h=", v_height)
    # print("frame_no2:", frame_no, "curr_frame_no2:", curr_frame_no)
    # print("frame count2:", videolength)
    # print("w2=", v_width2, "h2=", v_height2)


    # bgr_image_array = np.asarray(frame)
    # B, G, R = bgr_image_array.T
    # rgb_image_array = np.array((R, G, B)).T
    # rgb_image = Image.fromarray(rgb_image_array, mode='RGB')

    # print("R=",R,"size",np.size(R),frame[0,0])

    # frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    # img=cv.imread(frame)
    # print("img",img)
    # print("frame",frame)
    # print("frame",frame[480,640])

    # cv.namedWindow("output")
    # cv.setMouseCallback("output", on_mouse, 0)
    #
    # cv.namedWindow("ground_truth")
    # cv.setMouseCallback("ground_truth", on_mouse, 0)

    # mask = np.zeros_like(frame)
    # mask2 = np.zeros_like(frame2)

    found_pixel = 0
    found_pixel2 = 0

########################################################################################
#UDACITY_UDACITY_CODE
    #
    # for i in range(456, 647):
    #     for j in range(0, int(v_width)):
    #         if frame[i, j, 0] < 60 and frame[i, j, 1] < 60 and frame[i, j, 2] > 130:
    #             pixels_frame.add((i, j))
    #             found_pixel = found_pixel + 1
    #             # mask[i, j] = [0, 0, 255]
    #
    # # print(frame[i,j,0])
    #
    # for i in range(456, 647):
    #     for j in range(0, int(v_width2)):
    #         if frame2[i, j, 0] < 60 and frame2[i, j, 1] < 60 and frame2[i, j, 2] > 130:
    #             pixels_frame2.add((i, j))
    #             found_pixel2 = found_pixel2 + 1
    #             # mask2[i, j] = [0, 0, 255]
    #
    # frame_acc.append(compute_jaccard_index_2(pixels_frame, pixels_frame2))
    # print("lenght frame_acc", len(frame_acc))

# ########################################################################################
# #UDACITY_BIZIM_KOD
#
#     for i in range(461, 647):
#         for j in range(0, int(v_width)):
#             if frame[i, j, 0] < 60 and frame[i, j, 1] < 60 and frame[i, j, 2] > 130:
#                 pixels_frame.add((i, j))
#                 found_pixel = found_pixel + 1
#                 # mask[i, j] = [0, 0, 255]
#
#     # print(frame[i,j,0])
#
#     for i in range(461, 647):
#         for j in range(0, int(v_width2)):
#             if frame2[i, j, 0] < 60 and frame2[i, j, 1] < 60 and frame2[i, j, 2] > 130:
#                 pixels_frame2.add((i, j))
#                 found_pixel2 = found_pixel2 + 1
#                 # mask2[i, j] = [0, 0, 255]
#
#     frame_acc.append(compute_jaccard_index_2(pixels_frame, pixels_frame2))
#     print("lenght frame_acc", len(frame_acc))
#
# ########################################################################################
#BIZIM VIDEO
#
#     if curr_frame_no <= 111 or 128 <= curr_frame_no and curr_frame_no <= 173 or 185 <= curr_frame_no and curr_frame_no <= 250 or 259 <= curr_frame_no and curr_frame_no <= 345:
#         for i in range(y_ust[curr_frame_no-1], y_alt[curr_frame_no-1]):
#             for j in range(0, int(v_width)):
#                 if frame[i, j, 0] < 60 and frame[i, j, 1] < 60 and frame[i, j, 2] > 130:
#                     pixels_frame.add((i, j))
#                     found_pixel = found_pixel + 1
#                     # mask[i, j] = [0, 0, 255]
#         # print(frame[i,j,0])
#         for i in range(y_ust[curr_frame_no-1], y_alt[curr_frame_no-1]):
#             for j in range(0, int(v_width2)):
#                 if frame2[i, j, 0] < 60 and frame2[i, j, 1] < 60 and frame2[i, j, 2] > 130:
#                     pixels_frame2.add((i, j))
#                     found_pixel2 = found_pixel2 + 1
#                         # mask2[i, j] = [0, 0, 255]
#
#
#         frame_acc.append(compute_jaccard_index_2(pixels_frame, pixels_frame2))
#         print("lenght frame_acc", len(frame_acc))
#
########################################################################################
#BIZIM VIDEO-TÜM VIDEO

    for i in range(y_ust[curr_frame_no-1], y_alt[curr_frame_no-1]):
        for j in range(0, int(v_width)):
            if frame[i, j, 0] < 60 and frame[i, j, 1] < 60 and frame[i, j, 2] > 130:
                pixels_frame.add((i, j))
                found_pixel = found_pixel + 1
                # mask[i, j] = [0, 0, 255]
    # print(frame[i,j,0])
    for i in range(y_ust[curr_frame_no-1], y_alt[curr_frame_no-1]):
        for j in range(0, int(v_width2)):
            if frame2[i, j, 0] < 60 and frame2[i, j, 1] < 60 and frame2[i, j, 2] > 130:
                pixels_frame2.add((i, j))
                found_pixel2 = found_pixel2 + 1
                    # mask2[i, j] = [0, 0, 255]


    frame_acc.append(compute_jaccard_index_2(pixels_frame, pixels_frame2))
    print("lenght frame_acc", len(frame_acc))

########################################################################################
    for x in range(len(frame_acc)):
        print("frame_accuracy  x:", x, "value", frame_acc[x])

    acc_sum = 0
    #if curr_frame_no == 125:
    if curr_frame_no == 345:
        for z in range(0, len(frame_acc)):#int(curr_frame_no)):
            acc_sum = acc_sum + frame_acc[z]
        overall_accuracy=acc_sum/len(frame_acc)
        print("overall_accuracy=",overall_accuracy)
        stop1 = timeit.default_timer()
        print('runtime: ' + str(stop1 - start))
        break

    # Video
    # '''

    # if cv.waitKey(1) & 0xFF == ord('q'):
    #    break
    # if frame_no > 26: cv.imshow("output" , frame)

    # cv.putText(frame, str(frame_no), (5, 50), cv.FONT_HERSHEY_COMPLEX,2 ,255)
    # cv.putText(frame2, str(curr_frame_no2), (5, 50), cv.FONT_HERSHEY_COMPLEX,2 ,255)

    # cv.imshow("mask_output", mask)
    # cv.imshow("mask_ground", mask2)
    # cv.imshow("ground_truth", frame2)
    # if frame_no > -1: cv.imshow("output", frame)
    # if frame_no > -1: cv.waitKey()
