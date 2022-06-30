import os
import cv2
import sys
#import time
#import datetime
import logging
import numpy as np

inp_prev_file = ""
inp_curr_file = ""
out_mdet_file = ""
inp_curr_icam: int = 0
min_area_size: int = 1000
sum_area_size: int = 2000
show_bounding: bool = True
weighted_alpha: float = 0.5
debug_mode: bool = False

def show_usage():
    logging.info("usage")
    logging.info(" motion_detection_frm.py [-opt] [arg]")
    logging.info("options")
    logging.info(" --inp_prev_file  [arg] - input previous image from file")
    logging.info(" --inp_curr_file  [arg] - input current image from file")
    logging.info(" --out_mdet_file  [arg] - output image if we have found motion")
    logging.info(" --inp_curr_icam  [arg] - input camera index as current image")
    logging.info(" --min_area_size  [arg] - minimum area size of motion detector")
    logging.info(" --sum_area_size  [arg] - sum of area size around motion")
    logging.info(" --show_bounding  [arg] - draw green box around motion")
    logging.info(" --weighted_alpha [arg] - average filter between input images")
    logging.info(" --debug_mode     [arg] - debug mode flag")
    logging.info(" --help                 - print usage information and exit")

def show_settings():
    logging.info("motion detection settings:")
    logging.info("inp_prev_file:  %s", inp_prev_file)
    logging.info("inp_curr_file:  %s", inp_curr_file)
    logging.info("out_mdet_file:  %s", out_mdet_file)
    logging.info("inp_curr_icam:  %d", inp_curr_icam)
    logging.info("min_area_size:  %d", min_area_size)
    logging.info("sum_area_size:  %d", sum_area_size)
    logging.info("show_bounding:  %d", show_bounding)
    logging.info("weighted_alpha: %f", weighted_alpha)
    logging.info("debug_mode   :  %d", debug_mode)

def setup_argv():
    # setup motion detector from arguments
    global inp_prev_file, inp_curr_file, out_mdet_file, inp_curr_icam, min_area_size, sum_area_size, show_bounding, weighted_alpha, debug_mode
    argi: int = 0
    for arg in sys.argv[1:]:
        if arg=="--inp_prev_file":
            if len(sys.argv)>argi+2:
                inp_prev_file = sys.argv[argi+2]
        elif arg=="--inp_curr_file":
            if len(sys.argv)>argi+2:
                inp_curr_file = sys.argv[argi+2]
        elif arg=="--out_mdet_file":
            if len(sys.argv)>argi+2:
                out_mdet_file = sys.argv[argi+2]
        elif arg=="--inp_curr_icam":
            if len(sys.argv)>argi+2:
                inp_curr_icam = int(sys.argv[argi+2])
        elif arg=="--min_area_size":
            if len(sys.argv)>argi+2:
                min_area_size = int(sys.argv[argi+2])
        elif arg=="--sum_area_size":
            if len(sys.argv)>argi+2:
                sum_area_size = int(sys.argv[argi+2])
        elif arg=="--show_bounding":
            if len(sys.argv)>argi+2:
                show_bounding = int(sys.argv[argi+2])
        elif arg=="--weighted_alpha":
            if len(sys.argv)>argi+2:
                weighted_alpha = float(sys.argv[argi+2])
        elif arg=="--debug_mode":
            if len(sys.argv)>argi+2:
                debug_mode = int(sys.argv[argi+2])              
        elif arg=="--help" or arg=="-h" or arg=="/?" or arg=="?":
            show_usage()
            return False
        argi += 1
    return True

def get_image_resized(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def main():
    # logging settings
    #logging_file_time = datetime.datetime.now().strftime("%H_%M_%S")
    logging_file_name = "motion_detection_frm.log" #"motion_detection_frm_"+logging_file_time+".log"
    logging_file_handler = logging.FileHandler(logging_file_name)
    logging_stdout_handler = logging.StreamHandler(sys.stdout)
    logging_handlers = [logging_file_handler, logging_stdout_handler]
    if debug_mode:
        logging.basicConfig(
            level=logging.DEBUG,    
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=logging_handlers
            #filename=FILE_NAME_LOG,
            #stream=sys.stdout
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,    
            format='[%(asctime)s] %(levelname)s %(message)s',
            handlers=logging_handlers
        )
    
    # setup
    if not setup_argv():
        return

    logging.info("motion detection started")
    show_settings()

    video_capture = None

    # test/load valid data
    # inp_prev_image
    inp_prev_image = None 
    if not inp_prev_file or not os.path.exists(inp_prev_file):
        logging.error("inp_prev_file $s does not exists", inp_prev_file)
        return
    inp_prev_image = cv2.imread(inp_prev_file)
    if inp_prev_image is None:
        logging.error("inp_prev_image is empty")
        return
    inp_prev_image_width = inp_prev_image.shape[1]
    inp_prev_image_height = inp_prev_image.shape[0]
    # inp_curr_file
    inp_curr_image = None
    if inp_curr_file and not os.path.exists(inp_curr_file):
        logging.error("inp_curr_file $s does not exists", inp_curr_file)
        return
    if inp_curr_file:
        inp_curr_image = cv2.imread(inp_curr_file)
        if inp_curr_image is None:
            logging.error("inp_curr_image is empty")
            return
    else:
        video_capture = cv2.VideoCapture(inp_curr_icam) # value (0) selects the devices default camera
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, inp_prev_image_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, inp_prev_image_height)
        if not video_capture.isOpened():
            logging.error("camera don't work or can't be accessed")
            return
        ret, inp_curr_image = video_capture.read()
        # if the frame is empty, break immediately
        if not ret:
            logging.error("inp_curr_image is empty")
            return
    inp_curr_image_width = inp_curr_image.shape[1]
    inp_curr_image_height = inp_curr_image.shape[0]

    # test for size
    if inp_prev_image_width != inp_curr_image_width or inp_prev_image_height != inp_curr_image_height:
        inp_curr_image = get_image_resized(inp_curr_image, inp_prev_image_width, inp_prev_image_height)
        logging.warning("inp_curr_image resized")

    # ready to start

    # prepare images; grayscale and blur
    prepared_curr_image = cv2.cvtColor(inp_curr_image, cv2.COLOR_BGR2GRAY)
    prepared_curr_image = cv2.GaussianBlur(prepared_curr_image, (21, 21), 0) # (5, 5)
    prepared_prev_image = cv2.cvtColor(inp_prev_image, cv2.COLOR_BGR2GRAY)
    prepared_prev_image = cv2.GaussianBlur(prepared_prev_image, (21, 21), 0) # (5, 5)
    weighted_alpha = 0.5 # background lighting range
    prepared_prev_image =  cv2.addWeighted(prepared_curr_image, weighted_alpha, prepared_prev_image, (1.0 - weighted_alpha), 0)
    
    # compute the absolute difference between the current image and the previous image
    diff_image = cv2.absdiff(src1=prepared_prev_image, src2=prepared_curr_image)

    # dilate the thresholded image to fill in holes, a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_image = cv2.dilate(diff_image, kernel, 1) # cv2.dilate(diff_image, None, iterations=2)

    # only take different areas that are different enough
    # if change in between prev and current frame is greater than (20) it will draw white color (255)
    thresh_image = cv2.threshold(src=diff_image, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1] #cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)[1]
    if debug_mode:
        #cv2.imshow("thresh_image", thresh_image)
        cv2.imwrite("thresh_image.jpg", thresh_image)

    # find and optionally draw contours
    contours, _ = cv2.findContours(image=thresh_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) #cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image=inp_curr_image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # for contour in contours:
    #     if cv2.contourArea(contour) < 50:
    #         # too small
    #         continue
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     cv2.rectangle(img=inp_curr_image, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    logging.info("len(contours): %d", len(contours))
    
    # find contour with max area
    contour_indx_cur = 0
    contour_area_max = min_area_size
    contour_area_sum = 0
    contour_indx_max = -1
    contours_detected = list()
    # loop over the contours
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        # if the contour is too small, ignore it
        if contour_area < min_area_size:
            continue
        contour_area_sum += contour_area
        if contour_area > contour_area_max:
            contour_area_max = contour_area
            contour_indx_max = contour_indx_cur
        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        contours_detected.append((x, y, w, h))
        contour_indx_cur += 1
    logging.info("len(contours_detected): %d", len(contours_detected))
    logging.info("contour_area_sum: %d", contour_area_sum)
    # draw contour with max area
    if contour_indx_max >= 0 and contour_area_sum > sum_area_size:
        (x, y, w, h) = contours_detected[contour_indx_max]
        if show_bounding:
            # draw box around contour
            cv2.rectangle(inp_curr_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (xc, yc, wc, hc) in contours_detected:
                cv2.rectangle(inp_curr_image, (xc, yc), (xc + wc, yc + hc), (0, 200, 0), 1)
        logging.info("motion detected with bounding (%d, %d, %d, %d), area size: %d", x, y, x + w, y + h, contour_area_max)
        out_mdet_file_saved = cv2.imwrite(out_mdet_file, inp_curr_image)
        if out_mdet_file_saved:
            logging.info("output image %s with motion was saved", out_mdet_file)
        else:
            logging.error("could not save image %s with motion", out_mdet_file)
    else:
        logging.info("(contour_indx_max >= 0): %d", (contour_indx_max >= 0))
        logging.info("(contour_area_sum > sum_area_size): %d",  (contour_area_sum > sum_area_size))
        logging.info("no motion found anywhere")

    # cleanup the camera and close any open windows
    if video_capture is not None:  
        video_capture.release()
    #if debug_mode:
    #    cv2.destroyAllWindows()

    logging.info("motion detection finished")

if __name__ == '__main__':
    main()
