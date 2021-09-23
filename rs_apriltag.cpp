// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <ostream>
#include <Eigen/Dense>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include "apriltag.hpp" 

using namespace std; 
using namespace Eigen;
using namespace rs2;
using namespace cv;


void getopt_addition(getopt_t *getopt);

apriltag_family_t *init_family(const char *famname);

void get_opt_option(apriltag_detector_t *td, getopt_t *getopt);

void draw_marker(Mat *frame,apriltag_detection_t *det);

void print_pose(apriltag_pose_t *pose){
    //for R
    int r = pose->R->nrows, c=pose->R->ncols;
    for (int i = 0; i < r; i++){
       for (int j = 0; j < c; j++){
            // cout << i*c+j << endl;
            cout << pose->R->data[i*c+j] << " ";
       } 
       cout << endl;
    }
    cout << endl;
    
    //for t
    r = pose->t->nrows, c=pose->t->ncols;
    for (int i = 0; i < r; i++){
       for (int j = 0; j < c; j++){
            cout << pose->t->data[i*c+j] << " ";
       } 
       cout << endl;
    }
    cout << endl;

}

int main(int argc, char * argv[]) try
{
    // Init realsense library 


    auto config = rs2::config();
    config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_BGR8, 30);
     

    rs2::colorizer color_map;
    rs2::pipeline pipe;


    auto selection      = pipe.start();
    auto color_stream   = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto resolution     = std::make_pair(color_stream.width(), color_stream.height());
    auto int_params     = color_stream.get_intrinsics();

    
    // April tag initialization part
    getopt_t *getopt = getopt_create();
    getopt_addition(getopt);

    if (!getopt_parse(getopt, argc, argv, 1)  ||  getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    // Initialize tag detector with options
    const char *famname = getopt_get_string(getopt, "family");
    apriltag_family_t *tf = init_family(famname);
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    get_opt_option(td, getopt);

    apriltag_detection_info_t info;
    const double tag_size = 0.0428;
    info.tagsize = tag_size;
    info.fx = int_params.fx;
    info.fy = int_params.fy;
    info.cx = int_params.ppx;
    info.cy = int_params.ppy;


    // Fetch and Detect

    apriltag_pose_t pose;
    cout << "To capture the picture, press 'c' " << endl;
    Mat gray, frame;
    while (1) // To capture the picture, press 'c'
    {
        rs2::frameset data = pipe.wait_for_frames();
        rs2::frame   color = data.get_color_frame();
        const int w = color.as<rs2::video_frame>().get_width();
        const int h = color.as<rs2::video_frame>().get_height();
        Mat image(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

        cvtColor(image, frame, CV_RGB2BGR);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = { .width = gray.cols,
            .height = gray.rows,
            .stride = gray.cols,
            .buf = gray.data
        };

        zarray_t *detections = apriltag_detector_detect(td, &im);

        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            draw_marker(&frame, det);
            info.det = det;
            double err = estimate_tag_pose(&info, &pose);
            cout << "Tag Number: " << det->id << endl;
            print_pose(&pose);
        }

        imshow("Tag Detections", frame);
        if (waitKey(1) == 'c'){
            if (zarray_size(detections) == 1) break;
            else cout << "There shoule be only one marker in the frame." << endl;
        }

        apriltag_detections_destroy(detections);
        // Update the window with new data
        // imshow(window_name, image);
    }



    /* Transformation matrix */

    MatrixXd SO3(3,3), T_O_EE(4,4), T_EE_MARK(4,4),
             T_MARK_CAM_rot(4,4), T_MARK_CAM_trans(4,4),  T_O_CAM(4,4);
                     
    SO3                = MatrixXd::Identity(3,3);
    T_O_EE             = MatrixXd::Identity(4,4);
    T_EE_MARK          = MatrixXd::Identity(4,4);
    T_MARK_CAM_rot     = MatrixXd::Identity(4,4);
    T_MARK_CAM_trans   = MatrixXd::Identity(4,4);
    
    double EE_MARK_trans[4] = {-0.05, 0, 0.0085, 1}; // t = [-0.05, 0, 0.0085]', no rotation. 
    T_EE_MARK.col(3) = Map<Vector4d>(EE_MARK_trans);

    for (int i = 0; i < 3; i++){
       for (int j = 0; j < 3; j++){
            SO3(i,j) = pose.R->data[3*i+j];
       } 
    }
    T_MARK_CAM_rot.block(0,0,3,3) = SO3.inverse();

    double MARK_CAM_trans[4]; 
    MARK_CAM_trans[0] = -pose.t->data[0];
    MARK_CAM_trans[1] = -pose.t->data[1];
    MARK_CAM_trans[2] = -pose.t->data[2];
    MARK_CAM_trans[3] = 1;
    T_MARK_CAM_trans.col(3) = Map<Vector4d>(MARK_CAM_trans);



    T_O_CAM = T_O_EE * T_EE_MARK * T_MARK_CAM_rot * T_MARK_CAM_trans;
    cout << "Base to EE" << endl << T_O_EE << endl << endl;
    cout << "EE to marker" << endl << T_EE_MARK << endl << endl;
    cout << "marker to cam rotation" << endl << T_MARK_CAM_rot << endl << endl;
    cout << "marker to cam translation" << endl << T_MARK_CAM_trans << endl << endl;

    cout << endl << endl << "Final Result :" << endl << T_O_CAM << endl;


    /* Blocker */
    waitKey();


    /* destroy apriltag detector */
    apriltag_detector_destroy(td);
    if (!strcmp(famname, "tag36h11")) {
        tag36h11_destroy(tf);
    } else if (!strcmp(famname, "tag25h9")) {
        tag25h9_destroy(tf);
    } else if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tagCircle21h7_destroy(tf);
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tagCircle49h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tagStandard41h12_destroy(tf);
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tagStandard52h13_destroy(tf);
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tagCustom48h12_destroy(tf);
    }
    getopt_destroy(getopt);


    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}



void getopt_addition(getopt_t *getopt){
    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");

}

apriltag_family_t *init_family(const char *famname){
    apriltag_family_t *tf;
    if (!strcmp(famname, "tag36h11")) {
        tf = tag36h11_create();
    } else if (!strcmp(famname, "tag25h9")) {
        tf = tag25h9_create();
    } else if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else if (!strcmp(famname, "tagCircle21h7")) {
        tf = tagCircle21h7_create();
    } else if (!strcmp(famname, "tagCircle49h12")) {
        tf = tagCircle49h12_create();
    } else if (!strcmp(famname, "tagStandard41h12")) {
        tf = tagStandard41h12_create();
    } else if (!strcmp(famname, "tagStandard52h13")) {
        tf = tagStandard52h13_create();
    } else if (!strcmp(famname, "tagCustom48h12")) {
        tf = tagCustom48h12_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
        exit(-1);
    }
    return tf;
}

void get_opt_option(apriltag_detector_t *td, getopt_t *getopt){
    td->quad_decimate   = getopt_get_double(getopt, "decimate");
    td->quad_sigma      = getopt_get_double(getopt, "blur");
    td->nthreads        = getopt_get_int(getopt, "threads");
    td->debug           = getopt_get_bool(getopt, "debug");
    td->refine_edges    = getopt_get_bool(getopt, "refine-edges");
}



void draw_marker(Mat *frame,apriltag_detection_t *det){

    line(*frame, Point(det->p[0][0], det->p[0][1]),
             Point(det->p[1][0], det->p[1][1]),
             Scalar(0, 0xff, 0), 2);
    line(*frame, Point(det->p[0][0], det->p[0][1]),
             Point(det->p[3][0], det->p[3][1]),
             Scalar(0, 0, 0xff), 2);
    line(*frame, Point(det->p[1][0], det->p[1][1]),
             Point(det->p[2][0], det->p[2][1]),
             Scalar(0xff, 0, 0), 2);
    line(*frame, Point(det->p[2][0], det->p[2][1]),
             Point(det->p[3][0], det->p[3][1]),
             Scalar(0xff, 0, 0), 2);

    stringstream ss;
    ss << det->id;
    String text = ss.str();
    int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontscale = 1.0;
    int baseline;
    Size textsize = getTextSize(text, fontface, fontscale, 2,
                                    &baseline);
    putText(*frame, text, Point(det->c[0]-textsize.width/2,
                               det->c[1]+textsize.height/2),
            fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
}