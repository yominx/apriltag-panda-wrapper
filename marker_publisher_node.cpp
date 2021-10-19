// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <fstream>
#include <ostream>
#include <cstdlib>

#include <Eigen/Dense>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API

#include "apriltag.hpp"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>

#define MATRIX_PATH "/home/pandacom/catkin_ws/src/BASE_TO_CAM_Matrix.txt"

using namespace std; 
using namespace Eigen;
using namespace rs2;
using namespace cv;

apriltag_family_t *init_family(const char *famname);

void publish_position(ros::Publisher pub, geometry_msgs::PoseStamped *grasp_point, Matrix4d *T_O_GRASP, int frame_id);
void get_opt_option(apriltag_detector_t *td, getopt_t *getopt);
void draw_marker(Mat *frame,apriltag_detection_t *det);
void matrix_parser(Matrix4d *T_O_EE);
void make_marker_matrix(apriltag_pose_t *pose, Matrix4d *T_CAM_MARKER);
void make_grasp_pose(Matrix4d *T_O_MARKER, Matrix4d *T_O_GRASP);
void getopt_addition(getopt_t *getopt);

ros::Publisher grasp_pos;
ros::Publisher marker_pos;

int main(int argc, char * argv[]) try
{
    ros::init(argc, argv, "data_integation");
    ros::NodeHandle n;
    marker_pos = n.advertise<geometry_msgs::PoseStamped>("/marker_point", 10);
    grasp_pos = n.advertise<geometry_msgs::PoseStamped>("/grasp_point", 10);
    geometry_msgs::PoseStamped marker_point;
    geometry_msgs::PoseStamped grasp_point;
    ros::Rate loop_rate(100);

    /* Load transformation matrix from Base to Camera*/
    Matrix4d T_O_CAM, T_CAM_MARKER, T_MARKER_GRASP, T_O_GRASP, T_O_MARKER;
    matrix_parser(&T_O_CAM);
    T_MARKER_GRASP <<   1,  0,  0, 0,
                        0,  1,  0,-0.14,
                        0,  0,  1, 0.0274,
                        0,  0,  0, 1;
 
    /* Initialize realsense library */
    auto config = rs2::config();
    config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_BGR8, 30);
     
    rs2::colorizer color_map;
    rs2::pipeline pipe;

    auto selection      = pipe.start();
    auto color_stream   = selection.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    auto resolution     = std::make_pair(color_stream.width(), color_stream.height());
    auto int_params     = color_stream.get_intrinsics();

    
    /* April tag initialization part */
    getopt_t *getopt = getopt_create();
    getopt_addition(getopt);

    if (!getopt_parse(getopt, argc, argv, 1)  ||  getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    /* Initialize tag detector with options */
    const char *famname = getopt_get_string(getopt, "family");
    apriltag_family_t *tf = init_family(famname);
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    get_opt_option(td, getopt);

    /* Set intrinsic parameters */
    apriltag_detection_info_t info;
    const double tag_size = 0.0428;
    info.tagsize = tag_size;
    info.fx = int_params.fx;
    info.fy = int_params.fy;
    info.cx = int_params.ppx;
    info.cy = int_params.ppy;



    // cout << T_O_CAM << endl;
    // Fetch and Detect

    apriltag_pose_t pose;
    Mat gray, frame;

    while (ros::ok()) // To capture the picture, press 'c'
    {
        rs2::frameset data = pipe.wait_for_frames();
        rs2::frame   color = data.get_color_frame();
        const int w = color.as<rs2::video_frame>().get_width();
        const int h = color.as<rs2::video_frame>().get_height();
        Mat image(Size(w, h), CV_8UC3, (void*)color.get_data(), Mat::AUTO_STEP);

        cvtColor(image, frame, COLOR_RGB2BGR);
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

            make_marker_matrix(&pose, &T_CAM_MARKER);
            T_O_MARKER = T_O_CAM * T_CAM_MARKER;
            make_grasp_pose(&T_O_MARKER, &T_O_GRASP);
            

            if(det->id == 100) continue;
            cout << "Tag Number: " << det->id << endl;
            cout << "Matrix: " << endl <<  T_O_GRASP << endl;
            publish_position(marker_pos, &marker_point, &T_O_MARKER, det->id);
            publish_position(grasp_pos, &grasp_point, &T_O_GRASP, det->id);
            // marker_pose
            // cout << "Matrix: " << T_CAM_MARKER * T_MARKER_GRASP << endl;
            // print_pose(&pose);
        }



        imshow("Tag Detections", frame);
        if(waitKey(10) != -1) break;

        apriltag_detections_destroy(detections);
        // Update the window with new data
        // imshow(window_name, image);
        ros::spinOnce();
        loop_rate.sleep();
    }



    /* Transformation matrix */


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


void matrix_parser(Matrix4d *T_O_CAM){

    ifstream matrix(MATRIX_PATH);

    char *ptr;
    int i=0, j=0;
    for (string line; getline(matrix, line); i++, j=0) // 4 times (row)
    {
        cout << "Loading matrix..." << endl;
        char row_vector[256];
        strcpy(row_vector, line.c_str());

        for (ptr = strtok(row_vector, " "); ptr != NULL; ptr = strtok(NULL, " "), j++) {   
            // cout << i << j << endl;
            // cout << ptr << endl;
            (*T_O_CAM)(i,j) = atof(ptr);
        }
    }
    // 
    cout << (*T_O_CAM) << endl;

}


void make_marker_matrix(apriltag_pose_t *pose, Matrix4d *T_CAM_MARKER){
    (*T_CAM_MARKER) = Matrix4d::Identity(4,4);
    for (int i = 0; i < 3; i++){
       for (int j = 0; j < 3; j++){
            (*T_CAM_MARKER)(i,j) = pose->R->data[i*3+j];
       } 
    }
    
    for (int i = 0; i < 3; i++){
        (*T_CAM_MARKER)(i, 3) = pose->t->data[i];
    }
}

void make_grasp_pose(Matrix4d *T_O_MARKER, Matrix4d *T_O_GRASP)
{
    Vector3d x, y, z;
    Vector4d p, pm;
    Matrix4d T;
    pm << 0, -0.14, 0.0274, 1;
    z << 0, 0, -1;
    x << (*T_O_MARKER).coeff(0,1), (*T_O_MARKER).coeff(1,1), (*T_O_MARKER).coeff(2,1);
    y = z.cross(x);
    p = *T_O_MARKER * pm;
    *T_O_GRASP << x.coeff(0), y.coeff(0), z.coeff(0), p.coeff(0),
                x.coeff(1), y.coeff(1), z.coeff(1), p.coeff(1),
                x.coeff(2), y.coeff(2), z.coeff(2), p.coeff(2),
                0,          0,          0,          1;
}

void publish_position(ros::Publisher pub, geometry_msgs::PoseStamped *pose, Matrix4d *T_O_GRASP, int frame_id){
    pose->header.stamp = ros::Time();
    pose->header.frame_id = std::to_string(frame_id);
    pose->pose.position.x = (*T_O_GRASP)(0,3);
    pose->pose.position.y = (*T_O_GRASP)(1,3);
    pose->pose.position.z = (*T_O_GRASP)(2,3);

    Matrix3d ROTATION_MAT = (*T_O_GRASP).block(0,0,3,3);
    Quaterniond q(ROTATION_MAT);

    pose->pose.orientation.x = q.x();
    pose->pose.orientation.y = q.y();
    pose->pose.orientation.z = q.z();
    pose->pose.orientation.w = q.w();

    pub.publish(*pose);
}