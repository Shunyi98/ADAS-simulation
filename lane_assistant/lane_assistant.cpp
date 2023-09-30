#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

class LaneAssistant
{
    // insert your custom functions and algorithms here
public:
    LaneAssistant()
    {
    }

    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        // do stuff with data
        // send results via socket

        socket.send( tronis::SocketData( "Ego Fahrzeug Geschwindigkeit >>> " +
                                         to_string( ego_velocity_ ) ) );

        // send steering value via socket
        getSteeringInput( socket );

        // send throttle value via socket
        getThrottleInput( socket );

        // socket.send(tronis::SocketData( "Steering Input >>> " + std::to_string( steering_input )
        // ) ); socket.send( tronis::SocketData( "throttle Input >>> " +std::to_string( 0.7
        // /*throttle_input*/ ) ) );

        // double throttle_input= 0.7;
        // throttle_input = getThrottleInput();
        // socket.send( tronis::SocketData( std::to_string( steering_input ) + "," + std::to_string(
        // throttle_input ) ));

        return true;
    }

protected:
    std::string image_name_;
    std::string stringout;
    cv::Mat image_;
    tronis::LocationSub ego_location_;
    // tronis::LocationSub location_;
    tronis::OrientationSub ego_orientation_;
    tronis::ModelDataWrapper data_model;
    double ego_velocity_;

    // parameters in Aufgabe2
    Point ego_leftS, ego_leftE;     // ego left lane start and end point
    Point ego_rightS, ego_rightE;   // ego right lane start and end point
    Point directionS, directionE;   // car driving direction
    double rows = 512, cols = 720;  // original Picture size (720, 512).
    // vector<Point> left_points, right_points;

    // parameters in Aufgabe 3
    double steering_input;  // Send the steeering input value to Tronis Socket
    double steering_pc =
        0.9;                     // PID Controller:	partitial			when the car direction is
                                 // vertical to the lane, the maximum steering input value
    double steering_ic = 0.0;   // 0.1;
    double steering_dc = 0.5;  // 0.001;  // PID Controller:	differential
    double Err_steering;
    double dErr_steering;
    double iErr_steering = 0;
    double lastErr_steering = 0;

    // parameters in Aufgabe 4
    tronis::ObjectVector Objects_BBox;
    double throttle_input;  // Send the throttle input value to Tronis Socket
    double throttle_pc = 0.5;
    double throttle_dc = -0.002;
    double throttle_ic = -0.02;  // PID Controller:	intergral
    double Err_velocity;
    double lastErr_velocity = 0;
    double dErr_velocity;
    double sumErr_velocity = 0;
    double dist;

    //**************************
    // Aufgabe 2: lane detection
    vector<Vec4d> setLanes()
    {
        cv::Mat blur_img;  // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
        GaussianBlur( image_, blur_img, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

        cv::Mat gray_img;  // Convert the image to grayscale
        cvtColor( blur_img, gray_img, cv::COLOR_BGR2GRAY );

        cv::Mat binary_img;  // transfer the image to binary one
        cv::threshold( gray_img, binary_img, 120, 255, cv::THRESH_BINARY );

        cv::Mat edge_img;  // Edge detection
        Canny( binary_img, edge_img, 100, 200 );

        // Another way to get the edge image:	Sobel Derivitive()
        //	  Mat grad_x, grad_y;
        //      Mat abs_grad_x, abs_grad_y;
        //      Sobel( binary_img, grad_x, CV_16S, 1, 0, 1, 1, 0, BORDER_DEFAULT );
        //      Sobel( binary_img, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_DEFAULT );
        //      convertScaleAbs( grad_x, abs_grad_x );  // converting back to CV_8U
        //      convertScaleAbs( grad_y, abs_grad_y );
        //      addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edge_img );

        // set a polygon mask to get ROI
        cv::Mat mask = Mat::zeros( image_.size(), edge_img.type() );
        const int num = 6;
        Point points[1][num] = {Point( 0, rows ),
                                Point( 0, rows * 0.7 ),
                                Point( cols * 0.33, rows * 0.55 ),
                                Point( cols * 0.66, rows * 0.55 ),
                                Point( cols, rows * 0.7 ),
                                Point( cols, rows )};
        const Point* polygon = points[0];
        fillConvexPoly( mask, polygon, num, Scalar( 255 ) );

        cv::Mat roi_img;
        cv::bitwise_and( edge_img, mask, roi_img );
        imshow( "Canny output: Region of Interest", edge_img );

        vector<Vec4d> raw_lanes;  // will hold all the results of the detection
        HoughLinesP( roi_img, raw_lanes, 1, CV_PI / 180, 100, 25,
                     10 );  // Probabilistic Line Transform

        return raw_lanes;
    }

    void getLanes( vector<Vec4d> raw_lanes )
    {
        vector<Vec4d> left_lanes, right_lanes;
        Vec4f left_lane_function, right_lane_function;
        vector<Point> left_points, right_points;

        ego_leftS.y = 300;
        ego_rightS.y = 300;
        ego_leftE.y = 500;
        ego_rightE.y = 500;  // down 2 nodes,E are down

        double k, left_k, right_k;  // gradient
        Point left_b, right_b;

        /*for( auto lane : raw_lanes )  // divide the line set into left and right part based on the
                                      // line center point
        {
            double lane_center = ( lane[0] + lane[2] ) / 2; //x coordinates of start,end points

            if( lane_center < cols / 2 )
            {
                left_lanes.push_back( lane );
            }
            else
            {
                right_lanes.push_back( lane );
            }
        }*/
        for( auto lane : raw_lanes )
        {
            k = ( lane[1] - lane[3] ) / ( lane[0] - lane[2] );
            /* if( std::fabs( k ) < 0.1 )
             {
                 continue;
             }*/
            if( k <= 0 )  // If the slope is negative, left group.
            {
                left_lanes.push_back( lane );
            }
            else  // Otherwise, right group.
            {
                right_lanes.push_back( lane );
            }
        }

        // get the left lines
        for( auto left_lane : left_lanes )  // add all the points into a vector
        {
            left_points.push_back( Point( left_lane[0], left_lane[1] ) );
            left_points.push_back(
                Point( left_lane[2], left_lane[3] ) );  // x,y coordinates of start,end points
        }
        if( left_points.size() > 0 )  // fit a line with the method of least square
        {
            // fitLine(input vector, output line, distance type, distance parameter, radial
            // parameter, angle parameter) output (vx, vy, x, y)
            cv::fitLine( left_points, left_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );

            left_k = left_lane_function[1] / left_lane_function[0];
            left_b = Point( left_lane_function[2], left_lane_function[3] );

            ego_leftS.x = ( ego_leftS.y - left_b.y ) / left_k + left_b.x;
            ego_leftE.x = ( ego_leftE.y - left_b.y ) / left_k + left_b.x;
        }

        // get the right lines
        for( auto right_lane : right_lanes )
        {
            right_points.push_back( Point( right_lane[0], right_lane[1] ) );
            right_points.push_back( Point( right_lane[2], right_lane[3] ) );
        }
        if( right_points.size() > 0 )
        {
            cv::fitLine( right_points, right_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );

            right_k = right_lane_function[1] / right_lane_function[0];
            right_b = Point( right_lane_function[2], right_lane_function[3] );

            ego_rightS.x = ( ego_rightS.y - right_b.y ) / right_k + right_b.x;
            ego_rightE.x = ( ego_rightE.y - right_b.y ) / right_k + right_b.x;
        }

        // Aufgabe3
        directionS = ( ego_leftS + ego_rightS ) / 2;
        directionE = ( ego_leftE + ego_rightE ) / 2;
        
        // cv::Vec4d direction( directionS.x, directionS.y, directionE.x, directionE.y );
    }

    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        // do stuff

        vector<Vec4d> raw_lanes = setLanes();
        // vector<Vec4d> warning_lanes = setWarnings();
        getLanes( raw_lanes );
        // vector<Point> left_points, right_points;

        // Draw the lane lines and show results
        line( image_, ego_leftS, ego_leftE, Scalar( 0, 0, 255 ), 3, LINE_AA );
        line( image_, ego_rightS, ego_rightE, Scalar( 0, 0, 255 ), 3, LINE_AA );

        // Aufagbe3: Draw the driving direction lines and show results
        line( image_, Point( directionS.x, directionS.y ), Point( directionE.x, directionE.y ),
              Scalar( 0, 255, 0 ), 3, LINE_AA );
		//if(abs(steering_input) < 0.1)//abs( Err_steering ) < 30
  //      {
  //      line( image_, Point( directionS.x, directionS.y ), Point( directionE.x, directionE.y ),
  //            Scalar( 0, 255, 0 ), 3, LINE_AA );
  //      }
  //      else if( steering_input>0.1)

  //      {
  //          line( image_,left_k, Point( directionE.x, directionE.y ),
  //                Scalar( 0, 255, 0 ), 3, LINE_AA );
  //      }
  //      else
  //      {
  //          line( image_,right_k, Point( directionE.x, directionE.y ),
  //                Scalar( 0, 255, 0 ), 3, LINE_AA );
  //      }
    }

    //**************************
    // Aufgabe 3: Steering control

    void setSteeringInput()
    {
        double delta_time = 1.0 / 30.0;     // 1.0;
        if( directionS.x == directionE.x )  // when the car drives straight
        {
            steering_input = 0;
        }
        else
        {
            double slope =
                -( directionS.y - directionE.y ) /
                ( directionS.x - directionE.x );  // positive:	up right to down left
            
			double steering_winkel =
                M_PI_2 - abs( atan( slope ) );  // 0: vertical		pi/2:	horizontal
/*
			if(abs(steering_input)<0.1)
                            Err_steering = steering_winkel / M_PI_2 - 0;
                        else if(  steering_input > 0.1 )
                            Err_steering = abs( atan( slope ) ) - atan( left_k );
                       */
            Err_steering = steering_winkel / M_PI_2 - 0;
            dErr_steering = ( Err_steering - lastErr_steering ) / delta_time;
            // iErr_steering = Err_steering * delta_time + iErr_steering;
            lastErr_steering = Err_steering;

			//if( abs( steering_input ) < 0.1 )
            steering_input = steering_pc * abs( Err_steering ) + steering_dc * dErr_steering;//+iErr_steering*iErr_steering;
            std::cout << "steering_input= " << steering_input << std::endl;
            //else if( abs( steering_input ) > 0.1 ) //when car turning
            //                steering_input =
            //                    (steering_pc+0.2) * abs( Err_steering ) + 10*steering_dc * dErr_steering;

            //+steering_ic * iErr_steering;

            // cout << "steering_pc * abs( Err_steering )" << steering_pc * abs( Err_steering ) <<
            // endl;

            if( slope > 0 )  // drving to the right is positive
            {
                steering_input = -( steering_input );
            }
        }
        // std::cout << "Steering input = " << steering_input << std::endl;
        // 1
        // if( directionS.x == directionE.x )  // when the car drives straight
        //{
        //    steering_input = 0;
        //}
        // else
        //{
        //    double slope = -( directionS.x - directionE.x ) /
        //                   ( directionS.y - directionE.y );  // positive: up right to down left
        //    double steering_winkel = atan( slope );          // -pi/2 to pi/2
        //    double last_steering_winkel;
        //    Err_steering = steering_winkel;
        //    dErr_steering = ( steering_winkel - last_steering_winkel ) / delta_time;
        //    iErr_steering = steering_winkel * delta_time + iErr_steering;
        //    last_steering_winkel = steering_winkel;

        //    steering_input = steering_pc * steering_winkel + steering_ic * iErr_steering +
        //                     steering_dc * dErr_steering;

        //    // Optional: Check the sign of slope to determine if steering_input should be negated
        //     if (slope > 0)
        //     {
        //         steering_input = -steering_input;
        //     }
        //}
        // if( ego_leftS.x,ego_leftE.x )
        // 2
        //       if( directionS.x == directionE.x )  // when the car drives straight
        //      {
        //          steering_input = 0;
        //      }
        //       else
        //      {
        //            double slope = -( directionS.x - directionE.x ) / //S is upper point
        //                           ( directionS.y - directionE.y );  // positive:	up right to
        //                           down,left,this part shouldn't be 0,because direction is the
        //                           middle of the 2 sides,and in before the ones which k<0.1
        //                           areremoved
        //
        //
        ////double slope = -( directionS.x - directionE.x ) / ( -200.0 );  // S is upper point;

        //      double steering_winkel = atan( slope );  // 0: vertical		pi/2:	horizontal

        //      Err_steering = steering_winkel;
        //      dErr_steering = ( Err_steering - lastErr_steering );//  / delta_time;
        //      iErr_steering = Err_steering + iErr_steering;//Err_steering* delta_time +
        //      iErr_steering; lastErr_steering = Err_steering;

        //      steering_input = steering_pc * Err_steering + steering_ic * iErr_steering +
        //      steering_dc * dErr_steering; steering_input = std::min( 1.0, std::max( -1.0,
        //      std::abs( steering_input ) ) ) *
        //                       ( steering_input >= 0 ? 1 : -1 );
        // if( slope > 0 )
        //          {
        //              steering_input = -steering_input; //positive:	up right to down,turn left
        //              for "-",right for +
        //          }

        /*Err_steering = steering_winkel;
dErr_steering = ( Err_steering - lastErr_steering );
iErr_steering = Err_steering + iErr_steering;
lastErr_steering = Err_steering;*/
        

        //}
    }

    void getSteeringInput( tronis::CircularMultiQueuedSocket& socket )
    {
        setSteeringInput();
    }

    //**************************
    // Aufgabe 4: Throttle control

    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity * 3.6 * 1e-2;  // from cm/s to km/h
        return true;
    }

    bool processObject( tronis::BoxDataSub* sensorData )
    // double processObject( data_model.get_typed<tronis::BoxDataSub>() )

    {
        // do stuff
        // tronis::ModelDataWrapper data_model;
        // tronis::BoxDataSub* sensorData = data_model.get_typed<tronis::BoxDataSub>();
        tronis::LocationSub location;
        Objects_BBox = sensorData->Objects;
        // tronis::LocationSub location = tronis::ObjectSub& object->Pose.Location;
        for( size_t i = 0; i < Objects_BBox.size(); i++ )
        {
            tronis::ObjectSub& object = Objects_BBox[i];
            /*std::cout << object.ActorName.Value() << "at";
            std::cout << object.Pose.Location.ToString() << std::endl;
            std::cout << object.Type << std::endl;*/
            // cout << "dist" << dist << endl;
            /*if( object.Type == "1" )
            {*/

            tronis::LocationSub location = object.Pose.Location;
            float pos_x = location.X / 100;
            float pos_y = location.Y / 100;
            dist = sqrt( pow( pos_x, 2 ) + pow( pos_y, 2 ) );
            //cout << "dist" << dist << endl;
            //}
        }

        // return dist;
        // std::cout << object.ActorName.Value() << "at";
        // std::cout << object.Pose.Location.ToString() << std::endl;

        return true;
    }
    //   bool processObject( )
    //	{// do stuff
    //       return true;
    //}

    void setThrottleInput( double dist )
    {
        Err_velocity = 60 - ego_velocity_;  // set the max speed to 60km/h
        sumErr_velocity = lastErr_velocity + Err_velocity;
        dErr_velocity = Err_velocity - lastErr_velocity;
        lastErr_velocity = Err_velocity;

        double min_distance = 20;
        /*double throttle_input;
                throttle_input = 0.7;
        return throttle_input;*/
        /*
                        if( Objects_BBox.size() >0 )
                  {  */

        if( dist < min_distance )
        {
            if( abs( ego_velocity_ ) < 1 )  // make it absolutely stop
            {
                throttle_input = 0;
            }
            else  // urgent stop
            {
                throttle_input = -0.7;
            }
        }
        else if( dist < min_distance + 5 && ego_velocity_ > 40 ||
                 dist < min_distance + 10 && ego_velocity_ > 45 )  // keep the distance
        {
            throttle_input = -0.5;  // the environment resistance throttle would be -0.5
        }
        else
        {
            /*if( abs( ego_velocity_ ) > 50 & abs( Err_steering ) > 30 )
                throttle_input = -0.5;*/
            if( abs( ego_velocity_ ) > 50)  // to keep the cvelocity stable
            {
                throttle_input = throttle_pc * Err_velocity + throttle_dc * dErr_velocity +
                                 throttle_ic * sumErr_velocity;
                /*throttle_input = std::min( 1.0, std::max( -1.0, std::abs( throttle_input ) ) ) *
                                 ( throttle_input >= 0 ? 1 : -1 );*/
            }
            else  // make it reaccelerate faster after deaccelerating
            {
                throttle_input = 0.8;
            }
        }

     //        }

        // else
        //        {
        //            throttle_input = 0.7;
        //            if( abs( ego_velocity_ ) > 60 )  // to keep the cvelocity stable
        //            {
        //                throttle_input = throttle_pc * Err_velocity + throttle_dc * dErr_velocity
        //                +
        //                                 throttle_ic * sumErr_velocity;
        //            }
        //        }
    }
    void getThrottleInput( tronis::CircularMultiQueuedSocket& socket )
    {
        string prefix_throttle = "Throttle value >>> ";
        // setThrottleInput( );

        // Send throttle_input value via socket
        // socket.send( tronis::SocketData( prefix_throttle + std::to_string( throttle_input ) ) );

        // tronis::ModelDataWrapper data_model;
        // tronis::BoxDataSub* sensorData = data_model->GetModelType();
        // tronis::BoxDataSub* sensorData = data_model->GetDataTypeId();
        // tronis::BoxDataSub* sensorData = data_model.get_typed<tronis::BoxDataSub>();
        // Objects_BBox = sensorData->Objects;

        if( Objects_BBox.size() )  // 1.Situation with front car
        {
            for( size_t i = 0; i < Objects_BBox.size(); i++ )
            {
                tronis::ObjectSub& object = Objects_BBox[i];
                /*std::cout << object.ActorName.Value() << "at";
                std::cout << object.Pose.Location.ToString() << std::endl;*/

                if( object.ActorName.Value() == "Generic_Hatchback_2" )
                //if( object.Type == "1" )
                {
                    setThrottleInput( dist );
                    cout << "dist" << dist << endl;
                    std::cout << "throttle_input = " << throttle_input << std::endl;
                    cout << "the car ahead detected!" << endl;
                    cout << endl;
                    // double dist = processDistance( object.Pose.Location );
                    // double dist = processObject( tronis::BoxDataSub * sensorData );
                    // socket.send(
                    //   tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
                }
                else
                {
                    setThrottleInput( 100.0 );
                    cout << "lane or track got detected, please ignore !" << endl;
                    // socket.send(
                    //   tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
                }
            }
        }
        else  // 2.Situation without front car
        {
            //	set the initial throttle input in case there are no objects in the front
            setThrottleInput( 100.0 );
            // throttle_input = 0.7;
            cout << "no objects in the front !" << endl;
            /*cout << "directionS.y - directionE.y!" << directionS.y << "directionS.y -
               directionE.y!"
                 << directionE.y << endl;*/
            std::cout << "throttle_input = " << throttle_input << std::endl;
            // socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_input ) ) );
        }

        std::string stringout =
            std::to_string( steering_input ) + ";" + std::to_string( throttle_input );
        socket.send( tronis::SocketData( stringout ) );
    }

    // Helper functions, no changes needed
public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    break;
                }
                case tronis::TronisDataType::BoxData:  // Object:
                {
                    processObject( data_model.get_typed<tronis::BoxDataSub>() );
                    // tronis::/*tronis::BoxDataSub * sensorData*/ );
                    break;
                }
                default:
                {
                    std::cout << data_model->ToString() << std::endl;
                    break;
                }
            }
            return true;
        }
        else
        {
            std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    // Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat image )
    {
        cv::Mat out = image;
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
        cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
        cv::imshow( image_name.c_str(), out );
    }

    // Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image )
    {
        std::cout << "processImage" << std::endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );

        detectLanes();
        showImage( image_name_, image_ );

        return true;
    }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;

    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
