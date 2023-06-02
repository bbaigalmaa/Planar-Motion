//
//  main.cpp
//  MotionFlow
//
//  Created by Baigalmaa on 2023.03.16.
//

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/optflow.hpp>

#include <cstdlib>

#include <ostream>

using namespace cv;
using namespace std;
float threshhold = 0.0;

//get2randomPoints() - generating 2 random points without duplication
vector<Point2f> get2randomPoints(Mat image);

//Checking given points is in selected random points
bool pointCheck(Point2f point);

Point3f calCoefficientsIm(Point2f point1, Point2f point2);

Point3f intersectOnImEqu(Point3f coeff1, Point3f coeff2);

//Storing selected random points
vector<Point2f> selectedRandomPoints;

vector<tuple<Point2f, Point2f>> frameMovement;
vector<tuple<Point2f, Point2f>> frameMovement2;

tuple<Point2f, Point2f> findPoint(Point2f p);
tuple<Point2f, Point2f> readPoints(string readData);


int samplePointScale = 300;
Point2f bestIntersectPoint;

bool isBestInlier(Point2f point, vector<tuple<Point2f,Point2f>> bestInliers);

vector<tuple<Point2f, Point2f>> get2randomPointsFromVector();

vector<tuple<Point2f,Point2f>> findBestInliers(Point2f intersectPoint, vector<tuple<Point2f, Point2f>> frameMov, vector<tuple<Point2f,Point2f>> bestInliers, float threshhold);

vector<tuple<Point2f,Point2f>> bestInliersIm;

int main(int argc, const char * argv[]) {
  //  cv::Mat image1RGB = cv::imread(argv[1]);
    
    
    //Variable for distance between pixels, used for visualizing solution on image
    
    
    int numberOfImageframe = 23;
    
    string imageFolder = argv[1];
    string imgExtension = ".jpg";
    
    cv::Mat image1RGB = cv::imread(imageFolder + to_string(1) + imgExtension);
    cv::Mat imageFirst = cv::imread(imageFolder + to_string(1) + imgExtension, IMREAD_GRAYSCALE);
    threshhold = (float)imageFirst.cols/4;
    
    for (int i=0; i < imageFirst.rows; i++){
        for(int j=0; j < imageFirst.cols; j++){
            frameMovement.push_back( tuple<Point2f, Point2f>(Point2f(i,j), Point2f(i,j)));
        }
    }
    
    
    //Open text file to write points
    ofstream outputfile;
    ifstream bestInliers;
    ifstream bestIntersection;

    bestInliers.open(imageFolder + "img_best_inliers_" + to_string(numberOfImageframe) + "_frames_"  + to_string(threshhold)+ "_threshold.txt");
    bestIntersection.open(imageFolder + "img_best_intersection_" + to_string(numberOfImageframe) + "_frames_"  + to_string(threshhold)+ "_threshold.txt");
    
    string bestInlierValue = "";
    string bestIntersec;
    if(bestInliers.is_open() && bestIntersection.is_open()){
        while (getline (bestInliers, bestInlierValue)) {
          // Output the text from the file
            tuple<Point2f, Point2f> data = readPoints(bestInlierValue);
            bestInliersIm.push_back(data);
        }
        
        vector<float> t;
        while(getline(bestIntersection, bestIntersec, ' ')){
            t.emplace_back(stof(bestIntersec));
        }
            
        Point2f intersect = Point2f(t[0], t[1]);
        bestIntersectPoint = intersect;
        
    }
    else {
        cout << "best inliers are not created."<< endl;
    }
   
    
    
    for(int i=1; i<=numberOfImageframe; i++){
        
        cv::Mat image1, image2;
        image1 = cv::imread(imageFolder + to_string(i) + imgExtension, IMREAD_GRAYSCALE);
        
        int nextFrame = i+1;
       
        if(nextFrame <= numberOfImageframe){
            image2 = cv::imread(imageFolder + to_string(nextFrame) + imgExtension, IMREAD_GRAYSCALE);
            
        }
        else
            break;
       
       
        cv::Mat outputFarneback;// = cv::Mat(image1.size(), CV_32FC2);
        
        if(!image1.empty() && !image2.empty() ){
            
            cv::calcOpticalFlowFarneback(image1, image2, outputFarneback, 0.5, 1, 5, 13, 5, 1.5, 0);
            //cv::calcOpticalFlowFarneback(image1, image2, outputFarneback, 0.5, 1, 5, 13, 7, 1.5, 0);// -- slow motion
            
            //extract x and y channels
            cv::Mat xy[2];
            cv::split(outputFarneback, xy);
            
            Mat magnitude, angle;
            
            //getting angle and magnitude seperately
            cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);
            
            for( int j=0; j<frameMovement.size(); j++){
                Point2f p1  = get<0>(frameMovement[j]);
                Point2f p  = get<1>(frameMovement[j]);
                
                if(p.x >= 0 && p.y >= 0){
                    
                    
                    float mag = magnitude.at<float>(round(p.x), round(p.y));
                    
                    
                    float ang = angle.at<float>(round(p.x), round(p.y));
                    
                    
                    
                    Point2f pDest = Point2f(p.x + mag*cos(ang * CV_PI / 180.0),p.y +  mag*sin(ang * CV_PI / 180.0));
                    
                  
                    
                    tuple<Point2f, Point2f> mapping = tuple<Point2f, Point2f>(p1, pDest);
                
                
                    
                    frameMovement[j].swap(mapping);
                    
                }
            }
        }
    }
    
    
    /*
    for(int i=0; i<frameMovement.size(); i+=50){
        Point2f p1 = get<0>(frameMovement[i]);
        Point2f p2 =  get<1>(frameMovement[i]);
        
        
       // cout << p1 << endl;
        //cout << p2 << endl;
       // arrowedLine( image1RGB, p1, p2, CV_RGB(255,0,0), 1, CV_AVX, 0, 0.3 );
        arrowedLine( image1RGB, Point2f(p1.y, p1.x), Point2f(p2.y, p2.x), CV_RGB(255,0,0), 1, CV_AVX, 0, 0.3 );
        
       // if(p2.x >= 0 && p2.y >= 0 )
        //    frameMovement2.push_back(frameMovement[i]);
    }
    
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    imshow("Display window", image1RGB);*/
    
    // outputfile.open(imageFolder + "output" + to_string(i) + ".txt");
     /*outputfile.open(imageFolder + "output-" + to_string(numberOfImageframe) + "-frames.txt");
     
     for(int k=0; k<frameMovement.size(); k++){
         Point2f p1 = get<0>(frameMovement[k]);
         Point2f p2 =  get<1>(frameMovement[k]);
         
         outputfile << p1 << ", " << p2 << endl;
     }
     
     outputfile.close();
      
      */
    
    
    

    //Ransac iteration variables
    int iterationNumber = 0;
    int maximumIterationNumber = 100;//1000;
    
    srand((unsigned int)time(NULL));
    //Ransac starting from here
    while (iterationNumber++ < maximumIterationNumber)
    {
        //getting 2 random points on image
       // vector<Point2f> randomPoints = get2randomPoints(imageFirst);
        
        
       // if(get<1>(findPoint(randomPoints.at(0))).x == 0 && get<1>(findPoint(randomPoints.at(0))).y == 0 && //get<1>(findPoint(randomPoints.at(1))).x == 0 && get<1>(findPoint(randomPoints.at(1))).y == 0)
         ///   randomPoints = get2randomPoints(imageFirst);
        
        
        vector<tuple<Point2f, Point2f>> randomPoints  = get2randomPointsFromVector();
        
        cout << get<0>(randomPoints.at(0)) << " " << get<1>(randomPoints.at(0))  << endl;
        cout << get<0>(randomPoints.at(1)) << " " << get<1>(randomPoints.at(1))  << endl;
        
        //Implicit coefficients of 2 lines
        Point3f imCoefficientsOfRandPoint1 = calCoefficientsIm(get<0>(randomPoints.at(0)), get<1>(randomPoints.at(0)));
        
       // cout << "random point 1: " << get<0>(findPoint(randomPoints.at(0))) << ", " << get<1>(findPoint(randomPoints.at(0))) << endl;
        Point3f imCoefficientsOfRandPoint2 = calCoefficientsIm(get<0>(randomPoints.at(1)), get<1>(randomPoints.at(1)));
        
      //  cout << "random point 2: " << get<0>(findPoint(randomPoints.at(1))) << ", " << get<1>(findPoint(randomPoints.at(1))) << endl;
        //Line intersection on implicit equation (ax+by+c=0)
        Point3f homIntersect = intersectOnImEqu(imCoefficientsOfRandPoint1, imCoefficientsOfRandPoint2);
        //cout << "intersection point: " << homIntersect.x/homIntersect.z << " " << homIntersect.y/homIntersect.z << endl;
        
        //checking intersection point is inside image frame
        while(homIntersect.x/homIntersect.z > imageFirst.rows || homIntersect.x/homIntersect.z < 0  ||  homIntersect.y/homIntersect.z > imageFirst.cols || homIntersect.y/homIntersect.z < 0){
           // cout << "calculating intersect again" << endl;
            randomPoints  = get2randomPointsFromVector();
            imCoefficientsOfRandPoint1 = calCoefficientsIm(get<0>(randomPoints.at(0)), get<1>(randomPoints.at(0)));
            imCoefficientsOfRandPoint2 = calCoefficientsIm(get<0>(randomPoints.at(1)), get<1>(randomPoints.at(1)));
            homIntersect = intersectOnImEqu(imCoefficientsOfRandPoint1, imCoefficientsOfRandPoint2);
        }
        
       // cout << "intersection point: " << homIntersect.x/homIntersect.z << " " << homIntersect.y/homIntersect.z << endl;
       
        //arrowedLine( image1RGB, get<0>(randomPoints.at(0)),get<1>(randomPoints.at(0)), CV_RGB(0,0,255), 1, CV_AVX, 0, 0.3 );
        
        //arrowedLine( image1RGB, get<0>(randomPoints.at(1)),get<1>(randomPoints.at(1)), CV_RGB(0,0,255), 1, CV_AVX, 0, 0.3 );
        
        //finding inliers with implicit equation intersection
        if(homIntersect.z != 0)
            bestInliersIm = findBestInliers(Point2f(homIntersect.x/homIntersect.z, homIntersect.y/homIntersect.z), frameMovement, bestInliersIm, threshhold);
        
        //cout << "Best intersection points: " << bestIntersectPoint << endl;
        //circle( image1RGB, Point2f(homIntersect.x/homIntersect.z, homIntersect.y/homIntersect.z), 10, CV_RGB(0,0,255), FILLED);
        
            
    }
    
    outputfile.open(imageFolder + "img_best_inliers_" + to_string(numberOfImageframe) + "_frames_"  + to_string(threshhold)+ "_threshold.txt");
    
    
    
    
    for( int i=samplePointScale; i<frameMovement.size(); i+=samplePointScale){
        Point2f p  = get<0>(frameMovement[i]);
        Point2f p1  = get<1>(frameMovement[i]);

        
        if(isBestInlier(p, bestInliersIm)){
            arrowedLine( image1RGB, Point2f(p.y, p.x),Point2f(p1.y, p1.x), CV_RGB(0,255,0), 1, CV_AVX, 0, 0.3 );
            outputfile << p.x << " " << p.y << " " << p1.x << " " << p1.y << endl;
           // circle( image1RGB, Point2f(p.y, p.x), 2, CV_RGB(0,255,0), FILLED);
        } else
            arrowedLine( image1RGB, Point2f(p.y, p.x),Point2f(p1.y, p1.x), CV_RGB(255,0,0), 1, CV_AVX, 0, 0.3 );
            //circle( image1RGB, Point2f(p.y, p.x), 1, CV_RGB(255,0,0), FILLED);
        //cout << p1 << ", " << p << endl;
        //arrowedLine( image1RGB, Point2f(p.y, p.x),Point2f(p1.y, p1.x), CV_RGB(255,0,0), 1, CV_AVX, 0, 0.3 );
        //circle( image1RGB, p1, 2, CV_RGB(0,255,0), FILLED);
        //circle( image1RGB, p, 2, CV_RGB(255,0,0), FILLED);
    }
    outputfile.close();
    
    outputfile.open(imageFolder +  "img_best_intersection_" + to_string(numberOfImageframe) + "_frames_"  + to_string(threshhold)+ "_threshold.txt");
    cout << bestIntersectPoint << endl;
    circle( image1RGB, Point2f(bestIntersectPoint.y, bestIntersectPoint.x), 10, CV_RGB(255, 255, 0), FILLED);
    outputfile << bestIntersectPoint.x << " " << bestIntersectPoint.y << endl;
   // outputfile << bestInliersIm.size() << endl;
    
    outputfile.close();
//
    
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    imshow("Display window", image1RGB);
    
    //Writing result image to file
    imwrite(imageFolder + "img_" + to_string(numberOfImageframe) + "_frames_"  + to_string(threshhold)+ "_threshold_"+to_string(maximumIterationNumber) + "_iteration.jpg", image1RGB);
    
   
    
    cv::waitKey(0);
    return 0;
}

tuple<Point2f, Point2f> readPoints(string readData){
    vector<float> result;
    
    stringstream str(readData);
    string point;
    while(str >> point){
        result.emplace_back(stof(point));
    }
    Point2f p1 = Point2f(result[0], result[1]);
    Point2f p2 = Point2f(result[2], result[3]);
    
    return tuple<Point2f, Point2f>(p1, p2);
    
    
}

bool isBestInlier(Point2f point, vector<tuple<Point2f,Point2f>> bestInliers) {
    
    for(int i = 0; i<bestInliers.size() ; i++){
        Point2f bestInlier = get<0>(bestInliers[i]);
        
        if(point.x == bestInlier.x && point.y == bestInlier.y )
            return true;
    }
    
    return false;
}

vector<tuple<Point2f,Point2f>> findBestInliers(Point2f intersectPoint, vector<tuple<Point2f, Point2f>> frameMov, vector<tuple<Point2f,Point2f>> bestInliers, float threshhold){

    float intersecx = intersectPoint.x;
    float intersecy = intersectPoint.y;
    vector<tuple<Point2f,Point2f>> inliers;
    
   
    
    //Distance between 2 points through line and point(intersecting point)
    for (int i=0; i < frameMov.size(); i++){
       
            Point2f p = get<0>(frameMov[i]);
            Point2f pDest = get<1>(frameMov[i]);
            Point3f coeff = calCoefficientsIm(p, pDest);
            
            
            // Distance(line, point) = | a * x + b * y + c | / sqrt(a * a + b * b)
            float distance = abs(coeff.x * intersecx + coeff.y * intersecy + coeff.z)/sqrt(coeff.x*coeff.x + coeff.y*coeff.y);
            
            if (distance < threshhold)
                inliers.emplace_back(frameMov[i]);
    }
    
    
    if (inliers.size() > bestInliers.size()){
        bestInliers.swap(inliers);
        bestIntersectPoint = intersectPoint;
        inliers.clear();
        inliers.resize(0);
    }
    
    return bestInliers;
    
}


Point3f intersectOnImEqu(Point3f coeff1, Point3f coeff2){
    float a1,a2,b1,b2,c1,c2;
    
    a1 = coeff1.x;
    b1 = coeff1.y;
    c1 = coeff1.z;
    //c1 = 1;
    
    a2 = coeff2.x;
    b2 = coeff2.y;
    c2 = coeff2.z;
    //c2 = 1;
    
    return Point3f(b1*c2-b2*c1, a2*c1-a1*c2, a1*b2-a2*b1);
    
    
}

Point3f calCoefficientsIm(Point2f point1, Point2f point2){
    float a,b,c;
    Point2d v = point2 - point1;
    v = v / cv::norm(v);
    
    Point2d n;
    n.x = -v.y;
    n.y = v.x;
    
    a = n.x;
    b = n.y;
    c = -(a * point1.x + b * point1.y);
    
    return Point3f(a,b,c);
}

tuple<Point2f, Point2f> findPoint(Point2f p) {
    Point2f initial = Point2f(0,0);
    tuple<Point2f, Point2f> result = tuple<Point2f, Point2f>(initial, initial);
    for( int i=0; i<frameMovement.size(); i++){
        
        Point2f p1  = get<0>(frameMovement[i]);
        Point2f p2  = get<1>(frameMovement[i]);
        
        if(p1.x == p.x && p1.y == p.y){
            result = tuple<Point2f, Point2f>(p1,p2);
            return result;
        }
            
    }
    return result;
}

bool pointCheck(Point2f point){
    for (int i=0; i<selectedRandomPoints.size(); i++ ){
        if(selectedRandomPoints.at(i).x == point.x && selectedRandomPoints.at(i).y == point.y)
            return false;
    }
    
    return true;
}


vector<Point2f> get2randomPoints(Mat image) {
    vector<Point2f> result;
    
    //selectedRandomPoints
    while(result.size() <= 2)
    {
    
        int randomX1 = round((image.rows - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
        int randomY1 = round((image.cols - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

        if (pointCheck(Point2f(randomX1, randomY1))){
            result.push_back(Point2f(randomX1, randomY1));
            selectedRandomPoints.push_back(Point2f(randomX1, randomY1));
            
            int randomX2 = round((image.rows - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
            int randomY2 = round((image.cols - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
            
            if (pointCheck(Point2f(randomX2, randomX2))){
                if (randomX1 == randomX2 && randomY1 == randomY2){
                    randomX2 = round((image.rows - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
                    randomY2 = round((image.cols - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
                    
                    if (pointCheck(Point2f(randomX2, randomX2))){
                        result.push_back(Point2f(randomX2, randomY2));
                        selectedRandomPoints.push_back(Point2f(randomX2, randomY2));
                    }
                    
                }else {
                    result.push_back(Point2f(randomX2, randomY2));
                    selectedRandomPoints.push_back(Point2f(randomX2, randomY2));
                }
            }
            
        }
    }
    
    return result;
    
}

vector<tuple<Point2f, Point2f>> get2randomPointsFromVector() {
    vector<tuple<Point2f, Point2f>> ret;
    int min = 0;
    auto max = frameMovement.size() - 1;
    
    
    int randNum1 = min + (rand() % static_cast<int>(max - min + 1));
    
    int randNum2 = min + (rand() % static_cast<int>(max - min + 1));
    
    
    //int randNum1 = (frameMovement.size() - 1) * (rand() / RAND_MAX);
    
   // int randNum2 = (frameMovement.size() - 1) * (rand() / RAND_MAX);
    
    //(rand() % 10) + 1
    
    
   
    
    
    if(randNum1 != randNum2){
        ret.push_back(frameMovement[randNum1]);
        ret.push_back(frameMovement[randNum2]);
    }
    
    return ret;
    
}
