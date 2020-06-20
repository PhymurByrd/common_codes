

//
//  video_search_by_orb.cpp
//  smartcut_xcode
//
//  Created by 刘会淼 on 2020/6/13.
//  Copyright © 2020 phymur. All rights reserved.
//


#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#include "video_search.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;


// 把视频连续帧检查出来，并把信息（帧idx/ms信息）存起来
class VideoSeparate {
public:
    string      mSrcVideoPath;
    string      mOutVideoDir;
    string      mOutVideoInfo;
    
    string      mLastStardIdx;

    vector<Mat>     mShortKeyFrames;                        //short关键帧
    vector<int>     mShortInfoFrameIdxs;
    vector<Mat>     mShortDescriptors;                     //短视频所有的Descriptor
    vector<vector<KeyPoint>> mShortKeypointsArray;        //短视频所有的Keypoints组

    vector<double>  mShortMatchScore;                        //与当前short帧匹配的，最匹配的score分
    
    int             mFrameRange[2];                     //开始/结束帧idx
    int             mFrameInterval;                     //长视频匹配间隔
    Rect            mRoiRect;
    int             mWinWidth;
    int             mWinHeight;
    
    vector<int>     mMatchFrameIdx;                         //匹配的帧idx
    vector<int>     mContinuousFrameIdx;

    
    Ptr<FeatureDetector> mDetector;
    Ptr<DescriptorExtractor> mDescriptor;
    Ptr<DescriptorMatcher> mMatcher;

    VideoSeparate(String shortvideo, String outvideo, string infopath)
    {
        mDetector = ORB::create();
        mDescriptor = ORB::create();
        mMatcher = DescriptorMatcher::create ( "BruteForce-Hamming" );

        mSrcVideoPath = shortvideo;
        mOutVideoDir = outvideo;
        mOutVideoInfo = infopath;
        mFrameRange[0] = 0;
        mFrameRange[1] = 0;
        mFrameInterval = 1;
        mRoiRect = {0,0,0,0};
        mWinWidth = 300;
        mWinHeight = 300;
    }

    void setRoiRegion(int x, int y, int w, int h)
    {
        mRoiRect = {x, y, w, h};
    }

    void setFrameRange(int start, int end, int interval=1)
    {
        mFrameRange[0] = start;
        mFrameRange[1] = end;
        mFrameInterval = interval;
    }

    void setWindowSize(int w, int h=0)
    {
        mWinWidth = w;
        mWinHeight = h;
    }

    
    void StringSplitIdx(const string& str, const string seps, vector<int>* pieces) {
        pieces->clear();
        if (str.empty()) {
            return;
        }
        size_t pos = 0;
        
        size_t next = std::string::npos;
        for (char sep : seps)
            next = std::min(str.find(sep, pos), next);
        
        while (next != std::string::npos) {
            pieces->push_back(stoi(str.substr(pos, next - pos)));
            pos = next + 1;
            next = std::string::npos;
            for (char sep : seps)
                next = std::min(str.find(sep, pos), next);
        }
        if (!str.substr(pos).empty()) {
            pieces->push_back(stoi(str.substr(pos)));
        }
    }

    int videoCaptureSeek(VideoCapture *video_capture, int dst_idx)
    {
        Mat skip_img;
        int cur_idx = video_capture->get(CAP_PROP_POS_FRAMES);
        if ((dst_idx - cur_idx > 0) && (dst_idx - cur_idx < 20))
        {
            // read one by one, is faster then short seek
            for (;cur_idx<dst_idx; cur_idx++)
            {
                if (!video_capture->read(skip_img))
                {
                    cout << "Frame skip fail!" << endl;
                    return 0;
                }
            }
        }
        else
        {
            video_capture->set(CAP_PROP_POS_FRAMES, dst_idx);
            cur_idx = video_capture->get(CAP_PROP_POS_FRAMES);
        }
        return cur_idx;
    }

    void processVideo()
    {
        VideoCapture *video_capture = new VideoCapture(mSrcVideoPath);
        int total_frame_cnt = (int)video_capture->get(CAP_PROP_FRAME_COUNT);
        Mat last_img, next_img;
        int frame_idx = 0;

        vector<KeyPoint> last_keypoints, keypoints;
        Mat last_descriptors;
        int new_seq = 1;
        int seq_idx = 0;
        
        if (total_frame_cnt == 0)
            throw "video capture fail";
        
        FILE *fp_seq_idx = fopen(mOutVideoInfo.c_str(), "w+");
        if (fp_seq_idx == NULL)
            return;

        total_frame_cnt = (total_frame_cnt <= 0 ? 1 : total_frame_cnt);

        while (true)
        {
            last_img = next_img;
            last_keypoints = keypoints;
            if (!video_capture->read(next_img))
                break;
            
            frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);

            if (mFrameRange[0] && (frame_idx <= mFrameRange[0]))
            {
                //开始帧pos
                frame_idx = videoCaptureSeek(video_capture, mFrameRange[0]);
                if (!frame_idx)
                    break;
                new_seq = 1;
                continue;
            }
            
            if (mRoiRect.x + mRoiRect.y + mRoiRect.width + mRoiRect.height > 0)
            {
//                Point p1 = Point(mRoiRect.x, mRoiRect.y);
//                Point p2 = Point(mRoiRect.width + mRoiRect.x, mRoiRect.y + mRoiRect.height);
//                rectangle(next_img, p1, p2, cv::Scalar(0, 255, 255), 2, 4);
                
                mRoiRect.width = (mRoiRect.width == 0) ? next_img.cols - mRoiRect.x : mRoiRect.width;
                mRoiRect.height = (mRoiRect.height == 0) ? next_img.rows - mRoiRect.y : mRoiRect.height;
                next_img = next_img(mRoiRect);
            }
            if ((mWinWidth) && (next_img.cols > mWinWidth))
            {
                //倍数缩放
                mWinWidth = next_img.cols / round(next_img.cols * 1.0 / mWinWidth);
                mWinHeight = next_img.rows / round(next_img.cols * 1.0 / mWinWidth);
                
                Size ResImgSiz = Size(mWinWidth, mWinHeight);
                Mat ResImg = Mat(ResImgSiz, next_img.type());
                resize(next_img, ResImg, ResImgSiz, INTER_CUBIC);
                next_img = ResImg;
            }
            if (mDebug)
            {
                imshow("img", next_img);
                waitKey(10);
            }
            mDetector->detect (next_img, keypoints );
            if (keypoints.size() < 10)
            {
                //有效keypoints小于
                keypoints.clear();
                new_seq = 1;
                continue;
            }
            
            //-- 第二步:根据角点位置计算 BRIEF 描述子
            Mat descriptors;
            mDescriptor->compute ( next_img, keypoints, descriptors );
            if (new_seq)
            {
                last_descriptors = descriptors;
                new_seq = 0;
                continue;
            }
                
            vector<DMatch> matches;
            vector<DMatch> good_matches;

            //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
            mMatcher->match ( last_descriptors, descriptors, matches );
//                mMatcher->radiusMatch ( long_descriptors, short_descriptors, matches, 30.0f );
            last_descriptors = descriptors;
            new_seq = 0;
            //-- 第四步:匹配点对筛选
            double min_dist=10000, max_dist=0;

            //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
            for ( int i = 0; i < matches.size(); i++ )
            {
                double dist = matches[i].distance;
                if ( dist < min_dist ) min_dist = dist;
                if ( dist > max_dist ) max_dist = dist;
            }

            min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
            max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
            
    //        printf ( "-- Max dist : %f \n", max_dist );
    //        printf ( "-- Min dist : %f \n", min_dist );

            //good matches 对应的匹配点
            vector<Point2f> longSelfPoints;
            vector<Point2f> shortSelfPoints;

            //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
            for ( int i = 0; i < matches.size(); i++ )
            {
                // 这个distance是两个像素的内容差异
                if ( matches[i].distance <= min(max(3*min_dist, 30.0), 40.0))
                {
                    good_matches.push_back ( matches[i] );
//                    longSelfPoints.push_back(long_keypoints.at(matches[i].queryIdx).pt);
//                    shortSelfPoints.push_back(short_keypoints.at(matches[i].trainIdx).pt);
                }
            }
            
            cout << frame_idx << " seq: " << seq_idx <<" good_matches: " << good_matches.size() << " / " << matches.size() << " min_dist:" << min_dist << endl;

            if (good_matches.size() < matches.size()*15/100) //20% good match
            {
                int timems = video_capture->get(CAP_PROP_POS_MSEC);

                string info = to_string(frame_idx) + ":" + to_string(timems) + ",\n";
                fwrite(info.c_str(), info.size(), 1, fp_seq_idx);
                fflush(fp_seq_idx);
                seq_idx++;
                
                Mat img_goodmatch;
                drawMatches ( last_img, last_keypoints, next_img, keypoints, good_matches, img_goodmatch );
                putText(img_goodmatch, "Seq " + to_string(seq_idx), Point(0,100), 2, 1, Scalar(0, 0, 255));
                destroyAllWindows();
                imshow ( "goodmatch", img_goodmatch );
                waitKey();
            }
            if (0)
            {
                vector< DMatch > null_matches;
                Mat img_goodmatch;
                drawMatches ( last_img, last_keypoints, next_img, keypoints, good_matches, img_goodmatch );
                putText(img_goodmatch, "Seq " + to_string(seq_idx), Point(0,100), 2, 1, Scalar(0, 0, 255));
                destroyAllWindows();
                imshow ( "goodmatch", img_goodmatch );
                waitKey();

//                drawMatches ( last_img, last_keypoints, next_img, keypoints, null_matches, img_goodmatch );
//                imshow ( "keypoint", img_goodmatch );
//                waitKey(1);
            }


//            fprintf(stdout, "\t %s 分析中: %3d / %3d...\r", mShortVideoPath.c_str(), frame_idx, total_frame_cnt);
//            fflush(stdout);

            // seek逻辑
            if (mFrameInterval > 1)
            {
                // < 10时，seek比往后度还慢
                frame_idx = videoCaptureSeek(video_capture, frame_idx + mFrameInterval);
            }
            
            //提前结束帧pos
            if (mFrameRange[1] && (frame_idx >= mFrameRange[1]))
                break;
        }
        fclose(fp_seq_idx);
        fp_seq_idx = NULL;

        cout << "Process Short Video "<< mShortVideoPath <<" done!" << endl;
    }

};

int video_separate(const char *src_video, const char *dst_dir, const char *info_path)
{

    if (!src_video || !dst_dir || !info_path)
        return -1;
    
    cout << endl << ".... ORB 匹配开始...." << endl;
    
    VideoSeparate separator(src_video, dst_dir, info_path);
    
//    try {
        if (!mRebuild)
        {
            separator.setFrameRange(0, 0, 5);
            cout << endl << "1. 短视频分析中...." << endl;
            separator.setRoiRegion(0, 370, 720, 360);
            separator.setWindowSize(300);
            separator.processVideo();
            
        }
        else
        {
        }
        
        cout << endl << "4. 重建新视频中...." << endl;
//    } catch (...) {
//        cout << "catched an exception" << endl;
//        throw;
//    };

//    waitKey(0);

    return 0;
}

