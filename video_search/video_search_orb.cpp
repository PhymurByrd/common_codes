
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


class VideoSearchOrb {
public:
    
    String          mShortVideoPath;                        //短视频处理视频
    String          mLongVideoPath;                         //长视频处理视频
    String          mAllVideoPath;                          //长视频原视频
    String          mOutVideoPath;                          //输出视频

    vector<Mat>     mShortKeyFrames;                        //short关键帧
    vector<int>     mShortInfoFrameIdxs;
    vector<Mat>     mShortDescriptors;                     //短视频所有的Descriptor
    vector<vector<KeyPoint>> mShortKeypointsArray;        //短视频所有的Keypoints组

    vector<double>  mShortMatchScore;                        //与当前short帧匹配的，最匹配的score分
    
    int             mShortFrameRange[2];                    //开始/结束帧idx
    int             mLongFrameRange[2];                     //开始/结束帧idx
    int             mLongFrameInterval;                     //长视频匹配间隔
    
    vector<int>     mMatchFrameIdx;                         //匹配的帧idx
    vector<int>     mContinuousFrameIdx;

    
    Ptr<FeatureDetector> mDetector;
    Ptr<DescriptorExtractor> mDescriptor;
    Ptr<DescriptorMatcher> mMatcher;

    VideoSearchOrb(String shortvideo, String longvideo, String allvideo, String outvideo)
    {
        mDetector = ORB::create();
        mDescriptor = ORB::create();
        mMatcher = DescriptorMatcher::create ( "BruteForce-Hamming" );

        mShortVideoPath = shortvideo;
        mLongVideoPath = longvideo;
        mAllVideoPath = allvideo;
        mOutVideoPath = outvideo;
        mShortFrameRange[0] = 0;
        mShortFrameRange[1] = 0;
        mLongFrameRange[0] = 0;
        mLongFrameRange[1] = 0;
        mLongFrameInterval = 10;
    }
    
    void setFrameRange(int isShort, int start, int end, int interval=1)
    {
        if (isShort)
        {
            mShortFrameRange[0] = start;
            mShortFrameRange[1] = end;
        }
        else
        {
            mLongFrameRange[0] = start;
            mLongFrameRange[1] = end;
            mLongFrameInterval = interval;
        }
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


    void preProcessShortVideo()
    {
        VideoCapture *video_capture = new VideoCapture(mShortVideoPath);
        int total_frame_cnt = (int)video_capture->get(CAP_PROP_FRAME_COUNT);
        Mat next_img;
        int frame_idx = 0;
        
        if (total_frame_cnt == 0)
            throw "video capture fail";
        
        total_frame_cnt = (total_frame_cnt <= 0 ? 1 : total_frame_cnt);
        mShortMatchScore.resize(total_frame_cnt, 35);
        mMatchFrameIdx.resize(total_frame_cnt);

        while (video_capture->read(next_img)) {
            frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);

            if (mShortFrameRange[0] && (frame_idx <= mShortFrameRange[0]))
            {
                //开始帧pos
                video_capture->set(CAP_PROP_POS_FRAMES, frame_idx);
                frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);
                continue;
            }
            
            vector<KeyPoint> keypoints;

            mDetector->detect (next_img, keypoints );
            if (keypoints.size() < 10)
            {
                //有效keypoints小于
                keypoints.clear();
                continue;
            }

            //-- 第二步:根据角点位置计算 BRIEF 描述子
            Mat short_descriptors;
            mDescriptor->compute ( next_img, keypoints, short_descriptors );

//            BOWImgDescriptorExtractor *bowDE = new BOWImgDescriptorExtractor(mDescriptor,mMatcher);

            mShortKeyFrames.push_back(next_img.clone());
            mShortInfoFrameIdxs.push_back(frame_idx);
            mShortKeypointsArray.push_back(keypoints);
            mShortDescriptors.push_back(short_descriptors.clone());

            fprintf(stdout, "\t %s 分析中: %3d / %3d...\r", mShortVideoPath.c_str(), frame_idx, total_frame_cnt);
            fflush(stdout);

            //提前结束帧pos
            if (mShortFrameRange[1] && (frame_idx >= mShortFrameRange[1]))
                break;
        }
        cout << "Process Short Video "<< mShortVideoPath <<" done!" << endl;
    }
    
    
    void processLongVideo()
    {
        VideoCapture *video_capture = new VideoCapture(mLongVideoPath);
        int total_frame_cnt = video_capture->get(CAP_PROP_FRAME_COUNT);
        int long_frame_idx = 0;
        
        vector<KeyPoint> long_keypoints, short_keypoints;
        Mat long_descriptors, short_descriptors;
        Mat long_img, short_img;

        while (video_capture->read(long_img)) {
            Mat descriptors_2;

            long_frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);

            if (mLongFrameRange[0] && (long_frame_idx <= mLongFrameRange[0]))
            {
                //开始帧pos
                video_capture->set(CAP_PROP_POS_FRAMES, mLongFrameRange[0]);
                long_frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);
                continue;
            }
            imshow ( "long_img", long_img );
            waitKey(10);

            mDetector->detect (long_img, long_keypoints );
            if (long_keypoints.size() < 10)
            {
                //有效keypoints小于
                continue;
            }
            mDescriptor->compute ( long_img, long_keypoints, long_descriptors );

            //与short的每一帧对比，找到对应帧
            unsigned long max_good_matches = 0;
            float match_score = 1000.0, match_frameidx = 0;
            
            vector< DMatch > match_good_matches;
            Mat match_short_image;
            vector<KeyPoint> match_short_keypoints;

            for (int k=0; k<mShortKeyFrames.size(); k++)
            {
                short_img = mShortKeyFrames[k];
                vector<DMatch> matches;
                vector< DMatch > good_matches;
                vector< DMatch > good_temp_matches;

                short_keypoints = mShortKeypointsArray[k];
//                mDetector->detect (short_img, short_keypoints );
//                if (short_keypoints.size() < 10)
//                    continue;
                
                //特征点数差异过大，跳过
                if ((short_keypoints.size() * 2 < long_keypoints.size()) ||
                    (short_keypoints.size() /2 > long_keypoints.size()))
                    continue;

                short_descriptors = mShortDescriptors[k];
//                //-- 第二步:根据角点位置计算 BRIEF 描述子
//                mDescriptor->compute ( short_img, short_keypoints, short_descriptors );

                //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
                mMatcher->match ( long_descriptors, short_descriptors, matches );
//                mMatcher->radiusMatch ( long_descriptors, short_descriptors, matches, 30.0f );

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
//                    if ( matches[i].distance <= max ( 3*min_dist, 30.0 ) )
                    {
                        good_matches.push_back ( matches[i] );
                        longSelfPoints.push_back(long_keypoints.at(matches[i].queryIdx).pt);
                        shortSelfPoints.push_back(short_keypoints.at(matches[i].trainIdx).pt);
                    }
                    if ( matches[i].distance <= min(max(3*min_dist, 30.0), 40.0))
                    {
                        good_temp_matches.push_back ( matches[i] );
                    }
                }

                if (good_temp_matches.size() < matches.size()*3/10) //20% good match
                {
                    continue;
                }
                cout << "good_matches: " << good_temp_matches.size() << " / " << matches.size() << " max:" << max_good_matches << " min_dist:" << min_dist << endl;
//
//                if (good_matches.size() > max_good_matches)
//                    max_good_matches = good_matches.size();
                
                vector<uchar> inliers = vector<uchar>(longSelfPoints.size(),0);
                Mat homography = cv::findHomography(longSelfPoints, shortSelfPoints, inliers, FM_LMEDS, 0.5);

                if (homography.empty())
                    continue;
        //        cout << homography << endl << endl;

                vector<Point2f> longSelfPoints_new;
                //根据homography仿射变化出理想的 特征点s
                perspectiveTransform(longSelfPoints, longSelfPoints_new, homography);
                float avergError = 0.f;
                //计算仿射变化出的理想特征点，与实际特征点的dist
                for (int i = 0; i < longSelfPoints_new.size(); ++i)
                {
                    Point2f diff = (longSelfPoints_new[i] - shortSelfPoints[i]);
                    float dist = norm(diff);
                    avergError += dist;
                }
                avergError /= longSelfPoints_new.size();
//
                if (avergError < match_score)
                {
                    match_score = avergError;
                    match_frameidx = mShortInfoFrameIdxs[k];
                    match_good_matches = good_temp_matches;
                    match_short_image = short_img.clone();
                    match_short_keypoints = short_keypoints;
                }
                if (mShortMatchScore[k] > avergError)
                {
                    mShortMatchScore[k] = avergError;
                    mMatchFrameIdx[k] = long_frame_idx;
                    
                    cout << k << "-" << long_frame_idx << " avergError:" << avergError << " / " << good_matches.size() << " " << match_good_matches.size() << endl;
                    fprintf(stdout, "  分析中: %3d, %3d / %3d... %.2f\r", k, long_frame_idx, total_frame_cnt, avergError);
                    fflush(stdout);

                    if (mDebug)
                    {
                        vector< DMatch > null_matches;
                        Mat img_goodmatch;
                        drawMatches ( long_img, long_keypoints, short_img, short_keypoints, match_good_matches, img_goodmatch );
                        destroyAllWindows();
                        imshow ( "goodmatch", img_goodmatch );
                        waitKey();

                        drawMatches ( long_img, long_keypoints, short_img, short_keypoints, null_matches, img_goodmatch );
                        imshow ( "keypoint", img_goodmatch );
                        waitKey();
                    }
                }
            }
            
//            fprintf(stdout, "\t %s 分析中: %3d / %3d...\r", mLongVideoPath.c_str(), long_frame_idx, total_frame_cnt);
//            fflush(stdout);

            // seek逻辑
            if (mLongFrameInterval > 1)
            {
                video_capture->set(CAP_PROP_POS_FRAMES, long_frame_idx + mLongFrameInterval);
                long_frame_idx = video_capture->get(CAP_PROP_POS_FRAMES);
            }
            
            //提前结束帧pos
            if (mLongFrameRange[1] && (long_frame_idx >= mLongFrameRange[1]))
                break;
            
        }
        
        cout << "Match score "<< mShortMatchScore[0] << " " << mMatchFrameIdx[0] <<" done!" << endl;
        cout << "Process Short Video "<< mShortVideoPath <<" done!" << endl;
    }
    
    /// 从文件读入，然后smoooth下
    void monitorSmoothMatchFrame()
    {
        {
            std::ifstream in("frameidxs.txt");
            std::stringstream buffer;
            buffer << in.rdbuf();
            
            StringSplitIdx(buffer.str(), ",", &mMatchFrameIdx);
        }
        SmoothMatchFrame(false);
    }
    
    void SmoothMatchFrame(bool dump)
    {
        // 连续化 匹配帧
        // 1. 如果前后两针frameidx <TOLARENCE，连到一起。
        // 2.
        int i = 0;
        
        if (dump) //dump到文件，然后monitorSmoothMatchFrame 可以快速调整smooth策略。
        {
            FILE *fp = fopen("frameidxs.txt", "w+");
            if (fp == NULL)
                return;
            for (i=0; i<mMatchFrameIdx.size(); i++)
            {
                fwrite((to_string(mMatchFrameIdx[i]) + ",").c_str(), (to_string(mMatchFrameIdx[i]) + ",").size(), 1, fp);
            }
            fclose(fp);
            fp = NULL;
        }
        int seq_min = mMatchFrameIdx[0], seq_max = mMatchFrameIdx[0];
        for (i=1; (mMatchFrameIdx.size() > 2) && (i<mMatchFrameIdx.size()-2);i++)
        {
            if (mMatchFrameIdx[i] <= 0){
                mMatchFrameIdx[i] = mMatchFrameIdx[i-1];
                continue;
            }
            
            if (abs(mMatchFrameIdx[i-1] - mMatchFrameIdx[i]) < mLongFrameInterval+10) //abs <10
            {
                //如果下一个节点frameidx, delta<10，则记录为同一个seq
                seq_min = seq_min > mMatchFrameIdx[i] ? mMatchFrameIdx[i] : seq_min;
                seq_max = seq_max < mMatchFrameIdx[i] ? mMatchFrameIdx[i] : seq_max;
                continue;
            }
            else
            {
                //如果delta>=10，则有可能是new seq
                if (abs(mMatchFrameIdx[i-1] - mMatchFrameIdx[i+1]) < mLongFrameInterval+10)
                {
                    //如果后面一张，跟前一张是能对上的，说明当前张抖动，忽略。
                    mMatchFrameIdx[i] = (mMatchFrameIdx[i-1] + mMatchFrameIdx[i+1]) /2;
                    continue;
                }
                else
                {
                    //起 新seq
                    int temp_j = 0;
                    for (temp_j = seq_min; temp_j <= seq_max; temp_j++)
                    {
                        mContinuousFrameIdx.push_back(temp_j);
                    }
//                    cout << "Pushing seq " << to_string(seq_min) << "->" << to_string(seq_max) << endl;
                    fprintf(stdout, "Pushing seq: %3d -> %3d\n", seq_min, seq_max);
                    fflush(stdout);

                    seq_min = seq_max = mMatchFrameIdx[i];
                }
            }
        }
        int temp_j = 0;
        for (temp_j = seq_min; temp_j <= seq_max; temp_j++)
        {
            mContinuousFrameIdx.push_back(temp_j);
        }
        if (1) //dump到文件，然后monitorSmoothMatchFrame 可以快速调整smooth策略。
        {
            FILE *fp = fopen("smoothframeidxs.txt", "w+");
            if (fp == NULL)
                return;
            for (i=0; i<mContinuousFrameIdx.size(); i++)
            {
                fwrite((to_string(mContinuousFrameIdx[i]) + ",").c_str(), (to_string(mContinuousFrameIdx[i]) + ",").size(), 1, fp);
            }
            fclose(fp);
            fp = NULL;
        }

    }
    
        #define TMP_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

    /// 根据搜索的frameidx，重组新视频。
    void genNewVideo()
    {
        int i = 0;
        
        VideoCapture *video_capture = new VideoCapture(mAllVideoPath);
        int fps = video_capture->get(CAP_PROP_FPS);
        int width = video_capture->get(CAP_PROP_FRAME_WIDTH);
        int height = video_capture->get(CAP_PROP_FRAME_HEIGHT);
        VideoWriter *video_write = new VideoWriter(mOutVideoPath, TMP_MACRO('a', 'v', 'c', '1'), fps, Size(width, height));
        Mat next_img;
        
        for (i=0; i<mContinuousFrameIdx.size(); i++)
        {
            video_capture->set(CAP_PROP_POS_FRAMES, mContinuousFrameIdx[i]);
            
            if (video_capture->read(next_img))
                video_write->write(next_img);
            fprintf(stdout, "Rebuilding: %3d / %3ld\r", i, mContinuousFrameIdx.size());
            fflush(stdout);
//            cout << "writting " << to_string(i) << "/" << to_string(mContinuousFrameIdx.size()) << "\r";
        }
        video_write->release();
    }

};

/// <#Description#>
/// @param short_video 短视频，被查找视频
/// @param long_video 长视频。 长视频和短视频要统一做处理，尽量截取相同的画面内容。 水印和字幕，也会影响判断成功率
/// @param all_video 长视频的完整视频
int video_search_orb(const char *short_video, const char *long_video, const char *all_video, const char *out_video)
{

    if (!all_video || !long_video || !all_video || !out_video)
        return -1;
    
    cout << endl << ".... ORB 匹配开始...." << endl;
    
    VideoSearchOrb videosearch(short_video, long_video, all_video, out_video);
    
//    try {
        if (!mRebuild)
        {
//            videosearch.setFrameRange(1, 4*25+1, 15*25);       //short range
//            videosearch.setFrameRange(0, 45*24, 60*24, 1); //long range

            videosearch.setFrameRange(1, 4*25+1, 9*25+25);       //short range
            videosearch.setFrameRange(0, 45*24, 250*24, 1); //long range
            cout << endl << "1. 短视频分析中...." << endl;
            videosearch.preProcessShortVideo();
            
            cout << endl << "2. 长视频分析中...." << endl;
            videosearch.processLongVideo();
            videosearch.SmoothMatchFrame(true);
        }
        else
        {
            videosearch.monitorSmoothMatchFrame();
        }
        
        cout << endl << "4. 重建新视频中...." << endl;
        videosearch.genNewVideo();
//    } catch (...) {
//        cout << "catched an exception" << endl;
//        throw;
//    };

//    waitKey(0);

    return 0;
}

