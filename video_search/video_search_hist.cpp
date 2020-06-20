//
//  video_search.cpp
//  smartcut_xcode
//
//  Created by 刘会淼 on 2020/6/13.
//  Copyright © 2020 phymur. All rights reserved.
//

#include "video_search.hpp"

#include <stdio.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

int mRebuild = 0;//直接重建
int mDebug = 0;
int mInterval = 10; //检测间隔，10张
std::string mLongVideoPath;
std::string mShortVideoPath;

class HistogramND {
private:
    int hisSize[1], hisWidth, hisHeight;//直方图的大小,宽度和高度
    float range[2];//直方图取值范围
    const float *ranges;
    Mat channelsRGB[3];//分离的BGR通道
    MatND histNormRGB[3];//输出直方图分量 归一化
    
    vector<MatND> histNormAllRGB;
    vector<int>   histNormAllRGB_FrameIdx; //histNormAllRGB 对应的frame idx
    vector<int>   histNormAllRGB_FrameTimeMs; //histNormAllRGB 对应的ms

    vector<int>   mMatchFrameIdx;
    vector<int>   mMatchFrameTimeMs;
    vector<int>   mContinuousFrameIdx;
    int           mMaxMatchFrameIdx;

    String mVideoPath;
    bool   mSearchVideo; //长的那个视频置1，它是search的视频。 另外短的是被search的视频，置0
    Rect mRoiRect;
    
    int mFrameRange[2];
public:
    Mat mImage;//源图像

    HistogramND() {
        hisSize[0] = 256;
        hisWidth = 400;
        hisHeight = 400;
        range[0] = 5.0;
        range[1] = 250.0;
        ranges = &range[0];
        mSearchVideo = 0;
        mRoiRect = {0,0,0,0};
        mFrameRange[0] = 0;
        mFrameRange[1] = 0;
        mMaxMatchFrameIdx = 0;
    }
    
    void setVideo(String video_path, bool search)
    {
        mVideoPath = video_path;
        mSearchVideo = search;
    }
    
    void setRoiRegion(int x, int y, int w, int h)
    {
        mRoiRect = {x, y, w, h};
    }
    
    void setFrameRange(int start, int end)
    {
        mFrameRange[0] = start;
        mFrameRange[1] = end;
    }
 
    //导入图片
    bool importImage(String path) {
        mImage = imread(path);
        if (!mImage.data)
            return false;
        return true;
    }
    bool importImage(String path, Rect roi) {
        mImage = imread(path);
        if (!mImage.data)
            return false;
        mImage = mImage(roi);
        return true;
    }

    
    void importImage(Mat img) {
        mImage = img;
    }
    
    void importImage(Mat img, Rect roi) {
        roi.width = (roi.width == 0) ? img.cols : roi.width;
        roi.height = (roi.height == 0) ? img.rows : roi.height;
        mImage = img(roi);
    }

 
    //分离通道
    void splitChannels() {
        split(mImage, channelsRGB);
    }
 
    //计算直方图
    void getHistogram() {
        MatND histRGB[3];//输出直方图分量

        calcHist(&channelsRGB[0], 1, 0, Mat(), histRGB[0], 1, hisSize, &ranges);
        calcHist(&channelsRGB[1], 1, 0, Mat(), histRGB[1], 1, hisSize, &ranges);
        calcHist(&channelsRGB[2], 1, 0, Mat(), histRGB[2], 1, hisSize, &ranges);
 
        normalize(histRGB[0], histNormRGB[0]);//, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
        histNormAllRGB.push_back(histNormRGB[0].clone());
        normalize(histRGB[1], histNormRGB[1]);//, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
        histNormAllRGB.push_back(histNormRGB[1].clone());
        normalize(histRGB[2], histNormRGB[2]);//, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
        histNormAllRGB.push_back(histNormRGB[2].clone());
        
        //输出各个bin的值
//        for (int i = 0; i < hisSize[0]; ++i) {
//            cout << i << "   B:" << histRGB[0].at<float>(i);
//            cout << "   G:" << histRGB[1].at<float>(i);
//            cout << "   R:" << histRGB[2].at<float>(i) << endl;
//        }
    }
    
    double getMinHistogramDist(MatND *hstNormRGB1, MatND *hstNormRGB2)
    {
        double chisquared = 0.0;
        chisquared  = compareHist(hstNormRGB1[0], hstNormRGB2[0], HISTCMP_CHISQR);
        chisquared += compareHist(hstNormRGB1[1], hstNormRGB2[1], HISTCMP_CHISQR);
        chisquared += compareHist(hstNormRGB1[2], hstNormRGB2[2], HISTCMP_CHISQR);
        return chisquared;
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
    
    void getMinHistogramDist(HistogramND &long_hist)
    {
        double chisquared = 0.0, min_chisquared = 0.0;
        int min_frame_idx = 0;
        float min_frame_ms = 0.0f;
        VideoCapture *short_capture = NULL, *long_capture  = NULL;
        
        if (histNormAllRGB.size() == 0 || long_hist.histNormAllRGB.size() == 0)
        {
            return ;
        }
        
        int i = 0, j = 0;
        if (mDebug)
        {
            short_capture = new VideoCapture(mShortVideoPath);
            long_capture  = new VideoCapture(mLongVideoPath);
        }
        for (i=0; i<histNormAllRGB.size(); i+=3)
        {
            chisquared = 0.0;
            min_chisquared = 20.0;
            min_frame_idx = 0;
            min_frame_ms = 0.0f;

            for (j=0; j<long_hist.histNormAllRGB.size(); j+=3)
            {
                chisquared = getMinHistogramDist(&histNormAllRGB[i], &long_hist.histNormAllRGB[j]);
//                cout << "Frame Matchs: " << i/3 << " --> " << chisquared << endl;
                if (chisquared < min_chisquared || min_chisquared == 0.0)
                {
                    min_chisquared = chisquared;
                    min_frame_idx = long_hist.histNormAllRGB_FrameIdx[j/3];
                    min_frame_ms = long_hist.histNormAllRGB_FrameTimeMs[j/3];
                }
                
//                displayHistogram(&histNormAllRGB[i], "Dst");
//                displayHistogram(&all_hist.histNormAllRGB[min_frame_idx*3], "Min");
//                cvWaitKey(0);
            }
            fprintf(stdout, "Frame Matching: %d --> %d : %.2f\r", i/3, min_frame_idx, min_chisquared);
            fflush(stdout);
//            cout << "Frame Matchs: " << i/3 << " --> " << min_frame_idx << " : "<< min_chisquared << endl;
            mMatchFrameIdx.push_back(min_frame_idx);
            mMatchFrameTimeMs.push_back(min_frame_ms);
            if (mDebug && min_frame_idx)
            {
                short_capture->set(CAP_PROP_POS_FRAMES, i/3);
                Mat short_img, long_img;
                short_capture->read(short_img);
                    
                long_capture->set(CAP_PROP_POS_FRAMES, min_frame_idx);
                long_capture->read(long_img);
                imshow("short", short_img);
                imshow("long", long_img);
                waitKey();
                destroyAllWindows();
            }
            if (min_frame_idx > mMaxMatchFrameIdx)
                mMaxMatchFrameIdx = min_frame_idx;
        }
        return ;
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
#define GAP_TOLARENCE 25
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
            
            fp = fopen("frametimes.txt", "w+");
            if (fp == NULL)
                return;
            for (i=0; i<mMatchFrameTimeMs.size(); i++)
            {
                fwrite((to_string(mMatchFrameTimeMs[i]) + ",").c_str(), (to_string(mMatchFrameTimeMs[i]) + ",").size(), 1, fp);
            }
            fclose(fp);
            fp = NULL;

        }
        int seq_min = mMatchFrameIdx[0], seq_max = mMatchFrameIdx[0];
        for (i=1; i<mMatchFrameIdx.size()-2;i++)
        {
            if (mMatchFrameIdx[i] <= 0){
                mMatchFrameIdx[i] = mMatchFrameIdx[i-1];
                continue;
            }
            
            if (abs(mMatchFrameIdx[i-1] - mMatchFrameIdx[i]) < GAP_TOLARENCE) //abs <10
            {
                //如果下一个节点frameidx, delta<10，则记录为同一个seq
                seq_min = seq_min > mMatchFrameIdx[i] ? mMatchFrameIdx[i] : seq_min;
                seq_max = seq_max < mMatchFrameIdx[i] ? mMatchFrameIdx[i] : seq_max;
                continue;
            }
            else
            {
                //如果delta>=10，则有可能是new seq
                if (abs(mMatchFrameIdx[i-1] - mMatchFrameIdx[i+1]) < GAP_TOLARENCE)
                {
                    //如果后面一张，跟前一张是能对上的，说明当前张抖动，忽略。
                    mMatchFrameIdx[i] = (mMatchFrameIdx[i-1] + mMatchFrameIdx[i+1]) /2;
                    continue;
                }
                else
                {
                    //起 新seq
                    int temp_j = 0;
                    for (temp_j = seq_min; temp_j < seq_max; temp_j++)
                    {
                        mContinuousFrameIdx.push_back(temp_j);
                    }
//                    cout << "Pushing seq " << to_string(seq_min) << "->" << to_string(seq_max) << endl;
                    fprintf(stdout, "Pushing seq: %3d -> %3d\r", seq_min, seq_max);
                    fflush(stdout);

                    seq_min = seq_max = mMatchFrameIdx[i];
                }
            }
        }
        int temp_j = 0;
        for (temp_j = seq_min; temp_j < seq_max; temp_j++)
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
    void fetchNewVideo(string all_video_path, string out_video)
    {
        int i = 0;
        
        VideoCapture *video_capture = new VideoCapture(all_video_path);
        int fps = video_capture->get(CAP_PROP_FPS);
        int width = video_capture->get(CAP_PROP_FRAME_WIDTH);
        int height = video_capture->get(CAP_PROP_FRAME_HEIGHT);
        VideoWriter *video_write = new VideoWriter(out_video, TMP_MACRO('a', 'v', 'c', '1'), fps, Size(width, height));
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
    
 
    //显示直方图
    void displayHistogram(MatND *hstNormRGB, string title) {
        Mat rgbHist[3];
        for (int i = 0; i < 3; i++)
        {
            rgbHist[i] = Mat(hisWidth, hisHeight, CV_8UC3, Scalar::all(0));
        }
        for (int i = 0; i < hisSize[0]; i++)
        {
            int val = saturate_cast<int>(hstNormRGB[0].at<float>(i));
            rectangle(rgbHist[0], Point(i * 2 + 10, rgbHist[0].rows), Point((i + 1) * 2 + 10, rgbHist[0].rows - val), Scalar(0, 0, 255), 1, 8);
            val = saturate_cast<int>(hstNormRGB[1].at<float>(i));
            rectangle(rgbHist[1], Point(i * 2 + 10, rgbHist[1].rows), Point((i + 1) * 2 + 10, rgbHist[1].rows - val), Scalar(0, 255, 0), 1, 8);
            val = saturate_cast<int>(hstNormRGB[2].at<float>(i));
            rectangle(rgbHist[2], Point(i * 2 + 10, rgbHist[2].rows), Point((i + 1) * 2 + 10, rgbHist[2].rows - val), Scalar(255, 0, 0), 1, 8);
        }
//        subplot(3,1,1);
//        imshow(title + "R", rgbHist[0]);
////        subplot(3,1,2);
//        imshow(title + "G", rgbHist[1]);
////        subplot(3,1,3);
//        imshow(title + "B", rgbHist[2]);
//        imshow("image", image);
    }
    
    void processVideo()
    {
        VideoCapture *video_capture = new VideoCapture(mVideoPath);
        video_capture->get(CAP_PROP_FRAME_COUNT);
        Mat next_img;
        int frame_idx = 0;
        while (video_capture->read(next_img)) {
            frame_idx++;
                        
            //开始帧pos
            if (mFrameRange[0] && (frame_idx <= mFrameRange[0]))
                continue;
            
//            destroyAllWindows();
//            imshow("video_capture_" + to_string(frame_idx) + ".png", next_img);
//            cvWaitKey(10);

            if (mRoiRect.x + mRoiRect.y + mRoiRect.width + mRoiRect.height == 0)
                importImage(next_img);
            else
                importImage(next_img, mRoiRect);
            
//            namedWindow("cropped_" + to_string(frame_idx) + ".png");
//            moveWindow("cropped_" + to_string(frame_idx) + ".png", 700, 100);
//            imshow("cropped_" + to_string(frame_idx) + ".png", mImage);

            
            {
                MatND histRGB[3];//输出直方图分量
                split(mImage, channelsRGB);

                calcHist(&channelsRGB[0], 1, 0, Mat(), histRGB[0], 1, hisSize, &ranges);
                calcHist(&channelsRGB[1], 1, 0, Mat(), histRGB[1], 1, hisSize, &ranges);
                calcHist(&channelsRGB[2], 1, 0, Mat(), histRGB[2], 1, hisSize, &ranges);

                normalize(histRGB[0], histNormRGB[0]);          //, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
                histNormAllRGB.push_back(histNormRGB[0].clone());
                normalize(histRGB[1], histNormRGB[1]);          //, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
                histNormAllRGB.push_back(histNormRGB[1].clone());
                normalize(histRGB[2], histNormRGB[2]);          //, 0, hisWidth - 20, NORM_MINMAX);//NORM_MINMAX);
                histNormAllRGB.push_back(histNormRGB[2].clone());
                histNormAllRGB_FrameIdx.push_back(video_capture->get(CAP_PROP_POS_FRAMES));
                histNormAllRGB_FrameTimeMs.push_back(video_capture->get(CAP_PROP_POS_MSEC));
            }
            fprintf(stdout, "\t %s 分析中: %3d...\r", mVideoPath.c_str(), frame_idx);
            fflush(stdout);

            //提前结束帧pos
            if (mFrameRange[1] && (frame_idx >= mFrameRange[1]))
                break;
        }
        cout << "Process Video "<< mVideoPath <<" done!" << endl;

    }
};

/// <#Description#>
/// @param short_video 短视频，被查找视频
/// @param all_video 长视频。 长视频和短视频要统一做处理，尽量截取相同的画面内容。 水印和字幕，也会影响判断成功率
/// @param full_video 长视频的完整视频
int video_search_hist(const char *short_video, const char *all_video, const char *full_video, const char *out_video) {

    if (!all_video || !short_video)
        return -1;
    cout << endl << ".... Hist 匹配开始...." << endl;

    //all 是
    HistogramND all_hist, short_hist, min_hist;

    try {
        
        //设置video参数。
        all_hist.setVideo(all_video, 1);
//        all_hist.setFrameRange(1/*72*25*/, 80*25); //25fps: 60s -> 100s
//        all_hist.setRoiRegion(100, 40, 440, 280); //25fps: 60s -> 100s

        short_hist.setVideo(short_video, 0);
//        short_hist.setRoiRegion(0, 320, 0, 400);
//        short_hist.setFrameRange(2*25, 7*25+1);
        
        if (!mRebuild)
        {
            cout << endl << "1. 短视频分析中...." << endl;
            short_hist.processVideo();
            cout << endl << "2. 长视频分析中...." << endl;
            all_hist.processVideo();
            cout << endl << "3. 视频画面匹配中...." << endl;
            short_hist.getMinHistogramDist(all_hist);
            short_hist.SmoothMatchFrame(true);
        }
        else
        {
            //不解析，用dump的参数做smooth调试
            short_hist.monitorSmoothMatchFrame();
        }

        cout << endl << "4. 重建新视频中...." << endl;
        short_hist.fetchNewVideo(full_video ? full_video : all_video, out_video);
        cout << endl << "重建完成 " << out_video << " ...." << endl;
    } catch (...) {
        return IO_ERROR;
    };

//    waitKey(0);

    return SUCCESS;
}
