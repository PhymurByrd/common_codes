//
//  video_search.hpp
//  smartcut_xcode
//
//  Created by 刘会淼 on 2020/6/13.
//  Copyright © 2020 phymur. All rights reserved.
//

#ifndef video_search_hpp
#define video_search_hpp

#include <stdio.h>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"


extern int mRebuild ;//直接重建
extern int mDebug ;
extern std::string mLongVideoPath;
extern std::string mShortVideoPath;

//For trans_detection
typedef struct sVideoCapureParams{
    const char *src_path;    //输入的视频文件路径
    const char *out_path;    //截取视频的图片保存路径
    int max_shot_frames;     //视频截取帧数最大值(用于场景截取模式)
    double min_diff;         //视频差异的最小值(用于场景截取模式)
    int min_shot_frames;     //视频截取帧数最小值(用于平均帧提取模式)
} sVideoCapureParams;


typedef enum eVideoCapureCodes {
    UNKNOWN_ERROR = -500,
    PARAM_ERROR  = -102,
    MALLOC_ERROR = -10,
    DECODE_ERROR = -3,
    IO_ERROR = -1,
    SUCCESS = 0
} eVideoCapureCodes;

typedef enum eVideoCaptureMode {
    INTERLVAL_MODE = 0,
    TRANS_DETECTION_MODE  = 1
} eVideoCaptureMode;

int trans_detection(sVideoCapureParams *params);
int interval_capture(sVideoCapureParams *params);
extern int video_capture_frame(eVideoCaptureMode mode, sVideoCapureParams *params);

extern int video_search_orb(const char *short_video, const char *all_video, const char *full_video, const char *out_video);
extern int video_search_hist(const char *short_video, const char *all_video, const char *full_video, const char *out_video) ;

extern int video_search_main(int argc, char *argv[]);
extern int video_search_orb_main(int argc, char *argv[]);

extern int video_separate(const char *src_video, const char *dst_dir, const char *info_path);

#endif /* video_search_hpp */
