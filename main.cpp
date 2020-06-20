
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include "video_search/video_search.hpp"

using namespace std;
using namespace cv;


int main(int argc, char *argv[]) {
    int i = 0;
    string mode = "hist";
    if (argc < 2) {
        printf("Usage: %s <short_video> <long_video> <full_video> <out_video>\n", argv[1]);
        printf("使用方法: %s 短视频.mp4 长视频.mp4 长视频-未处理.mp4 输出视频名.mp4\n", argv[1]);
        return -1;
    }
    
    for (i=5; i<argc; i++)
    {
        if (strncmp(argv[i], "debug" , 5) == 0)
        {
            mDebug = 1;
            mLongVideoPath = argv[2];
            mShortVideoPath = argv[1];
        }
        else if (strncmp(argv[i], "rebuild" , 7) == 0)
        {
            mRebuild = 1;
        }
        else if (strncmp(argv[i], "orb" , 3) == 0)
        {
            mode = argv[i];
        }
        else if (strncmp(argv[i], "hist" , 4) == 0)
        {
            mode = argv[i];
        }
        else if (strncmp(argv[i], "sep" , 3) == 0)
        {
            mode = argv[i];
        }
    }
    
    if (mode == "sep")
    {
        video_separate(argv[1], argv[2], argv[3]);
    }
    else if (mode == "orb")
    {
        video_search_orb(argv[1], argv[2], argv[3], argv[4]);
    }
    else
    {
        video_search_hist(argv[1], argv[2], argv[3], argv[4]);
    }

    return 0;
}
