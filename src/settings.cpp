#include "settings.h"

int gaussianNewtonTrackingIterationNum  = 10;
int subGradientTrackingIterationNum = 50 ;
int optimizedPyramidLevel = 3 ;
bool useGaussianNewton = true ;
bool printDebugInfo = true ;
bool visualizeTrackingDebug = true ;
bool enable_pubPointCloud = false ;
bool enable_pubKeyFrameOdom = false ;
bool visaulizeGraphStructure = true ;
bool enable_pubTF = true ;
bool enable_LoopClosure = true ;
bool enable_crossCheckTracking = false ;
bool enable_histogramEqualization = false ;
double trustTrackingErrorThreshold = 2.0 ;
int cannyThreshold1 = 100 ;
int cannyThreshold2 = 200 ;
double errorAngleThreshold = 10.0/180*PI;
double errorTranslationThreshold = 1.5;
bool adaptiveCannyThreshold = true ;

float KFDistWeight = 4;
float KFUsageWeight = 3;
int maxLoopClosureCandidates = 10;

float minUseGrad = 5;
float cameraPixelNoise2 = 4*4;
float depthSmoothingFactor = 1;

int maxOptimizationIterations = 100;
int propagateKeyFrameDepthCount = 0;
float loopclosureStrictness = 1.5;
float relocalizationTH = 0.7;

bool onUAV = false;
double edgeProportion = 0.15;
bool frontMarginalization = false ;
double loopClosureInterval = 0.1 ;
double visualWeight = 10000 ;
int edgeIterationNum = 5 ;

double bias_g_x_0 = 0.0;
double bias_g_y_0 = 0.0;
double bias_g_z_0 = 0.0;
double bias_a_x_0 = 0.0;
double bias_a_y_0 = 0.0;
double bias_a_z_0 = 0.0;
bool denseOrNot = true;

double huber_r_v = 0.05 ;
double huber_r_w = 1.0/180.0*PI ;
bool IMUorNot = true ;

void handleKey(char k)
{
    char kkk = k;
    switch(kkk)
    {
    case 'a': case 'A':
//		autoRun = !autoRun;		// disabled... only use for debugging & if you really, really know what you are doing
        break;
    case 's': case 'S':
//		autoRunWithinFrame = !autoRunWithinFrame; 	// disabled... only use for debugging & if you really, really know what you are doing
        break;
//    case 'd': case 'D':
//        debugDisplay = (debugDisplay+1)%6;
//        printf("debugDisplay is now: %d\n", debugDisplay);
//        break;
//    case 'e': case 'E':
//        debugDisplay = (debugDisplay-1+6)%6;
//        printf("debugDisplay is now: %d\n", debugDisplay);
//        break;
//    case 'o': case 'O':
//        onSceenInfoDisplay = !onSceenInfoDisplay;
//        break;
//    case 'r': case 'R':
//        printf("requested full reset!\n");
//        fullResetRequested = true;
//        break;
//    case 'm': case 'M':
//        printf("Dumping Map!\n");
//        dumpMap = true;
//        break;
//    case 'p': case 'P':
//        printf("Tracking all Map-Frames again!\n");
//        doFullReConstraintTrack = true;
//        break;
//    case 'l': case 'L':
//        printf("Manual Tracking Loss Indicated!\n");
//        manualTrackingLossIndicated = true;
//        break;
    }

}
