#pragma once

#include <string.h>
#include <string>
#include "Eigen/Core"

#define PROJECT_TO_IMU_CENTER
#define DENSE

#define ALIGN __attribute__((__aligned__(16)))
#define SSEE(val,idx) (*(((float*)&val)+idx))
#define DIVISION_EPS 1e-10f
#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))

const float var_weight = 1.0;
const float huber_d = 5.0;

/** ============== Bundle Adjustment Paramters ======================= */
#define variablesNumInState 9
//#define slidingWindowSize 60
#define	INDEX(i, j, n, m)		( (i)*(m) + (j)  )
#define	STATE_SZ(i)		(  (i)*variablesNumInState )
#define maxIterationBA 100
const double PI = acos(-1.0);

//const double huber_r_v = 500 ;
//const double huber_r_w = PI ;
const Eigen::Matrix3d acc_cov = 1e-4 * Eigen::Matrix3d::Identity();
const Eigen::Matrix3d gra_cov = 1e-4 * Eigen::Matrix3d::Identity();
const Eigen::Matrix3d gyr_cov = 1e-4 * Eigen::Matrix3d::Identity();
//const Eigen::Matrix3d acc_cov = 0.23/1000.0*9.8 * 0.23/1000.0*9.8 * 200 * Eigen::Matrix3d::Identity();
//const Eigen::Matrix3d gra_cov = 0.23/1000.0*9.8 * 0.23/1000.0*9.8 * 200 * Eigen::Matrix3d::Identity();
//const Eigen::Matrix3d gyr_cov = 0.0135/180.0*PI * 0.0135/180.0*PI * 25 * Eigen::Matrix3d::Identity();
const int frameInfoListSize = 200 ;
//#define DENSE_TRACKING_WEIGHT 0.000001
#define RECORD_RESULT
#define MINIMUM_3DPOINTSFORTRACKING 100

/** ============== Depth Variance Handeling ======================= */
#define SUCC_VAR_INC_FAC (1.01f) // before an ekf-update, the variance is increased by this factor.
#define FAIL_VAR_INC_FAC 1.1f // after a failed stereo observation, the variance is increased by this factor.
#define MAX_VAR (0.5f*0.5f) // initial variance on creation - if variance becomes larter than this, hypothesis is removed.

#define VAR_GT_INIT_INITIAL 0.01f*0.01f	// initial variance vor Ground Truth Initialization
#define VAR_RANDOM_INIT_INITIAL (0.5f*MAX_VAR)	// initial variance vor Random Initialization


#define SE3TRACKING_MIN_LEVEL 0
#define SE3TRACKING_MAX_LEVEL 4

#define QUICK_KF_CHECK_LVL (SE3TRACKING_MAX_LEVEL-1)

#define PYRAMID_LEVELS SE3TRACKING_MAX_LEVEL
#define MIN_GOODPERALL_PIXEL_ABSMIN 0.001
#define MIN_GOODPERGOODBAD_PIXEL (0.5f)
#define MIN_GOODPERALL_PIXEL (0.04f)
#define MAX_DIFF_CONSTANT (40.0f*40.0f)
#define MAX_DIFF_GRAD_MULT (0.5f*0.5f)
#define MIN_ABS_GRAD_CREATE (minUseGrad)
#define MIN_ABS_GRAD_DECREASE (minUseGrad)
#define MIN_NUM_MAPPED -1

extern int gaussianNewtonTrackingIterationNum ;
extern int subGradientTrackingIterationNum ;
extern int optimizedPyramidLevel ;
extern bool printDebugInfo ;
extern bool useGaussianNewton ;
extern bool visualizeTrackingDebug ;
extern bool visaulizeGraphStructure ;
extern bool enable_pubPointCloud ;
extern bool enable_pubKeyFrameOdom ;
extern bool enable_pubTF ;
extern bool enable_LoopClosure;
extern bool enable_crossCheckTracking ;
extern double trustTrackingErrorThreshold ;
extern int cannyThreshold1 ;
extern int cannyThreshold2 ;
extern bool enable_histogramEqualization ;
extern double edgeProportion ;
extern double errorAngleThreshold ;
extern double errorTranslationThreshold ;
extern bool adaptiveCannyThreshold ;

extern bool onUAV ;
extern bool frontMarginalization ;
extern double loopClosureInterval ;
extern double visualWeight ;

extern double bias_g_x_0 ;
extern double bias_g_y_0 ;
extern double bias_g_z_0 ;
extern double bias_a_x_0 ;
extern double bias_a_y_0 ;
extern double bias_a_z_0 ;
extern bool denseOrNot ;
extern bool IMUorNot ;


extern double huber_r_v ;
extern double huber_r_w ;

//#define PRINT_DEBUG_INFO
//#define PUB_TF
//#define USE_GAUSSIANNEWTON
//#define VISUALIZE_TRACKING_DEBUG
////#define PUB_POINT_CLOUD
////#define PUB_KEYFRAME_ODOM
//#define PUB_GRAPH
//#define GAUSSIANNEWTONTRACKING_NUM 10

extern float KFDistWeight;
extern float KFUsageWeight;
extern int maxLoopClosureCandidates;

extern float minUseGrad;
extern float cameraPixelNoise2;
extern float depthSmoothingFactor;
extern int edgeIterationNum ;

const bool useAffineLightningEstimation = true ;
const bool enablePrintDebugInfo = false ;
const bool printTrackingIterationInfo = false ;



void handleKey(char k);


class DenseDepthTrackerSettings
{
public:
    inline DenseDepthTrackerSettings()
    {
        // Set default settings
        if (PYRAMID_LEVELS > 6)
            printf("WARNING: Sim3Tracker(): default settings are intended for a maximum of 6 levels!");

        lambdaSuccessFac = 0.5f;
        lambdaFailFac = 2.0f;

        const float stepSizeMinc[6] = {1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8};
        const int maxIterations[6] = {20, 50, 100, 100, 100, 100};


        for (int level = 0; level < PYRAMID_LEVELS; ++ level)
        {
            lambdaInitial[level] = 0;
            stepSizeMin[level] = stepSizeMinc[level];
            convergenceEps[level] = 0.999f;
            maxItsPerLvl[level] = maxIterations[level];
        }

        lambdaInitialTestTrack = 0;
        stepSizeMinTestTrack = 1e-3;
        convergenceEpsTestTrack = 0.98;
        maxItsTestTrack = 5;

        var_weight = 1.0;
        huber_d = 3;
    }

    float lambdaSuccessFac;
    float lambdaFailFac;
    float lambdaInitial[PYRAMID_LEVELS];
    float stepSizeMin[PYRAMID_LEVELS];
    float convergenceEps[PYRAMID_LEVELS];
    int maxItsPerLvl[PYRAMID_LEVELS];

    float lambdaInitialTestTrack;
    float stepSizeMinTestTrack;
    float convergenceEpsTestTrack;
    float maxItsTestTrack;

    float huber_d;
    float var_weight;
};
