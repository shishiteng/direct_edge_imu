#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "settings.h"

inline void fillCvMat(cv::Mat* mat, cv::Vec3b color)
{
    for(int y=0;y<mat->size().height;y++)
        for(int x=0;x<mat->size().width;x++)
            mat->at<cv::Vec3b>(y,x) = color;
}

inline float getInterpolatedElementEigen(const Eigen::MatrixXf& mat, const float x, const float y)
{
    //stats.num_pixelInterpolations++;

    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    float res =   dxdy * mat(iy+1, ix+1)
                + (dy-dxdy) * mat(iy+1, ix)
                + (dx-dxdy) * mat(iy, ix+1)
                + (1-dx-dy+dxdy) * mat(iy, ix) ;

    return res;
}

inline Eigen::Vector2f getInterpolatedElement42(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    float dx = x - ix;
    float dy = y - iy;
    float dxdy = dx*dy;
    const Eigen::Vector4f* bp = mat +ix+iy*width;


    return dxdy * *(const Eigen::Vector2f*)(bp+1+width)
            + (dy-dxdy) * *(const Eigen::Vector2f*)(bp+width)
            + (dx-dxdy) * *(const Eigen::Vector2f*)(bp+1)
            + (1-dx-dy+dxdy) * *(const Eigen::Vector2f*)(bp);
}

inline cv::Mat getDepthRainbowPlot(const float* idepth, const float* idepthVar, const float* gray, int width, int height)
{
	cv::Mat res = cv::Mat(height,width,CV_8UC3);
	if(gray != 0)
	{
		cv::Mat keyFrameImage(height, width, CV_32F, const_cast<float*>(gray));
		cv::Mat keyFrameImage8u;
		keyFrameImage.convertTo(keyFrameImage8u, CV_8UC1);
        cv::cvtColor(keyFrameImage8u, res, CV_GRAY2RGB);
	}
	else
        fillCvMat(&res,cv::Vec3b(255,170,168));

	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			float id = idepth[i + j*width];

			if(id >=0 && idepthVar[i + j*width] >= 0)
			{

				// rainbow between 0 and 4
				float r = (0-id) * 255 / 1.0; if(r < 0) r = -r;
				float g = (1-id) * 255 / 1.0; if(g < 0) g = -g;
				float b = (2-id) * 255 / 1.0; if(b < 0) b = -b;

				uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
				uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
				uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);

				res.at<cv::Vec3b>(j,i) = cv::Vec3b(255-rc,255-gc,255-bc);
			}
		}
	return res;
}

inline cv::Mat getVarRedGreenPlot(const float* idepthVar, const float* gray, int width, int height)
{
	float* idepthVarExt = (float*)Eigen::internal::aligned_malloc(width*height*sizeof(float));

	memcpy(idepthVarExt,idepthVar,sizeof(float)*width*height);

	for(int i=2;i<width-2;i++)
		for(int j=2;j<height-2;j++)
		{
			if(idepthVar[(i) + width*(j)] <= 0)
				idepthVarExt[(i) + width*(j)] = -1;
			else
			{
				float sumIvar = 0;
				float numIvar = 0;
				for(int dx=-2; dx <=2; dx++)
					for(int dy=-2; dy <=2; dy++)
					{
						if(idepthVar[(i+dx) + width*(j+dy)] > 0)
						{
							float distFac = (float)(dx*dx+dy*dy)*(0.075*0.075)*0.02;
							float ivar = 1.0f/(idepthVar[(i+dx) + width*(j+dy)] + distFac);
							sumIvar += ivar;
							numIvar += 1;
						}
					}
				idepthVarExt[(i) + width*(j)] = numIvar / sumIvar;
			}

		}


	cv::Mat res = cv::Mat(height,width,CV_8UC3);
	if(gray != 0)
	{
		cv::Mat keyFrameImage(height, width, CV_32F, const_cast<float*>(gray));
		cv::Mat keyFrameImage8u;
		keyFrameImage.convertTo(keyFrameImage8u, CV_8UC1);
		cv::cvtColor(keyFrameImage8u, res, CV_GRAY2RGB);
	}
	else
		fillCvMat(&res,cv::Vec3b(255,170,168));

	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			float idv = idepthVarExt[i + j*width];

			if(idv > 0)
			{
				float var= sqrt(idv);

				var = var*60*255*0.5 - 20;
				if(var > 255) var = 255;
				if(var < 0) var = 0;

				res.at<cv::Vec3b>(j,i) = cv::Vec3b(0,255-var, var);
			}
		}

	Eigen::internal::aligned_free((void*)idepthVarExt);

	return res;
}

inline void printMessageOnCVImage(cv::Mat &image, std::string line1 )
{
	for(int x=0;x<image.cols;x++)
		for(int y=image.rows-30; y<image.rows;y++)
			image.at<cv::Vec3b>(y,x) *= 0.5;

	cv::putText(image, line1, cvPoint(10,image.rows-18),
        CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200,200,250), 1, 8);
}

// reads interpolated element from a uchar* array
// SSE2 optimization possible
inline float getInterpolatedElement(const float* const mat, const float x, const float y, const int width)
{
	//stats.num_pixelInterpolations++;

	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const float* bp = mat +ix+iy*width;


	float res =   dxdy * bp[1+width]
				+ (dy-dxdy) * bp[width]
				+ (dx-dxdy) * bp[1]
				+ (1-dx-dy+dxdy) * bp[0];

	return res;
}

inline Eigen::Vector3f getInterpolatedElement43(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector4f* bp = mat +ix+iy*width;


	return dxdy * *(const Eigen::Vector3f*)(bp+1+width)
	        + (dy-dxdy) * *(const Eigen::Vector3f*)(bp+width)
	        + (dx-dxdy) * *(const Eigen::Vector3f*)(bp+1)
			+ (1-dx-dy+dxdy) * *(const Eigen::Vector3f*)(bp);
}

inline Eigen::Vector4f getInterpolatedElement44(const Eigen::Vector4f* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const Eigen::Vector4f* bp = mat +ix+iy*width;


	return dxdy * *(bp+1+width)
	        + (dy-dxdy) * *(bp+width)
	        + (dx-dxdy) * *(bp+1)
			+ (1-dx-dy+dxdy) * *(bp);
}

inline void setPixelInCvMat(cv::Mat* mat, cv::Vec3b color, int xx, int yy, int lvlFac)
{
	for(int x=xx*lvlFac; x < (xx+1)*lvlFac && x < mat->size().width;x++)
		for(int y=yy*lvlFac; y < (yy+1)*lvlFac && y < mat->size().height;y++)
			mat->at<cv::Vec3b>(y,x) = color;
}

inline cv::Vec3b getGrayCvPixel(float val)
{
	if(val < 0) val = 0; if(val>255) val=255;
	return cv::Vec3b(val,val,val);
}

inline float getRefFrameScore(float distanceSquared, float usage, float KFDistWeight, float KFUsageWeight)
{
    return distanceSquared*KFDistWeight*KFDistWeight
            + (1-usage)*(1-usage) * KFUsageWeight * KFUsageWeight;
}

inline void calculateInvDepthImage(cv::Mat disparity, cv::Mat& iDepthImage, cv::Mat& iVar, float baseline, float f)
{
    int n = disparity.rows ;
    int m = disparity.cols ;

    iDepthImage.create(n, m, CV_32F);
    iVar.create(n, m, CV_32F);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float d = disparity.at<float>(i, j);
            if ( d < 3.0 ) {
                iDepthImage.at<float>(i, j) = -1;
                iVar.at<float>(i, j) = -1 ;
            }
            else {
                //depthImage.at<float>(i, j) = baseline * f / d;
                iDepthImage.at<float>(i, j) = d / (baseline * f) ;//inverse depth
                //printf("iDepthImage = %f\n", iDepthImage.at<float>(i, j) ) ;
                iVar.at<float>(i, j) = VAR_GT_INIT_INITIAL ;
            }
        }
    }
}

inline float SQ(float a){
    return a*a;
}

inline Eigen::Matrix3d vectorToSkewMatrix(const Eigen::Vector3d& w)
{
  Eigen::Matrix3d skewW(3, 3);
  skewW(0, 0) = skewW(1, 1) = skewW(2, 2) = 0;
  skewW(0, 1) = -w(2);
  skewW(1, 0) = w(2);
  skewW(0, 2) = w(1);
  skewW(2, 0) = -w(1);
  skewW(1, 2) = -w(0);
  skewW(2, 1) = w(0);

  return skewW;
}

inline void RtoEulerAngles(Eigen::Matrix3d R, double a[3])
{
    double theta = acos(0.5*(R(0, 0) + R(1, 1) + R(2, 2) - 1.0));
    a[0] = (R(2, 1) - R(1, 2)) / (2.0* sin(theta));
    a[1] = (R(0, 2) - R(2, 0)) / (2.0* sin(theta));
    a[2] = (R(1, 0) - R(0, 1)) / (2.0* sin(theta));
}

inline void R_to_ypr(const Eigen::Matrix3d& R, double angle[3])
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    //Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0)*cos(y) + n(1)*sin(y));
    double r = atan2(a(0)*sin(y) - a(1)*cos(y), -o(0)*sin(y) + o(1)*cos(y));
    angle[0] = y;
    angle[1] = p;
    angle[2] = r;
    //ypr(0) = y;
    //ypr(1) = p;
    //ypr(2) = r;
}
