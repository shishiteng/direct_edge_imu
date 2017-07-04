/**

* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "SE3Tracker.h"

namespace lsd_slam
{
#define USESSE true
#if defined(ENABLE_SSE)
#define callOptimized(function, arguments) (USESSE ? function##SSE arguments : function arguments)
#else
#define callOptimized(function, arguments) function arguments
#endif


SE3Tracker::SE3Tracker(int w, int h, Eigen::Matrix3f K)
{
	width = w;
	height = h;

	this->K = K;
	fx = K(0,0);
	fy = K(1,1);
	cx = K(0,2);
    cy = K(1,2);

	KInv = K.inverse();
	fxi = KInv(0,0);
	fyi = KInv(1,1);
	cxi = KInv(0,2);
	cyi = KInv(1,2);


	buf_warped_residual = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dx = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_dy = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_x = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_y = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_warped_z = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_d = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_idepthVar = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
	buf_weight_p = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

	buf_warped_size = 0;

	debugImageWeights = cv::Mat(height,width,CV_8UC3);
	debugImageResiduals = cv::Mat(height,width,CV_8UC3);
	debugImageSecondFrame = cv::Mat(height,width,CV_8UC3);
	debugImageOldImageWarped = cv::Mat(height,width,CV_8UC3);
	debugImageOldImageSource = cv::Mat(height,width,CV_8UC3);



	lastResidual = 0;
	iterationNumber = 0;
	pointUsage = 0;
	lastGoodCount = lastBadCount = 0;

	diverged = false;
}

SE3Tracker::~SE3Tracker()
{
	debugImageResiduals.release();
	debugImageWeights.release();
	debugImageSecondFrame.release();
	debugImageOldImageSource.release();
	debugImageOldImageWarped.release();

	Eigen::internal::aligned_free((void*)buf_warped_residual);
	Eigen::internal::aligned_free((void*)buf_warped_dx);
	Eigen::internal::aligned_free((void*)buf_warped_dy);
	Eigen::internal::aligned_free((void*)buf_warped_x);
	Eigen::internal::aligned_free((void*)buf_warped_y);
	Eigen::internal::aligned_free((void*)buf_warped_z);

	Eigen::internal::aligned_free((void*)buf_d);
	Eigen::internal::aligned_free((void*)buf_idepthVar);
	Eigen::internal::aligned_free((void*)buf_weight_p);
}



// tracks a frame.
// first_frame has depth, second_frame DOES NOT have depth.
SE3 SE3Tracker::trackFrameOnPermaref(Frame*reference,
        Frame*frame,
        SE3 referenceToFrameOrg)
{

	Sophus::SE3f referenceToFrame = referenceToFrameOrg.cast<float>();

//	boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
//	boost::unique_lock<boost::mutex> lock2 = boost::unique_lock<boost::mutex>(reference->permaRef_mutex);

    affineEstimation_a = 1;
    affineEstimation_b = 0;

	LGS6 ls;
	diverged = false;
	trackingWasGood = true;

    callOptimized(calcResidualAndBuffers, (reference->permaRef_posData, reference->permaRef_colorAndVarData, 0, reference->permaRefNumPts, frame, referenceToFrame, QUICK_KF_CHECK_LVL));
	if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width>>QUICK_KF_CHECK_LVL)*(height>>QUICK_KF_CHECK_LVL))
	{
//        printf("reference->permaRefNumPts=%d\n", reference->permaRefNumPts) ;
//        printf("buf_warped_size=%d\n",buf_warped_size ) ;
//        std::cout << referenceToFrameOrg.rotationMatrix() << std::endl ;
//        std::cout << referenceToFrameOrg.translation() << std::endl ;
		diverged = true;
		trackingWasGood = false;
		return SE3();
	}
	if(useAffineLightningEstimation)
	{
		affineEstimation_a = affineEstimation_a_lastIt;
		affineEstimation_b = affineEstimation_b_lastIt;
	}


	float lastErr = callOptimized(calcWeightsAndResidual,(referenceToFrame));

	float LM_lambda = settings.lambdaInitialTestTrack;

	for(int iteration=0; iteration < settings.maxItsTestTrack; iteration++)
	{

		callOptimized(calculateWarpUpdate,(ls));


		int incTry=0;
		while(true)
		{
			// solve LS system with current lambda
			Vector6 b = -ls.b;
			Matrix6x6 A = ls.A;
			for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
			Vector6 inc = A.ldlt().solve(b);
			incTry++;

			// apply increment. pretty sure this way round is correct, but hard to test.
			Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;

			// re-evaluate residual
            //puts("e") ;
            Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
            Eigen::Vector3f transVec = referenceToFrame.translation();
//            std::cout << rotMat << "\n" ;
//            std::cout << transVec.transpose() << "\n" ;
            calcResidualAndBuffers(reference->permaRef_posData,
                                   reference->permaRef_colorAndVarData,
                                   0,
                                   reference->permaRefNumPts,
                                   frame,
                                   new_referenceToFrame,
                                   QUICK_KF_CHECK_LVL) ;
//            callOptimized(calcResidualAndBuffers, (reference->permaRef_posData,
//                                                   reference->permaRef_colorAndVarData,
//                                                   0,
//                                                   reference->permaRefNumPts,
//                                                   frame,
//                                                   new_referenceToFrame,
//                                                   QUICK_KF_CHECK_LVL) );
            if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width>>QUICK_KF_CHECK_LVL)*(height>>QUICK_KF_CHECK_LVL))
			{
//                printf("buf_warped_size=%d\n",buf_warped_size ) ;
//                std::cout << referenceToFrameOrg.rotationMatrix() << std::endl ;
//                std::cout << referenceToFrameOrg.translation() << std::endl ;
				diverged = true;
				trackingWasGood = false;
				return SE3();
			}
			float error = callOptimized(calcWeightsAndResidual,(new_referenceToFrame));

			// accept inc?
			if(error < lastErr)
			{
				// accept inc
				referenceToFrame = new_referenceToFrame;
				if(useAffineLightningEstimation)
				{
					affineEstimation_a = affineEstimation_a_lastIt;
					affineEstimation_b = affineEstimation_b_lastIt;
				}
				// converged?
				if(error / lastErr > settings.convergenceEpsTestTrack)
					iteration = settings.maxItsTestTrack;


				lastErr = error;


				if(LM_lambda <= 0.2)
					LM_lambda = 0;
				else
					LM_lambda *= settings.lambdaSuccessFac;

				break;
			}
			else
			{
				if(!(inc.dot(inc) > settings.stepSizeMinTestTrack))
				{
					iteration = settings.maxItsTestTrack;
					break;
				}

				if(LM_lambda == 0)
					LM_lambda = 0.2;
				else
					LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
			}
		}
	}

	lastResidual = lastErr;

	trackingWasGood = !diverged
			&& lastGoodCount / (frame->width(QUICK_KF_CHECK_LVL)*frame->height(QUICK_KF_CHECK_LVL)) > MIN_GOODPERALL_PIXEL
			&& lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

	return toSophus(referenceToFrame);
}




// tracks a frame.
// first_frame has depth, second_frame DOES NOT have depth.
SE3 SE3Tracker::trackFrame(TrackingReference* reference,
        Frame*frame,
        const SE3& RefToFrame_initialEstimate)
{
	diverged = false;
	trackingWasGood = true;
    affineEstimation_a = 1;
    affineEstimation_b = 0;

	// ============ track frame ============
    Sophus::SE3f referenceToFrame = RefToFrame_initialEstimate.cast<float>();
	LGS6 ls;


	int numCalcResidualCalls[PYRAMID_LEVELS];
	int numCalcWarpUpdateCalls[PYRAMID_LEVELS];
	float last_residual = 0;
	for(int lvl=SE3TRACKING_MAX_LEVEL-1;lvl >= SE3TRACKING_MIN_LEVEL;lvl--)
	{
		numCalcResidualCalls[lvl] = 0;
		numCalcWarpUpdateCalls[lvl] = 0;

		reference->makePointCloud(lvl);

        callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, referenceToFrame, lvl ));
		if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN * (width>>lvl)*(height>>lvl))
		{
			diverged = true;
			trackingWasGood = false;
            return RefToFrame_initialEstimate;
		}

		if(useAffineLightningEstimation)
		{
			affineEstimation_a = affineEstimation_a_lastIt;
			affineEstimation_b = affineEstimation_b_lastIt;
		}
		float lastErr = callOptimized(calcWeightsAndResidual,(referenceToFrame));

		numCalcResidualCalls[lvl]++;


		float LM_lambda = settings.lambdaInitial[lvl];

		for(int iteration=0; iteration < settings.maxItsPerLvl[lvl]; iteration++)
		{

			callOptimized(calculateWarpUpdate,(ls));

			numCalcWarpUpdateCalls[lvl]++;

			iterationNumber = iteration;

			int incTry=0;
			while(true)
			{
                //printf("incTry=%d buf_warped_size=%d\n", incTry, buf_warped_size ) ;
				// solve LS system with current lambda
				Vector6 b = -ls.b;
				Matrix6x6 A = ls.A;
				for(int i=0;i<6;i++) A(i,i) *= 1+LM_lambda;
				Vector6 inc = A.ldlt().solve(b);
				incTry++;

				// apply increment. pretty sure this way round is correct, but hard to test.
				Sophus::SE3f new_referenceToFrame = Sophus::SE3f::exp((inc)) * referenceToFrame;
				//Sophus::SE3f new_referenceToFrame = referenceToFrame * Sophus::SE3f::exp((inc));

//                                    std::cout << new_referenceToFrame.rotationMatrix() << std::endl ;
//                                    std::cout << new_referenceToFrame.translation() << std::endl ;

				// re-evaluate residual
                callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, new_referenceToFrame, lvl));
				if(buf_warped_size < MIN_GOODPERALL_PIXEL_ABSMIN* (width>>lvl)*(height>>lvl))
				{
					diverged = true;
					trackingWasGood = false;
                    printf("buf_warped_size=%d\n",buf_warped_size ) ;
                    std::cout << new_referenceToFrame.rotationMatrix() << std::endl ;
//                    std::cout << new_referenceToFrame.translation() << std::endl ;
                    return RefToFrame_initialEstimate;
				}

				float error = callOptimized(calcWeightsAndResidual,(new_referenceToFrame));
				numCalcResidualCalls[lvl]++;


				// accept inc?
				if(error < lastErr)
				{
					// accept inc
					referenceToFrame = new_referenceToFrame;
					if(useAffineLightningEstimation)
					{
						affineEstimation_a = affineEstimation_a_lastIt;
						affineEstimation_b = affineEstimation_b_lastIt;
					}


					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						// debug output
						printf("(%d-%d): ACCEPTED increment of %f with lambda %.1f, residual: %f -> %f\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr, error);

						printf("         p=%.4f %.4f %.4f %.4f %.4f %.4f\n",
								referenceToFrame.log()[0],referenceToFrame.log()[1],referenceToFrame.log()[2],
								referenceToFrame.log()[3],referenceToFrame.log()[4],referenceToFrame.log()[5]);
					}

					// converged?
					if(error / lastErr > settings.convergenceEps[lvl])
					{
                        if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (last residual reduction too small).\n",
									lvl,iteration);
						}
						iteration = settings.maxItsPerLvl[lvl];
					}

					last_residual = lastErr = error;


					if(LM_lambda <= 0.2)
						LM_lambda = 0;
					else
						LM_lambda *= settings.lambdaSuccessFac;

					break;
				}
				else
				{
					if(enablePrintDebugInfo && printTrackingIterationInfo)
					{
						printf("(%d-%d): REJECTED increment of %f with lambda %.1f, (residual: %f -> %f)\n",
								lvl,iteration, sqrt(inc.dot(inc)), LM_lambda, lastErr, error);
					}

					if(!(inc.dot(inc) > settings.stepSizeMin[lvl]))
					{
						if(enablePrintDebugInfo && printTrackingIterationInfo)
						{
							printf("(%d-%d): FINISHED pyramid level (stepsize too small).\n",
									lvl,iteration);
                        }
						iteration = settings.maxItsPerLvl[lvl];
						break;
					}

					if(LM_lambda == 0)
						LM_lambda = 0.2;
					else
						LM_lambda *= std::pow(settings.lambdaFailFac, incTry);
				}
			}
		}
	}


	lastResidual = last_residual;

	trackingWasGood = !diverged
			&& lastGoodCount / (frame->width(SE3TRACKING_MIN_LEVEL)*frame->height(SE3TRACKING_MIN_LEVEL)) > MIN_GOODPERALL_PIXEL
			&& lastGoodCount / (lastGoodCount + lastBadCount) > MIN_GOODPERGOODBAD_PIXEL;

    if(trackingWasGood){
        reference->keyframe->numFramesTrackedOnThis++;
        //printf("reference->keyframe->numFramesTrackedOnThis=%d\n", reference->keyframe->numFramesTrackedOnThis ) ;
    }

    frame->initialTrackedResidual = lastResidual / pointUsage;
    return toSophus(referenceToFrame) ;
}

#if defined(ENABLE_SSE)
float SE3Tracker::calcWeightsAndResidualSSE(
		const Sophus::SE3f& referenceToFrame)
{
	const __m128 txs = _mm_set1_ps((float)(referenceToFrame.translation()[0]));
	const __m128 tys = _mm_set1_ps((float)(referenceToFrame.translation()[1]));
	const __m128 tzs = _mm_set1_ps((float)(referenceToFrame.translation()[2]));

	const __m128 zeros = _mm_set1_ps(0.0f);
	const __m128 ones = _mm_set1_ps(1.0f);


	const __m128 depthVarFacs = _mm_set1_ps((float)settings.var_weight);// float depthVarFac = var_weight;	// the depth var is over-confident. this is a constant multiplier to remedy that.... HACK
	const __m128 sigma_i2s = _mm_set1_ps((float)cameraPixelNoise2);


	const __m128 huber_res_ponlys = _mm_set1_ps((float)(settings.huber_d/2));

	__m128 sumResP = zeros;


	float sumRes = 0;

	for(int i=0;i<buf_warped_size-3;i+=4)
	{
//		float px = *(buf_warped_x+i);	// x'
//		float py = *(buf_warped_y+i);	// y'
//		float pz = *(buf_warped_z+i);	// z'
//		float d = *(buf_d+i);	// d
//		float rp = *(buf_warped_residual+i); // r_p
//		float gx = *(buf_warped_dx+i);	// \delta_x I
//		float gy = *(buf_warped_dy+i);  // \delta_y I
//		float s = depthVarFac * *(buf_idepthVar+i);	// \sigma_d^2


		// calc dw/dd (first 2 components):
		__m128 pzs = _mm_load_ps(buf_warped_z+i);	// z'
		__m128 pz2ds = _mm_rcp_ps(_mm_mul_ps(_mm_mul_ps(pzs, pzs), _mm_load_ps(buf_d+i)));  // 1 / (z' * z' * d)
		__m128 g0s = _mm_sub_ps(_mm_mul_ps(pzs, txs), _mm_mul_ps(_mm_load_ps(buf_warped_x+i), tzs));
		g0s = _mm_mul_ps(g0s,pz2ds); //float g0 = (tx * pz - tz * px) / (pz*pz*d);

		 //float g1 = (ty * pz - tz * py) / (pz*pz*d);
		__m128 g1s = _mm_sub_ps(_mm_mul_ps(pzs, tys), _mm_mul_ps(_mm_load_ps(buf_warped_y+i), tzs));
		g1s = _mm_mul_ps(g1s,pz2ds);

		 // float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		__m128 drpdds = _mm_add_ps(
				_mm_mul_ps(g0s, _mm_load_ps(buf_warped_dx+i)),
				_mm_mul_ps(g1s, _mm_load_ps(buf_warped_dy+i)));

		 //float w_p = 1.0f / (sigma_i2 + s * drpdd * drpdd);
		__m128 w_ps = _mm_rcp_ps(_mm_add_ps(sigma_i2s,
				_mm_mul_ps(drpdds,
						_mm_mul_ps(drpdds,
								_mm_mul_ps(depthVarFacs,
										_mm_load_ps(buf_idepthVar+i))))));


		//float weighted_rp = fabs(rp*sqrtf(w_p));
		__m128 weighted_rps = _mm_mul_ps(_mm_load_ps(buf_warped_residual+i),
				_mm_sqrt_ps(w_ps));
		weighted_rps = _mm_max_ps(weighted_rps, _mm_sub_ps(zeros,weighted_rps));


		//float wh = fabs(weighted_rp < huber_res_ponly ? 1 : huber_res_ponly / weighted_rp);
		__m128 whs = _mm_cmplt_ps(weighted_rps, huber_res_ponlys);	// bitmask 0xFFFFFFFF for 1, 0x000000 for huber_res_ponly / weighted_rp
		whs = _mm_or_ps(
				_mm_and_ps(whs, ones),
				_mm_andnot_ps(whs, _mm_mul_ps(huber_res_ponlys, _mm_rcp_ps(weighted_rps))));




		// sumRes.sumResP += wh * w_p * rp*rp;
		if(i+3 < buf_warped_size)
			sumResP = _mm_add_ps(sumResP,
					_mm_mul_ps(whs, _mm_mul_ps(weighted_rps, weighted_rps)));

		// *(buf_weight_p+i) = wh * w_p;
		_mm_store_ps(buf_weight_p+i, _mm_mul_ps(whs, w_ps) );
	}
	sumRes = SSEE(sumResP,0) + SSEE(sumResP,1) + SSEE(sumResP,2) + SSEE(sumResP,3);

	return sumRes / ((buf_warped_size >> 2)<<2);
}
#endif

float SE3Tracker::calcWeightsAndResidual(
		const Sophus::SE3f& referenceToFrame)
{
	float tx = referenceToFrame.translation()[0];
	float ty = referenceToFrame.translation()[1];
	float tz = referenceToFrame.translation()[2];

	float sumRes = 0;

	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);	// x'
		float py = *(buf_warped_y+i);	// y'
		float pz = *(buf_warped_z+i);	// z'
		float d = *(buf_d+i);	// d
		float rp = *(buf_warped_residual+i); // r_p
		float gx = *(buf_warped_dx+i);	// \delta_x I
		float gy = *(buf_warped_dy+i);  // \delta_y I
		float s = settings.var_weight * *(buf_idepthVar+i);	// \sigma_d^2


		// calc dw/dd (first 2 components):
		float g0 = (tx * pz - tz * px) / (pz*pz*d);
		float g1 = (ty * pz - tz * py) / (pz*pz*d);


		// calc w_p
		float drpdd = gx * g0 + gy * g1;	// ommitting the minus
		float w_p = 1.0f / ((cameraPixelNoise2) + s * drpdd * drpdd);

		float weighted_rp = fabs(rp*sqrtf(w_p));

		float wh = fabs(weighted_rp < (settings.huber_d/2) ? 1 : (settings.huber_d/2) / weighted_rp);

		sumRes += wh * w_p * rp*rp;


		*(buf_weight_p+i) = wh * w_p;
	}

	return sumRes / buf_warped_size;
}

#if defined(ENABLE_SSE)
float SE3Tracker::calcResidualAndBuffersSSE(
		const Eigen::Vector3f* refPoint,
		const Eigen::Vector2f* refColVar,
		int* idxBuf,
		int refNum,
		Frame* frame,
		const Sophus::SE3f& referenceToFrame,
        int level)
{
    return calcResidualAndBuffers(refPoint, refColVar, idxBuf, refNum, frame, referenceToFrame, level);
}
#endif


float SE3Tracker::calcResidualAndBuffers(
		const Eigen::Vector3f* refPoint,
		const Eigen::Vector2f* refColVar,
		int* idxBuf,
		int refNum,
		Frame* frame,
		const Sophus::SE3f& referenceToFrame,
        int level)
{


	int w = frame->width(level);
	int h = frame->height(level);
	Eigen::Matrix3f KLvl = frame->K(level);
	float fx_l = KLvl(0,0);
	float fy_l = KLvl(1,1);
	float cx_l = KLvl(0,2);
	float cy_l = KLvl(1,2);

	Eigen::Matrix3f rotMat = referenceToFrame.rotationMatrix();
	Eigen::Vector3f transVec = referenceToFrame.translation();
	
	const Eigen::Vector3f* refPoint_max = refPoint + refNum;


	const Eigen::Vector4f* frame_gradients = frame->gradients(level);

	int idx=0;

	float sumResUnweighted = 0;

	bool* isGoodOutBuffer = idxBuf != 0 ? frame->refPixelWasGood() : 0;

	int goodCount = 0;
	int badCount = 0;

	float sumSignedRes = 0;



	float sxx=0,syy=0,sx=0,sy=0,sw=0;

	float usageCount = 0;

	for(;refPoint<refPoint_max; refPoint++, refColVar++, idxBuf++)
	{

		Eigen::Vector3f Wxp = rotMat * (*refPoint) + transVec;
		float u_new = (Wxp[0]/Wxp[2])*fx_l + cx_l;
		float v_new = (Wxp[1]/Wxp[2])*fy_l + cy_l;

		// step 1a: coordinates have to be in image:
		// (inverse test to exclude NANs)
        if(!(u_new > 2 && v_new > 2 && u_new < w-3 && v_new < h-3))
		{
			if(isGoodOutBuffer != 0)
				isGoodOutBuffer[*idxBuf] = false;
			continue;
		}

        Eigen::Vector3f resInterp = getInterpolatedElement43(frame_gradients, u_new, v_new, w);

		float c1 = affineEstimation_a * (*refColVar)[0] + affineEstimation_b;
		float c2 = resInterp[2];
		float residual = c1 - c2;

		float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
		sxx += c1*c1*weight;
		syy += c2*c2*weight;
		sx += c1*weight;
		sy += c2*weight;
		sw += weight;

		bool isGood = residual*residual / (MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*(resInterp[0]*resInterp[0] + resInterp[1]*resInterp[1])) < 1;

		if(isGoodOutBuffer != 0)
			isGoodOutBuffer[*idxBuf] = isGood;


		*(buf_warped_x+idx) = Wxp(0);
		*(buf_warped_y+idx) = Wxp(1);
		*(buf_warped_z+idx) = Wxp(2);

		*(buf_warped_dx+idx) = fx_l * resInterp[0];
		*(buf_warped_dy+idx) = fy_l * resInterp[1];
		*(buf_warped_residual+idx) = residual;

		*(buf_d+idx) = 1.0f / (*refPoint)[2];
		*(buf_idepthVar+idx) = (*refColVar)[1];
		idx++;


		if(isGood)
		{
			sumResUnweighted += residual*residual;
			sumSignedRes += residual;
			goodCount++;
		}
		else
			badCount++;

		float depthChange = (*refPoint)[2] / Wxp[2];	// if depth becomes larger: pixel becomes "smaller", hence count it less.
		usageCount += depthChange < 1 ? depthChange : 1;

	}

	buf_warped_size = idx;

	pointUsage = usageCount / (float)refNum;
	lastGoodCount = goodCount;
	lastBadCount = badCount;
	lastMeanRes = sumSignedRes / goodCount;

	affineEstimation_a_lastIt = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
	affineEstimation_b_lastIt = (sy - affineEstimation_a_lastIt*sx)/sw;

	return sumResUnweighted / goodCount;
}


#if defined(ENABLE_SSE)
void SE3Tracker::calculateWarpUpdateSSE( LGS6 &ls)
{
	ls.initialize(width*height);

//	printf("wupd SSE\n");
	for(int i=0;i<buf_warped_size-3;i+=4)
	{

		__m128 val1, val2, val3, val4;
		__m128 J61, J62, J63, J64, J65, J66;

		// redefine pz
		__m128 pz = _mm_load_ps(buf_warped_z+i);
		pz = _mm_rcp_ps(pz);						// pz := 1/z


		__m128 gx = _mm_load_ps(buf_warped_dx+i);
		val1 = _mm_mul_ps(pz, gx);			// gx / z => SET [0]
		//v[0] = z*gx;
		J61 = val1;



		__m128 gy = _mm_load_ps(buf_warped_dy+i);
		val1 = _mm_mul_ps(pz, gy);					// gy / z => SET [1]
		//v[1] = z*gy;
		J62 = val1;


		__m128 px = _mm_load_ps(buf_warped_x+i);
		val1 = _mm_mul_ps(px, gy);
		val1 = _mm_mul_ps(val1, pz);	//  px * gy * z
		__m128 py = _mm_load_ps(buf_warped_y+i);
		val2 = _mm_mul_ps(py, gx);
		val2 = _mm_mul_ps(val2, pz);	//  py * gx * z
		val1 = _mm_sub_ps(val1, val2);  // px * gy * z - py * gx * z => SET [5]
		//v[5] = -py * z * gx +  px * z * gy;
		J66 = val1;


		// redefine pz
		pz = _mm_mul_ps(pz,pz); 		// pz := 1/(z*z)

		// will use these for the following calculations a lot.
		val1 = _mm_mul_ps(px, gx);
		val1 = _mm_mul_ps(val1, pz);		// px * z_sqr * gx
		val2 = _mm_mul_ps(py, gy);
		val2 = _mm_mul_ps(val2, pz);		// py * z_sqr * gy


		val3 = _mm_add_ps(val1, val2);
		val3 = _mm_sub_ps(_mm_setr_ps(0,0,0,0),val3);	//-px * z_sqr * gx -py * z_sqr * gy
		//v[2] = -px * z_sqr * gx -py * z_sqr * gy;	=> SET [2]
		J63 = val3;


		val3 = _mm_mul_ps(val1, py); // px * z_sqr * gx * py
		val4 = _mm_add_ps(gy, val3); // gy + px * z_sqr * gx * py
		val3 = _mm_mul_ps(val2, py); // py * py * z_sqr * gy
		val4 = _mm_add_ps(val3, val4); // gy + px * z_sqr * gx * py + py * py * z_sqr * gy
		val4 = _mm_sub_ps(_mm_setr_ps(0,0,0,0),val4); //val4 = -val4.
		//v[3] = -px * py * z_sqr * gx +
		//       -py * py * z_sqr * gy +
		//       -gy;		=> SET [3]
		J64 = val4;


		val3 = _mm_mul_ps(val1, px); // px * px * z_sqr * gx
		val4 = _mm_add_ps(gx, val3); // gx + px * px * z_sqr * gx
		val3 = _mm_mul_ps(val2, px); // px * py * z_sqr * gy
		val4 = _mm_add_ps(val4, val3); // gx + px * px * z_sqr * gx + px * py * z_sqr * gy
		//v[4] = px * px * z_sqr * gx +
		//	   px * py * z_sqr * gy +
		//	   gx;				=> SET [4]
		J65 = val4;

		if(i+3<buf_warped_size)
		{
			ls.updateSSE(J61, J62, J63, J64, J65, J66, _mm_load_ps(buf_warped_residual+i), _mm_load_ps(buf_weight_p+i));
		}
		else
		{
			for(int k=0;i+k<buf_warped_size;k++)
			{
				Vector6 v6;
				v6 << SSEE(J61,k),SSEE(J62,k),SSEE(J63,k),SSEE(J64,k),SSEE(J65,k),SSEE(J66,k);
				ls.update(v6, *(buf_warped_residual+i+k), *(buf_weight_p+i+k));
			}
		}


	}

	// solve ls
	ls.finish();

}
#endif



void SE3Tracker::calculateWarpUpdate(
		LGS6 &ls)
{
//	weightEstimator.reset();
//	weightEstimator.estimateDistribution(buf_warped_residual, buf_warped_size);
//	weightEstimator.calcWeights(buf_warped_residual, buf_warped_weights, buf_warped_size);
//
	ls.initialize(width*height);
	for(int i=0;i<buf_warped_size;i++)
	{
		float px = *(buf_warped_x+i);
		float py = *(buf_warped_y+i);
		float pz = *(buf_warped_z+i);
		float r =  *(buf_warped_residual+i);
		float gx = *(buf_warped_dx+i);
		float gy = *(buf_warped_dy+i);
		// step 3 + step 5 comp 6d error vector

		float z = 1.0f / pz;
		float z_sqr = 1.0f / (pz*pz);
		Vector6 v;
		v[0] = z*gx + 0;
		v[1] = 0 +         z*gy;
		v[2] = (-px * z_sqr) * gx +
			  (-py * z_sqr) * gy;
		v[3] = (-px * py * z_sqr) * gx +
			  (-(1.0 + py * py * z_sqr)) * gy;
		v[4] = (1.0 + px * px * z_sqr) * gx +
			  (px * py * z_sqr) * gy;
		v[5] = (-py * z) * gx +
			  (px * z) * gy;

		// step 6: integrate into A and b:
		ls.update(v, r, *(buf_weight_p+i));
	}

	// solve ls
	ls.finish();
	//result = ls.A.ldlt().solve(ls.b);


}



}

