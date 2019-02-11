/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *      02-10-19    :   Modified Version by Russell Marineau
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include <thread>
#include <mutex>
#include "stdio-wrapper.h"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
/*---
  static int *stages_array;
  static int *rectangles_array;
  static int *weights_array;
  static int *alpha1_array;
  static int *alpha2_array;
  static int *tree_thresh_array;
  static int *stages_thresh_array;
  static int **scaled_rectangles_array;
  ---*/
  int *stages_array;
int *rectangles_array;
int *weights_array;
int *alpha1_array;
int *alpha2_array;
int *tree_thresh_array;
int *stages_thresh_array;
int **scaled_rectangles_array;

std::vector<MyRect> allCandidates;
//int **stage_sum;

//CONSTANT VARIABLES MUST BE SIZED AT COMPILE TIME SO CHOOSE A SUFFICIANTLY LARGE NODES VALUE
//LIMIT OF 16000 ints
#define NODES 3000
#define STAGES 100
 __constant__ int dalpha1_array[NODES];
 __constant__ int dalpha2_array[NODES];
 __constant__ int dstages_thresh_array[STAGES];
 __constant__ int dweights_array[NODES*3];
 __constant__ int dstages_array[STAGES];

__device__ int dtree_thresh_array[NODES];
//__device__ int stage_sum[300];

//int clock_counter = 0;
//float n_features = 0;


int iter_counter = 0;
int all_nodes;
std::mutex push_back_mutex;

/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

void ScaleImage_Invoker(myCascade* cascade, float factor, int sum_row, int sum_col, int** new_scaled_rect_array, cudaStream_t thisStream);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);


/* sets images for haar classifier cascade */
int** setImageForCascadeClassifier(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum, cudaStream_t thisStream);

/* rounding function */
__host__ __device__ inline  int  myRound( float value )
{
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

void doThreadWork(MyImage* img, MySize* minSize, MySize* maxSize, myCascade* cascade, float factor, MySize* winSize, MySize* sz, MySize* sz1)
{
	/* if a minSize different from the original detection window is specified, continue to the next scaling */
	if (winSize->width < minSize->width || winSize->height < minSize->height)
		return;
	cudaSetDevice(0);
	cudaStream_t thisStream;
	cudaStreamCreate(&thisStream);

	MyImage *img1 = new MyImage;
	MyIntImage *sum1 = new MyIntImage;
	MyIntImage *sqsum1 = new MyIntImage;
	cudaStreamAttachMemAsync(thisStream, img1);
	//cudaMemPrefetchAsync(img1, sizeof(MyImage), 0, thisStream);
	cudaStreamAttachMemAsync(thisStream, sum1);
	//cudaMemPrefetchAsync(sum1, sizeof(MyIntImage), 0, thisStream);
	cudaStreamAttachMemAsync(thisStream, sqsum1);
	//cudaMemPrefetchAsync(sqsum1, sizeof(MyIntImage), 0, thisStream);

	/* malloc for img1: unsigned char */
	createImage(img->width, img->height, img1, thisStream);
	/* malloc for sum1: unsigned char */
	createSumImage(img->width, img->height, sum1, thisStream);
	/* malloc for sqsum1: unsigned char */
	createSumImage(img->width, img->height, sqsum1, thisStream);


	/*************************************
	* Set the width and height of
	* img1: normal image (unsigned char)
	* sum1: integral image (int)
	* sqsum1: squared integral image (int)
	* see image.c for details
	************************************/
	setImage(sz->width, sz->height, img1);
	setSumImage(sz->width, sz->height, sum1);
	setSumImage(sz->width, sz->height, sqsum1);

	/***************************************
	* Compute-intensive step:
	* building image pyramid by downsampling
	* downsampling using nearest neighbor
	**************************************/
	nearestNeighbor(img, img1);

	/***************************************************
	* Compute-intensive step:
	* At each scale of the image pyramid,
	* compute a new integral and squared integral image
	***************************************************/
	integralImages(img1, sum1, sqsum1);

	/* sets images for haar classifier cascade */
	/**************************************************
	* Note:
	* Summing pixels within a haar window is done by
	* using four corners of the integral image:
	* http://en.wikipedia.org/wiki/Summed_area_table
	*
	* This function loads the four corners,
	* but does not do compuation based on four coners.
	* The computation is done next in ScaleImage_Invoker
	*************************************************/
	myCascade *tempcascade = new myCascade;
	cudaStreamAttachMemAsync(thisStream, tempcascade);
	//cudaMemPrefetchAsync(tempcascade, sizeof(myCascade), 0, thisStream);

	// classifier properties
	tempcascade->n_stages = 25;
	tempcascade->total_nodes = 2913;
	tempcascade->orig_window_size.height = 24;
	tempcascade->orig_window_size.width = 24;

	int** new_scaled_rect_array = setImageForCascadeClassifier(tempcascade, sum1, sqsum1, thisStream);

	//cudaStreamAttachMemAsync(thisStream, new_scaled_rect_array);

	/* print out for each scale of the image pyramid */
	//printf("detecting faces, iter := %d\n", iter_counter);

	/****************************************************
	* Process the current scale with the cascaded fitler.
	* The main computations are invoked by this function.
	* Optimization oppurtunity:
	* the same cascade filter is invoked each time
	***************************************************/

	ScaleImage_Invoker(tempcascade, factor, sz->height, sz->width, new_scaled_rect_array, thisStream);
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* img, MySize* minSize, MySize* maxSize, myCascade* cascade,
                                   float scaleFactor, int minNeighbors)
{
	//cudaSetDevice(0);
    /* group overlaping windows */
    const float GROUP_EPS = 0.4f;
    /* pointer to input image */
    /***********************************
     * create structs for images
     * see haar.h for details
     * img1: normal image (unsigned char)
     * sum1: integral image (int)
     * sqsum1: square integral image (int)
     **********************************/
    

    /********************************************************
     * allCandidates is the preliminaray face candidate,
     * which will be refined later.
     *
     * std::vector is a sequential container
     * http://en.wikipedia.org/wiki/Sequence_container_(C++)
     *
     * Each element of the std::vector is a "MyRect" struct
     * MyRect struct keeps the info of a rectangle (see haar.h)
     * The rectangle contains one face candidate
     *****************************************************/
	allCandidates.clear();

    /* scaling factor */
    float factor;

    /* maxSize */
    if( maxSize->height == 0 || maxSize->width == 0 )
        {
            maxSize->height = img->height;
            maxSize->width = img->width;
        }

    /* window size of the training set */
    MySize winSize0 = cascade->orig_window_size;

    

    /* initial scaling factor */
    factor = 1;

	std::vector<std::thread> threads;
	iter_counter = 0;

    /* iterate over the image pyramid */
    for( factor = 1; ; factor *= scaleFactor )
        {
		/* size of the image scaled up */
		MySize* winSize = new MySize;
		winSize->width = myRound(winSize0.width*factor);
		winSize->height = myRound(winSize0.height*factor);
		//printf(" w=%d h=%d ", winSize.width, winSize.height); // 24x24 when factor=1;

		/* size of the image scaled down (from bigger to smaller) */
		MySize* sz = new MySize;
		sz->width = int(float(img->width) / factor);
		sz->height = int(float(img->height) / factor);
		//printf("img_w=%d img_h=%d factor=%f", img->width, img->height, factor);
		//printf(" w=%d h=%d ", sz.width, sz.height);

		/* difference between sizes of the scaled image and the original detection window */
		MySize* sz1 = new MySize;
		sz1->width = sz->width - winSize0.width;
		sz1->height = sz->height - winSize0.height;

		/* if the actual scaled image is smaller than the original detection window, break */
		if (sz1->width < 0 || sz1->height < 0)
			break;

				threads.push_back(std::thread(doThreadWork, img, minSize, maxSize, cascade, factor, winSize, sz, sz1));
				/* iteration counter */
				iter_counter++;
                //ScaleImage_Invoker(tempcascade, factor, sz->height, sz->width, allCandidates, new_scaled_rect_array);
        } /* end of the factor loop, finish all scales in pyramid*/
    
	for (int i = 0; i < iter_counter; i++)
	{
		threads[i].join();
	}

    if( minNeighbors != 0)
        {
            groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
        }

    //freeImage(img1);
    //freeSumImage(sum1);
    //freeSumImage(sqsum1);
    return allCandidates;
}



/***********************************************
 * Note:
 * The int_sqrt is softwar integer squre root.
 * GPU has hardware for floating squre root (sqrtf).
 * In GPU, it is wise to convert an int variable
 * into floating point, and use HW sqrtf function.
 * More info:
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
 **********************************************/
/*****************************************************
 * The int_sqrt is only used in runCascadeClassifier
 * If you want to replace int_sqrt with HW sqrtf in GPU,
 * simple look into the runCascadeClassifier function.
 *****************************************************/
__device__ unsigned int int_sqrt (unsigned int value)
{
    int i;
    unsigned int a = 0, b = 0, c = 0;
    for (i=0; i < (32 >> 1); i++)
        {
            c<<= 2;
#define UPPERBITS(value) (value>>30)
            c += UPPERBITS(value);
#undef UPPERBITS
            value <<= 2;
            a <<= 1;
            b = (a<<1) | 1;
            if (c >= b)
                {
                    c -= b;
                    a++;
                }
        }
    return a;
}


int ** setImageForCascadeClassifier( myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum, cudaStream_t thisStream)
{
	//cudaSetDevice(0);
    int i, j, k;
    MyRect equRect;
    int r_index = 0;
    int w_index = 0;
    MyRect tr;

	int ** new_scaled_rect_array;

	gpuErrchk(cudaMallocManaged(&new_scaled_rect_array, sizeof(int*)*all_nodes*(12 + 0)));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, new_scaled_rect_array);
	//cudaMemPrefetchAsync(new_scaled_rect_array, sizeof(int*)*all_nodes*(12 + 0), 0, thisStream);

    cascade->sum = *sum;
    cascade->sqsum = *sqsum;

    equRect.x = equRect.y = 0;
    equRect.width = cascade->orig_window_size.width;
    equRect.height = cascade->orig_window_size.height;

    cascade->inv_window_area = equRect.width*equRect.height;

	gpuErrchk(cudaMallocManaged(&(cascade->p0), sizeof(int) * sum->width * sum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->p0));
	//cudaMemPrefetchAsync(cascade->p0, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->p1), sizeof(int) * sum->width * sum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->p1));
	//cudaMemPrefetchAsync(cascade->p1, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->p2), sizeof(int) * sum->width * sum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->p2));
	//cudaMemPrefetchAsync(cascade->p2, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->p3), sizeof(int) * sum->width * sum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->p3));
	//cudaMemPrefetchAsync(cascade->p3, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->pq0), sizeof(int) * sqsum->width * sqsum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->pq0));
	//cudaMemPrefetchAsync(cascade->pq0, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->pq1), sizeof(int) * sqsum->width * sqsum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->pq1));
	//cudaMemPrefetchAsync(cascade->pq1, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->pq2), sizeof(int) * sqsum->width * sqsum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->pq2));
	//cudaMemPrefetchAsync(cascade->pq2, sizeof(int) * sum->width * sum->height, 0, thisStream);
	gpuErrchk(cudaMallocManaged(&(cascade->pq3), sizeof(int) * sqsum->width * sqsum->height));
	//cudaDeviceSynchronize();
	cudaStreamAttachMemAsync(thisStream, (cascade->pq3));
	//cudaMemPrefetchAsync(cascade->pq3, sizeof(int) * sum->width * sum->height, 0, thisStream);

    cascade->p0 = (sum->data) ;
	//gpuErrchk(cudaMemPrefetchAsync(cascade->p0, sizeof(int) * sum->width * sum->height, 0, thisStream));
    cascade->p1 = (sum->data +  equRect.width - 1) ;
	//cudaMemPrefetchAsync(cascade->p1, sizeof(int) * sum->width * sum->height, 0, thisStream);
    cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
	//cudaMemPrefetchAsync(cascade->p2, sizeof(int) * sum->width * sum->height, 0, thisStream);
    cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
	//cudaMemPrefetchAsync(cascade->p3, sizeof(int) * sum->width * sum->height, 0, thisStream);
    cascade->pq0 = (sqsum->data);
	//cudaMemPrefetchAsync(cascade->pq0, sizeof(int) * sqsum->width * sqsum->height, 0, thisStream);
    cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
	//cudaMemPrefetchAsync(cascade->pq1, sizeof(int) * sqsum->width * sqsum->height, 0, thisStream);
    cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
	//cudaMemPrefetchAsync(cascade->pq2, sizeof(int) * sqsum->width * sqsum->height, 0, thisStream);
    cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);
	//cudaMemPrefetchAsync(cascade->pq3, sizeof(int) * sqsum->width * sqsum->height, 0, thisStream);

	

    /****************************************
     * Load the index of the four corners
     * of the filter rectangle
     **************************************/

    /* loop over the number of stages */
    for( i = 0; i < cascade->n_stages; i++ ) // 25 inside info.txt on 1st line
        {
            /* loop over the number of haar features */
            //printf(" %d",stages_array[i]);
            for( j = 0; j < stages_array[i]; j++ ) // 9,16,27,... inside info.txt
                {
                    int nr = 3;
                    /* loop over the number of rectangles */
                    for( k = 0; k < nr; k++ )
                        {
                            tr.x      = rectangles_array[r_index + k*4];
                                tr.y      = rectangles_array[r_index + 1 + k*4];
                                tr.width  = rectangles_array[r_index + 2 + k*4];
                                tr.height = rectangles_array[r_index + 3 + k*4];
                                if (k < 2)
                                    {
										new_scaled_rect_array[r_index + k*4]     = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
										new_scaled_rect_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
										new_scaled_rect_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
										new_scaled_rect_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
                                    }
                                else // k == 2
                                    {
                                        if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
                                            {
												new_scaled_rect_array[r_index + k*4]     = NULL ;
												new_scaled_rect_array[r_index + k*4 + 1] = NULL ;
												new_scaled_rect_array[r_index + k*4 + 2] = NULL;
												new_scaled_rect_array[r_index + k*4 + 3] = NULL;
                                            }
                                        else
                                            {
												new_scaled_rect_array[r_index + k*4]     = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
												new_scaled_rect_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
												new_scaled_rect_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
												new_scaled_rect_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
                                            }
                                    } /* end of branch if(k<2) */
                        } /* end of k loop*/
                    r_index+=12;
                    w_index+=3;
                } /* end of j loop */
        } /* end i loop */
	return new_scaled_rect_array;
}


/****************************************************
 * evalWeakClassifier:
 * the actual computation of a haar filter.
 * More info:
 * http://en.wikipedia.org/wiki/Haar-like_features
 ***************************************************/
__device__ int evalWeakClassifier(int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index, int** scaled_rectangles_array)
{
    // called by runCascadeClassifier() described later;

    /* the node threshold is multiplied by the standard deviation of the image */
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int t = dtree_thresh_array[tree_index] * variance_norm_factor;

	int sum = *(scaled_rectangles_array[r_index] + p_offset);
	sum -= *(scaled_rectangles_array[r_index + 1] + p_offset);
	sum -= *(scaled_rectangles_array[r_index + 2] + p_offset);
	sum += *(scaled_rectangles_array[r_index + 3] + p_offset);
    sum *= dweights_array[w_index];

    sum += (*(scaled_rectangles_array[r_index+4] + p_offset)
            - *(scaled_rectangles_array[r_index+5] + p_offset)
            - *(scaled_rectangles_array[r_index + 6] + p_offset)
            + *(scaled_rectangles_array[r_index + 7] + p_offset))
        * dweights_array[w_index + 1];

    if ((scaled_rectangles_array[r_index+8] != NULL))
        sum += (*(scaled_rectangles_array[r_index+8] + p_offset)
                - *(scaled_rectangles_array[r_index + 9] + p_offset)
                - *(scaled_rectangles_array[r_index + 10] + p_offset)
                + *(scaled_rectangles_array[r_index + 11] + p_offset))
            * dweights_array[w_index + 2];

    if(sum >= t)
		return dalpha2_array[tree_index];
    else
		return dalpha1_array[tree_index];

}



__device__ int runCascadeClassifier( myCascade* _cascade, MyPoint pt, int start_stage, int** scaled_rectangles_array)
{

    int p_offset, pq_offset;
    int i, j;
    unsigned int mean;
    unsigned int variance_norm_factor;
    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    int stage_sum;
    myCascade* cascade;
    cascade = _cascade;
    
        // pt is
        p_offset = pt.y * (cascade->sum.width) + pt.x;
        pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

        /**************************************************************************
         * Image normalization
         * mean is the mean of the pixels in the detection window
         * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
         * inv_window_area is 1 over the total number of pixels in the detection window
         *************************************************************************/

        variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
        mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

        variance_norm_factor = (variance_norm_factor*cascade->inv_window_area);
        variance_norm_factor =  variance_norm_factor - mean*mean;

        /***********************************************
         * Note:
         * The int_sqrt is softwar integer squre root.
         * GPU has hardware for floating squre root (sqrtf).
         * In GPU, it is wise to convert the variance norm
         * into floating point, and use HW sqrtf function.
         * More info:
         * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
         **********************************************/
        if( variance_norm_factor > 0 )
            variance_norm_factor = sqrtf(variance_norm_factor);
        else
            variance_norm_factor = 1;

        /**************************************************
         * The major computation happens here.
         * For each scale in the image pyramid,
         * and for each shifted step of the filter,
         * send the shifted window through cascade filter.
         *
         * Note:
         *
         * Stages in the cascade filter are independent.
         * However, a face can be rejected by any stage.
         * Running stages in parallel delays the rejection,
         * which induces unnecessary computation.
         *
         * Filters in the same stage are also independent,
         * except that filter results need to be merged,
         * and compared with a per-stage threshold.
         *************************************************/



        for( i = start_stage; i < cascade->n_stages; i++ ) // 25
            {

                /****************************************************
                 * A shared variable that induces false dependency
                 *
                 * To avoid it from limiting parallelism,
                 * we can duplicate it multiple times,
                 * e.g., using stage_sum_array[number_of_threads].
                 * Then threads only need to sync at the end
                 ***************************************************/
                stage_sum = 0;

				//const int arrsize = dstages_array[i];

				//int* stage_sum[259] = { 0 };
				/*int steps = dstages_array[i];
				int blocksize = 1024;
				int numblocks = (int)steps / blocksize;

				if (numblocks < 1)
					numblocks = 1;

				evalWeakClassifier << <numblocks, blocksize >> >(variance_norm_factor, p_offset, haar_counter, w_index, r_index, scaled_rectangles_array);*/


                for( j = 0; j < dstages_array[i]; j++ ) // 9,16,27,... inside info.txt
                    {
                        /**************************************************
                         * Send the shifted window to a haar filter.
                         **************************************************/
                        // p_offset
                        stage_sum += evalWeakClassifier(variance_norm_factor, p_offset, haar_counter, w_index, r_index, scaled_rectangles_array);
                        //n_features++;
                        haar_counter++;
                        w_index+=3;
                        r_index+=12;
                    } /* end of j loop */

                /**************************************************************
                 * threshold of the stage.
                 * If the sum is below the threshold,
                 * no faces are detected,
                 * and the search is abandoned at the i-th stage (-i).
                 * Otherwise, a face is detected (1)
                 **************************************************************/

                /* the number "0.4" is empirically chosen for 5kk73 */
                if( stage_sum < 0.4*dstages_thresh_array[i] ){
                    return -i;
                } /* end of the per-stage thresholding */
				/*int final_sum = 0;
				for (int x = 0; i < dstages_array[i]; i++)
					final_sum += stage_sum[x];

				if (final_sum < 0.4*dstages_thresh_array[i])
					    return -i;*/

            } /* end of i loop */
        return 1;
}

__global__ void
//__launch_bounds__(128, 16)
ShiftFilterCuda(int maxX, myCascade &cascade, float factor, MySize winSize, MyRect* rects, int** scaled_rectangles_array)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	MyPoint p;
	p.x = i % maxX;
	p.y = (i - p.x) / maxX;

	int result = runCascadeClassifier(&cascade, p, 0, scaled_rectangles_array);

	if (result > 0)
	{
		rects[i].x = myRound(p.x * factor);
		rects[i].y = myRound(p.y * factor);
		rects[i].width = winSize.width;
		rects[i].height = winSize.height;
	}
}


void ScaleImage_Invoker( myCascade* cascade, float factor, int sum_row, int sum_col, int** new_scaled_rect_array, cudaStream_t thisStream)
{
	//cudaSetDevice(0);
    MyPoint p;
    int result;
    int y1, y2, x2, x, y, step;

    MySize winSize0 = cascade->orig_window_size;
    MySize winSize;

    winSize.width =  myRound(winSize0.width*factor);
    winSize.height =  myRound(winSize0.height*factor);
    y1 = 0;

    /********************************************
     * When filter window shifts to image boarder,
     * some margin need to be kept
     *********************************************/
    y2 = sum_row - winSize0.height;
    x2 = sum_col - winSize0.width;

    /********************************************
     * Step size of filter window shifting
     * Reducing step makes program faster,
     * but decreases quality of detection.
     * example:
     * step = factor > 2 ? 1 : 2;
     *
     * For 5kk73,
     * the factor and step can be kept constant,
     * unless you want to change input image.
     *
     * The step size is set to 1 for 5kk73,
     * i.e., shift the filter window by 1 pixel.
     *******************************************/
    step = 1;

    /**********************************************
     * Shift the filter window over the image.
     * Each shift step is independent.
     * Shared data structure may limit parallelism.
     *
     * Some random hints (may or may not work):
     * Split or duplicate data structure.
     * Merge functions/loops to increase locality
     * Tiling to increase computation-to-memory ratio
     *********************************************/
	
	int steps = x2 * (y2 - y1);
	if (steps == 0)
		return;
	int blocksize = 64;
	int numblocks = (int)steps / blocksize;

	if (numblocks < 1)
		numblocks = 1;

	MyRect* rectList = new MyRect[numblocks * blocksize];
	cudaStreamAttachMemAsync(thisStream, rectList);
	//gpuErrchk(cudaMemPrefetchAsync(rectList, sizeof(MyRect) * numblocks * blocksize, 0, thisStream));
	/*for (int i = 0; i < numblocks * blocksize; i++)
		rectList[i] = new MyRect;*/

	//int stage_sum_size = 0;

	//for (int i = 0; i < cascade->n_stages; i++) // 25
	//	stage_sum_size += dstages_array[i];

	//gpuErrchk(cudaMallocManaged(&stage_sum, sizeof(int*)*stage_sum_size));

	//gpuErrchk(cudaDeviceSynchronize());

	//cudaStreamSynchronize(thisStream);

	ShiftFilterCuda << <numblocks, blocksize, 0, thisStream >> >(x2, *cascade, factor, winSize, rectList, new_scaled_rect_array);

	gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaStreamSynchronize(thisStream));

	/*cudaFree(cascade->p0);
	cudaFree(cascade->p1);
	cudaFree(cascade->p2);
	cudaFree(cascade->p3);
	cudaFree(cascade->pq0);
	cudaFree(cascade->pq1);
	cudaFree(cascade->pq2);
	cudaFree(cascade->pq3);*/

	//gpuErrchk(cudaDeviceSynchronize())

	for (int i = 0; i < numblocks * blocksize; i++)
	{
		if (rectList[i].height != 0)
		{
			push_back_mutex.lock();
			allCandidates.push_back(rectList[i]);
			push_back_mutex.unlock();
		}
			
	}
    //for( x = 0; x < x2; x += step ) // cris: <= or =?
    //    for( y = y1; y < y2; y += step )
    //        {
    //            p.x = x;
    //            p.y = y;

    //            /*********************************************
    //             * Optimization Oppotunity:
    //             * The same cascade filter is used each time
    //             ********************************************/
    //            result = runCascadeClassifier( cascade, p, 0 );

    //            /*******************************************************
    //             * If a face is detected,
    //             * record the coordinates of the filter window
    //             * the "push_back" function is from std:vec, more info:
    //             * http://en.wikipedia.org/wiki/Sequence_container_(C++)
    //             *
    //             * Note that, if the filter runs on GPUs,
    //             * the push_back operation is not possible on GPUs.
    //             * The GPU may need to use a simpler data structure,
    //             * e.g., an array, to store the coordinates of face,
    //             * which can be later memcpy from GPU to CPU to do push_back
    //             *******************************************************/
				//steps++;
    //            if( result > 0 )
    //                {
				//		MyRect* r = new MyRect;
				//		r->x = myRound(x*factor);
				//		r->y = myRound(y*factor);
				//		r->width = winSize.width;
				//		r->height = winSize.height;
    //                    vec->push_back(r);
    //                }
    //        }
	//printf("detecting faces, steps for iter := %d\n", steps);
}



/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
    int x, y, s, sq, t, tq;
    unsigned char it;
    int height = src->height;
    int width = src->width;
    unsigned char *data = src->data;
    int * sumData = sum->data;
    int * sqsumData = sqsum->data;
    for( y = 0; y < height; y++)
        {
            s = 0;
            sq = 0;
            /* loop over the number of columns */
            for( x = 0; x < width; x ++)
                {
                    it = data[y*width+x];
                    /* sum of the current row (integer)*/
                    s += it;
                    sq += it*it;

                    t = s;
                    tq = sq;
                    if (y != 0)
                        {
                            t += sumData[(y-1)*width+x];
                            tq += sqsumData[(y-1)*width+x];
                        }
                    sumData[y*width+x]=t;
                    sqsumData[y*width+x]=tq;
                }
        }
}

/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{

    int y;
    int j;
    int x;
    int i;
    unsigned char* t;
    unsigned char* p;
    int w1 = src->width;
    int h1 = src->height;
    int w2 = dst->width;
    int h2 = dst->height;

    int rat = 0;

    unsigned char* src_data = src->data;
    unsigned char* dst_data = dst->data;


    int x_ratio = (int)((w1<<16)/w2) +1;
    int y_ratio = (int)((h1<<16)/h2) +1;

    for (i=0;i<h2;i++)
        {
            t = dst_data + i*w2;
            y = ((i*y_ratio)>>16);
            p = src_data + y*w1;
            rat = 0;
            for (j=0;j<w2;j++)
                {
                    x = (rat>>16);
                    *t++ = p[x];
                    rat += x_ratio;
                }
        }
}

void readTextClassifier()//(myCascade * cascade)
{
    /*number of stages of the cascade classifier*/
    int stages = 0;
    /*total number of weak classifiers (one node each)*/
    int total_nodes = 0;
    int i, j, k, l;
    char mystring [12];
    int r_index = 0;
    int w_index = 0;
    int tree_index = 0;
    FILE *finfo = fopen("info.txt", "r");

    /**************************************************
  / how many stages are in the cascaded filter?
  / the first line of info.txt is the number of stages
  / (in the 5kk73 example, there are 25 stages)
    **************************************************/
    if ( fgets (mystring , 12 , finfo) != NULL )
        {
            stages = atoi(mystring);
        }
    i = 0;

    stages_array = (int *)malloc(sizeof(int)*stages);

    /**************************************************
     * how many filters in each stage?
     * They are specified in info.txt,
     * starting from second line.
     * (in the 5kk73 example, from line 2 to line 26)
     *************************************************/
    while ( fgets (mystring , 12 , finfo) != NULL )
        {
            stages_array[i] = atoi(mystring);
            total_nodes += stages_array[i];
            i++;
        }
    fclose(finfo);


    /* TODO: use matrices where appropriate */
    /***********************************************
     * Allocate a lot of array structures
     * Note that, to increase parallelism,
     * some arrays need to be splitted or duplicated
     **********************************************/
    //printf("\n total_nodes = %d", total_nodes);
	
	rectangles_array = (int *)malloc(sizeof(int)*total_nodes*(12 + 0)); // total_nodes = 2913
	gpuErrchk(cudaMallocManaged(&scaled_rectangles_array, sizeof(int*)*total_nodes*(12 + 0)));
	//cudaDeviceSynchronize();
	all_nodes = total_nodes;
	
            if (scaled_rectangles_array == NULL) {
                printf("ERROR: malloc failed!\n");
                exit(1);
            }
            weights_array = (int *)malloc(sizeof(int)*total_nodes*3);
            alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
            alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
            tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
            stages_thresh_array = (int*)malloc(sizeof(int)*stages);
            FILE *fp = fopen("class.txt", "r");

            /******************************************
             * Read the filter parameters in class.txt
             *
             * Each stage of the cascaded filter has:
             * 18 parameter per filter x tilter per stage
             * + 1 threshold per stage
             *
             * For example, in 5kk73,
             * the first stage has 9 filters,
             * the first stage is specified using
             * 18 * 9 + 1 = 163 parameters
             * They are line 1 to 163 of class.txt
             *
             * The 18 parameters for each filter are:
             * 1 to 4: coordinates of rectangle 1
             * 5: weight of rectangle 1
             * 6 to 9: coordinates of rectangle 2
             * 10: weight of rectangle 2
             * 11 to 14: coordinates of rectangle 3
             * 15: weight of rectangle 3
             * 16: threshold of the filter
             * 17: alpha 1 of the filter
             * 18: alpha 2 of the filter
             ******************************************/

            /* loop over n of stages */
            for (i = 0; i < stages; i++) // 25 stages
                {    /* loop over n of trees */
                    for (j = 0; j < stages_array[i]; j++) // stages_array[i] = 9,16,27,... inside info.txt
                        {   /* loop over n of rectangular features */
                            for(k = 0; k < 3; k++)
                                {   /* loop over the n of vertices */
                                    for (l = 0; l < 4; l++)
                                        {
                                            if (fgets (mystring , 12 , fp) != NULL) {
                                                rectangles_array[r_index] = atoi(mystring);
                                                    //printf(" %d", rectangles_array[r_index]);
                                                    } else {
                                                break;
                                                    }
                                            r_index++;
                                        } /* end of l loop */

                                        if (fgets (mystring , 12 , fp) != NULL)
                                            {
                                                weights_array[w_index] = atoi(mystring);
                                                /* Shift value to avoid overflow in the haar evaluation */
                                                /*TODO: make more general */
                                                /*weights_array[w_index]>>=8; */
                                            }
                                        else
                                            break;

                                            w_index++;
                                } /* end of k loop */

                                if (fgets (mystring , 12 , fp) != NULL)
                                    tree_thresh_array[tree_index]= atoi(mystring);
                                else
                                    break;
                                if (fgets (mystring , 12 , fp) != NULL)
                                    alpha1_array[tree_index]= atoi(mystring);
                                else
                                    break;
                                if (fgets (mystring , 12 , fp) != NULL)
                                    alpha2_array[tree_index]= atoi(mystring);
                                else
                                    break;
                                tree_index++;
                                if (j == stages_array[i]-1)
                                    {
                                        if (fgets (mystring , 12 , fp) != NULL)
                                            stages_thresh_array[i] = atoi(mystring);
                                        else
                                            break;
                                    }
                        } /* end of j loop */
                } /* end of i loop */
            fclose(fp);

			gpuErrchk(cudaMemcpyToSymbol(dtree_thresh_array, tree_thresh_array, sizeof(int)*total_nodes, 0, cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpyToSymbol(dalpha1_array, alpha1_array, sizeof(int)*total_nodes, 0, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpyToSymbol(dalpha2_array, alpha2_array, sizeof(int)*total_nodes));
			gpuErrchk(cudaMemcpyToSymbol(dweights_array, weights_array, sizeof(int)*total_nodes*3));
			gpuErrchk(cudaMemcpyToSymbol(dstages_thresh_array, stages_thresh_array, sizeof(int)*stages));
			gpuErrchk(cudaMemcpyToSymbol(dstages_array, stages_array, sizeof(int)*stages));
}


void releaseTextClassifier()
{
    free(stages_array);
    //free(rectangles_array);
    //free(scaled_rectangles_array);
    free(weights_array);
    free(tree_thresh_array);
    free(alpha1_array);
    free(alpha2_array);
    free(stages_thresh_array);
}
/* End of file. */
