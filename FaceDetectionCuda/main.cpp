/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
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

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"
#include "managed.cuh"

#define INPUT_FILENAME "face9.pgm"
#define OUTPUT_FILENAME "output.pgm"

using namespace std;


int main (int argc, char *argv[])
{
	std::clock_t start;
	double duration;

	

	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		printf("  concurrent: %i\n\n", prop.concurrentManagedAccess);
	}

	cudaSetDevice(0);
	

    int flag;

    //int mode = 1;
    int i;

    /* detection parameters */
    float scaleFactor = 1.25;
    int minNeighbours = 1;


    printf("-- entering main function --\r\n");

    printf("-- loading image --\r\n"); 

    MyImage imageObj;
    MyImage *image = &imageObj;

    flag = readPgm((char *)INPUT_FILENAME, image);
    if (flag == -1)
        {
            printf( "Unable to open input image\n");
            return 1;
        }

    printf("-- loading cascade classifier --\r\n");

    myCascade *cascade = new myCascade;
	MySize *minSize = new MySize;
	minSize->width = 20;
	minSize->height = 20;
	MySize *maxSize = new MySize;
	maxSize->width = 0;
	maxSize->height = 0;

    // classifier properties
    cascade->n_stages=25;
    cascade->total_nodes=2913;
    cascade->orig_window_size.height = 24;
    cascade->orig_window_size.width = 24;


    readTextClassifier();

    std::vector<MyRect> result;
	std::vector<double> times;

    printf("-- detecting faces --\r\n");
	start = std::clock();
	int total_loops = 1;
	for (int loops = 0; loops < total_loops; loops++)
	{
		
		result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);
	}

	duration = ((std::clock() - start) / (double)CLOCKS_PER_SEC) / (total_loops + 1);

	for (i = 0; i < result.size(); i++)
	{
		MyRect r = result[i];
		drawRectangle(image, r);
	}
	

        printf("-- saving output --\r\n");
        flag = writePgm((char *)OUTPUT_FILENAME, image);

        printf("-- image saved --\r\n");

        // delete image and free classifier
        releaseTextClassifier();

            freeImage(image);

			

			printf("Runtime := %f\n", duration);
            return 0;
}
