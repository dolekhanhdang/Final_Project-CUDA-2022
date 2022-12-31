#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <math.h>

#define NUM_REMOVE = 1000;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void readPnm(char* fileName, int& width, int& height, uchar3*& pixels)
{
    FILE* f = fopen(fileName, "r");
    if (f == NULL)
    {
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    char type[3];
    fscanf(f, "%s", type);

    if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // In this exercise, we assume 1 byte per value
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    pixels = (uchar3*)malloc(width * height * sizeof(uchar3));
    for (int i = 0; i < width * height; i++)
        fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

    fclose(f);
}

// void writePnm(uchar3 * pixels, int numChannels, int width, int height, 
// 		char * fileName)
// {
// 	FILE * f = fopen(fileName, "w");
// 	if (f == NULL)
// 	{
// 		printf("Cannot write %s\n", fileName);
// 		exit(EXIT_FAILURE);
// 	}	

// 	if (numChannels == 1)
// 		fprintf(f, "P2\n");
// 	else if (numChannels == 3)
// 		fprintf(f, "P3\n");
// 	else
// 	{
// 		fclose(f);
// 		printf("Cannot write %s\n", fileName);
// 		exit(EXIT_FAILURE);
// 	}

// 	fprintf(f, "%i\n%i\n255\n", width, height); 

// 	for (int i = 0; i < width * height * numChannels; i++)
// 		fprintf(f, "%hhu\n", pixels[i]);

// 	fclose(f);
// }

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void writeImportantMatrix(char * fileName, uint8_t * matrix, int width, int height){
    FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fprintf(f,"%hhu ",matrix[i * width + j]);
        }
        fprintf(f,"\n");
    }
    fclose(f);
}

char* concatStr(const char* s1, const char* s2)
{
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void checkCorrectness(uint8_t * out, uint8_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void seamCal(uint8_t* inPixels, int width, int height, int* traceBack,uint8_t* Sums)
{
    if (inPixels == NULL)
        return;

    int last_row = height - 1;
    int index = last_row * width;
    for(int col = 0; col < width; col++){
        traceBack[index + col] = INT_MAX;
        Sums[index + col] = inPixels[index + col];
    }

    for (int row = height - 2; row >= 0; row--)
    {
        for (int col = 0; col < width; col++)
        {
            // if (row == (height - 1))
            // {
            //     traceBack[row * width + col] = INT_MAX;
            //     Sums     [row * width + col] = inPixels[row * width + col];
            // }
            // else
            // {
            int inPixel  = inPixels[row * width + col];
            int minIndex = col;
            int minValue = inPixel + Sums[(row + 1) * width + minIndex];
            if (col - 1 >= 0)
            {
                int checkValue = inPixel + Sums[(row + 1) * width + (col-1)];
                if (checkValue < minValue)
                {
                    minIndex = col - 1;
                    minValue = checkValue;
                }
            }
            if (col + 1 < width)
            {
                int checkValue = inPixel + Sums[(row + 1) * width + (col + 1)];
                if (checkValue < minValue)
                {
                    minIndex = col + 1;
                    minValue = checkValue;
                }
            }
            traceBack[row * width + col] = minIndex;
            Sums     [row * width + col] = minValue;
            // }
        }
    }
}

void seamRemove(uchar3* inPixels, int* Seam, int width, int height, uchar3* newInPixels)
{
    for (int row = 0;row < height;row++)
    {
        int avoid = Seam[row];
        for (int col = 0; col < width; col++)
        {
            if (col == avoid)
                continue;
            if (col < avoid)
                newInPixels[row * (width - 1) + col] = inPixels[row * width + col];
            else
                newInPixels[row * (width - 1) + col-1] = inPixels[row * width + col];
        }
    }
}

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, 
		uint8_t * outPixels)
{
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if(r < height && c < width){
        int i = r * width + c;
        outPixels[i] = inPixels[i].x * 0.299 + inPixels[i].y * 0.585 + inPixels[i].z * 0.114;
    }
}

__global__ void convolutionKernel(uint8_t * inPixels, int width, int height,
    int* filter_x_Sobel, int* filter_y_Sobel, int filterWidth,
    uint8_t * outPixels){
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if(r < height && c < width){
        // int i = r * width + c;
        uint8_t outPixel = 0;
 
        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                float filterVal_x = filter_x_Sobel[filterR * filterWidth + filterC];
                float filterVal_y = filter_y_Sobel[filterR * filterWidth + filterC];

                int inPixelsR = r - filterWidth / 2 + filterR;
                int inPixelsC = c - filterWidth / 2 + filterC;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];

                outPixel += abs(filterVal_x * inPixel);
                outPixel += abs(filterVal_y * inPixel);
            }
        }
        outPixels[r * width + c] = outPixel;
    }
}

__global__ void seamCalKernel(uint8_t * inPixels, int width, int height, int seamCalHeight, uint8_t * outMatrix,int * traceBack){
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;

    
    if(c < width && r < height){
        if(r == height - 1)
            outMatrix[r * width + c] = inPixels[r * width + c];
        else if(r == seamCalHeight && r != height -1){
            int minIdx = (r + 1) * width + c;
            int minValue = inPixels[minIdx];

            if(c > 0){
                if(minValue > inPixels[minIdx - 1]){
                    minValue = inPixels[minIdx - 1];
                    minIdx = minIdx - 1;
                }
            }
            
            if(c < width - 1){
                if(minValue > inPixels[minIdx + 1]){
                    minValue = inPixels[minIdx + 1];
                    minIdx = minIdx + 1;
                }
            }

            outMatrix[r * width + c] = inPixels[r * width + c] + minValue;
            traceBack[r * width + c] = minIdx;
        }
    }
}

__global__ void seamRemoveKernel(uchar3* inPixels, int* seam, int width, int height, uchar3* newInPixels,
    uint8_t * greyInPixels, uint8_t * newGreyInPixels){
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if(r < height && c < width){
        int avoid = seam[r];
        if (c < avoid)
        {
            newInPixels[r * (width - 1) + c] = inPixels[r * width + c];
            newGreyInPixels[r* (width - 1) + c] = greyInPixels[r * width + c];
        }
            
        if (c > avoid){
            newInPixels[r * (width - 1) + c-1] = inPixels[r * width + c];
            newGreyInPixels[r * (width - 1) + c-1] = greyInPixels[r * width + c];
        }
    }
}

void seamCarvingByDevice(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int colRemoveNum, dim3 blockSize=dim3(1)){
    
    // int seamCal_blockSize, seamCal_gridSize;
    uint8_t * importanceMat;
    int * traceBack;

    int * d_traceBack;
    int * seam = (int *)malloc(height * sizeof(int)); 
    GpuTimer timer;
    timer.Start();
    // convert to greyscale image
    uchar3 * d_inPixels, * d_newInPixels;
    uint8_t * d_greyInPixels, * d_pixelImportanceMat, * d_importanceMat2Bottom, * d_newGreyInPixels;
    int * d_seam;

    CHECK(cudaMalloc(&d_inPixels,height * width * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_greyInPixels, height * width * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_seam,height * sizeof(int)));
    CHECK(cudaMemcpy(d_inPixels,inPixels,height * width * sizeof(uchar3),cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    convertRgb2GrayKernel<<<gridSize,blockSize>>>(d_inPixels,width,height,d_greyInPixels);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if(errSync != cudaSuccess){
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    if(errAsync != cudaSuccess){
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Convert to grey scale processing time (use device): %f ms\n\n",time);

     // Define filter
    int filterWidth = 3;
    int filter_x_Sobel[9] = { 1 , 0 , -1, 2 , 0 , -2,  1 ,  0 , -1 };
    int filter_y_Sobel[9] = { 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 };

    int * d_filter_x_Sobel, * d_filter_y_Sobel;
    CHECK(cudaMalloc(&d_filter_x_Sobel,filterWidth * filterWidth * sizeof(int)));
    CHECK(cudaMalloc(&d_filter_y_Sobel,filterWidth * filterWidth * sizeof(int)));

    CHECK(cudaMemcpy(d_filter_x_Sobel,filter_x_Sobel,
        filterWidth * filterWidth * sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_y_Sobel,filter_y_Sobel,
        filterWidth * filterWidth * sizeof(int),cudaMemcpyHostToDevice));

    timer.Start();
    for(int numRemove = 1; numRemove < colRemoveNum; numRemove++){
        // device pixel importance matrix
        CHECK(cudaMalloc(&d_pixelImportanceMat, height * width * sizeof(uint8_t)));

        // convolution with x, y sobel filter
        convolutionKernel<<<gridSize,blockSize>>>(d_greyInPixels,width,height,d_filter_x_Sobel,
            d_filter_y_Sobel, filterWidth, d_pixelImportanceMat);
        cudaError_t errSync1 = cudaGetLastError();
        cudaError_t errAsync1 = cudaDeviceSynchronize();
        if(errSync1 != cudaSuccess){
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync1));
        }
        if(errAsync1 != cudaSuccess){
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync1));
        }

        
        CHECK(cudaMalloc(&d_importanceMat2Bottom,height * width * sizeof(uint8_t)));
        CHECK(cudaMalloc(&d_traceBack, height * width * sizeof(int)));

        for(int inRow = height - 1; inRow >= 0; inRow -= 1){
            seamCalKernel<<<gridSize,blockSize>>>(d_pixelImportanceMat,width,height,inRow,d_importanceMat2Bottom,d_traceBack);
        }

        importanceMat = (uint8_t *)malloc(width * height * sizeof(uint8_t));
        CHECK(cudaMemcpy(importanceMat,d_importanceMat2Bottom,width * height * sizeof(uint8_t),cudaMemcpyDeviceToHost));
        int minEnergyValue = importanceMat[0];
        int minEnergyIndex = 0;
        for(int c = 1; c < width; c++){
            if(minEnergyValue > importanceMat[c]){
                minEnergyValue = importanceMat[c];
                minEnergyIndex = c;
            }
        }

        traceBack = (int *)malloc(width *height * sizeof(int));
        CHECK(cudaMemcpy(traceBack,d_traceBack,width *height * sizeof(int),cudaMemcpyDeviceToHost));
        seam[0] = minEnergyIndex;
        minEnergyIndex = traceBack[minEnergyIndex];
        for(int r = 0; r < height - 1; r++){
            seam[r + 1] = traceBack[minEnergyIndex];
            minEnergyIndex = traceBack[minEnergyIndex];
        }
        CHECK(cudaMemcpy(d_seam,seam,height * sizeof(int),cudaMemcpyHostToDevice));
        
        width -= 1;
        CHECK(cudaMalloc(&d_newInPixels,width * height * sizeof(uchar3)));
        CHECK(cudaMalloc(&d_newGreyInPixels,width * height * sizeof(uint8_t)));

        seamRemoveKernel<<<gridSize,blockSize>>>(d_inPixels,d_seam,width,height,d_newInPixels,d_greyInPixels,d_newGreyInPixels);
        gridSize.x = (width - 1) / blockSize.x + 1;

        uchar3 * temp = d_inPixels;
        d_inPixels = d_newInPixels;
        d_newInPixels = temp;

        uint8_t * temp1 = d_greyInPixels;
        d_greyInPixels = d_newGreyInPixels;
        d_newGreyInPixels = temp1;

        CHECK(cudaFree(d_newGreyInPixels));
        CHECK(cudaFree(d_newInPixels));
        CHECK(cudaFree(d_pixelImportanceMat));
        CHECK(cudaFree(d_importanceMat2Bottom));
        CHECK(cudaFree(d_traceBack));
    }
    timer.Stop();
    time = timer.Elapsed();
    printf("Seam carving processing time (use device): %f ms\n\n",time);
    CHECK(cudaMemcpy(outPixels,d_inPixels,width * height * sizeof(uchar3),cudaMemcpyDeviceToHost));
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    if (argc != 3 && argc != 5)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    int width, height;
    uchar3 * inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    uchar3 * outPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    // uint8_t * outMatrix = (uint8_t *)malloc(width * height);
    dim3 blockSize(32, 32); // Default
    if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}

    seamCarvingByDevice(inPixels,width,height, outPixels,100,blockSize);

    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension

    if(outPixels != NULL){
        writePnm(outPixels, width - 100, height, concatStr(outFileNameBase, "seam_carving_device.pnm"));
        // writePnm(outMatrix, 1, width, height, concatStr(outFileNameBase, "_convolution_device.pnm"));
    }
    free(inPixels);
    free(outPixels);
}
