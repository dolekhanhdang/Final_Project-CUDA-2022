﻿#include <stdio.h>
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

void writePnm(uchar3* pixels, int width, int height, char* fileName)
{
    FILE* f = fopen(fileName, "w");
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

void writePnm_gray(uint8_t* pixels, int numChannels, int width, int height,
    char* fileName)
{
    FILE* f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if (numChannels == 1)
        fprintf(f, "P2\n");
    else if (numChannels == 3)
        fprintf(f, "P3\n");
    else
    {
        fclose(f);
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fprintf(f, "%i\n%i\n255\n", width, height);

    for (int i = 0; i < width * height * numChannels; i++)
        fprintf(f, "%hhu\n", pixels[i]);

    fclose(f);
}

char* concatStr(const char* s1, const char* s2)
{
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}
// Parallel functions

__global__ void gray_kernel(uchar3* inPixels, int width, int height, int* outPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row >= height) || (col >= width))
        return;
    int index = row * width + col;
    uchar3 inPixel = inPixels[index];
    outPixels[index] = 0.299f * inPixel.x + 0.587f * inPixel.y + 0.114f * inPixel.z;
}

__global__ void convolution_kernel(int* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, int* outPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row >= height) || (col >= width))
        return;
    int index = row * width + col;
    int distanceToKernelCol = (filterWidth - 3) / 2 + 1;
    int rowCount = -distanceToKernelCol;
    int columnCount = -distanceToKernelCol;
    int result_x = 0;
    int result_y = 0;
    for (int filterIndex = 0; filterIndex < (filterWidth * filterWidth); filterIndex++)
    {
        int checkRow = row + rowCount;
        int checkColumn = col + columnCount;
        if (checkRow < 0)
            checkRow = 0;
        if (checkRow >= height)
            checkRow = height - 1;
        if (checkColumn < 0)
            checkColumn = 0;
        if (checkColumn >= width)
            checkColumn = width - 1;
        result_x += inPixels[checkRow * width + checkColumn] * filter_x_Sobel[filterIndex];
        result_y += inPixels[checkRow * width + checkColumn] * filter_y_Sobel[filterIndex];
        if ((filterIndex + 1) % filterWidth == 0)
        {
            rowCount += 1;
            columnCount = -distanceToKernelCol;
        }
        else columnCount += 1;
    }
    outPixels[index] = abs(result_x)+abs(result_y);
}

__global__ void convolution_kernel_1(int* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, int* outPixels)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < height && c < width) {
        // int i = r * width + c;
        uint8_t outPixel_x = 0;
        uint8_t outPixel_y = 0;

        for (int filterR = 0; filterR < filterWidth; filterR++)
        {
            for (int filterC = 0; filterC < filterWidth; filterC++)
            {
                int filterVal_x = filter_x_Sobel[filterR * filterWidth + filterC];
                int filterVal_y = filter_y_Sobel[filterR * filterWidth + filterC];

                int inPixelsR = r - filterWidth / 2 + filterR;
                int inPixelsC = c - filterWidth / 2 + filterC;
                inPixelsR = min(max(0, inPixelsR), height - 1);
                inPixelsC = min(max(0, inPixelsC), width - 1);
                uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];

                outPixel_x += filterVal_x * inPixel;
                outPixel_y += filterVal_y * inPixel;
            }
        }
        outPixels[r * width + c] = abs(outPixel_x) + abs(outPixel_y);
    }
}

float computeError(uchar3* a1, uchar3* a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
    {
        err += abs((int)a1[i].x - (int)a2[i].x);
        err += abs((int)a1[i].y - (int)a2[i].y);
        err += abs((int)a1[i].z - (int)a2[i].z);
    }
    err /= (n * 3);
    return err;
}

float computeError_1(int* a1, int* a2, int n)
{
    float err = 0;
    for (int i = 0; i < n; i++)
        err += abs((int)a1[i] - (int)a2[i]);
    err /= n;
    return err;
}

void printError(int* deviceResult, int* hostResult, int width, int height)
{
    float err = computeError_1(deviceResult, hostResult, width * height);
    printf("Error: %f\n", err);
}


// Sequence functions
void convolution(uchar3* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, int* outPixels)
{
    printf("    Convolution begin \n");
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
    {
        for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
        {
            uint8_t outPixel_x = 0;
            uint8_t outPixel_y = 0;
            for (int filterR = 0; filterR < filterWidth; filterR++)
            {
                for (int filterC = 0; filterC < filterWidth; filterC++)
                {
                    float filterVal_x = filter_x_Sobel[filterR * filterWidth + filterC];
                    float filterVal_y = filter_y_Sobel[filterR * filterWidth + filterC];
                    int inPixelsR = outPixelsR - filterWidth / 2 + filterR;
                    int inPixelsC = outPixelsC - filterWidth / 2 + filterC;
                    inPixelsR = min(max(0, inPixelsR), height - 1);
                    inPixelsC = min(max(0, inPixelsC), width - 1);
                    uchar3 inPixel = inPixels[inPixelsR * width + inPixelsC];
                    uint8_t grayFilter = 0.299f * inPixel.x + 0.587f * inPixel.y + 0.114f * inPixel.z;
                    outPixel_x += filterVal_x * grayFilter;
                    outPixel_y += filterVal_y * grayFilter;
                }
            }
            outPixels[outPixelsR * width + outPixelsC] = abs(outPixel_x) + abs(outPixel_y);
        }
    }
}

void seamTraceBack(int* traceBack, int* Sums, int* Seam, int width, int height)
{
    if ((traceBack == NULL) || (Sums == NULL) || (Seam == NULL))
        return;
    int Min = Sums[0];
    int indexMin = 0;
    for (int i = 0;i < width;i++)
    {
        int check = Sums[i];
        if (check < Min)
        {
            Min = check;
            indexMin = i;
        }
    }
    printf("    indexMin = %d - min = %d \n", indexMin, Min);
    Seam[0] = indexMin;
    for (int i = 1;i < height;i++)
    {
        Seam[i] = traceBack[(i - 1) * width + indexMin];
        indexMin = Seam[i];
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
            {
                newInPixels[row * (width - 1) + col] = inPixels[row * width + col];
            }
            else
            {
                newInPixels[row * (width - 1) + col-1] = inPixels[row * width + col];
            }
        }
    }
}

void seamCal(int* inPixels, int width, int height, int* traceBack,int* Sums)
{
    if (inPixels == NULL)
        return;
    for (int row = height - 1; row >= 0; row--)
    {
        for (int col = 0; col < width; col++)
        {
            if (row == (height - 1))
            {
                traceBack[row * width + col] = INT_MAX;
                Sums     [row * width + col] = inPixels[row * width + col];
            }
            else
            {
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
            }
        }
    }
}

void seamCarving(uchar3* inPixels, int width, int height, uchar3*& outPixels, int numColRemove)
{
    for (int i = 1;i <= numColRemove; i++)
    {
        printf("Seam = %d \n", i);
        //Define varivable
        int  new_width     = width - i + 1;
        int* traceBack    = (int*)   malloc(new_width * height * sizeof(int));
        int* Sums         = (int*)   malloc(new_width * height * sizeof(int));
        int* Seam         = (int*)   malloc(            height * sizeof(int));
        uchar3* newPixels = (uchar3*)malloc((new_width -1) * height * sizeof(uchar3));
        // Define filter
        int filterWidth = 3;
        float filter_x_Sobel[9] = { 1 , 0 , −1, 2 , 0 , −2,  1 ,  0 , −1 };
        float filter_y_Sobel[9] = { 1 , 2 , 1 , 0 , 0 , 0 , −1 , −2 , −1 };
        // Gray filter and convolution ( calculate Importance matrix )
        int* gray   = (int*)malloc(new_width * height * sizeof(int));
        convolution       (inPixels , new_width, height, filter_x_Sobel, filter_y_Sobel, filterWidth, gray);
        seamCal           (gray     , new_width, height,traceBack,Sums);
        seamTraceBack     (traceBack, Sums, Seam, new_width, height);
        seamRemove        (inPixels , Seam, new_width, height, newPixels);
        uchar3* temp    = inPixels;
        inPixels        = newPixels;
        newPixels       = temp;
        if (i > 1)
            free(newPixels);
        free(traceBack);
        free(Sums);
        free(Seam);
        free(gray);
        if (i == numColRemove)
        {
            outPixels = inPixels;
            printf("DONE ! \n");
            return;
        }
    }
}

void seamCarving_CUDA(uchar3* inPixels, int width, int height, int* outPixels, int numColRemove, dim3 blockSize = dim3(1, 1))
{
    GpuTimer timer;
    // Filter
    int   filterWidth = 3;
    float filter_x_Sobel[9] = { 1 , 0 , −1, 2 , 0 , −2,  1 ,  0 , −1 };
    float filter_y_Sobel[9] = { 1 , 2 , 1 , 0 , 0 , 0 , −1 , −2 , −1 };

    // Allocate device memories
    uchar3* d_in;
    int* gray, * d_out;
    float* d_filter_x, * d_filter_y;
    size_t pixelsSize = width * height * sizeof(uchar3);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    size_t outputSize = width * height * sizeof(int);
    CHECK(cudaMalloc(&d_in, pixelsSize));
    CHECK(cudaMalloc(&gray, outputSize));
    CHECK(cudaMalloc(&d_out, outputSize));
    CHECK(cudaMalloc(&d_filter_x, filterSize));
    CHECK(cudaMalloc(&d_filter_y, filterSize));
    //CHECK(cudaMalloc(&gray      , pixelsSize));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, inPixels, pixelsSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_x, filter_x_Sobel, filterSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_y, filter_y_Sobel, filterSize, cudaMemcpyHostToDevice));

    // Call kernel
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    timer.Start();

    gray_kernel << <gridSize, blockSize >> > (d_in, width, height, gray);
    convolution_kernel_1 << <gridSize, blockSize >> > (gray, width, height, d_filter_x, d_filter_y, filterWidth, d_out);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Kernel time: %f ms\n", time);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memory
    CHECK(cudaMemcpy(outPixels, d_out, outputSize, cudaMemcpyDeviceToHost));

 
    int* True_Pixels = (int*)malloc(width * height * sizeof(int));
    convolution(inPixels, width, height, filter_x_Sobel, filter_y_Sobel, filterWidth, True_Pixels);
    printError(outPixels, True_Pixels, width,height);
    free(True_Pixels);
    //Free device memory

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(gray));
    CHECK(cudaFree(d_filter_x));
    CHECK(cudaFree(d_filter_y));

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
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    // Read input image file
    int width, height;
    uchar3* inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);
    dim3 blockSize(32, 32); // Default
    if (argc == 5)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }

    uchar3* outPixels = NULL;
    //uchar3* outPixels = (uchar3*)malloc(width * height * sizeof(uchar3));
    GpuTimer timer;
    timer.Start();
    int numRemove = 100;
    numRemove = atoi(argv[3]);
    seamCarving(inPixels, width, height, outPixels, numRemove);
    //uint8_t testSeam[15] = { 1, 4, 3, 5, 2, 3, 2, 5, 2, 3, 5, 3, 4, 2, 1}; // Only for test purpose
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    // Write results to files
    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    if(outPixels != NULL)
        writePnm(outPixels, width- numRemove, height, concatStr(outFileNameBase, "_host.pnm"));
    printf("HMM \n");
    // Free memories
    free(inPixels );
    free(outPixels);
}

