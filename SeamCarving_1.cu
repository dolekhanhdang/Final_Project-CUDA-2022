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

__global__ void gray_kernel(uchar3* inPixels, int width, int height, uint8_t* outPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row >= height) || (col >= width))
        return;
    int index = row * width + col;
    uchar3 inPixel = inPixels[index];
    outPixels[index] = 0.299f * inPixel.x + 0.587f * inPixel.y + 0.114f * inPixel.z;
}

__global__ void convolution_kernel(uint8_t* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, uint8_t* outPixels)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((row >= height) || (col >= width))
        return;
    int index = row * width + col;
    int distanceToKernelCol = (filterWidth - 3) / 2 + 1;
    int rowCount = -distanceToKernelCol;
    int columnCount = -distanceToKernelCol;
    uint8_t result = 0;
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
        result += abs(inPixels[checkRow * width + checkColumn] * filter_x_Sobel[filterIndex]);
        result += abs(inPixels[checkRow * width + checkColumn] * filter_y_Sobel[filterIndex]);
        if ((filterIndex + 1) % filterWidth == 0)
        {
            rowCount += 1;
            columnCount = -distanceToKernelCol;
        }
        else columnCount += 1;
    }
    outPixels[index] = result;
}

void seamCarving_CUDA(uchar3* inPixels, int width, int height, uint8_t* outPixels, int numColRemove, dim3 blockSize = dim3(1,1))
{
    GpuTimer timer;
    // Filter
    int   filterWidth       = 3;
    float filter_x_Sobel[9] = { 1 , 0 , −1, 2 , 0 , −2,  1 ,  0 , −1 };
    float filter_y_Sobel[9] = { 1 , 2 , 1 , 0 , 0 , 0 , −1 , −2 , −1 };

    // Allocate device memories
    uchar3 * d_in;
    uint8_t* gray, *d_out;
    float  * d_filter_x, *d_filter_y;
    size_t pixelsSize = width       * height      * sizeof(uchar3);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    size_t outputSize = width       * height      * sizeof(uint8_t);
    CHECK(cudaMalloc(&d_in      , pixelsSize));
    CHECK(cudaMalloc(&gray      , outputSize));
    CHECK(cudaMalloc(&d_out     , outputSize));
    CHECK(cudaMalloc(&d_filter_x, filterSize));
    CHECK(cudaMalloc(&d_filter_y, filterSize));
    //CHECK(cudaMalloc(&gray      , pixelsSize));

    // Copy data to device memories
    CHECK(cudaMemcpy(d_in      , inPixels       , pixelsSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_x, filter_x_Sobel , filterSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_y, filter_y_Sobel , filterSize, cudaMemcpyHostToDevice));

    // Call kernel
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    timer.Start();

    gray_kernel        << <gridSize, blockSize >> > (d_in, width, height, gray);
    convolution_kernel << <gridSize, blockSize >> > (gray, width, height, d_filter_x, d_filter_y, filterWidth, d_out);

    timer.Stop();
    float time = timer.Elapsed();
    printf("Kernel time: %f ms\n", time);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memory
    CHECK(cudaMemcpy(outPixels, d_out, outputSize, cudaMemcpyDeviceToHost));

    //Free device memory

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(gray));
    CHECK(cudaFree(d_filter_x));
    CHECK(cudaFree(d_filter_y));

}


// Sequence functions
void convolution(uchar3* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, uint8_t* outPixels)
{
    printf("    Convolution begin \n");
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
    {
        for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
        {
            uint8_t outPixel = 0;
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
                    outPixel += abs(filterVal_x * grayFilter) + abs(filterVal_y* grayFilter);

                }
            }
            outPixels[outPixelsR * width + outPixelsC] = outPixel;
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

void seamCal(uint8_t* inPixels, int width, int height, int* traceBack,int* Sums)
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
        uint8_t* gray   = (uint8_t*)malloc(new_width * height * sizeof(uint8_t));
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

    // Read input image file
    int width, height;
    uchar3* inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);
    /*
    // Calculation
    uchar3* outPixels = NULL;
    GpuTimer timer;
    timer.Start();
    seamCarving(inPixels, width, height, outPixels, 1000);
    //uint8_t testSeam[15] = { 1, 4, 3, 5, 2, 3, 2, 5, 2, 3, 5, 3, 4, 2, 1}; // Only for test purpose
    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
    // Write results to files
    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    //writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));
    if(outPixels != NULL)
        writePnm(outPixels, width-1000, height, concatStr(outFileNameBase, "_device.pnm"));
    printf("HMM \n");
    */
    dim3 blockSize(32, 32);
    uint8_t* outPixels = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    seamCarving_CUDA(inPixels, width, height, outPixels, 0, blockSize);
    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    if (outPixels != NULL)
        writePnm_gray(outPixels,1, width, height, concatStr(outFileNameBase, "_device.pnm"));
    // Free memories
    free(inPixels );
    free(outPixels);
}

