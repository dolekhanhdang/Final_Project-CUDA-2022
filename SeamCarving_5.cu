#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <math.h>

// #define NUM_REMOVE = 1000;
// #define NUM_REMOVE = 1000;

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

    if (strcmp(type, "P3") != 0)
    {
        fclose(f);
        printf("Cannot read %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    fscanf(f, "%i", &width);
    fscanf(f, "%i", &height);

    int max_val;
    fscanf(f, "%i", &max_val);
    if (max_val > 255) // We assume 1 byte per value
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

void writePnm(uint8_t* pixels, int numChannels, int width, int height,
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

void writePnm(uchar3* pixels, int width, int height,
    char* fileName)
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


void writeImportantMatrix(char* fileName, int* matrix, int width, int height) {
    FILE* f = fopen(fileName, "w");
    if (f == NULL)
    {
        printf("Cannot write %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(f, "%d ", matrix[i * width + j]);
        }
        fprintf(f, "\n");
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

void printError(uchar3* deviceResult, uchar3* hostResult, int width, int height)
{
    float err = computeError(deviceResult, hostResult, width * height);
    printf("Error: %f\n", err);
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

__global__ void seamRemoveKernel(uchar3* inPixels, int width, int height, int* seam, uchar3* outPixels) {

    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;

    if (r < height && c < width) {
        int inSeam = seam[r];
        int index = r * width + c;
        if (index < inSeam) {
            outPixels[r * (width - 1) + c] = inPixels[index];
        }
        else {
            if (c < width - 1) {
                outPixels[r * (width - 1) + c] = inPixels[index + 1];
            }
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
    //printf("    indexMin = %d - min = %d \n", indexMin, Min);
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
                newInPixels[row * (width - 1) + col - 1] = inPixels[row * width + col];
            }
        }
    }
}

__global__ void seamCal(int* inPixels,int* inReal, int width, int height, int* traceBack, int* Sums, int stride)
{
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width      * height;
    if ((row >= height) || (col >= width))
        return;
    int new_row = row * stride * 2;
    if (height % 2 != 0)
        new_row += 1;
    int index = new_row * width + col;
    
    if (new_row == (height - 1))
    {
        traceBack[index] = INT_MAX;
        Sums[index] = inPixels[index];
    }
    else if( new_row + stride < height)
    {
        int inPixel      = inPixels[index];
        int minIndex     = col;
        int colTraceback = col;
        int pointTraceBack = (new_row + (stride / 2 - 1)) * width;
        if (stride > 1)
        {
            inPixel      = inReal[pointTraceBack + width  + traceBack[pointTraceBack + col]];
            minIndex     = traceBack[pointTraceBack + col];
            colTraceback = traceBack[pointTraceBack + col];
        }
        int minValueTraceBack  = inPixel         + inPixels[(new_row + stride) * width + minIndex];
        int min_Final          = inPixels[index] + inPixels[(new_row + stride) * width + minIndex];
        if (colTraceback - 1 >= 0)
        {
            int checkValue = inPixel + inPixels[(new_row + stride) * width + (colTraceback - 1)];
            if (checkValue < minValueTraceBack)
            {
                minIndex          = colTraceback - 1;
                min_Final         = inPixels[index] + inPixels[(new_row + stride) * width + (colTraceback - 1)];
                minValueTraceBack = checkValue;
                //printf("%d - %d -%d - %d\n", traceBack[pointTraceBack + col], inPixel, minValueTraceBack, checkValue);
            }
        }
        if (colTraceback + 1 < width)
        {
            int checkValue = inPixel + inPixels[(new_row + stride) * width + (colTraceback + 1)];
            if (checkValue < minValueTraceBack)
            {
                //printf("%d - %d -%d - %d\n", traceBack[pointTraceBack + col], inPixel, minValueTraceBack, checkValue);
                minIndex          = colTraceback + 1;
                minValueTraceBack = checkValue;
                min_Final         = inPixels[index] + inPixels[(new_row + stride) * width + (colTraceback + 1)];
            }
        }
        traceBack[(new_row+stride-1)*width+col] = minIndex;
        Sums[new_row * width + col] = min_Final;
    }
    if ( (stride * 2) >= (height*2))
    {
        if ((row == 0) & (height % 2 != 0))
        {
            int inPixel = inReal[row * width + col];
            int minIndex = col;
            int minValue = inPixel + inPixels[(row + 1) * width + minIndex];
            if (col - 1 >= 0)
            {
                int checkValue = inPixel + inPixels[(row + 1) * width + (col - 1)];
                if (checkValue < minValue)
                {
                    minIndex = col - 1;
                    minValue = checkValue;
                }
            }
            if (col + 1 < width)
            {
                int checkValue = inPixel + inPixels[(row + 1) * width + (col + 1)];
                if (checkValue < minValue)
                {
                    minIndex = col + 1;
                    minValue = checkValue;
                }
            }
            traceBack[row * width + col] = minIndex;
            Sums[row * width + col] = minValue;
        }
    }
    
}

void print_array(int* in, int width, int height)
{
    for (int i = 0;i < height;i++)
    {
        for (int j = 0; j < width;j++)
        {
            printf("%d ", in[i * width + j]);
        }
        printf("\n");
    }
    printf("\n --------------- \n");
}

void visualizeSeam(uchar3* inPixels, int width, int height, int* seam) {
    uchar3* visInPixels = (uchar3*)malloc(width * height * sizeof(uchar3));
    memcpy(visInPixels, inPixels, width * height * sizeof(uchar3));

    for (int r = 0; r < height; r++) {
        visInPixels[seam[r]].x = 255;
        visInPixels[seam[r]].y = 0;
        visInPixels[seam[r]].z = 0;
    }

    char fname[] = "visSeam.pnm";
    writePnm(visInPixels, width, height, fname);
    free(visInPixels);
}

void seamCarving_CUDA(uchar3* inPixels, int width, int height, uchar3*& outPixels, int numColRemove, dim3 blockSize = dim3(1, 1))
{
    printf("CUDA START \n");
    GpuTimer timer;
    // Filter
    int   filterWidth = 3;
    float filter_x_Sobel[9] = { 1 , 0 , −1, 2 , 0 , −2,  1 ,  0 , −1 };
    float filter_y_Sobel[9] = { 1 , 2 , 1 , 0 , 0 , 0 , −1 , −2 , −1 };

    
    // Define varivable
    uchar3* d_inPixels, * d_out;
    int* gray, * d_Sums, * d_TraceBack, *convolution, *d_Seam;
    float* d_filter_x, * d_filter_y;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    // Allocate filter device memory
    CHECK(cudaMalloc(&d_filter_x, filterSize));
    CHECK(cudaMalloc(&d_filter_y, filterSize));

    // Copy filter to device memories
    CHECK(cudaMemcpy(d_filter_x, filter_x_Sobel, filterSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_y, filter_y_Sobel, filterSize, cudaMemcpyHostToDevice));

    timer.Start();
    for (int i = 1;i <= numColRemove;i++)
    {
        //printf("Seam ---- %d \n", i);
        //Define Size
        int  new_width = width - i + 1;
        size_t pixelsSize = new_width * height * sizeof(uchar3);
        size_t outputSize = (new_width - 1) * height * sizeof(uchar3);
        size_t SumSize    = new_width * height * sizeof(int);
        //Define Variable
        int* TraceBack    = (int*)malloc(new_width * height * sizeof(int));
        int* Sums         = (int*)malloc(new_width * height * sizeof(int));
        int* Seam         = (int*)malloc(height * sizeof(int));
        uchar3* newPixels = (uchar3*)malloc((new_width - 1) * height * sizeof(uchar3));
        // Allocate device memories
        CHECK(cudaMalloc(&d_inPixels, pixelsSize));
        CHECK(cudaMalloc(&gray, pixelsSize));
        CHECK(cudaMalloc(&convolution, pixelsSize));
        CHECK(cudaMalloc(&d_Seam, height * sizeof(int)));

        CHECK(cudaMalloc(&d_out, outputSize));


        CHECK(cudaMalloc(&d_Sums, SumSize));
        CHECK(cudaMalloc(&d_TraceBack, SumSize));

        // Copy data to device memories
        CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));
        

        // Call kernel
        dim3 gridSize((new_width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
        //printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

        // Gray kernel
        //printf("Gray \n");
        gray_kernel << <gridSize, blockSize >> > (d_inPixels, new_width, height, gray);

        // Convolution kernel
        //printf("Convolution \n");
        convolution_kernel << <gridSize, blockSize >> > (gray, new_width, height, d_filter_x, d_filter_y, filterWidth, convolution);

        // SeamCal kennel
        //printf("SeamCal \n");
        dim3 gridSize_1((new_width - 1) /blockSize.x + 1, (height - 1) / blockSize.y + 1);
        seamCal <<<gridSize_1, blockSize >>> (convolution,convolution, new_width, height, d_TraceBack, d_Sums, 1);
        for (int stride = 2;stride < height;stride *= 2)
        {
            dim3 gridSize_2((new_width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
            seamCal << <gridSize_2, blockSize >> > (d_Sums, convolution, new_width, height, d_TraceBack, d_Sums, stride);
        }

        // Copy traceBack and Sums to host
        //printf("Copy Traceback \n");
        CHECK(cudaMemcpy(Sums, d_Sums, SumSize, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(TraceBack, d_TraceBack, SumSize, cudaMemcpyDeviceToHost));


        //Find seam
        //printf("Seam Find \n");
        seamTraceBack(TraceBack, Sums, Seam, new_width, height);
        CHECK(cudaMemcpy(d_Seam, Seam, height * sizeof(int), cudaMemcpyHostToDevice));
        //char fname3[] = "importantMat.txt";
        //writeImportantMatrix(fname3, Sums, new_width, height);
        
        //Remove Seam
        //printf("Seam Remove \n");
        seamRemove(inPixels, Seam, new_width, height, newPixels);
        
        //seamRemoveKernel << <gridSize, blockSize >> > (d_inPixels, width, height, d_Seam, d_out);
        //Copy new pixels
        //CHECK(cudaMemcpy(newPixels, d_out, outputSize, cudaMemcpyDeviceToHost));
        uchar3* temp = inPixels;
        inPixels = newPixels;
        newPixels = temp;
        if (i > 1)
            free(newPixels);
        free(TraceBack);
        free(Sums);
        free(Seam);
        CHECK(cudaFree(d_inPixels));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(gray));
        CHECK(cudaFree(convolution));
        CHECK(cudaFree(d_Seam));
        CHECK(cudaFree(d_Sums));
        CHECK(cudaFree(d_TraceBack));
        if (i == numColRemove)
        {
            CHECK(cudaFree(d_filter_x));
            CHECK(cudaFree(d_filter_y));
            timer.Stop();
            float time = timer.Elapsed();
            printf("Kernel time: %f ms\n", time);
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError());
            outPixels = inPixels;
            printf("DONE ! \n");
            return;
        }

    }
}


void checkNewAlgo()
{
    //Define varivable
    //int inPixels[5*4] = { 1, 4, 3, 5, 2, 3, 2, 5, 2, 3, 5, 3, 4, 2, 1, 6 , 3 , 2 , 5 , 4 };
    int inPixels[5 * 4] = { 1,9,9,9,9,9,1,9,9,9,9,9,1,9,9,9,9,9,1,9};
    int width        = 5;
    int height       = 4;
    int* d_inPixels, * d_Sums, * d_TraceBack;
    size_t pixelsSize = width * height * sizeof(int);
    int* Sums       = (int*)malloc(width * height * sizeof(int));
    int* TraceBack  = (int*)malloc(width * height * sizeof(int));
    //Allocate device memory
    CHECK(cudaMalloc(&d_inPixels , pixelsSize));
    CHECK(cudaMalloc(&d_Sums     , pixelsSize));
    CHECK(cudaMalloc(&d_TraceBack, pixelsSize));

    //Copy from host to device
    CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));

    dim3 blockSize(2, 2);
    GpuTimer timer;
    timer.Start();
    // Call kernel
    dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
    seamCal << <gridSize, blockSize >>> (d_inPixels, d_inPixels, width, height, d_TraceBack, d_Sums,1);
    for (int stride = 2;stride <height*2;stride*=2)
    {
        dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
        seamCal << <gridSize, blockSize >> > (d_Sums, d_inPixels, width, height, d_TraceBack, d_Sums, stride);
    }
 
    timer.Stop();
    float time = timer.Elapsed();
    printf("Kernel time: %f ms\n", time);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess) {
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    }

    //Copy from device to host
    CHECK(cudaMemcpy(Sums     , d_Sums     , pixelsSize, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(TraceBack, d_TraceBack, pixelsSize, cudaMemcpyDeviceToHost));
    
    print_array(Sums, width, height);
    print_array(TraceBack, width, height);
    //Free memory
    free(Sums);
    free(TraceBack);

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_Sums));
    CHECK(cudaFree(d_TraceBack));
}

int main(int argc, char** argv)
{
    // PRINT OUT DEVICE INFO
    if (argc != 4 && argc != 6)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    printDeviceInfo();

    int width, height;
    uchar3* inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    dim3 blockSize(32, 32); // Default
    if (argc == 6)
    {
        blockSize.x = atoi(argv[4]);
        blockSize.y = atoi(argv[5]);
    }

    //CHECK NEW FUNCTIONS
   //checkNewAlgo();
   
    uchar3* outPixels = NULL;
    int numRemove = 100;
    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    numRemove = atoi(argv[3]);
    seamCarving_CUDA(inPixels, width, height, outPixels, numRemove, blockSize);

    //uchar3* truePixels;
    //readPnm(argv[4], width, height, truePixels);
    //printf("\nImage size (width x height): %i x %i\n", width, height);
    //printError(outPixels, truePixels, width, height);
    if (outPixels != NULL)
        writePnm(outPixels, width - numRemove, height, concatStr(outFileNameBase, "_device.pnm"));
    printf("HMM \n");
  
    free(inPixels);
    free(outPixels);

}
