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

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
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

void convertRgb2Gray_convolution_ByDevice(uchar3 * inPixels, int width, int height,
    uint8_t * outPixels, uint8_t * outMatrix, dim3 blockSize=dim3(1)){
    
    GpuTimer timer;
    timer.Start();

    uchar3 * d_inPixels;
    uint8_t * d_outPixels, * d_outMatrix;
    // size_t nBytes = height * width * sizeof(uchar3);
    CHECK(cudaMalloc(&d_inPixels,height * width * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_outPixels, height * width * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_outMatrix, height * width * sizeof(uint8_t)));

    CHECK(cudaMemcpy(d_inPixels,inPixels,height * width * sizeof(uchar3),cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    convertRgb2GrayKernel<<<gridSize,blockSize>>>(d_inPixels,width,height,d_outPixels);
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


    timer.Start();
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

    convolutionKernel<<<gridSize,blockSize>>>(d_outPixels,width,height,d_filter_x_Sobel,
        d_filter_y_Sobel, filterWidth, d_outMatrix);
    cudaError_t errSync1 = cudaGetLastError();
    cudaError_t errAsync1 = cudaDeviceSynchronize();
    if(errSync1 != cudaSuccess){
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync1));
    }
    if(errAsync1 != cudaSuccess){
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync1));
    }

    timer.Stop();
    time = timer.Elapsed();
    printf("Convolution processing time (use device): %f ms\n\n",time);
    
    CHECK(cudaMemcpy(outMatrix,d_outMatrix, height * width * sizeof(uint8_t),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outPixels,d_outPixels, height * width * sizeof(uint8_t),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_outMatrix));
    CHECK(cudaFree(d_filter_x_Sobel));
    CHECK(cudaFree(d_filter_y_Sobel));
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

    uint8_t * outPixels = (uint8_t *)malloc(width * height);
    uint8_t * outMatrix = (uint8_t *)malloc(width * height);
    dim3 blockSize(32, 32); // Default
    if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	} 
    convertRgb2Gray_convolution_ByDevice(inPixels,width,height,outPixels,outMatrix,blockSize);

    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension

    if(outPixels != NULL && outMatrix != NULL){
        writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_greyscale_device.pnm"));
        writePnm(outMatrix, 1, width, height, concatStr(outFileNameBase, "_convolution_device.pnm"));
    }
        

    free(inPixels);
    free(outPixels);
    free(outMatrix);
}