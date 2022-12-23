#include <stdio.h>
#include <stdint.h>
#include <malloc.h>


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

char* concatStr(const char* s1, const char* s2)
{
    char* result = (char*)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void convolution(uchar3* inPixels, int width, int height, float* filter_x_Sobel, float* filter_y_Sobel, int filterWidth, uchar3* outPixels)
{
    for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
    {
        for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
        {
            float3 outPixel = make_float3(0, 0, 0);
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
                    outPixel.x += (filterVal_x + filterVal_y) * inPixel.x;
                    outPixel.y += (filterVal_x + filterVal_y) * inPixel.y;
                    outPixel.z += (filterVal_x + filterVal_y) * inPixel.z;
                    float test = inPixel.x;
                    if (outPixelsR * 512 + outPixelsC == -1)
                        printf("%d - %f - x = %f - y = %f - z = %f \n", filterR * filterWidth + filterC, test, outPixel.x, outPixel.y, outPixel.z);
                }
            }
            outPixels[outPixelsR * width + outPixelsC] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
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

    // Calculation
    int filterWidth = 3;
    float filter_x_Sobel[9] = { 1 ,0, −1,2, 0, −2,1, 0, −1};
    float filter_y_Sobel[9] = { 1, 2, 1, 0, 0, 0, −1, −2, −1};
    uchar3* outPixels = (uchar3*)malloc(width * height * sizeof(uchar3));
    convolution(inPixels,width,height,filter_x_Sobel,filter_y_Sobel,filterWidth, outPixels);

    // Write results to files
    char* outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    //writePnm(correctOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));
    writePnm(outPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(inPixels);
    free(outPixels);
}