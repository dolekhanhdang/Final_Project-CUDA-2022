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

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels)
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
    int * outPixels){
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    if(r < height && c < width){
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

__global__ void calImportantMatKernel(int * inPixels, int width, int inHeight, int * traceBack, int * outPixels){
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    // int r = blockDim.y * blockIdx.y + threadIdx.y;

    if(c < width){
        int minIdx = (inHeight + 1) * width + c;
        int minValue = outPixels[minIdx];

        if(c > 0){
            if(minValue > outPixels[minIdx - 1]){
                minValue = outPixels[minIdx - 1];
                minIdx -= 1;
            }
        }

        if(c < width - 1){
            if(minValue > outPixels[(inHeight + 1) * width + c + 1]){
                minValue = outPixels[(inHeight + 1) * width + c + 1];
                minIdx = (inHeight + 1) * width + c + 1;
            }
        }
        
        outPixels[inHeight * width + c] = minValue + inPixels[inHeight * width + c];
        traceBack[inHeight * width + c] = minIdx;
    }
}

__global__ void seamRemoveKernel(uchar3 * inPixels, uint8_t * inGreyPixels, int width, int height, int * seam,
     uchar3 * outPixels, uint8_t * outGreyPixels){
    
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    int r = blockDim.y * blockIdx.y + threadIdx.y;

    if(r < height && c < width){
        int inSeam = seam[r];
        int index = r * width + c;
        if(index < inSeam){
            outPixels[r * (width - 1) + c] = inPixels[index];
            outGreyPixels[r * (width - 1) + c] = inGreyPixels[index];
        }
        else{
            if(c < width - 1){
                outPixels[r * (width - 1) + c] = inPixels[index + 1];
                outGreyPixels[r * (width - 1) + c] = inGreyPixels[index + 1];
            } 
        }
    }
}

void convertRGB2GreyByDevice(uchar3 * inPixels, int width, int height, uint8_t * outPixels, dim3 blockSize=dim3(1)){
    uchar3 * d_inPixels;
    uint8_t * d_outPixels;

    CHECK(cudaMalloc(&d_inPixels, height * width * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_outPixels, height * width * sizeof(uint8_t)));

    CHECK(cudaMemcpy(d_inPixels,inPixels,height * width * sizeof(uchar3),cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // convert to grey scale
    GpuTimer timer;
    timer.Start();
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
    // printf("Convert to grey scale processing time (use device): %f ms\n\n",time);

    CHECK(cudaMemcpy(outPixels,d_outPixels,height * width * sizeof(uint8_t),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
}

void convolutionByDevice(uint8_t * inPixels, int width, int height, int * outPixels, dim3 blockSize=dim3(1)){

    // allocate device memory
    uint8_t * d_inPixels;
    int * d_outPixels;

    CHECK(cudaMalloc(&d_inPixels, height * width * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_outPixels, height * width * sizeof(int)));

    CHECK(cudaMemcpy(d_inPixels,inPixels,height * width * sizeof(uint8_t),cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // define filters
    int filterWidth = 3;
    int filter_x_Sobel[] = { 1 , 0 , -1, 2 , 0 , -2,  1 ,  0 , -1 };
    int filter_y_Sobel[] = { 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 };

    // for(int filterR = 0; filterR < filterWidth; filterR++){
    //     for(int filterC = 0; filterC < filterWidth;filterC++){
    //         float i = filterR - filterWidth/2;
    //         float j = filterC - filterWidth/2;
    //         filter_x_Sobel[filterR * filterWidth + filterC] = i / (i*i + j*j);
    //         filter_y_Sobel[filterR * filterWidth + filterC] = j / (i*i + j*j);
    //     }
    // }

    int * d_filter_x_Sobel, * d_filter_y_Sobel;
    CHECK(cudaMalloc(&d_filter_x_Sobel,filterWidth * filterWidth * sizeof(int)));
    CHECK(cudaMalloc(&d_filter_y_Sobel,filterWidth * filterWidth * sizeof(int)));

    CHECK(cudaMemcpy(d_filter_x_Sobel,filter_x_Sobel,filterWidth * filterWidth * sizeof(int),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter_y_Sobel,filter_y_Sobel,filterWidth * filterWidth * sizeof(int),cudaMemcpyHostToDevice));

    // convolution kernel
    GpuTimer timer;
    timer.Start();
    convolutionKernel<<<gridSize,blockSize>>>(d_inPixels,width,height,d_filter_x_Sobel,d_filter_y_Sobel, filterWidth, d_outPixels);
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
    // printf("Convolution processing time (use device): %f ms\n\n",time);

    CHECK(cudaMemcpy(outPixels,d_outPixels,height * width * sizeof(int),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_filter_x_Sobel));
    CHECK(cudaFree(d_filter_y_Sobel));
}

void calImportantMatByDevice(int * inPixels, int width, int height, int * traceBack, int * outPixels,int blockSize=32){
     // allocate device memory
    int * d_inPixels, * d_outPixels;
    int * d_traceBack;

    CHECK(cudaMalloc(&d_inPixels, height * width * sizeof(int)));
    CHECK(cudaMalloc(&d_outPixels, height * width * sizeof(int)));
    CHECK(cudaMalloc(&d_traceBack, (height-1) * width * sizeof(int)));

    CHECK(cudaMemcpy(d_inPixels,inPixels,height * width * sizeof(int),cudaMemcpyHostToDevice));
    // copy the last line of inPixels (host) to d_outPixels (device)
    CHECK(cudaMemcpy(d_outPixels + (height - 1) * width, inPixels + (height - 1) * width,width * sizeof(int),cudaMemcpyHostToDevice));

    int gridSize = (width - 1) / blockSize + 1;

    // calImportantMat kernel
    GpuTimer timer;
    timer.Start();

    for(int inHeight = height - 2; inHeight >= 0; inHeight--){
        calImportantMatKernel<<<gridSize,blockSize>>>(d_inPixels,width,inHeight,d_traceBack,d_outPixels);
        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if(errSync != cudaSuccess){
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }
        if(errAsync != cudaSuccess){
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        }
    }
    timer.Stop();
    float time = timer.Elapsed();
    // printf("Calculating cummulative matrix processing time (use device): %f ms\n\n",time);
    
    CHECK(cudaMemcpy(outPixels,d_outPixels,height * width * sizeof(int),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(traceBack,d_traceBack, (height - 1)*width*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_traceBack));
}

void findSeam(int * cumEnergy, int width, int height, int * traceBack, int * seam){
    int minValue = cumEnergy[0];
    int minIdx = 0;

    for(int i = 1; i < width; i++){
        if(minValue > cumEnergy[i]){
            minIdx = i;
            minValue = cumEnergy[i];
        }
    }

    seam[0] = minValue;
    for(int i = 1; i < height; i++){
        seam[i] = traceBack[minIdx];
        minIdx = traceBack[minIdx];
    }
}

void visualizeSeam(uchar3 * inPixels, int width, int height, int * seam){
    uchar3 * visInPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    memcpy(visInPixels,inPixels,width * height * sizeof(uchar3));

    for(int r = 0; r < height; r++){
        visInPixels[seam[r]].x = 255;
        visInPixels[seam[r]].y = 0;
        visInPixels[seam[r]].z = 0;
    }

    char fname[] = "visSeam.pnm";
    writePnm(visInPixels,width,height,fname);
    free(visInPixels);
}

void removeSeamByDevice(uchar3 * inPixels, uint8_t * inGreyPixels, int width, int height, uchar3 * outPixels, uint8_t * outGreyPixels,
    int * seam, dim3 blockSize=dim3(1)){

    uchar3 * d_inPixels, * d_outPixels;
    uint8_t * d_inGreyPixels, * d_outGreyPixels;
    int * d_seam;

    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_outPixels, (width - 1) * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_inGreyPixels, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_outGreyPixels, (width - 1) * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_seam, height * sizeof(int)));

    CHECK(cudaMemcpy(d_inPixels,inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_inGreyPixels,inGreyPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_seam, seam,height * sizeof(int),cudaMemcpyHostToDevice));

    dim3 gridSize((width - 1) / blockSize.x + 1,
        (height - 1) / blockSize.y + 1);

    // remove seam kernel
    GpuTimer timer;
    timer.Start();

    seamRemoveKernel<<<gridSize,blockSize>>>(d_inPixels,d_inGreyPixels,width,height,d_seam,d_outPixels,d_outGreyPixels);
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if(errSync != cudaSuccess){
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    if(errAsync != cudaSuccess){
        printf("Async kernel error while removing seam: %s\n", cudaGetErrorString(errAsync));
    }


    timer.Stop();
    float time = timer.Elapsed();
    // printf("Remove seam processing time (use device): %f ms\n\n",time);

    CHECK(cudaMemcpy(outPixels,d_outPixels,(width - 1)  * height * sizeof(uchar3),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(outGreyPixels,d_outGreyPixels,(width - 1)*height*sizeof(uint8_t),cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_inPixels));
    CHECK(cudaFree(d_inGreyPixels));
    CHECK(cudaFree(d_outPixels));
    CHECK(cudaFree(d_outGreyPixels));
    CHECK(cudaFree(d_seam));
}

// uchar3 * outPixels remember to add this argument !!!
void seamCarvingByDevice(uchar3 * inPixels, int width, int height, int numRemove, dim3 blockSize=dim3(1)){

    uint8_t * inGreyPixels = (uint8_t*)malloc(height * width * sizeof(uint8_t));
    int * outPixels = (int*)malloc(height * width * sizeof(int)); // energy pixel matrix
    int * cummulativeEnergy2Bottom = (int *)malloc(height * width * sizeof(int));
    int * traceBack = (int*)malloc((height - 1) * width * sizeof(int));
    int * seam = (int *)malloc(height * sizeof(int));

    uchar3 * newInPixels; //= (uchar3*)malloc((width - 1)*height * sizeof(uchar3));
    uint8_t * newInGreyPixels; //= (uint8_t*)malloc((width - 1)*height * sizeof(uint8_t));

    // int originWidth = width;
    uchar3 * temp;
    uint8_t * temp1;

    convertRGB2GreyByDevice(inPixels,width,height,inGreyPixels, blockSize);

    for(int i = 0; i < numRemove; i++){
        convolutionByDevice(inGreyPixels,width,height,outPixels,blockSize);
        calImportantMatByDevice(outPixels,width,height,traceBack,cummulativeEnergy2Bottom,blockSize.x);
        findSeam(cummulativeEnergy2Bottom,width,height,traceBack,seam);

        // visualizeSeam(inPixels,width,height,seam);

        newInPixels = (uchar3*)malloc((width - 1)*height * sizeof(uchar3));
        newInGreyPixels = (uint8_t*)malloc((width - 1)*height * sizeof(uint8_t));
    
        removeSeamByDevice(inPixels,inGreyPixels,width,height,newInPixels,newInGreyPixels,seam,blockSize);

        width -= 1;

        temp = inPixels;
        inPixels = newInPixels;
        newInPixels = temp;

        temp1 = inGreyPixels;
        inGreyPixels = newInGreyPixels;
        newInGreyPixels = temp1;

        free(outPixels);
        outPixels = (int*)malloc(height * width * sizeof(int)); // energy pixel matrix
        free(traceBack);
        traceBack = (int*)malloc((height - 1) * width * sizeof(int));
        free(cummulativeEnergy2Bottom);
        cummulativeEnergy2Bottom = (int *)malloc(height * width * sizeof(int));
        free(newInPixels);
        free(newInGreyPixels);
    }
    

    char fname[] = "seams_removed.pnm";

    writePnm(inPixels,width,height,fname);


    // char fname1[] = "grey_scale.pnm";
    // char fname2[] = "convolution.txt";
    // char fname3[] = "importantMat.txt";
    // writePnm(inGreyPixels,1,width,height,fname1);
    // writeImportantMatrix(fname2,outPixels,width,height);
    // writeImportantMatrix(fname3,cummulativeEnergy2Bottom,width,height);

    free(inGreyPixels);
    free(outPixels);
    free(cummulativeEnergy2Bottom);
    free(traceBack);
    free(seam);
    free(newInPixels);
    free(newInGreyPixels);

}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    if (argc != 3 && argc != 5)
    {
        printf("The number of arguments is invalid\n");
        return EXIT_FAILURE;
    }

    // printDeviceInfo();

    int width, height;
    uchar3 * inPixels;
    readPnm(argv[1], width, height, inPixels);
    printf("\nImage size (width x height): %i x %i\n", width, height);

    dim3 blockSize(32, 32); // Default
    if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}

    int numRemove = 500;
    numRemove = atoi(argv[2]);
    seamCarvingByDevice(inPixels,width,height,numRemove, blockSize);

    free(inPixels);
}
