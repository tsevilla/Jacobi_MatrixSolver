//
//  main.c
//  Jacobi_MatrixSolver
//
//  Created by Thomas Sevilla on 8/17/14.
//  Copyright (c) 2014 Thomas Sevilla. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc/malloc.h>
#include <OpenCL/OpenCL.h>
#include "Solver.cl.h"


//Reading in the matrix------------------------------------------------------------------------------------------------
void initmat(float *A, float *B, float *x)
{
    FILE *fileIN;
    
    if((fileIN = fopen("/Users/Tom/Documents/Jacobi_MatrixSolver/in.txt", "r")) == NULL)
    {
        fprintf(stdout,"File does not work/open");
    }
    
    int size_A_Matrix= 0;
    fscanf(fileIN,"%i", &size_A_Matrix);
    
    
    int Ndim=size_A_Matrix;
    int Pdim=size_A_Matrix;
    
    int i, j;
    
    printf("A Matrix:\n");
	for (i = 0; i < Ndim; i++){
        printf("Row #%i: \n",i+1);
		for (j = 0; j < Pdim; j++){
            fscanf(fileIN, "%f\n", &A[i*Ndim+j]);
        printf("%f\n",A[i*Ndim+j]);
        }
    }
    printf("Was Scanned\n");
    
    printf("B Matrix:\n");
	for (i = 0; i < Pdim; i++){
			fscanf(fileIN, "%f\n", &B[i]);
       printf("%f\n",B[i]);
        }
     printf("Was Scanned\n");
    //Initilize intial guess for x's to 0;
    
    printf("X Matrix: \n");
    for (i = 0; i < Ndim; i++){
        //printf("Row #%i: ",i+1);
        for (j = 0; j < Pdim; j++){
            x[i*Ndim+j]=0;
            printf("%f\n",x[i*Ndim+j]);
        }
    }

    
}
//--------------------------------------------------------------------------------------------------------------------------------


void OpenCl_Calc(dispatch_queue_t queue,void* d_a, void * d_b, void * d_output,float * x_host, int size_A_matrix)
{
    //Kernel call:
    dispatch_sync(queue,^{
        size_t preferred_wgs_mult;
        gcl_get_kernel_block_workgroup_info(Ax_BSolver_kernel, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferred_wgs_mult),&preferred_wgs_mult, NULL);
        cl_ndrange range= {2,{0,0,0},{1,1,0},{preferred_wgs_mult,preferred_wgs_mult,0}};
        Ax_BSolver_kernel(&range,(cl_int)size_A_matrix,(cl_float*)d_a,(cl_float*)d_b, (cl_float*)d_output);
        
        //Getting the data out of the kernel code:
        gcl_memcpy(x_host, d_output, sizeof(cl_float)*size_A_matrix);
    });
    
}


//MAIN CODE
int main()
{
    //Input of matrix ----------------------------------------------------------------------------------------------------------------
    char name[128];

    FILE *fileIN;
    
    if((fileIN = fopen("/Users/Tom/Documents/Jacobi_MatrixSolver/in.txt", "r")) == NULL)
    {
        fprintf(stdout," File does not work/open");
        return -1;
    }
    
    int size_A_matrix = 0;
    fscanf(fileIN,"%i", &size_A_matrix);
    printf("%i\n", size_A_matrix);
   //--------------------------------------------------------------------------------------------------------------------------------
    
    //Sizes of the different matrices from the equation Ax=B
    
    int szA, szB, sz_x;
    
    szA = size_A_matrix * size_A_matrix; //Square matrix
    szB = size_A_matrix;
    sz_x = size_A_matrix;
    
    //Memory allocation for the Matrices:
    
    float *A_host=(float*)malloc(sizeof(cl_float)*szA);
    float *B_host=(float*)malloc(sizeof(cl_float)*szB);
    float *x_host=(float*)malloc(sizeof(cl_float)*szA);
    
    //Load data into Host Matrices
    initmat(A_host, B_host, x_host);
    
    
    //Dispatch the OpenCl queue
    dispatch_queue_t queue= gcl_create_dispatch_queue(CL_DEVICE_TYPE, NULL);
    
    if(queue==NULL)
    {
        queue=gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU,NULL);
    }
    
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
    
    fprintf(stdout, "Created a dispatch queue using the %s\n", name);
    

    void* d_a  = gcl_malloc(sizeof(cl_float)*szA, A_host,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    void* d_b  = gcl_malloc(sizeof(cl_float)*szB, B_host,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    void* d_output = gcl_malloc(sizeof(cl_float)*szA, NULL, CL_MEM_WRITE_ONLY);

//OpenCl Calculations
    int iterations=10;
    
    for (int i=0; i<iterations; i++)
    {
OpenCl_Calc(queue,d_a,d_b,d_output,x_host,size_A_matrix);

OpenCl_Calc(queue,d_output,d_b,d_a,x_host,size_A_matrix);
       
    }
    
    //Print Function:
    for (int i = 0; i < size_A_matrix; i++){
        printf("Row #%i: \n",i+1);
        for (int j = 0; j < size_A_matrix; j++){
            printf("%f\n",x_host[i*size_A_matrix+j]);
        }
    }
    
    //MemoryCleanUp
    
    gcl_free(d_a);
    gcl_free(d_b);
    gcl_free(d_output);
    
    // And the same goes for system memory, as usual.
    free(x_host);
    free(A_host);
    free(B_host);
//    
//    // Finally, release your queue just as you would any GCD queue.    // 11
 dispatch_release(queue);
    
    
  
}

