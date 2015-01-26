
kernel void Ax_BSolver(const int width,global float* A,global float* B, global float* x)
{
    int j = get_global_id(1)+1;
    int i = get_global_id(0)+1;
    
    if(i<width-1){
        
        float temp= (A[i+(j-1)*width]+A[i+(j+1)*width]+A[i-1+j*width]+A[i+1+j*width]-B[i+j*width]*width*width)/4.0;
        
        x[i+j*width]=temp;
      
    }

}

