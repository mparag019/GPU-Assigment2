#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;
using std::endl;

typedef long long ll;


__global__ void dkernel(long int* g_mat, long int *g_ans, long int* g_filter, int m, int n, int k){

    extern __shared__ long int fil_cpy[];


    int i = blockIdx.x;
    int j = threadIdx.x;

    if (threadIdx.x == 0){
        for(int l = 0; l < k * k; l++){
            fil_cpy[l] = g_filter[l];
        }
    }

    __syncthreads();
    
    for(int p = i - k/2; p < i - k/2 + k; p++){
        for(int q = j - k/2; q < j - k/2 + k; q++){
            if (p >= 0 && p <= m - 1 && q >= 0 && q <= n - 1){
                g_ans[i * n + j] += (g_mat[p * n + q] * fil_cpy[(p - i + k/2) * k + q - j + k/2]);
            }
        }
    }
}


int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    long int* g_mat = new long int[m * n];
    long int* g_filter = new long int[k * k];
    long int* g_ans = new long int[m * n];

    cudaMalloc(&g_mat,m * n * sizeof(long int));
    cudaMalloc(&g_filter,k * k * sizeof(long int));
    cudaMalloc(&g_ans,m * n * sizeof(long int));

    cudaMemcpy(g_mat, h_mat, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_filter, h_filter, k * k * sizeof(long int), cudaMemcpyHostToDevice);

    
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch

    dkernel<<<m, n, k * k * sizeof(long int)>>>(g_mat, g_ans, g_filter, m, n, k);

    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch

    cudaDeviceSynchronize();

    cudaMemcpy(h_ans, g_ans, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    
    
    
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */
 

    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}
