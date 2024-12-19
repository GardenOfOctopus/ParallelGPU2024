#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <cmath>

#include "mpi.h"
#include "types.h"

#define N 128
#define M (N + 1)
#define T 0.001
#define K 20
#define Lx 1.0
#define Ly 1.0
#define Lz 1.0
#define NUM_THREADS 8

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define index(i, j, k) ((i) + (j)*(bl_size_.x+2) + (k)*(bl_size_.x+2)*(bl_size_.y+2))

double hx, hy, hz;
double tau;
int proc_num;
int my_rank;
struct Bl_coords bl_coords;
struct Bl_size bl_size, usual_bl_size;
struct Bl_iterations bl_it;

__device__ double hx_, hy_, hz_;
__device__ double tau_;
__device__ int proc_num_;
__device__ int my_rank_;
__device__ struct Bl_coords bl_coords_;
__device__ struct Bl_size bl_size_, usual_bl_size_;
__device__ struct Bl_iterations bl_it_;

FILE *myfile;

void print_U(double * U){
        int i, j, k;

        for (i = 0; i < bl_size.x + 2; i++){
                for (j = 0; j < bl_size.y + 2; j++){
                        for (k = 0; k < bl_size.z + 2; k++){
                                //fprintf(myfile, "%6.4f ", b[i][j][k]);
                                printf("%.10f ", U[(i) + (j)*(bl_size.x+2) + (k)*(bl_size.x+2)*(bl_size.y+2)]);
                        }
                        //fprintf(myfile, "\n");
                        printf("\n");
                }
                //fprintf(myfile,"\n");
                printf("\n");
        }  
        //fprintf(myfile, "__________________________________\n");
        printf("__________________________________\n");
}

struct BoundaryInf{
        int ix, iy, iz, jx, jy, jz;
        int bx_r, by_r, bz_r;
        int bx_s, by_s, bz_s;  

        int x_size;
        int y_size;
};

class Boundary{      

        void calculate_boundary_matrix_parameters(){

                if (type == NORTH){
                        inf.ix = 1; inf.iy = 0; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 0; inf.jz = 1;
                        inf.bx_s = 1; inf.by_s = bl_size.y; inf.bz_s = 1;
                        inf.bx_r = 1; inf.by_r = bl_size.y + 1; inf.bz_r = 1;
                        inf.x_size = bl_size.x ; inf.y_size = bl_size.z;
                } else if (type == SOUTH){
                        inf.ix = 1; inf.iy = 0; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 0; inf.jz = 1;
                        inf.bx_s = 1; inf.by_s = 1; inf.bz_s = 1;
                        inf.bx_r = 1; inf.by_r = 0; inf.bz_r = 1;
                        inf.x_size = bl_size.x; inf.y_size = bl_size.z;
                } else if (type == EAST){
                        inf.ix = 0; inf.iy = 1; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 0; inf.jz = 1;
                        inf.bx_s = bl_size.x; inf.by_s = 1; inf.bz_s = 1;
                        inf.bx_r = bl_size.x + 1; inf.by_r = 1; inf.bz_r = 1;
                        inf.x_size = bl_size.y; inf.y_size = bl_size.z;
                } else if (type == WEST){
                        inf.ix = 0; inf.iy = 1; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 0; inf.jz = 1;
                        inf.bx_s = 1; inf.by_s = 1; inf.bz_s = 1;
                        inf.bx_r = 0; inf.by_r = 1; inf.bz_r = 1;
                        inf.x_size = bl_size.y; inf.y_size = bl_size.z;
                } else if (type == TOP){
                        inf.ix = 1; inf.iy = 0; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 1; inf.jz = 0;
                        inf.bx_s = 1; inf.by_s = 1; inf.bz_s = bl_size.z;
                        inf.bx_r = 1; inf.by_r = 1; inf.bz_r = bl_size.z + 1;
			inf.x_size = bl_size.x; inf.y_size = bl_size.y;
                } else if (type == BOTTOM){
                        inf.ix = 1; inf.iy = 0; inf.iz = 0; 
                        inf.jx = 0; inf.jy = 1; inf.jz = 0;
                        inf.bx_s = 1; inf.by_s = 1; inf.bz_s = 1;
                        inf.bx_r = 1; inf.by_r = 1; inf.bz_r = 0;
			inf.x_size = bl_size.x; inf.y_size = bl_size.y;
                }
        }

public:
        BoundaryType type;
        bool has_neighbour;
        int neighbour_rank;
        MPI_Request request;
        MPI_Status status;

        double * recv_matrix;
        double * send_matrix;

        BoundaryInf inf;  

        Boundary(int type_, int neighbour_rank_){
                type = (BoundaryType) type_;
							
		has_neighbour = false;
		neighbour_rank = neighbour_rank_;
		inf.x_size = 0;
		inf.y_size = 0;
                if (neighbour_rank_ == -1) 
			return;

		has_neighbour = true;

                calculate_boundary_matrix_parameters();

                recv_matrix = (double *) malloc(inf.x_size * inf.y_size * sizeof(double));
                send_matrix = (double *) malloc(inf.x_size * inf.y_size * sizeof(double));
        }
};

__global__  void fill_un1_by_recv_(double * buf_, double * Un1_, BoundaryInf inf){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;

        if (i >= 0 && i < inf.x_size && j >= 0 && j < inf.y_size){ // ????? = 0
                Un1_[index(i*inf.ix + j*inf.jx + inf.bx_r,i*inf.iy + j*inf.jy + inf.by_r,i*inf.iz + j*inf.jz + inf.bz_r)] = buf_[j * inf.x_size + i];
        } 
}

__global__  void fill_send_by_un1_(double * Un1_, double * buf_, BoundaryInf inf){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;

        if (i >= 0 && i < inf.x_size && j >= 0 && j < inf.y_size){ // ????? = 0
                buf_[j * inf.x_size + i] = Un1_[index(i*inf.ix + j*inf.jx + inf.bx_s, i*inf.iy + j*inf.jy + inf.by_s, i*inf.iz + j*inf.jz + inf.bz_s)];
        }
}

std::vector<Boundary> my_boundaries;

int calculate_neighbour_rank(int type, int a, int b, int c){

        if (type == NORTH && bl_coords.y + 1 != b){
                return my_rank + a;
        } else if (type == SOUTH && bl_coords.y != 0){
                return my_rank - a;
        } else if (type == EAST && bl_coords.x + 1 != a){
                return my_rank + 1;
        } else if (type == WEST && bl_coords.x != 0){
                return my_rank - 1;
        } else if (type == TOP && bl_coords.z + 1 != c){
                return my_rank + a * b;
        } else if (type == BOTTOM && bl_coords.z != 0){
                return my_rank - a * b;
        }
        return -1;
}

void calculate_begin_end(){
        bl_it.i_begin = 1; bl_it.i_end = (bl_size.x + 2) - 1; 
        bl_it.j_begin = 1; bl_it.j_end = (bl_size.y + 2) - 1; 
        bl_it.k_begin = 1; bl_it.k_end = (bl_size.z + 2) - 1;

        if (!my_boundaries[EAST].has_neighbour) bl_it.i_end--;
        if (!my_boundaries[WEST].has_neighbour) bl_it.i_begin++;
        if (!my_boundaries[NORTH].has_neighbour) bl_it.j_end--;
        if (!my_boundaries[SOUTH].has_neighbour) bl_it.j_begin++;
        if (!my_boundaries[TOP].has_neighbour) bl_it.k_end--;
        if (!my_boundaries[BOTTOM].has_neighbour) bl_it.k_begin++;
}

// Находим блок процесса по его номеру: размеры блока, его расположение, соседей, инфу о границах
void configure_my_block(int my_rank, int a, int b, int c){

        bl_coords.x = my_rank % a;
        bl_coords.y = (my_rank / a) % b;
        bl_coords.z = my_rank / (a * b);

        bl_size.x = M / a;
        bl_size.y = M / b;
        bl_size.z = M / c;

        if (M % a != 0) bl_size.x++;
        if (M % b != 0) bl_size.y++;
        if (M % c != 0) bl_size.z++;

        usual_bl_size.x = bl_size.x;
        usual_bl_size.y = bl_size.y;
        usual_bl_size.z = bl_size.z;

		
        if (bl_coords.x == a - 1) bl_size.x = (M - (a - 1) * bl_size.x);
        if (bl_coords.y == b - 1) bl_size.y = (M - (b - 1) * bl_size.y);
        if (bl_coords.z == c - 1) bl_size.z = (M - (c - 1) * bl_size.z);

        for (int i = 0; i < 6; i++){
                my_boundaries.push_back(Boundary(i, calculate_neighbour_rank(i, a, b, c)));
        }

        calculate_begin_end();
}

void step();
void init();
void calculate_error(int step_num, double * U);
void print_error_matrix(int step_num);
void exchange_data(double *Un1_, double* buf_);

double Uanalytical(double i_local, double j_local, double k_local, double t){
        double x = (bl_coords.x * usual_bl_size.x + (i_local - 1)) * hx;
        double y = (bl_coords.y * usual_bl_size.y + (j_local - 1)) * hy;
        double z = (bl_coords.z * usual_bl_size.z + (k_local - 1)) * hz;
        return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(M_PI * z / Lz) * cos(M_PI * sqrt(1 / pow(Lx, 2) + 1 / pow(Ly, 2) + 1 / pow(Lz, 2)) * t);
}

__device__ double Uanalytical_(double i_local, double j_local, double k_local, double t){
        double x = (bl_coords_.x * usual_bl_size_.x + (i_local - 1)) * hx_;
        double y = (bl_coords_.y * usual_bl_size_.y + (j_local - 1)) * hy_;
        double z = (bl_coords_.z * usual_bl_size_.z + (k_local - 1)) * hz_;
        return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(M_PI * z / Lz) * cos(M_PI * sqrt(1 / pow(Lx, 2) + 1 / pow(Ly, 2) + 1 / pow(Lz, 2)) * t);
}

__device__ double fi_(double i, double j, double k){
        return Uanalytical_(i, j, k, 0);
}

__device__ double laplas_operator_(double* b, int i, int j, int k){
        double x_comp = (b[index(i-1,j,k)] - 2 * b[index(i,j,k)] + b[index(i+1,j,k)]) / pow(hx_, 2);
        double y_comp = (b[index(i,j-1,k)] - 2 * b[index(i,j,k)] + b[index(i,j+1,k)]) / pow(hy_, 2);
        double z_comp = (b[index(i,j,k-1)] - 2 * b[index(i,j,k)] + b[index(i,j,k+1)]) / pow(hz_, 2);
        return  x_comp + y_comp + z_comp;
}

__device__ double fi_laplas_(int i, int j, int k){
        double x_comp = (fi_((i-1),j,k) - 2 * fi_(i,j,k) + fi_((i+1),j,k)) / pow(hx_, 2);
        double y_comp = (fi_(i,(j-1),k) - 2 * fi_(i,j,k) + fi_(i,(j+1),k)) / pow(hy_, 2);
        double z_comp = (fi_(i,j,(k-1)) - 2 * fi_(i,j,k) + fi_(i,j,(k+1))) / pow(hz_, 2);
        return  x_comp + y_comp + z_comp;
}

void calculate_time(double delta){
        MPI_Barrier(MPI_COMM_WORLD);
        double reduced_time;
        MPI_Reduce(&delta, &reduced_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (my_rank != 0){
            return;
        }
        printf("Time: %.20f\n", reduced_time);
}

std::vector<int> get_multipliers(int n){
        int div = 2;
        std::vector<int> result;
        while (n > 1){
                while (n % div == 0){
                        result.push_back(div);
                        n = n / div;
                }
                if (div == 2) div++;
                else div += 2;
        }
        return result;
}

void split_tasks(int & a, int & b, int &c){
        std::vector<int> mults = get_multipliers(proc_num);
				
        int len = mults.size();
 
        if (len == 0){
		a = 1; b = 1; c = 1;
	} else if (len == 1){
                a = mults[0]; b = 1; c = 1;
	} else if (len == 2){
                a = mults[0]; b = mults[1]; c = 1;
        } else if (len == 3){
                a = mults[0]; b = mults[1]; c = mults[2];
        } else if (len == 4){
                a = mults[0] * mults[1]; b = mults[2]; c = mults[3];
        } else if (len == 5){
                a = mults[0] * mults[4]; b = mults[1]*mults[3]; c = mults[3];
        } else{
                int n = len / 3;
                for (int i = 0; i < n; i ++){
                        a = mults[3*i] * mults[(n - 1) - 3*i];
                        b = mults[3*i + 1] * mults[(n - 1) - (3*i + 1)];
                        c = mults[3*i + 2] * mults[(n - 1) - (3*i + 2)];
                }
                if (n % 3 == 1){
                        a = a * mults[n / 2];
                }
                if (n % 3 == 2){
                        b = b * mults[n / 2];
                        a = a * mults[n / 2 - 1];
                }
        }
}  

void print_information(){
	std:: cout << "Proc_num: " << proc_num << "\n";
	std::cout << "My rank: " << my_rank << "\n";
        std::cout << "My block coords: " << bl_coords.x << " " << bl_coords.y << " " << bl_coords.z << "\n";
        std::cout << "My block size: " << bl_size.x << " " << bl_size.y << " " << bl_size.z << "\n";
	std::cout << "My iterations: " << bl_it.i_begin << " " << bl_it.i_end << " " << bl_it.j_begin << " " << bl_it.j_end << " " << bl_it.k_begin << " " << bl_it.k_end << "\n";

        for (int i = 0; i < 6; i++){
                std::cout << "Boundary " << i << "\n";
                std::cout << "    type=" << my_boundaries[i].type << "\n";
                std::cout << "    has_neighbour=" << my_boundaries[i].has_neighbour << "\n";
                std::cout << "    neighbour_rank=" << my_boundaries[i].neighbour_rank << "\n";
                std::cout << "    x_size=" << my_boundaries[i].inf.x_size << "\n";
                std::cout << "    y_size=" << my_boundaries[i].inf.y_size << "\n";
        }
}

__global__ void init_(double* Un2_, double* Un1_){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        if (i >= bl_it_.i_begin && i < bl_it_.i_end && j >= bl_it_.j_begin && j < bl_it_.j_end && k >= bl_it_.k_begin && k < bl_it_.k_end) {
                Un2_[index(i,j,k)] = fi_(i, j, k);
                Un1_[index(i,j,k)] = Un2_[index(i,j,k)] + pow(tau_, 2) * fi_laplas_(i, j, k)/ 2;
        }
}

__global__ void step_(double* Un2_, double* Un1_, double* Un_){
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int j = threadIdx.y + blockDim.y * blockIdx.y;
        int k = threadIdx.z + blockDim.z * blockIdx.z;

        if (i >= bl_it_.i_begin && i < bl_it_.i_end && j >= bl_it_.j_begin && j < bl_it_.j_end && k >= bl_it_.k_begin && k < bl_it_.k_end) {
                Un_[index(i,j,k)] = 2 * Un1_[index(i,j,k)] - Un2_[index(i,j,k)] + pow(tau_, 2) * laplas_operator_(Un1_, i, j, k);
        }
}

int main(int argc, char ** argv){

        tau = T / K;
        hx = Lx / N;
        hy = Ly / N;
        hz = Lz / N;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        int a, b, c;
        split_tasks(a, b, c);
        configure_my_block(my_rank, a, b, c);
        //print_information();

        double * U = (double *)malloc(sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2));
        double* temp;
        double* Un2_;
        double* Un1_;
        double* Un_;
        double* buf_;

        gpuErrchk(cudaMalloc((void **)&Un2_, sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2)));
        gpuErrchk(cudaMalloc((void **)&Un1_, sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2)));
        gpuErrchk(cudaMalloc((void **)&Un_, sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2)));
        gpuErrchk(cudaMalloc((void **)&buf_, sizeof(double) * pow(max(bl_size.x, max(bl_size.y, bl_size.z)), 2)));
        gpuErrchk(cudaMemcpyToSymbol(tau_, &tau, sizeof(double)));
        gpuErrchk(cudaMemcpyToSymbol(hx_, &hx, sizeof(double)));
        gpuErrchk(cudaMemcpyToSymbol(hy_, &hy, sizeof(double)));
        gpuErrchk(cudaMemcpyToSymbol(hz_, &hz, sizeof(double)));
        gpuErrchk(cudaMemcpyToSymbol(bl_it_, &bl_it, sizeof(int) * 6));
        gpuErrchk(cudaMemcpyToSymbol(bl_size_, &bl_size, sizeof(int) * 3));
        gpuErrchk(cudaMemcpyToSymbol(usual_bl_size_, &usual_bl_size, sizeof(int) * 3));
        gpuErrchk(cudaMemcpyToSymbol(bl_coords_, &bl_coords, sizeof(int) * 3));

        double start = MPI_Wtime();

        dim3 num_threads(NUM_THREADS, NUM_THREADS, NUM_THREADS);
        dim3 num_blocks((bl_size.x + 2) / NUM_THREADS + 1, (bl_size.y + 2) / NUM_THREADS + 1, (bl_size.z + 2) / NUM_THREADS + 1);
        double start_init = MPI_Wtime();
        init_<<<num_blocks, num_threads>>>(Un2_, Un1_);
        gpuErrchk(cudaDeviceSynchronize());
        printf("init_ ended, time: %.20f\n", MPI_Wtime() - start_init);

        //gpuErrchk(cudaMemcpy(U, Un1_, sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2), cudaMemcpyDeviceToHost));
        //print_U(U);

        for (int t = 2; t <= K; t++){
                double start_exchange = MPI_Wtime();
                exchange_data(Un1_, buf_);
                printf("exchange_data() t=%d ended, time:%.20f\n", t, MPI_Wtime() - start_exchange);

                double start_step = MPI_Wtime();
                step_<<<num_blocks, num_threads>>>(Un2_, Un1_, Un_);
                gpuErrchk(cudaDeviceSynchronize());
                printf("step_ t=%d ended, time: %.20f\n", t, MPI_Wtime() - start_step);

                //if (t % (K / 1) == 0){
                //        double start_memcpy = MPI_Wtime();
                //        gpuErrchk(cudaMemcpy(U, Un_, sizeof(double) * (bl_size.x + 2) * (bl_size.y + 2) * (bl_size.z + 2), cudaMemcpyDeviceToHost));
                //        printf("cudaMemcpy() ended, time: %.20f\n", MPI_Wtime() - start_memcpy);
//
                //        double start_calc = MPI_Wtime();
                //        calculate_error(t, U);
                //        printf("calculate_error() ended, time: %.20f\n", MPI_Wtime() - start_calc);
                //}

                temp = Un2_;
                Un2_ = Un1_;
                Un1_ = Un_;
                Un_ = temp;
        }


        double delta = MPI_Wtime() - start;
        calculate_time(delta);

        gpuErrchk(cudaFree(Un2_));
        gpuErrchk(cudaFree(Un1_));
        gpuErrchk(cudaFree(Un_));
        gpuErrchk(cudaFree(buf_));

        free(U);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();

        return 0;
}


void calculate_error(int step_num, double * U){
        double error = 0.0;
        double diff;

        int i, j, k;
        
        for (i = bl_it.i_begin; i < bl_it.i_end; i++){
                for (j = bl_it.j_begin; j < bl_it.j_end; j++){
                        for (k = bl_it.k_begin; k < bl_it.k_end; k++){
                                diff = std::fabs(U[(i) + (j)*(bl_size.x+2) + (k)*(bl_size.x+2)*(bl_size.y+2)] - Uanalytical(i, j, k, tau * step_num));
                                if (diff > error)
                                    error = diff;
                        }
                }
        }
        double reduced_error;
        MPI_Reduce(&error, &reduced_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (my_rank != 0){
            return;
        }
        printf("Error: %.20f\n", reduced_error);
}

void print_error_matrix(int step_num){
        int i, j, k;
        
        for (i = bl_it.i_begin; i < bl_it.i_end; i++){
                for (j = bl_it.j_begin; j < bl_it.j_end; j++){
                        for (k = bl_it.k_begin; k < bl_it.k_end; k++){
                                printf("%10.10f ", std::fabs(Uanalytical(i, j, k, tau * step_num)));;
                        }
                        printf("\n");
                }
                printf("\n");
        }
}


void exchange_data(double *Un1_, double* buf_){
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){

                        dim3 num_threads(NUM_THREADS, NUM_THREADS);
                        dim3 num_blocks(my_boundaries[i].inf.x_size / NUM_THREADS + 1, my_boundaries[i].inf.y_size / NUM_THREADS + 1);
                        fill_send_by_un1_<<<num_blocks, num_threads>>>(Un1_, buf_, my_boundaries[i].inf);
                        gpuErrchk(cudaDeviceSynchronize())
                        int matrix_size = my_boundaries[i].inf.x_size * my_boundaries[i].inf.y_size;
                        cudaMemcpy(my_boundaries[i].send_matrix, buf_, sizeof(double) * matrix_size, cudaMemcpyDeviceToHost); //??????

                        MPI_Isend(my_boundaries[i].send_matrix, matrix_size, MPI_DOUBLE, my_boundaries[i].neighbour_rank, my_rank, MPI_COMM_WORLD, &my_boundaries[i].request);
                }
        }

        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){
                        int matrix_size = my_boundaries[i].inf.x_size * my_boundaries[i].inf.y_size;
                        MPI_Status status;
                        MPI_Recv(my_boundaries[i].recv_matrix, matrix_size, MPI_DOUBLE, my_boundaries[i].neighbour_rank, my_boundaries[i].neighbour_rank, MPI_COMM_WORLD, &status);
                }
        }

        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){
                        MPI_Status status;
                        MPI_Wait(&my_boundaries[i].request, &status);

                        dim3 num_threads(NUM_THREADS, NUM_THREADS);
                        dim3 num_blocks(my_boundaries[i].inf.x_size / NUM_THREADS + 1, my_boundaries[i].inf.y_size / NUM_THREADS + 1);
                        int matrix_size = my_boundaries[i].inf.x_size * my_boundaries[i].inf.y_size;
                        cudaMemcpy(buf_, my_boundaries[i].recv_matrix, sizeof(double) * matrix_size, cudaMemcpyHostToDevice);
                        fill_un1_by_recv_<<<num_blocks, num_threads>>>(buf_, Un1_, my_boundaries[i].inf);
                        gpuErrchk(cudaDeviceSynchronize())
                }
        }
}
      
                                                
