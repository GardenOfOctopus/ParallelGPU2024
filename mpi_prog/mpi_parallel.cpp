#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>

#include "mpi.h"

#define N 128
#define M (N+1)
#define T 0.001
#define K 20
#define Lx 1.0
#define Ly 1.0
#define Lz 1.0

//#define M_PI 3.14159265358979323846

double tau;

typedef std::vector<double> Line;
typedef std::vector<Line> Matrix;
typedef std::vector<Matrix> Block;

Block Un2;
Block Un1;
Block Un;

double hx, hy, hz;

int proc_num;
int my_rank;
struct Bl_coords{
        int x; int y; int z;
} bl_coords;

struct Bl_size{
        int x; int y; int z;
};

Bl_size bl_size, usual_bl_size;

enum BoundaryType{
        NORTH,
        EAST, 
        SOUTH, 
	WEST,
        TOP,
        BOTTOM, 
};

FILE *myfile;

void print_U(Block &b){
        int i, j, k;

        for (i = 0; i < bl_size.x + 2; i++){
                for (j = 0; j < bl_size.y + 2; j++){
                        for (k = 0; k < bl_size.z + 2; k++){
                                printf("%.10f ", b[i][j][k]);
                        }
                        printf("\n");
                }
                printf("\n");
        }  
        printf("__________________________________\n");
}

class Boundary{

        int ix, iy, iz, jx, jy, jz;
        int bx_r, by_r, bz_r;
        int bx_s, by_s, bz_s;

        void calculate_boundary_matrix_parameters(){

                if (type == NORTH){
                        ix = 1; iy = 0; iz = 0; 
                        jx = 0; jy = 0; jz = 1;
                        bx_s = 1; by_s = bl_size.y; bz_s = 1;
                        bx_r = 1; by_r = bl_size.y + 1; bz_r = 1;
                        x_size = bl_size.x ; y_size = bl_size.z;
                } else if (type == SOUTH){
                        ix = 1; iy = 0; iz = 0; 
                        jx = 0; jy = 0; jz = 1;
                        bx_s = 1; by_s = 1; bz_s = 1;
                        bx_r = 1; by_r = 0; bz_r = 1;
                        x_size = bl_size.x; y_size = bl_size.z;
                } else if (type == EAST){
                        ix = 0; iy = 1; iz = 0; 
                        jx = 0; jy = 0; jz = 1;
                        bx_s = bl_size.x; by_s = 1; bz_s = 1;
                        bx_r = bl_size.x + 1; by_r = 1; bz_r = 1;
                        x_size = bl_size.y; y_size = bl_size.z;
                } else if (type == WEST){
                        ix = 0; iy = 1; iz = 0; 
                        jx = 0; jy = 0; jz = 1;
                        bx_s = 1; by_s = 1; bz_s = 1;
                        bx_r = 0; by_r = 1; bz_r = 1;
                        x_size = bl_size.y; y_size = bl_size.z;
                } else if (type == TOP){
                        ix = 1; iy = 0; iz = 0; 
                        jx = 0; jy = 1; jz = 0;
                        bx_s = 1; by_s = 1; bz_s = bl_size.z;
                        bx_r = 1; by_r = 1; bz_r = bl_size.z + 1;
			x_size = bl_size.x; y_size = bl_size.y;
                } else if (type == BOTTOM){
                        ix = 1; iy = 0; iz = 0; 
                        jx = 0; jy = 1; jz = 0;
                        bx_s = 1; by_s = 1; bz_s = 1;
                        bx_r = 1; by_r = 1; bz_r = 0;
			x_size = bl_size.x; y_size = bl_size.y;
                }
        }

public:
        BoundaryType type;
        bool has_neighbour;
        int neighbour_rank;
        MPI_Request request;
        MPI_Status status;

        int x_size;
        int y_size;

        double * recv_matrix;
        double * send_matrix;

        Boundary(int type_, int neighbour_rank_){
                type = (BoundaryType) type_;
							
		has_neighbour = false;
		neighbour_rank = neighbour_rank_;
		x_size = 0;
		y_size = 0;
                if (neighbour_rank_ == -1) 
			return;

		has_neighbour = true;

                calculate_boundary_matrix_parameters();

                recv_matrix = (double *) malloc(x_size * y_size * sizeof(double));
                send_matrix = (double *) malloc(x_size * y_size * sizeof(double));
        }

        void fill_send_by_un1(){

                for (int i = 0; i < x_size; i++){
                        for (int j = 0; j < y_size; j++){
                              send_matrix[j * x_size + i] = Un1[i*ix + j*jx + bx_s][i*iy + j*jy + by_s][i*iz + j*jz + bz_s];
                        }
                }
        }

        void fill_un1_by_recv(){

                for (int i = 0; i < x_size; i++){
                        for (int j = 0; j < y_size; j++){
                                Un1[i*ix + j*jx + bx_r][i*iy + j*jy + by_r][i*iz + j*jz + bz_r] = recv_matrix[j * x_size + i];
                        }
                }
        }
};

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

int i_begin, i_end, j_begin, j_end, k_begin, k_end;

void calculate_begin_end(){
        i_begin = 1; i_end = (bl_size.x + 2) - 1; j_begin = 1; j_end = (bl_size.y + 2) - 1; k_begin = 1; k_end = (bl_size.z + 2) - 1;

        if (!my_boundaries[EAST].has_neighbour) i_end--;
        if (!my_boundaries[WEST].has_neighbour) i_begin++;
        if (!my_boundaries[NORTH].has_neighbour) j_end--;
        if (!my_boundaries[SOUTH].has_neighbour) j_begin++;
        if (!my_boundaries[TOP].has_neighbour) k_end--;
        if (!my_boundaries[BOTTOM].has_neighbour) k_begin++;
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
void calculate_error(int step_num);
void print_error_matrix(int step_num);
void exchange_data();

double Uanalytical(double i_local, double j_local, double k_local, double t){
        double x = (bl_coords.x * usual_bl_size.x + (i_local - 1)) * hx;
        double y = (bl_coords.y * usual_bl_size.y + (j_local - 1)) * hy;
        double z = (bl_coords.z * usual_bl_size.z + (k_local - 1)) * hz;
        return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(M_PI * z / Lz) * cos(M_PI * sqrt(1 / pow(Lx, 2) + 1 / pow(Ly, 2) + 1 / pow(Lz, 2)) * t);
}

double fi(double i, double j, double k){
        double ret = Uanalytical(i, j, k, 0);
        return ret;
}


double laplas_operator(Block &b, int i, int j, int k){
        double x_comp = (b[i-1][j][k] - 2 * b[i][j][k] + b[i+1][j][k]) / pow(hx, 2);
        double y_comp = (b[i][j-1][k] - 2 * b[i][j][k] + b[i][j+1][k]) / pow(hy, 2);
        double z_comp = (b[i][j][k-1] - 2 * b[i][j][k] + b[i][j][k+1]) / pow(hz, 2);
        return  x_comp + y_comp + z_comp;
        //return b[i-1][j][k] + b[i][j][k] + b[i+1][j][k] + b[i][j-1][k] + b[i][j+1][k] + b[i][j][k-1] + b[i][j][k+1];
}

double fi_laplas(int i, int j, int k){
        double x_comp = (fi((i-1),j,k) - 2 * fi(i,j,k) + fi((i+1),j,k)) / pow(hx, 2);
        double y_comp = (fi(i,(j-1),k) - 2 * fi(i,j,k) + fi(i,(j+1),k)) / pow(hy, 2);
        double z_comp = (fi(i,j,(k-1)) - 2 * fi(i,j,k) + fi(i,j,(k+1))) / pow(hz, 2);
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
        std:: cout << "Split is a=" << a << " b=" << b << " c=" << c << "\n";
        if (a * b * c != proc_num){
                printf("Alarm! Haos! Entropy!\n");
        }
}  

void print_information(){
	std:: cout << "Proc_num: " << proc_num << "\n";
	std::cout << "My rank: " << my_rank << "\n";
        std::cout << "My block coords: " << bl_coords.x << " " << bl_coords.y << " " << bl_coords.z << "\n";
        std::cout << "My block size: " << bl_size.x << " " << bl_size.y << " " << bl_size.z << "\n";
	std::cout << "My iterations: " <<i_begin << " " << i_end << " " << j_begin << " " << j_end << " " << k_begin << " " << k_end << "\n";

        for (int i = 0; i < 6; i++){
                std::cout << "Boundary " << i << "\n";
                std::cout << "    type=" << my_boundaries[i].type << "\n";
                std::cout << "    has_neighbour=" << my_boundaries[i].has_neighbour << "\n";
                std::cout << "    neighbour_rank=" << my_boundaries[i].neighbour_rank << "\n";
                std::cout << "    x_size=" << my_boundaries[i].x_size << "\n";
                std::cout << "    y_size=" << my_boundaries[i].y_size << "\n";
        }
}

int main(int argc, char ** argv){

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        int a, b, c;
        split_tasks(a, b, c);

        configure_my_block(my_rank, a, b, c);

        print_information();

        tau = T / K;
        hx = Lx / N;
        hy = Ly / N;
        hz = Lz / N;

        Un2 = Block(bl_size.x + 2, Matrix(bl_size.y + 2, Line(bl_size.z + 2, 0)));
        Un1 = Block(bl_size.x + 2, Matrix(bl_size.y + 2, Line(bl_size.z + 2, 0)));
        Un = Block(bl_size.x + 2, Matrix(bl_size.y + 2, Line(bl_size.z + 2, 0)));

        double error;
        double start = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        init();

        for (int t = 2; t <= K; t++){
                exchange_data();
                
                step(); 

                if (t % (K / 1) == 0)
                        calculate_error(t);

                Un2.swap(Un1);
                Un1.swap(Un);
        }


        double delta = MPI_Wtime() - start;
        calculate_time(delta);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();

        return 0;
}

void calculate_error(int step_num){
        double error = 0.0;
        double diff;

        int i, j, k;
        
        for (i = i_begin; i < i_end; i++){
                for (j = j_begin; j < j_end; j++){
                        for (k = k_begin; k < k_end; k++){
                                diff = fabs(Un[i][j][k] - Uanalytical(i, j, k, tau * step_num));
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


void init(){
        std::cout << "Init() start " << my_rank << "\n";
        int i, j, k;
        
        for (i = i_begin; i < i_end; i++){
                for (j = j_begin; j < j_end; j++){
                        for (k = k_begin; k < k_end; k++){
                                Un2[i][j][k] = fi(i, j, k); // U0
                                Un1[i][j][k] = Un2[i][j][k] + pow(tau, 2) * fi_laplas(i, j, k)/ 2; // U1 
                        }
                }
        }
        std::cout << "Init() end " << my_rank << "\n";
}

void step(){
        std::cout << "Step() start " << my_rank << "\n";
        int i, j, k;

        
        for (i = i_begin; i < i_end; i++){
                for (j = j_begin; j < j_end; j++){
                        for (k = k_begin; k < k_end; k++){
                                Un[i][j][k] = 2 * Un1[i][j][k] - Un2[i][j][k] + pow(tau, 2) * laplas_operator(Un1, i, j, k);
                        }
                }
        }
        std::cout << "Step() end " << my_rank << "\n";
}

void exchange_data(){
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "Exchange() start " << my_rank << "\n";
        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){
                        my_boundaries[i].fill_send_by_un1();
                        int matrix_size = my_boundaries[i].x_size * my_boundaries[i].y_size;
                        MPI_Isend(my_boundaries[i].send_matrix, matrix_size, MPI_DOUBLE, my_boundaries[i].neighbour_rank, my_rank, MPI_COMM_WORLD, &my_boundaries[i].request);
                }
        }

        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){
                        int matrix_size = my_boundaries[i].x_size * my_boundaries[i].y_size;
                        MPI_Status status;
                        MPI_Recv(my_boundaries[i].recv_matrix, matrix_size, MPI_DOUBLE, my_boundaries[i].neighbour_rank, my_boundaries[i].neighbour_rank, MPI_COMM_WORLD, &status);
                }
        }

        for (int i = 0; i < 6; i++){
                if (my_boundaries[i].has_neighbour){
                        MPI_Status status;
                        MPI_Wait(&my_boundaries[i].request, &status);
                        my_boundaries[i].fill_un1_by_recv();
                }
        }
        
        std::cout << "Exchange() end " << my_rank << "\n";
}
      
                                                
