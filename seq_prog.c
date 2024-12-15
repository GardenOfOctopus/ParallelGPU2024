#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
//#include <omp.h>

#define N 512
#define T 0.001
#define K 20
#define Lx 1.0
#define Ly 1.0
#define Lz 1.0

#define M_PI 3.14159265358979323846

double hx, hy, hz;
double tau;

double (* Un2)[N + 1][N + 1];
double (* Un1)[N + 1][N + 1];
double (* Un)[N + 1][N + 1];
double (* temp)[N+1][N+1];

double Uanalytical(double x, double y, double z, double t);
double fi(double x, double y, double z);
void step();
void init();
double calculate_error();

double Uanalytical(double x, double y, double z, double t){
    return sin(M_PI * x / Lx) * sin(M_PI * y / Ly) * sin(M_PI * z / Lz) * cos(M_PI * sqrt(1 / pow(Lx, 2) + 1 / pow(Ly, 2) + 1 / pow(Lz, 2)) * t);
}

double fi(double x, double y, double z){
    return Uanalytical(x, y, z, 0);
}

double fi_index(int i, int j, int k){
    return Uanalytical(i*hx, j*hy, k*hz, 0);
}

double laplas_operator(double (* matrix)[N+1][N+1], int i, int j, int k){
	double x_comp = (matrix[i-1][j][k] - 2 * matrix[i][j][k] + matrix[i+1][j][k]) / pow(hx, 2);
	double y_comp = (matrix[i][j-1][k] - 2 * matrix[i][j][k] + matrix[i][j+1][k]) / pow(hy, 2);
	double z_comp = (matrix[i][j][k-1] - 2 * matrix[i][j][k] + matrix[i][j][k+1]) / pow(hz, 2);
	return  x_comp + y_comp + z_comp;
}

double fi_laplas(int i, int j, int k){
        double x_comp = (fi_index((i-1),j,k) - 2 * fi_index(i,j,k) + fi_index((i+1),j,k)) / pow(hx, 2);
        double y_comp = (fi_index(i,(j-1),k) - 2 * fi_index(i,j,k) + fi_index(i,(j+1),k)) / pow(hy, 2);
        double z_comp = (fi_index(i,j,(k-1)) - 2 * fi_index(i,j,k) + fi_index(i,j,(k+1))) / pow(hz, 2);
        return  x_comp + y_comp + z_comp;
}

void print_U(double (* matrix)[N+1][N+1]){
        int i, j, k;

        for (i = 0; i <= N; i++){
                for (j = 0; j <= N; j++){
                        for (k = 0; k <= N; k++){
                                printf("%.10f ", matrix[i][j][k]);
                        }
                        printf("\n");
                }
                printf("\n");
        }  
        printf("__________________________________\n");
}

int main(int argc, char ** argv){

	Un2 = malloc((N + 1) * (N + 1) * (N + 1) * sizeof(double)); // Un-2
	Un1 = malloc((N + 1) * (N + 1) * (N + 1) * sizeof(double)); // Un-1
    Un = malloc((N + 1) * (N + 1) * (N + 1) * sizeof(double)); 

	tau = T / K;
	hx = Lx / N;
    hy = Ly / N;
    hz = Lz / N;

	double error;

	struct timeval t1, t2, elapsed_time;
	gettimeofday(&t1, NULL);

	init();

	for (int t = 2; t <= K; t++){
		step();

		error = calculate_error(t);

        if (t % (K / 1) == 0)
            printf("Error: %.20f\n",error);
		
		temp = Un2;
    	Un2 = Un1;
    	Un1 = Un;
    	Un = temp;
	}

	gettimeofday(&t2, NULL);
	timersub(&t2, &t1, &elapsed_time);
	printf("Time: %f\n", elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0);


	return 0;
}

double calculate_error(int step_num){
	double error = 0.0;
	double diff;
        
	for (int i = 0; i <= N; i++){
	    for (int j = 0; j <= N; j++){
		    for (int k = 0; k <= N; k++){
				diff = Un[i][j][k] - Uanalytical(i * hx, j * hy, k * hz, tau * step_num);
				if (diff < 0){
					diff = diff * (-1);
				}
                if (diff > error){
                    error = diff;
				}
			}
		}
	}
	return error;
}

void init(){
	for (int i = 0; i <= N; i++){
		for (int j = 0; j <= N; j++){
			for (int k = 0; k <= N; k++){
				Un2[i][j][k] = fi(i * hx, j * hy, k * hz); // U0
            }
        }
    }

	for (int i = 1; i <= N - 1; i++){
		for (int j = 1; j <= N - 1; j++){
			for (int k = 1; k <= N - 1; k++){
				Un1[i][j][k] = Un2[i][j][k] + pow(tau, 2) * fi_laplas(i, j, k)/ 2; // U1
            }
        }
    }
}

void step(){
	for (int i = 1; i <= N - 1; i++){
		for (int j = 1; j <= N - 1; j++){
			for (int k = 1; k <= N - 1 ; k++) {	
				Un[i][j][k] = 2 * Un1[i][j][k] - Un2[i][j][k] + pow(tau, 2) * laplas_operator(Un1, i, j, k);
			}
		}
	}
}
