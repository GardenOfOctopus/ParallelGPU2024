ARCH = sm_60
HOST_COMP = mpicc
CUDA_COMP = nvcc
TARGET = cuda

SRC = cuda_parallel.cu

MPI_FLAGS = -O3 -std=c++11
CUDA_FLAGS = -Xptxas -dlcm=cg -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

all: $(TARGET)

$(TARGET): $(SRC)
	module load SpectrumMPI; $(CUDA_COMP) $(MPI_FLAGS) $(CUDA_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean