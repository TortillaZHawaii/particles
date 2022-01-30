CC=nvcc

GLUT_LIBS=-lGL -lGLU -lglut
CU_LIBS=-I"/usr/local/cuda/samples/common/inc"

LDLIBS=${GLUT_LIBS} ${CU_LIBS}

FILES=main.o cpu.o gpu.o shm.o shared.o setup.o

all: particles

particles: ${FILES}
	${CC} ${FILES} -o particles ${LDLIBS}

# MAIN
main.o: main.cu
	${CC} -c main.cu -o main.o ${LDLIBS} -dc

setup.o: setup.cu setup.cuh
	${CC} -c setup.cu -o setup.o ${LDLIBS} -dc

# KERNELS
cpu.o: kernels/cpu.cu kernels/cpu.cuh
	${CC} -c kernels/cpu.cu -o cpu.o ${LDLIBS} -dc

gpu.o: kernels/gpu.cu kernels/gpu.cuh
	${CC} -c kernels/gpu.cu -o gpu.o ${LDLIBS} -dc

shm.o: kernels/shm.cu kernels/shm.cuh
	${CC} -c kernels/shm.cu -o shm.o ${LDLIBS} -dc

shared.o: kernels/shared.cu kernels/shared.cuh
	${CC} -c kernels/shared.cu -o shared.o ${LDLIBS} -dc


# CLEAN
clear:
	clean

clean:
	rm -f particles *.o
