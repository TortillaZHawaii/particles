CC=nvcc

GLUT_LIBS=-lGL -lGLU -lglut
CU_LIBS=-I"/usr/local/cuda/samples/common/inc"

LDLIBS=${GLUT_LIBS} ${CU_LIBS}

FILES=main.o

all: particles

particles: ${FILES}
	${CC} ${FILES} -o particles ${LDLIBS}

main.o: main.cu
	${CC} -c main.cu -o main.o ${LDLIBS}

# remember about -dc
# calculations_utils.o: calculations_utils.cuh
# 	${CC} -c calculations_utils.cu -o calculations_utils.o ${LDLIBS} -dc

clean:
	rm -f particles *.o
