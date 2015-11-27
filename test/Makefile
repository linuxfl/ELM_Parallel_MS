#
# Makefile for regression test
#

INC_CLAS        = /opt/OpenBLAS/include/
LIB_CLAS        = /opt/OpenBLAS/lib/

CFLAGS		= -Wall -I${DIR_INC} -I${INC_CLAS}
LDFLAGS		= -lm -fopenmp

regression_test:regression_test.o ../src/matrix.o ../src/elm.o
	gcc regression_test.o ../src/matrix.o ../src/elm.o ${LDFLAGS} -o $@ -L ${LIB_CLAS} -lopenblas
regression_test.o ./src/matrix.o ./src/elm.o:regression_test.c ../src/matrix.c ../src/elm.c
	gcc $(CFLAGS) -c $< -o $@ -fopenmp
.PHONY:clean

clean:
	rm *~ *.o
	rm regression_test
