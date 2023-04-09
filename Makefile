LOGIN=xmocar00
NAME=parkmeans
CC=mpic++
RUN=mpirun
CPU_CNT=32
OPEN_MPI_PATH=/usr/local/share/OpenMPI

build:
	$(CC) --prefix $(OPEN_MPI_PATH) -o $(NAME) $(NAME).cc

run:
	$(RUN) --prefix $(OPEN_MPI_PATH) -oversubscribe -np $(CPU_CNT) $(NAME) 

clean:
	rm -rf $(NAME).o $(NAME)

