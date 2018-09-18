#My own makefile for ML lab 7

#Declare variables
CC=g++
LIBS=-lm

#First create ".exe" called backpropagation
backpropagation2: backpropagation2.o
	$(CC) backpropagation2.o -o backpropagation2 $(LIBS)

#Need to make backpropagation2.o file though
backpropagation2.o: backpropagation2.cpp
	$(CC) -c backpropagation2.cpp


#Other rules

#Clean .o and exe
clean:
	@rm -f *.o
	@rm -f backpropagation2

#To run program
run:
	./backpropagation2
