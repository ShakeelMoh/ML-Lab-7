//ML Lab 7
//Back propagation 2

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>

using namespace std;

float e = 2.71828182845904523536;

//Calculates sigmoid output
float sigmoid(float x){

   float output = 1/(1+(pow(e, -x)));
   return output;

}

int main (int argc, char *argv[]) {

   cout << "Back propagation ANN Part 2\n" << endl;



   return 0;

}
