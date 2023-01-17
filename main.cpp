#include <iostream>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv []){
	
	double N_v = 10 ; // The number of nodes in V-Direction. 
	double N_x = 5 ; // The number of nodes in X_Direction. 
	double dv = 2 ; // The size of the cell in V_Direction. 
	double dx = 1.25 ; // The size of the cell in X_Direction. 
   

	// <---|---|---|--- . . . ---|->V
	//    -20 -18 -16           20


	// <-|-----|-------|------|------|------|-> X
  //   0   1.256   2.512  3.768  5.024  6.28 (2Pi)


	
  Kokkos::initialize( argc, argv );
  { 


  }
  Kokkos::finalize();

	return 0;

}
