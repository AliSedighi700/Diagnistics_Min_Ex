#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

// in order to define the coordinates, we should define the number of nodes in the grid and the size of each cell. 

	int N_v = 10 ; // The number of nodes in V-Direction. 
	double N_x = 5 ; // The number of nodes in X_Direction. 
	int  dv = 2 ; // The size of the cell in V_Direction. 
	double dx = 1.25 ; // The size of the cell in X_Direction. 
   

	// <---|---|---|--- . . . ---|->V
	//    -20 -18 -16           20


	// <-|-----|-------|------|------|------|-> X
  //   0   1.256   2.512  3.768  5.024  6.28 (2Pi)


	
  Kokkos::initialize( argc, argv );
  {
		int V_max = -20; // maximum value of velocity. 
   	double X_max = 2 *  M_PI ; //maximum value of position. 


		std::vector<int> Velocities ; 
		std::vector<double> Positions ; 

		double x = 0 ; 
		int v = 0 ; 

		for(double i = 1 ; i < X_max ; i++) // definition of x_coordinate. 
		{
			x += dx ;
			Positions.push_back(x) ; 
		}
		
    for(double number : Positions ) 
			std::cout << number << std::endl ;

		for(int i = 0  ; i < V_max ; i++)
		{
			v += dv ; 
			Velocities.push_back(v) ; 
		}


    for(double number : Velocities ) 
			std::cout << number << std::endl ;



  }
  Kokkos::finalize();

	return 0;

}
