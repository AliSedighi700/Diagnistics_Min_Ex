#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

// in order to define the coordinates, we should define the number of nodes in the grid and the size of each cell. 

	int N_v = 10 ; // The number of nodes in V-Direction. 
	double N_x = 5 ; // The number of nodes in X_Direction. 
	int  dv = 2 ; // The size of the cell in V_Direction. 
	double dx = 1.256 ; // The size of the cell in X_Direction. 
  const float e = 2.718228183 ; 

	// <---|---|---|--- . . . ---|->V
	//    -20 -18 -16           20


	// <-|-----|-------|------|------|------|-> X
  //   0   1.256   2.512  3.768  5.024  6.28 (2Pi)


	
  Kokkos::initialize( argc, argv );
  {
		int V_max = 20; // maximum value of velocity. 
   	double X_max = 6.28 ; //maximum value of position. 


		std::vector<int> v_1 ; 
		std::vector<int> v_2 ; 
		std::vector<int> v_3 ; 
		
		std::vector<double> p_1 ;
   	std::vector<double> p_2 ;
    std::vector<double> p_3 ;
   


		double x = 0.0 ; 
		int v = -20 ; 

		for(double i = 0 ; i < X_max ; i++) // definition of x_coordinate. 
		{ 
			x += dx ;
			p_1.push_back(x) ;

			if(x >= X_max)
			break ; 
		}
		
    for(double number : p_1 )    // check whether the elements are correct. 
			std::cout << number << std::endl ;

    
		v_1.push_back(v) ; 

		for(int i = 0  ; i < V_max  ; i++) // Definition of v-coordinate.
		{
			v += dv ; 
			v_1.push_back(v) ; 
		}

   for(int number : v_1) // check the v-elements are correct. 
	   std::cout << number << std::endl ; 

   Kokkos::View<double ******> f{} ; //Distribution_Function definition (6D View).

   float M_Dist = ( 1 / sqrt(pow( 1 / (2 * M_PI),   3)) )  * 4 * M_PI * pow(v,2) * pow(e, 1 );

  }
  Kokkos::finalize();

	return 0;

}
