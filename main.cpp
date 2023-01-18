#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

// in order to define the coordinates, we should define the number of nodes in the grid and the size of each cell. 

	int N_v = 10 ; // The number of nodes in V-Direction. 
	double N_x = 5 ; // The number of nodes in X_Direction. 
	int  dv = 2 ; // The size of the cell in V_Direction. 
	double dx = (2/N_x) * M_PI ; // The size of the cell in X_Direction. 
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
   


		double x_1 = 0 ; 
		for(double i = 0 ; i < X_max ; i++)  // X-1 definition. 
		{ 
			x_1 += dx ;
			p_1.push_back(x_1) ;

			if(x_1 >= X_max)
			break ; 
		}

		double x_2 = 0 ; 
  	for(double i = 0 ; i < X_max ; i++) // X-2 definition. 
		{ 
			x_2 += dx ;
			p_2.push_back(x_2) ;

			if(x_2 >= X_max)
			break ; 
		}


		double x_3 = 0 ; 		
		for(double i = 0 ; i < X_max ; i++)  // X-3 definition. 
		{ 
		  x_3 += dx ;
			p_3.push_back(x_3) ;

			if(x_3 >= X_max)
			break ; 
		}
	

	  for(double number : p_1)
			std::cout << number << std::endl ; 

    int V_1 = -20 ; 
		v_1.push_back(V_1) ; 
		for(int i = 0  ; i < V_max  ; i++) // Definition of v1-coordinate.
		{
			V_1 += dv ; 
			v_1.push_back(V_1) ; 
		}


    int V_2 = -20 ; 		
		v_2.push_back(V_2) ; 
  	for(int i = 0  ; i < V_max  ; i++) // Definition of v2-coordinate.
		{
			V_2 += dv ; 
			v_2.push_back(V_2) ; 
		}


    int V_3 = -20 ; 
		v_3.push_back(V_3) ; 
  	for(int i = 0  ; i < V_max  ; i++) // Definition of v3-coordinate.
		{
			V_3 += dv ; 
			v_3.push_back(V_3) ; 
		}





   Kokkos::View<double ******> f{} ; //Distribution_Function definition (6D View).

   float M_Dist = ( 1 / sqrt(pow( 1 / (2 * M_PI),   3)) ) * pow(e, 1 );

  }
  Kokkos::finalize();

	return 0;

}





