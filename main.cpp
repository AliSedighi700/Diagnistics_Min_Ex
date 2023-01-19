#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

// in order to define the coordinates, we should define the number of nodes in the grid and the size of each cell. 

	size_t N_v = 10 ; // The number of nodes in V-Direction. 
	size_t  N_x = 5 ; // The number of nodes in X_Direction. 
	double dx = (2.0/N_x) * M_PI ; // The size of the cell in X_Direction. 
  const float e = 2.718228183 ; 

	// <---|---|---|--- . . . ---|->V
	//    -20 -18 -16           20


	// <-|-----|-------|------|------|------|-> X
  //   0   1.256   2.512  3.768  5.024  6.28 (2Pi)


	
  Kokkos::initialize( argc, argv );
  {
    int V_max = 20; // maximum value of velocity. 
   	double X_max = 6.28 ; //maximum value of position. 
    int dv = V_max / N_v ; 

    std::vector<int> v_1 ; 
    std::vector<int> v_2 ; 
    std::vector<int> v_3 ; 
		
    std::vector<double> p_1 ;
   	std::vector<double> p_2 ;
    std::vector<double> p_3 ;

    double x = 0 ; 
    for(double i = 0 ; i < X_max ; i++)  // X-1 definition. 
    { 
      x += dx ;
      p_1.push_back(x) ;

      if(x >= X_max)
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



   	Kokkos::View<double ******> f{"Distribution", N_v, N_v, N_v, N_x, N_x, N_x } ; //Distribution_Function definition (6D View).
    
    float M_Dist = ( 1 / sqrt(pow( 1 / (2 * M_PI),   3)) );


// feed the 6D View with values if the distribution function .
    Kokkos::parallel_for(
		        "rho",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_v, N_v, N_v, N_x, N_x, N_x}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){ 
								  f(i,j,k,l,m,n) = M_Dist * pow( e , -0.5 * pow( v_1[i], 2 )) * pow( e , -0.5 * pow( v_2[j] , 2 )) * pow( e, -0.5 * pow( v_3[k] , 2 )) ;
              }); 

// Integration. due to race condition, we do the triple integral over velocity space with 3 parallel and 3 serial loop using Kokko parallel_for. 

    Kokkos::View<double ***> Sum {"label", N_x, N_x, N_x} ; 

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x, N_x, N_x}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < v_1.size() ; i3++)
				  	for(size_t i4 = 0 ; i4 < v_2.size() ; i4++)
					    for(size_t i5 = 0 ; i5 < v_3.size() ; i5++ )
							  Sum(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) ; 
          Sum(i0, i1, i2) *= pow(dv, 3)  ; 
         
				 });
  }
  Kokkos::finalize();

	return 0; 

  
}




