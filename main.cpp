#include <iostream>

#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

  std::array<size_t, 3 > N_v = {45,45, 45} ; // The number of nodes in V-Direction. 
	size_t N_x = 25 ; // The number of nodes in X_Direction. 
	double dx = (2.0/N_x) * M_PI ; // The size of the cell in X_Direction. 

  
  Kokkos::initialize( argc, argv );
  {
   	double X_max = 6.28 ; //maximum value of position. 
		double V_max = 8 ; 

		std::array<double, 3> dv = {} ;

    for(int i = 0 ; i < 3 ; i++)
		  dv[i] = 2*V_max / (N_v[i] - 1) ;

    std::vector<double> v_1(N_v[0]); 

    std::vector<double> v_2(N_v[1]) ; 

    std::vector<double> v_3(N_v[2]) ;
		
    std::vector<double> p_1 ;
		p_1.reserve(N_x) ;

   	std::vector<double> p_2 ;
		p_2.reserve(N_x) ;

    std::vector<double> p_3 ;
		p_3.reserve(N_x) ;

    
    for(double i = 0 ; i <  N_x ; i++)  // X-1 definition. 
		{
      p_1.emplace_back( i * dx);
			p_2.emplace_back(i * dx) ;
			p_3.emplace_back(i * dx) ;
    }

    std::array< std::vector<double>, 3> V{v_1,v_2,v_3}; 

    for(int  j = 0 ; j < 3; j++)
		  for(int i = 0 ; i < N_v[j]; i++)
		    V[j][i] = -1 * V_max + i * dv[j] ;


   	Kokkos::View<double ******> f{"Distribution", N_x, N_x, N_x, N_v[0], N_v[1], N_v[2]} ; //Distribution_Function definition (6D View).
    
    float M_Dist = ( sqrt(pow( 1 / (2 * M_PI),   3)) );

		std::array<double, 3> u_0 = {1,2,3} ;  
    
    // feed the 6D View with values of the distribution function .
    Kokkos::parallel_for(
		        "rho",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_x, N_x, N_x, N_v[0], N_v[1], N_v[2]}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){
								  double vx = V[0][l] - u_0[0] ; 
								  double vy = V[1][m] - u_0[1] ; 
								  double vz = V[2][n] - u_0[2] ;
								  f(i,j,k,l,m,n) = M_Dist * exp( -0.5 * (vx * vx + vy * vy + vz * vz)) ;  
              }); 

    // Integration. due to race condition, we do the triple integral over velocity space with 3 parallel and 3 serial loop using Kokko parallel_for. 

    Kokkos::View<double ***> Sum_E ("Energy", N_x, N_x, N_x) ; // define a view to put energy values in it. 
    Kokkos::View<double ***> Sum_rho ("rho", N_x, N_x, N_x) ;	// define a view to put particle number density values in it. 				
    


		std::array< Kokkos::View<double ***>,3> U{Kokkos::View<double ***>{"u1", N_x, N_x, N_x}, // wee need multidimentional array for values of the flow. 
		                                          Kokkos::View<double ***>{"u2", N_x, N_x, N_x},
																							Kokkos::View<double ***>{"u3", N_x, N_x, N_x}} ; 


   	std::array< Kokkos::View<double ***>,3> heat{Kokkos::View<double ***>{"h1", N_x, N_x, N_x}, // wee need multidimentional array for values of the heat flux.
		                                             Kokkos::View<double ***>{"h2", N_x, N_x, N_x},
																								 Kokkos::View<double ***>{"h3", N_x, N_x, N_x}} ; 

   	std::array<std::array< Kokkos::View<double ***>,3>, 3> stress{Kokkos::View<double ***>{"s11", N_x, N_x, N_x}, //we need multidimentional array for stress tensor. 
		                                                              Kokkos::View<double ***>{"s12", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"s13", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"s21", N_x, N_x, N_x},  
                                                                  Kokkos::View<double ***>{"s22", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"s23", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"s31", N_x, N_x, N_x}, 
                                                                  Kokkos::View<double ***>{"s32", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"s33", N_x, N_x, N_x}}; 

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x, N_x, N_x}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < V[0].size() ; i3++) // sum over v1.
            for(size_t i4 = 0 ; i4 < V[1].size() ; i4++)// sum over v2.
              for(size_t i5 = 0 ; i5 < V[2].size() ; i5++ )// sum over v3.
              {

							  double d3v = (dv[0] * dv[1] * dv[2] ) ; 
								double vsqu = ((V[0][i3] * V[0][i3]) + (V[1][i4] * V[1][i4]) + (V[2][i5] * V[2][i5])) ; 

                std::array< size_t , 3>  index{i3, i4, i5} ; 

                Sum_E(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * vsqu  * d3v ; // calculating the energy. 

                Sum_rho(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * d3v  ; //calculating the number density. 
								 
								for(int i = 0 ; i < 3 ; i++)
								{
                  U[i](i0, i1, i2) += (f(i0, i1, i2, i3, i4, i5) * V[i][index[i]]) * d3v; //calculating the flow. 

	    					  for(int j = 0; j < 3 ; j++)
									{
						        stress[i][j](i0 ,i1 ,i2) += f(i0, i1, i2, i3, i4, i5) * (V[i][index[i]] * V[j][index[j]]) * d3v ; // calculating the stress tensor. 
								  }
   		              heat[i](i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * V[i][index[i]] * vsqu ;  
							  }
              }


				 });

				 std::cout << " The particle number density: " << Sum_rho(0,0,0) <<  "\n" ;  
				 std::cout << " The Energy: " << Sum_E(0,0,0)  << "\n" ; 


				 for(int i = 0 ; i < 3 ; i ++)
				   for(int j = 0 ; j < 3; j++)
					  {
             std::cout << "stress[ " << i+1 <<" ]" << "[ " << j+1 << " ]" << ":" << stress[i][j](0,0,0) << "\n"; 
						}

				 for(int i = 0 ; i < 3 ; i++)
         {
				   std::cout << "heat:  " << i + 1<< ": "  << heat[i](0,0,0) << "\n" ; 
	       }

				 for(int i = 0 ; i < 3 ; i++)
         {
				   std::cout << "flow:  " << i + 1<< ": "  << U[i](0,0,0) << "\n" ; 
	       }


   }
  Kokkos::finalize();

	return 0 ; 

  
}
