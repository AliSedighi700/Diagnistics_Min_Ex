#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 



int main(int argc, char* argv []){

	size_t N_v = 11 ; // The number of nodes in V-Direction. 
	size_t  N_x = 10 ; // The number of nodes in X_Direction. 
	double dx = (2.0/N_x) * M_PI ; // The size of the cell in X_Direction. 
  const float e = 2.71828183 ; 


  Kokkos::initialize( argc, argv );
  {
   	double X_max = 6.28 ; //maximum value of position. 
		double V_max = 4 ; 
    double dv = 2*V_max / (N_v - 1) ; 

    std::vector<double> v_1 ; 
    std::vector<double> v_2 ; 
    std::vector<double> v_3 ; 
		
    std::vector<double> p_1 ;
   	std::vector<double> p_2 ;
    std::vector<double> p_3 ;


    for(double i = 0 ; i <  N_x ; i++)  // X-1 definition. 
      p_1.push_back(i * dx) ;


    for(double i = 0 ; i <  N_x ; i++)  // X-2 definition. 
      p_2.push_back(i * dx) ;
	 
      
    for(double i = 0 ; i < N_x ; i++)  // X-3 definition. 
			p_3.push_back(i * dx) ;


		for(double i = 0  ; i < N_v  ; i++) // Definition of v1-coordinate.
			v_1.push_back(-1 * V_max + i *dv) ;

 
	  
		for(double i = 0  ; i < N_v  ; i++) // Definition of v2-coordinate.
			v_2.push_back(-1 * V_max + i * dv) ;


		for(double i = 0  ; i < N_v  ; i++) // Definition of v3-coordinate.
			v_3.push_back(-1 * V_max + i * dv) ; 




    std::array< std::vector<double>, 3> V{v_1,v_2,v_3}; 

   	Kokkos::View<double ******> f{"Distribution", N_x, N_x, N_x, N_v, N_v, N_v} ; //Distribution_Function definition (6D View).
    
    float M_Dist = ( sqrt(pow( 1 / (2 * M_PI),   3)) );

    // feed the 6D View with values if the distribution function .
    Kokkos::parallel_for(
		        "rho",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_x, N_x, N_x, N_v, N_v, N_v}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){ 
								  f(i,j,k,l,m,n) = M_Dist * pow( e , -0.5 * pow( v_1[l], 2 )) * pow( e , -0.5 * pow( v_2[m] , 2 )) * pow( e, -0.5 * pow( v_3[n] , 2 )) ;
              }); 

    // Integration. due to race condition, we do the triple integral over velocity space with 3 parallel and 3 serial loop using Kokko parallel_for. 

    Kokkos::View<double ***> Sum_E ("Energy", N_x, N_x, N_x) ; // define a view to put energy values in it. 
		Kokkos::View<double ***> Sum_rho ("rho", N_x, N_x, N_x) ;	// define a view to put particle number density values in it. 				
    


		std::array< Kokkos::View<double ***>,3> U{Kokkos::View<double ***>{"u1", N_x, N_x, N_x}, // wee need multidimentional array for values of the flow. 
																							Kokkos::View<double ***>{"u2", N_x, N_x, N_x},
																							Kokkos::View<double ***>{"u3", N_x, N_x, N_x}} ; 

   	std::array<std::array< Kokkos::View<double ***>,3>, 3> stress{Kokkos::View<double ***>{"u11", N_x, N_x, N_x}, // wee need multidimentional array for values of the stress tensor. 
																							                    Kokkos::View<double ***>{"u12", N_x, N_x, N_x},
																							                    Kokkos::View<double ***>{"u13", N_x, N_x, N_x},
																																	Kokkos::View<double ***>{"u21", N_x, N_x, N_x}, // wee need multidimentional array for values of the stress tensor. 
                                                                  Kokkos::View<double ***>{"u22", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"u23", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"u31", N_x, N_x, N_x}, // wee need multidimentional array for values of the stress tensor. 
                                                                  Kokkos::View<double ***>{"u32", N_x, N_x, N_x},
                                                                  Kokkos::View<double ***>{"u33", N_x, N_x, N_x}}; 

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x, N_x, N_x}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < V[1].size() ; i3++) // sum over v1.
				  	for(size_t i4 = 0 ; i4 < v_2.size() ; i4++)// sum over v2.
					    for(size_t i5 = 0 ; i5 < v_3.size() ; i5++ )// sum over v3.
							{
								Sum_rho(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) ; //calculating the number density. 

							  Sum_E(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * ((v_1[i3] * v_1[i3]) + (v_2[i4] * v_2[i4]) + (v_3[i5] * v_3[i5]))  ;  //calculating the energy. 

                for(int i = 0 ; i < 3 ; i++)
								{
								  U[i](i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * V[i][i3]  ; //what about i1,i2,i3? we need to iterate over them? //calculating the flow.
	              }
                
								for(int i = 0 ; i < 3 ; i++)
								{
	    					  for(int j = 0; j < 3 ; j++)
									{
						          stress[i][j](i0 ,i1 ,i2) += f(i0, i1, i2, i3, i4, i5) * (V[i][i3] * V[j][i3]) ;
								  }
							 }
              }


					Sum_rho(i0, i1, i2) *= (dv * dv * dv) ; 			
          Sum_E(i0, i1, i2) *= (dv * dv * dv)  ; 

					for(int i = 0 ; i < 3 ; i++)
					{
					  U[i](i0, i1, i2) *= (dv * dv * dv) ; 
					}

					for(int i = 0 ; i < 3 ; i++)
					  for(int j = 0 ; j < 3 ; j++)
						{
					    stress[i][j](i0,i1,i2) *= (dv * dv * dv) ; 
					  }
         
				 });

				 std::cout << " The particle number density: " << Sum_rho(0,0,0) <<  "\n" ;  
				 std::cout << " The Energy: " << Sum_E(0,0,0)  << "\n" ; 

				 for(int i = 0 ; i < 3 ; i++)
         {
				   std::cout << "u " << i << ": "  << U[i](0,0,0) << "\n" ; 
				 }

				 for(int i = 0 ; i < 3 ; i ++)
				   for(int j = 0 ; j < 3; j++)
					  {
             std::cout << "stress[ " << i+1 <<" ]" << "[ " << j+1 << " ]" << ":" << stress[i][j](0,0,0) << "\n"; 
						}



   }
  Kokkos::finalize();

	return 0 ; 

  
}




