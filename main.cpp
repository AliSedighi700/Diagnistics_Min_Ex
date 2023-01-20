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

//    for(double i = 0 ; i < N_x ; i++)
	//	{
    //   std::cout << p_1[i] << " " <<  p_2[i] << " " << p_3[i] << "\n" ; 
	//	}

    //for(double i = 0 ; i < N_v ; i++)
	 	//{
       //std::cout << v_1[i] << " " <<  v_2[i] << " " << v_3[i] << "\n" ; 
		//}



   	Kokkos::View<double ******> f{"Distribution", N_x, N_x, N_x, N_v, N_v, N_v} ; //Distribution_Function definition (6D View).
    
    float M_Dist = ( sqrt(pow( 1 / (2 * M_PI),   3)) );


    ////////calculating the particle density///////////
    Kokkos::parallel_for(
		        "rho",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_x, N_x, N_x, N_v, N_v, N_v}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){ 
								  f(i,j,k,l,m,n) = M_Dist * pow( e , -0.5 * pow( v_1[l], 2 )) * pow( e , -0.5 * pow( v_2[m] , 2 )) * pow( e, -0.5 * pow( v_3[n] , 2 )) ;
              }); 

		Kokkos::View<double ***> Sum_rho ("for rho", N_x, N_x, N_x) ;					

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x, N_x, N_x}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < v_1.size() ; i3++)
				  	for(size_t i4 = 0 ; i4 < v_2.size() ; i4++)
					    for(size_t i5 = 0 ; i5 < v_3.size() ; i5++ )
							  Sum_rho(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) ; 
          Sum_rho(i0, i1, i2) *= (dv * dv * dv) ;      
				 });



				 ////calculating the energy////////

    // feed the 6D View with values if the distribution function .
    Kokkos::parallel_for(
		        "rho",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_x, N_x, N_x, N_v, N_v, N_v}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){ 
								  f(i,j,k,l,m,n) = M_Dist * pow( e , -0.5 * pow( v_1[l], 2 )) * pow( e , -0.5 * pow( v_2[m] , 2 )) * pow( e, -0.5 * pow( v_3[n] , 2 )) ;
              }); 

    // Integration. due to race condition, we do the triple integral over velocity space with 3 parallel and 3 serial loop using Kokko parallel_for. 

    Kokkos::View<double ***> Sum_E {"Energy", N_x, N_x, N_x} ; 

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x, N_x, N_x}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < v_1.size() ; i3++)
				  	for(size_t i4 = 0 ; i4 < v_2.size() ; i4++)
					    for(size_t i5 = 0 ; i5 < v_3.size() ; i5++ )
							  Sum_E(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * ((v_1[i3] * v_1[i3]) + (v_2[i4] * v_2[i4]) + (v_3[i5] * v_3[i5])) ;  
          Sum_E(i0, i1, i2) *= (dv * dv * dv)  ; 
					
         
				 });

				 std::cout << " The particle number density: " << Sum_rho(0,0,0) <<  "\n" ;  
				 std::cout << " The Energy: " << Sum_E(0,0,0)  << "\n" ; 
  }
  Kokkos::finalize();

	return 0 ; 

  
}




