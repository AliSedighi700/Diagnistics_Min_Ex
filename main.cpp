#include <iostream>
#include <vector> 
#include <Kokkos_Core.hpp>
#include <cmath> 

int main(int argc, char* argv []){

  std::array<size_t, 3 > N_v = {21, 21, 21} ; // The number of nodes in V-Direction. 
	std::array<size_t, 3>  N_x = {3,3,3} ; // The number of nodes in X_Direction. 
  
  Kokkos::initialize( argc, argv );
  {
   	double X_max = 6.28 ; //maximum value of position. 
		double V_max = 8 ; 

		std::array<double, 3> dv = {} ;
    std::array<double, 3> dx = {} ; 
		
    for(int i = 0 ; i < 3 ; i++)
    {
		  dv[i] = 2*V_max / (N_v[i] - 1) ; //velocity space valume element.
			dx[i] = (2.0/N_x[i]) * M_PI ; //position space valume element. 
    }

    std::array< std::vector<double>, 3> V{}; 
		std::array< std::vector<double>, 3> X{};

    for(int  j = 0 ; j < 3; j++)
		{
			V[j] = std::vector<double>(N_v[j]);
		  for(int i = 0 ; i < N_v[j]; i++)
		    V[j][i] = -1 * V_max + i * dv[j] ;
		  
			X[j] = std::vector<double>(N_x[j]);
			for(int i = 0 ; i < N_x[j]; i++)
			  X[j][i] = i * dx[j] ; 
    }

   	Kokkos::View<double ******> f{"Distribution", N_x[0], N_x[1], N_x[2], N_v[0], N_v[1], N_v[2]} ; //Distribution_Function definition (6D View).
    
    double M_Dist = ( sqrt(pow( 1 / (2 * M_PI),   3)) );

		std::array<double, 3> u_0 = {2,1,1} ;// for shifting the Maxwellian. 
    
    // feed the 6D View with values of the distribution function. 
    Kokkos::parallel_for(
		        "InitDistFunc",
						Kokkos::MDRangePolicy<Kokkos::Rank<6>>(
						  {0,0,0,0,0,0}, {N_x[0], N_x[1], N_x[2], N_v[0], N_v[1], N_v[2]}),
						    KOKKOS_LAMBDA(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n){
								  double vx = V[0][l] - u_0[0] ; 
								  double vy = V[1][m] - u_0[1] ; 
								  double vz = V[2][n] - u_0[2] ;
								  f(i,j,k,l,m,n) = M_Dist * exp( -0.5 * ( (vx * vx) + (vy * vy) + (vz * vz) )) ;  
              }); 

    Kokkos::View<double ***> Sum_E ("Energy", N_x[0], N_x[1], N_x[2]) ; // define a view to put energy values in it. 
    Kokkos::View<double ***> Sum_rho ("rho", N_x[0], N_x[1], N_x[2]) ;	// define a view to put particle number density values in it. 				

		std::array< Kokkos::View<double ***>,3> U{} ; // flow = empty view to fee. 
   	std::array< Kokkos::View<double ***>,3> heat{} ; // heat flux = empty view to feed. 

   	std::array<std::array< Kokkos::View<double ***>,3>, 3> stress{}; // stress tensor = rmpty view to feed. 


		for(size_t i=0; i < 3 ; i++)
		{
			std::string label{"u"+std::to_string(i)};//Vector component labels. 
			U[i] = Kokkos::View<double ***>{label , N_x[0], N_x[1], N_x[2]} ;// feeding the flow View. 

			label = std::string{"heat"+std::to_string(i)};//Vector component labels. 
			heat[i] =Kokkos::View<double ***>{label, N_x[0], N_x[1], N_x[2]} ;//feeding the heat flux View. 

			for(size_t j=0; j < 3; j++)
			{
				label = std::string{"heat"+std::to_string(i) + std::to_string(j)}; // Tensor component labels. 
				stress[i][j] = Kokkos::View<double ***>{label, N_x[0], N_x[1], N_x[2]}; // feeding the stress tensor View. 
			}	
		}		

		Kokkos::parallel_for(
		        "Sumd3v",
						  Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{N_x[0], N_x[1], N_x[2]}),
							    KOKKOS_LAMBDA(size_t i0, size_t i1, size_t i2){
					
					for(size_t i3 = 0 ; i3 < V[0].size() ; i3++) // sum over v1.
            for(size_t i4 = 0 ; i4 < V[1].size() ; i4++)// sum over v2.
              for(size_t i5 = 0 ; i5 < V[2].size() ; i5++ )// sum over v3.
              {
							  double d3v = (dv[0] * dv[1] * dv[2] ) ; 
								double v2 = ((V[0][i3] * V[0][i3]) + (V[1][i4] * V[1][i4]) + (V[2][i5] * V[2][i5])) ; 
                std::array< size_t , 3>  index{i3, i4, i5} ; 

                Sum_E(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * v2 * d3v ; // calculating the energy. 

                Sum_rho(i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * d3v  ; //calculating the number density. 
								 
								for(int i = 0 ; i < 3 ; i++)
								{
                  U[i](i0, i1, i2) += (f(i0, i1, i2, i3, i4, i5) * V[i][index[i]]) * d3v; //calculating the flow. 
   		            heat[i](i0, i1, i2) += f(i0, i1, i2, i3, i4, i5) * V[i][index[i]] * v2 * d3v;  //calculating the heat flux. 

		    					for(int j = 0; j < 3 ; j++)
						        stress[i][j](i0 ,i1 ,i2) += f(i0, i1, i2, i3, i4, i5) * (V[i][index[i]] * V[j][index[j]]) * d3v ; // calculating the stress tensor. 
							  }
              }});

				 std::cout << " The particle number density: " << Sum_rho(0,0,0) <<  "\n" ;  
				 std::cout << " The Energy: " << Sum_E(0,0,0)  << "\n" ; 

				 for(int i = 0 ; i < 3 ; i ++)
				 {
				   for(int j = 0 ; j < 3; j++)
					 {
             std::cout << "stress[ " << i+1 <<" ]" << "[ " << j+1 << " ]" << ":" << stress[i][j](0,0,0) << "\n"; 
					 }
         }

         for(int i = 0 ; i < 3 ; i++)
				 {
           std::cout << "heat " << i + 1<< ": "  << heat[i](0,0,0) << "\n" ;
				 }


				 for(int i = 0 ; i < 3 ; i++)
				 {
			     std::cout << "flow:  " << i + 1<< ": "  << U[i](0,0,0) << "\n" ; 
         }
          
// In this section we make a test to figure out whther all the results in the configuration space are quivalent to the analytical values. 
         
         int flag = 1 ; 
         
         for (int i = 0 ; i < N_x[0] ; i++)
				 {
				   for(int j = 0 ; j < N_x[1] ; j++)
					 {
					   for(int k = 0 ; k < N_x[2] ; k++)
						 {
						   if (abs(Sum_rho(i, j , k)-1.) > 1e-10) // cheack for rho
							 {
							   std::cout << "Error: result != solution-rho, error"<< (abs(Sum_rho(i,j,k)-1)<1)  << "\n" ; 
								 flag = 1 ; 
								 break ; 
							 }

							 if(flag == 1)
							   break ; 
						 }
						 if(flag==1)
						   break ; 
					 }
					 if(flag==1)
					   break ; 
				 }

   }
  Kokkos::finalize();
	return 0 ; 
}
