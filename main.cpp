#include <iostream>

#include <Kokkos_Core.hpp>

int main(int argc, char* argv []){

  Kokkos::initialize( argc, argv );
  {
		Kokkos::View<double******> f{};
		std::cout << "Hello World\n";
  }
  Kokkos::finalize();

	return 0;

}
