#npca_task:	0=binary_data_npca 1=L1_npca 2=L2_npca 3=L3_npca
npca:	npca.cpp wymlp.hpp wyhash.h makefile
	g++ npca.cpp -o npca -Ofast -march=native -Wall -static -Dnpca_hidden=32 -Dnpca_depth=3 -Dnpca_type=float
