comp=xlf

ifeq '${comp}' 'pgi'
	# PGI compiler
	export OMPI_FC=pgf90
	F90=mpif90
	CC=pgcc
	CUDAFLAGS= -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,cuda9.1 -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt -lcufft
	#CUDAFLAGS= -Mcuda=cc60,nordc,maxregcount:48,ptxinfo,loadcache:L1 -fast -Mfprelaxed
	F90FLAGS=-O3 -mp 
	#CUDAFLAGS+= -acc
	#LINKFLAGS = -ta=tesla:pinned
else
	# xlcuf compiler
	export OMPI_FC=xlf_r
	F90=mpif90
	CC=xlc_r
	#CUDAFLAGS= -qcuda -qtgtarch=sm_60 -W@,"-v,--maxrregcount=48" -qpath=IL:/home/dappelh/ibmcmp/specialfixes
	CUDAFLAGS= -qcuda -qtgtarch=sm_70 -W@,"-v,--maxrregcount=64" -lnvToolsExt
	CUDAFLAGS+= -qcheck -qsigtrap -g1 -lcufft
	#F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr8 -qtune=pwr8 -qsmp=omp -qoffload -lessl
	F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qoffload
endif



build: driver.o nvtx_mod.o overlapfft_mod.o
	${F90} -o ${comp}test driver.o overlapfft_mod.o nvtx_mod.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}

driver.o: driver.F90 nvtx_mod.o overlapfft_mod.o
	${F90} -c -o driver.o driver.F90 ${F90FLAGS} ${CUDAFLAGS}

overlapfft_mod.o: overlapfft_mod.F90 nvtx_mod.o
	${F90} -c -o overlapfft_mod.o overlapfft_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

nvtx_mod.o: nvtx_mod.F90
	${F90} -c -o nvtx_mod.o nvtx_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

clean:
	rm -f ${comp}test *.o *.lst *.mod 

