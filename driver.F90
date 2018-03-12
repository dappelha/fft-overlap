program main
  use overlapfft_mod
  use cudafor
  use nvtx_mod
  use mpi

  implicit none

  !include "mpif.h"

  ! full problem size:
  integer ::n, nx, ny, nz, nplanes
  complex, pinned, allocatable :: a(:,:,:), b(:,:,:)


  integer :: ierr
  integer :: s
  integer :: numtasks, taskid

  real :: res

  call MPI_INIT (ierr)
  
  call MPI_COMM_SIZE (MPI_COMM_WORLD,numtasks,ierr)
  call MPI_COMM_RANK (MPI_COMM_WORLD,taskid,ierr)

  ! figure out sizes per mpi rank:
  n = 2*1024
  nx = n
  ny = n
  nz = n
  nplanes = n/numtasks

  allocate( a(nx,ny,nplanes),b(nx,ny,nplanes) )

  ierr = cudaDeviceSynchronize()

  a = 1

  call  cufft3dTest(a,b,nx,ny,nz,nplanes,numtasks)

  ierr = cudaDeviceSynchronize()

  do s = 1, nplanes
     res = maxval(abs(a(:,:,s)-b(:,:,s)/(nx*ny)))
     if (res > 0.0001) write(*,"(A,I4,A,G8.3)") "plane = ",s,"     Max error C2C: ", res
  enddo

  if(taskid == 0) print*, "completed"

  call MPI_FINALIZE(ierr)


end program main
