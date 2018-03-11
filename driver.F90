program main
  use overlapfft_mod
  use cudafor
  use nvtx_mod
  implicit none

  integer, parameter :: nx=12*1024, ny=12*1024, nz = 8
  complex, pinned, allocatable :: a(:,:,:), b(:,:,:)


  integer :: ierr
  integer :: z

  real :: res(nz)

  allocate( a(nx,ny,nz),b(nx,ny,nz) )

  ierr = cudaDeviceSynchronize()

  a = 1

  call  cufft2dTest_2stream(a,b,nx,ny,nz)

  ierr = cudaDeviceSynchronize()

  do z = 1, nz
     res(z) = maxval(abs(a(:,:,z)-b(:,:,z)/(nx*ny)))
     write(*,"(A,I,A,G8.3)") "z = ",z,"     Max error C2C: ", res(z)
  enddo





end program main
