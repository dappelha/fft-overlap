program main
  use overlapfft_mod
  use cudafor
  use nvtx_mod
  implicit none

  integer, parameter :: nx=6*1024, ny=6*1024, nz = 36
  complex, pinned, allocatable :: a(:,:,:), b(:,:,:)

  integer :: ierr

  real, dimension(3) :: res

  allocate( a(nx,ny,nz),b(nx,ny,nz) )

  ierr = cudaDeviceSynchronize()

  a = 1

  call  cufft2dTest_manystream(a,b,nx,ny,nz)

  ierr = cudaDeviceSynchronize()


  res(1) = maxval(abs(a-b/(nx*ny)))
  write(*,"(A,G8.3)") 'Max error C2C: ', res(1)
  print *,'Max error C2C: ', res(1)

  a = 1

  call  cufft2dTest_2stream(a,b,nx,ny,nz)

  ierr = cudaDeviceSynchronize()

  res(1) = maxval(abs(a-b/(nx*ny)))
  write(*,"(A,G8.3)") 'Max error C2C: ', res(1)
  print *,'Max error C2C: ', res(1)




end program main
