program cufft2dTest
  use cufft
  use cudafor

  implicit none
  integer, parameter :: nx=6*256, ny=6*256, nz = 36
  complex, pinned, allocatable :: a(:,:,:), b(:,:,:)
  complex, device, allocatable :: d_a(:,:,:), d_b(:,:,:)
  integer, allocatable :: plan(:)
  integer :: ierr
  real, dimension(3) :: res, exp
  integer :: z, s, nstreams
  !  integer(kind=cuda_stream_kind) :: transfer, compute
  integer(kind=cuda_stream_kind), allocatable :: mystream(:)
  type(cudaEvent), allocatable :: HtoDcomplete(:), DtoHcomplete(:)

  nstreams = 3
  allocate( mystream(nstreams) )
  allocate( HtoDcomplete(nstreams) )
  allocate( DtoHcomplete(nstreams) )
  
  allocate( plan(nstreams) )

  do s = 1,nstreams
     ierr = cudaStreamCreate( mystream(s) )
     ierr = cudaEventCreate(HtoDcomplete(s))
     ierr = cudaEventCreate(DtoHcomplete(s))
     !mystream(s) = cudaforGetDefaultStream()
  enddo

  allocate( a(nx,ny,nz),b(nx,ny,nz) )

  allocate( d_a(nx,ny,nstreams), d_b(nx,ny,nstreams) )

  a = 1

  do s = 1,nstreams
     ! set as many plans as I have streams
     ierr = cufftPlan2D(plan(s),ny,nx,CUFFT_C2C)
     ierr = cufftSetStream(plan(s),mystream(s))
  enddo

  do z=1,nz
     s = mod(z,nstreams) + 1
     ierr = cudaMemcpyAsync(d_a(:,:,s),a(:,:,z),nx*ny,mystream(s))
     ierr = cudaEventRecord(HtoDcomplete(s),mystream(s))

     ierr = ierr + cufftExecC2C(plan(s),d_a(:,:,s),d_b(:,:,s),CUFFT_FORWARD)
     ierr = ierr + cufftExecC2C(plan(s),d_b(:,:,s),d_b(:,:,s),CUFFT_INVERSE)
     ierr = cudaMemcpyAsync(b(:,:,z), d_b(:,:,s), nx*ny, mystream(s))
  enddo
  res(1) = maxval(abs(a-b/(nx*ny)))
  print *,'Max error C2C: ', res(1)
end program cufft2dTest
