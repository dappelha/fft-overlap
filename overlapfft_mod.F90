module overlapfft_mod
  use cufft
  use cudafor

contains
  subroutine cufft2dTest_manystream(a,b,nx,ny,nz)

  implicit none
  integer, intent(in) :: nx,ny,nz
  complex, pinned, intent(inout) :: a(nx,ny,nz), b(nx,ny,nz)
  complex, device, allocatable :: d_a(:,:,:), d_b(:,:,:)
  integer, allocatable :: plan(:)
  integer :: ierr
  
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

  
  allocate( d_a(nx,ny,nstreams), d_b(nx,ny,nstreams) )


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

  ierr = cudaDeviceSynchronize()

  deallocate(d_a)
  deallocate(d_b)
  
end subroutine cufft2dTest_manystream


subroutine cufft2dTest_2stream(a,b,nx,ny,nz)

  implicit none
  integer, intent(in) :: nx,ny,nz
  complex, pinned, intent(inout) :: a(nx,ny,nz), b(nx,ny,nz)
  complex, device, allocatable :: d_a(:,:,:), d_b(:,:,:)
  integer, allocatable :: plan(:)
  integer :: ierr
  integer :: z, s, nbuffers
  integer(kind=cuda_stream_kind) :: transfer_stream, compute_stream
  type(cudaEvent), allocatable :: HtoDcomplete(:), DtoHcomplete(:), fft_complete(:)

  nbuffers = 2 ! double buffer

  allocate( HtoDcomplete(nbuffers) )
  allocate( DtoHcomplete(nbuffers) )
  
  allocate( plan(nbuffers) )

  ierr = cudaStreamCreate(transfer_stream)
  ierr = cudaStreamCreate(compute_stream)

  do s = 1,nbuffers
     !ierr = cudaStreamCreate( mystream(s) )
     ierr = cudaEventCreate(HtoDcomplete(s))
     ierr = cudaEventCreate(DtoHcomplete(s))
     ierr = cudaEventCreate(fft_complete(s))
     !mystream(s) = cudaforGetDefaultStream()
  enddo

  
  allocate( d_a(nx,ny,nbuffers), d_b(nx,ny,nbuffers) )


  do s = 1,nbuffers
     ! set as many plans as I have buffers
     ierr = cufftPlan2D(plan(s),ny,nx,CUFFT_C2C)
     ierr = cufftSetStream(plan(s),compute_stream)
  enddo

  do z=1,nz
     s = mod(z,nbuffers) + 1
     ierr = cudaMemcpyAsync(d_a(:,:,s),a(:,:,z),nx*ny,transfer_stream)
     ierr = cudaEventRecord(HtoDcomplete(s),transfer_stream)

     ! compute stream waits for HtoD to complete in transfer stream 
     ierr = cudaStreamWaitEvent(compute_stream, HtoDcomplete(s),0)
     ierr = ierr + cufftExecC2C(plan(s),d_a(:,:,s),d_b(:,:,s),CUFFT_FORWARD)
     ierr = ierr + cufftExecC2C(plan(s),d_b(:,:,s),d_b(:,:,s),CUFFT_INVERSE)
     ierr = cudaEventRecord(fft_complete(s),compute_stream)

     ! transfer_stream waits until compute is finished
     ierr = cudaStreamWaitEvent(transfer_stream, fft_complete(s),0)
     ierr = cudaMemcpyAsync(b(:,:,z), d_b(:,:,s), nx*ny, transfer_stream)
  enddo

  ierr = cudaDeviceSynchronize()

  deallocate(d_a)
  deallocate(d_b)


end subroutine cufft2dTest_2stream

end module overlapfft_mod
