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
  
  integer :: z
  integer :: s, current, previous, nstreams
  !  integer(kind=cuda_stream_kind) :: transfer, compute
  integer(kind=cuda_stream_kind), allocatable :: mystream(:)
  type(cudaEvent), allocatable :: HtoDcomplete(:), DtoHcomplete(:), fft_complete(:)

  nstreams = 2
  allocate( mystream(nstreams) )

  ! allocate events
  allocate( HtoDcomplete(nstreams) )
  allocate( DtoHcomplete(nstreams) )
  allocate( fft_complete(nstreams) )
  
  allocate( plan(nstreams) )

  do s = 1,nstreams
     ierr = cudaStreamCreate( mystream(s) )
     ierr = cudaEventCreate(HtoDcomplete(s))
     ierr = cudaEventCreate(DtoHcomplete(s))
     ierr = cudaEventCreate(fft_complete(s))
     !mystream(s) = cudaforGetDefaultStream()
  enddo

  
  allocate( d_a(nx,ny,nstreams), d_b(nx,ny,nstreams) )


  do s = 1,nstreams
     ! set as many plans as I have streams
     ierr = cufftPlan2D(plan(s),ny,nx,CUFFT_C2C)
     ierr = cufftSetStream(plan(s),mystream(s))
  enddo

  ! initialize previous stream. 
  ! CUDA Events evaluate as completed if they have never been started,
  ! so initial event checks will not wait.
  previous = 1

  do z=1,nz
     current = mod(z,nstreams) + 1
     print *, "previous = ", previous
     print *, "current = ", current
     ! hold current stream until previous HtoD event completes
     ierr = cudaStreamWaitEvent(mystream(current),HtoDcomplete(previous) ,0)
     ! copy the data
     ierr = cudaMemcpyAsync(d_a(:,:,current),a(:,:,z),nx*ny, mystream(current) )
     ! record that current HtoD has finished
     ierr = cudaEventRecord(HtoDcomplete(current), mystream(current) )

     ! don't start this fft until previous fft is complete:
     ierr = cudaStreamWaitEvent( mystream(current), fft_complete(previous) ,0)
     ierr = ierr + cufftExecC2C(plan(current),d_a(:,:,current),d_b(:,:,current),CUFFT_FORWARD)
     ierr = ierr + cufftExecC2C(plan(current),d_b(:,:,current),d_b(:,:,current),CUFFT_INVERSE)
     ! record that current fft has completed:
     ierr = cudaEventRecord(fft_complete(current), mystream(current) )

     ! perform DtoH as long as previous DtoH has finished
     ierr = cudaStreamWaitEvent( mystream(current), DtoHcomplete(previous) ,0)
     ! also wait until previous HtoD has finished
     ierr = cudaStreamWaitEvent(mystream(current),HtoDcomplete(previous) ,0)
     ierr = cudaMemcpyAsync(b(:,:,z), d_b(:,:,current), nx*ny, mystream(current))
     ierr = cudaEventRecord(DtoHcomplete(current), mystream(current) )

     ! set previous to current for next iteration:
     previous = current
  enddo

  ierr = cudaDeviceSynchronize()

  deallocate(d_a)
  deallocate(d_b)
  
end subroutine cufft2dTest_manystream


subroutine cufft2dTest_2stream(a,b,nx,ny,nz)

  ! this routine transfers in only one direction at a time
  ! this is good for Power9 which has maximum bandwidth in only one direction
  ! and the result from an fft on the gpu can be communicated as soon as possible

  implicit none
  integer, intent(in) :: nx,ny,nz
  complex, pinned, intent(inout) :: a(nx,ny,nz), b(nx,ny,nz)
  complex, device, allocatable :: d_a(:,:,:), d_b(:,:,:)
  integer, allocatable :: plan(:)
  integer :: ierr
  integer :: z, s, current, next, nbuffers
  integer(kind=cuda_stream_kind) :: transfer_stream, compute_stream
  type(cudaEvent), allocatable :: HtoDcomplete(:), DtoHcomplete(:), fft_complete(:)

  nbuffers = 2 ! double buffer

  ! allocate events:
  allocate( HtoDcomplete(nbuffers) )
  allocate( DtoHcomplete(nbuffers) )
  allocate( fft_complete(nbuffers) )
  
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

  ! stage the first plane into the GPU
  z = 1
  current = mod(z,nbuffers) + 1
  ierr = cudaMemcpyAsync(d_a(:,:,current),a(:,:,z),nx*ny,transfer_stream)
  ierr = cudaEventRecord(HtoDcomplete(current),transfer_stream)

  do z=1,nz
     current = mod(z,nbuffers) + 1
     next = mod(z+1,nbuffers) + 1
     print *, "current = ", current
     print *, "next = ", next

     ! compute stream waits for HtoD to complete in transfer stream 
     ierr = cudaStreamWaitEvent(compute_stream, HtoDcomplete(current),0)
     ierr = ierr + cufftExecC2C(plan(current),d_a(:,:,current),d_b(:,:,current),CUFFT_FORWARD)
     ierr = ierr + cufftExecC2C(plan(current),d_b(:,:,current),d_b(:,:,current),CUFFT_INVERSE)
     ierr = cudaEventRecord(fft_complete(current),compute_stream)

     ! While fft is called, transfer next buffer to GPU (unless last z)
     if (z /= nz) then
        ierr = cudaMemcpyAsync(d_a(:,:,next),a(:,:,z),nx*ny,transfer_stream)
        ierr = cudaEventRecord(HtoDcomplete(next),transfer_stream)
     endif
     
     ! transfer_stream waits until compute is finished
     ierr = cudaStreamWaitEvent(transfer_stream, fft_complete(current),0)
     ierr = cudaMemcpyAsync(b(:,:,z), d_b(:,:,current), nx*ny, transfer_stream)
  enddo

  ierr = cudaDeviceSynchronize()

  deallocate(d_a)
  deallocate(d_b)


end subroutine cufft2dTest_2stream

end module overlapfft_mod
