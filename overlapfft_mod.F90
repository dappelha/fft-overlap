module overlapfft_mod
  use cufft
  use cudafor
  use nvtx_mod
  use mpi

  character(len=30) :: str

contains
  subroutine cufft2dTest_manystream(a,b,nx,ny,nz)

  implicit none
  integer, intent(in) :: nx,ny,nz
  complex, pinned, allocatable, intent(inout) :: a(:,:,:), b(:,:,:)
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
  complex, pinned, allocatable, intent(inout) :: a(:,:,:), b(:,:,:)
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


subroutine cufft3dTest(a,b,nx,ny,nz,nplanes,numtasks)

  ! this routine uses nvlink in only one direction at a time
  ! this is good for Power9 which has maximum bandwidth in only one direction
  ! and the result from an fft on the gpu can then be communicated as soon as possible

  implicit none
  integer, intent(in) :: nx,ny,nz,nplanes,numtasks
  complex, pinned, allocatable, intent(inout) :: a(:,:,:), b(:,:,:)
  complex, pinned, allocatable :: temp(:,:,:)
  complex, device, allocatable :: d_a(:,:,:), d_b(:,:,:)
  integer, allocatable :: plan(:)
  integer :: ierr
  integer :: s,ss,p, current, next, buf, nbuffers
  integer(kind=cuda_stream_kind) :: transfer_stream, compute_stream
  type(cudaEvent), allocatable :: HtoDcomplete(:), DtoHcomplete(:), fft_complete(:)
  integer :: sendrequest(0:numtasks-1,nplanes), recvrequest(0:numtasks-1,nplanes)
  logical :: recieved(nplanes)
  integer(C_INT64_T) :: rangeid(nplanes) ! for nvtx range markers

  allocate(temp(nx,nplanes,nz))

  nbuffers = 2 ! double buffer

  ! allocate events:
  allocate( HtoDcomplete(nbuffers) )
  allocate( DtoHcomplete(nbuffers) )
  allocate( fft_complete(nbuffers) )
  
  allocate( plan(nbuffers) )

  ! create a transfer and a compute stream
  ierr = cudaStreamCreate(transfer_stream)
  ierr = cudaStreamCreate(compute_stream)

  ! Create an event for each buffer
  do buf = 1,nbuffers
     ierr = cudaEventCreate(HtoDcomplete(buf))
     ierr = cudaEventCreate(DtoHcomplete(buf))
     ierr = cudaEventCreate(fft_complete(buf))
     !mystream(s) = cudaforGetDefaultStream()
  enddo

  ! allocate GPU buffers
  allocate( d_a(nx,ny,nbuffers), d_b(nx,ny,nbuffers) )


  do buf = 1,nbuffers
     ! set as many plans as I have buffers
     ierr = cufftPlan2D(plan(buf),ny,nx,CUFFT_C2C)
     ierr = cufftSetStream(plan(buf),compute_stream)
  enddo

  ! stage the first plane into the GPU
  s = 1
  current = mod(s,nbuffers) + 1
  ierr = cudaMemcpyAsync(d_a(:,:,current),a(:,:,s),nx*ny,transfer_stream)
  ierr = cudaEventRecord(HtoDcomplete(current),transfer_stream)

  ! initialize that no planes have been mpi communicated yet.
  recieved(:) = .false.

  do s=1,nplanes
     current = mod(s,nbuffers) + 1
     next = mod(s+1,nbuffers) + 1
     !print *, "current = ", current
     !print *, "next = ", next

     ! compute stream waits for HtoD to complete in transfer stream 
     ierr = cudaStreamWaitEvent(compute_stream, HtoDcomplete(current),0)
     ierr = ierr + cufftExecC2C(plan(current),d_a(:,:,current),d_b(:,:,current),CUFFT_FORWARD)
     ierr = ierr + cufftExecC2C(plan(current),d_b(:,:,current),d_b(:,:,current),CUFFT_INVERSE)
     ierr = cudaEventRecord(fft_complete(current),compute_stream)

     ! While fft is called, transfer next buffer to GPU (unless last plane)
     if (s /= nplanes) then
        ierr = cudaMemcpyAsync(d_a(:,:,next),a(:,:,s),nx*ny,transfer_stream)
        ierr = cudaEventRecord(HtoDcomplete(next),transfer_stream)
     endif
     
     ! transfer_stream waits until compute is finished
     ierr = cudaStreamWaitEvent(transfer_stream, fft_complete(current),0)
     ierr = cudaMemcpyAsync(b(:,:,s), d_b(:,:,current), nx*ny, transfer_stream)

     ! need to MPI all to all communicate non-blocking here
     ! we are finished using a(:,:,z). Can use this to store. Actually don't know if transfer is finished yet.
     ! we have b(:,:,z). now 
     ! process 0 needs b(:,1:nslabs,z)
     ! process 1 needs b(:,nslabs+1:2*nslabs,z)
     ! process 2 needs b(:,2*nslabs+1:3*nslabs,z)
     ! process p needs b(:,p*nslabs+1:(p+1)*nslabs,z)
     
     ! will store in a(z,:,1:nslabs) (variables are (z,x,y). last dim corresponds to y, but is number of slabs.)


     ! TODO:I have not ensured the message is ready to be sent yet....

     ! try my own all to all as numtask messages 
     write(str,"(A,I4)") "MPI plane",s
     call nvtxStartRange(str,color=s,id=rangeid(s))
     do p=0, numtasks-1
        call MPI_Isend(b(1,p*nplanes+1,s),nx*nplanes,MPI_COMPLEX,p,0,MPI_COMM_WORLD,sendrequest(p,s),ierr)
     enddo
     do p=0, numtasks-1
        call MPI_Irecv(temp(1,1, p*nplanes+s),nx*nplanes,MPI_COMPLEX,p,0,MPI_COMM_WORLD,recvrequest(p,s),ierr)
     enddo

     do ss=1,s
        if( .not. recieved(ss) ) then
           ! for each past plane s, check if the n messages for that plane have completed
           call MPI_testall(numtasks,recvrequest(:,ss),recieved(ss), MPI_STATUSES_IGNORE,ierr)
           if(recieved(ss)) then
              write(0,*) "recieved plane ", ss
              ! hopefully first planes sent are first recieved, below will always make it look like that.
              call nvtxEndRange(id=rangeid(ss))
           endif
        endif
     enddo

  enddo

  ! wait for any remaining planes to finish communicating
  do s=1,nplanes
     if( .not. recieved(s) ) then
        call MPI_waitall(numtasks,recvrequest(:,s), MPI_STATUSES_IGNORE,ierr)
        write(0,*) "waited to recieve plane ", s
        call nvtxEndRange(id=rangeid(s))
     endif
  enddo

  ierr = cudaDeviceSynchronize()

  deallocate(d_a)
  deallocate(d_b)


end subroutine cufft3dTest



end module overlapfft_mod
