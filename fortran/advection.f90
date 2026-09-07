! 1D linear advection, explicit first-order upwind scheme, periodic BC.
! Same equation/IC as the Python CFD solver and the PINN, so runtimes
! and final fields can be compared directly.
!
! Usage: advection_fortran [nx]
program advection
  implicit none

  real(8), parameter :: c = 1.0d0
  real(8), parameter :: cfl = 0.8d0
  real(8), parameter :: t_final = 1.0d0
  real(8), parameter :: pi = 3.14159265358979323846d0

  integer :: nx, nt, i, n
  real(8) :: dx, dt
  real(8), allocatable :: u(:), u_new(:), x(:)
  integer(8) :: count_start, count_end, count_rate
  character(len=32) :: arg

  nx = 4000
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg)
    read(arg, *) nx
  end if

  dx = 1.0d0 / real(nx, 8)
  dt = cfl * dx / c
  nt = ceiling(t_final / dt)

  allocate(u(0:nx-1), u_new(0:nx-1), x(0:nx-1))

  do i = 0, nx - 1
    x(i) = real(i, 8) * dx
    u(i) = sin(2.0d0 * pi * x(i))
  end do

  call system_clock(count_start, count_rate)

  do n = 1, nt
    u_new(0) = u(0) - c * dt / dx * (u(0) - u(nx - 1))
    do i = 1, nx - 1
      u_new(i) = u(i) - c * dt / dx * (u(i) - u(i - 1))
    end do
    u = u_new
  end do

  call system_clock(count_end)

  print '(A,I0)', 'nx=', nx
  print '(A,I0)', 'nt=', nt
  print '(A,F12.6)', 'elapsed_seconds=', real(count_end - count_start, 8) / real(count_rate, 8)

  open(unit=10, file='results/fortran_output.csv', status='replace')
  write(10, '(A)') 'x,u'
  do i = 0, nx - 1
    write(10, '(F14.10,A,F14.10)') x(i), ',', u(i)
  end do
  close(10)

end program advection
