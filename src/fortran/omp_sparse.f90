module omp_sparse_mod

    use omp_lib
    implicit none

contains

subroutine dense_dot_sparse_v4(dense_mat, csc_indptr, csc_indices, csc_data, m, k, n, result)
    use omp_lib
    implicit none

    !f2py intent(in) :: m, k, n
    !f2py depend(m, k) :: dense_mat
    !f2py intent(in) :: csc_indptr, csc_indices, csc_data
    !f2py depend(n) :: csc_indptr
    !f2py intent(out) :: result
    !f2py depend(m, n) :: result

    integer, intent(in) :: m, k, n
    real(kind=8), dimension(m, k), intent(in) :: dense_mat
    integer, dimension(n + 1), intent(in) :: csc_indptr
    integer, dimension(:), intent(in) :: csc_indices
    real(kind=8), dimension(:), intent(in) :: csc_data
    real(kind=8), dimension(m, n), intent(out) :: result

    integer :: j, l, row_idx, nnz
    real(kind=8) :: sparse_val

    nnz = size(csc_indices)
    result = 0.0d0

    !$omp parallel default(private) shared(result, dense_mat, csc_indptr, csc_indices, csc_data, m, k, n)

    !$omp single
    print *, "omp_sparse: Using", omp_get_num_threads(), "threads in OpenMP"
    !$omp end single

    !$omp do schedule(dynamic, 16)
    do j = 1, n ! Loop over columns of sparse matrix
        do l = csc_indptr(j) + 1, csc_indptr(j + 1)
            row_idx = csc_indices(l) + 1  ! Convert 0-based to 1-based
            sparse_val = csc_data(l)

            ! Vectorized operation for better performance
            result(:, j) = result(:, j) + dense_mat(:, row_idx) * sparse_val
        end do
    end do
    !$omp end do

    !$omp end parallel

end subroutine

subroutine dense_dot_sparse_v11(dense_mat, row_start_col, row_segment_len, seg_data, m, k, n, result)
    use omp_lib
    implicit none

    !f2py intent(in) :: m, k, n
    !f2py depend(m, k) :: dense_mat
    !f2py intent(in) :: row_start_col, row_segment_len, seg_data
    !f2py depend(k) :: row_start_col, row_segment_len
    !f2py intent(out) :: result
    !f2py depend(m, n) :: result

    integer, intent(in) :: m, k, n
    real(kind=8), dimension(m, k), intent(in) :: dense_mat
    integer, dimension(k), intent(in) :: row_start_col, row_segment_len
    real(kind=8), dimension(:), intent(in) :: seg_data
    real(kind=8), dimension(m, n), intent(out) :: result

    integer :: i, j, start_col, seg_len, row_data_start
    integer, dimension(k) :: data_offsets
    real(kind=8) :: sparse_val

    result = 0.0d0

    ! Pre-calculate data offsets for each row
    data_offsets(1) = 1
    do i = 2, k
        data_offsets(i) = data_offsets(i-1) + row_segment_len(i-1)
    end do

    !$omp parallel default(private) shared(result, dense_mat, row_start_col, row_segment_len, seg_data, data_offsets, m, k, n)

    !$omp single
    print *, "omp_sparse v11: Using", omp_get_num_threads(), "threads in OpenMP"
    !$omp end single

    ! Parallelize over output columns instead of input rows to avoid race conditions
    !$omp do schedule(static) private(i, j, start_col, seg_len, sparse_val, row_data_start)
    do j = 1, n ! Loop over columns of result matrix
        do i = 1, k ! Loop over rows of sparse matrix (which are columns of dense matrix)
            start_col = row_start_col(i) + 1  ! Convert 0-based to 1-based
            seg_len = row_segment_len(i)
            row_data_start = data_offsets(i)

            if (seg_len > 0) then
                ! Check if this column j is within the segment for row i
                if (j >= start_col .and. j < start_col + seg_len) then
                    ! Calculate position within the segment
                    sparse_val = seg_data(row_data_start + (j - start_col))

                    ! No race condition since each thread handles different columns
                    result(:, j) = result(:, j) + dense_mat(:, i) * sparse_val
                end if
            end if
        end do
    end do
    !$omp end do

    !$omp end parallel

end subroutine

end module omp_sparse_mod
