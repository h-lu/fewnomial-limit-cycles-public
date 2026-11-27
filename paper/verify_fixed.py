"""Recover complete fields and verify the finite brackets used in the low-support table."""
from research.cross_visibility.verify import check_archive,check_nine_estimates
from research.tail_interpolation.verify import check_field
from research.tail_interpolation.witness import check_brackets,check_transfer

if __name__=="__main__":
    check_nine_estimates()  # includes check_brackets and check_transfer for the common density
    check_field(False)
    check_archive(False)
