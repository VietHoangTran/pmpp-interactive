# Lab 20.3, a bug hunt in lab 20.2's flash attention kernel. B_c is TILE_WIDTH: 32 as in the
# book (Metal chooses B_c at run time, in launch.mm). RACE: two of the three defects are
# missing barriers. No benchmark.
pmpp_add_lab(ch20/lab3_flash_hunt VARIANTS work solution RACE TILE 32)
