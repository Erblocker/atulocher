ifndef AVX
        AVX_DEF=$(shell $(CC) -mavx2 -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(findstring AVX2,$(AVX_DEF)),AVX2)
        AVX=on
endif
endif

