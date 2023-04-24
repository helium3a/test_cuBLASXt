# Test cuBlasXt vs cblas

## compile and run with command:
``` shell
nvcc -o test test.cu -lcublas -lcurand -Xlinker /usr/lib/x86_64-linux-gnu/libopenblas.so.0 && ./test
```