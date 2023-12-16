#include <stdio.h>
#include <iostream>
#include "common/gemm_tb.hh"

#define rank 2

struct MemRefDescriptor {
    uint64_t allocated;
    void* aligned;
    uint64_t offset;
    uint64_t shape[rank];
    uint64_t strides[rank];

    friend std::ostream& operator<<(std::ostream& os, const MemRefDescriptor& desc)
    {
        os << "desc@: " << &desc << std::endl;
        os << "\tallocated: " << desc.allocated << std::endl;
        os << "\taligned: " << desc.aligned << std::endl;
        os << "\toffset: " << desc.offset << std::endl;
        os << "\tshape: [";
        for (int i = 0; i < rank; i++) { os << desc.shape[i]; if (i < rank - 1) os << ", "; }
        os << "]" << std::endl;
        os << "\tstrides: [";
        for (int i = 0; i < rank; i++) { os << desc.strides[i]; if (i < rank - 1) os << ", "; }
        os << "]" << std::endl;
        return os;
    }
};


extern "C" void matmul_f32(struct MemRefDescriptor *pDA, struct MemRefDescriptor *pDB, struct MemRefDescriptor *pDC)
{
    uint32_t M = pDA->shape[0];
    uint32_t N = pDB->shape[1];
    uint32_t K = pDA->shape[1];
    sw_gemm_nn<float_t>(M, N, K, 1, (void*)(pDA->aligned), K, (void*)(pDB->aligned), N, (void*)(pDC->aligned), N);
}


extern "C" void hwacc_debug(struct MemRefDescriptor *pDA, struct MemRefDescriptor *pDB, struct MemRefDescriptor *pDC)
{
    std::cout << "pDA  " << *pDA << "\n";
    std::cout << "pDB  " << *pDB << "\n";
    std::cout << "pDC  " << *pDC << "\n";
    matmul_f32(pDA, pDB, pDC);
    std::cout << "\n";
}

