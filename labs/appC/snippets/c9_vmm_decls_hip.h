// The same steps: hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemSetAccess.
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr,
                                unsigned long long flags);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
// Our facts file quotes these two declarations; hipMemCreate and hipMemMap are named, not quoted.
