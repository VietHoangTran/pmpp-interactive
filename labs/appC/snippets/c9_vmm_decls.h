// 1. Reserve a range of virtual addresses.
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags);
// 2. Allocate physical memory on a device; you get a handle, not an address.
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags);
// 3. Put that memory behind part of the range.
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags);
// 4. Let devices read and write it. The book's list leaves this one out.
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
// Teardown, in this order: cuMemUnmap, cuMemRelease, cuMemAddressFree.
