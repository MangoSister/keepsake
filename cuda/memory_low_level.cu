#include "memory_low_level.cuh"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <aclapi.h>
#include <sddl.h>
#include <windows.h>
#include <winternl.h>
#endif

namespace ksc
{

// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/simpleVulkanMMAP

// Windows-specific LPSECURITYATTRIBUTES
void set_default_security_descriptor(CUmemAllocationProp *prop)
{
#if defined(__linux__)
    return;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
    static OBJECT_ATTRIBUTES objAttributes;
    static bool objAttributesConfigured = false;

    if (!objAttributesConfigured) {
        PSECURITY_DESCRIPTOR secDesc;
        BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(sddl, SDDL_REVISION_1, &secDesc, NULL);
        if (result == 0) {
            fprintf(stderr, "IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n", GetLastError());
        }

        InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);

        objAttributesConfigured = true;
    }

    prop->win32HandleMetaData = &objAttributes;
    return;
#endif
}

CudaShareableLowLevelMemory cuda_alloc_device_low_level(size_t size, int device)
{
    // `ipcHandleTypeFlag` specifies the platform specific handle type this sample
    // uses for importing and exporting memory allocation. On Linux this sample
    // specifies the type as CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR meaning that
    // file descriptors will be used. On Windows this sample specifies the type as
    // CU_MEM_HANDLE_TYPE_WIN32 meaning that NT HANDLEs will be used. The
    // ipcHandleTypeFlag variable is a convenience variable and is passed by value
    // to individual requests.
#if defined(__linux__)
    CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_WIN32;
#endif

    CUmemAllocationProp allocProp = {};
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = device;
    allocProp.win32HandleMetaData = nullptr;
    allocProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;

    // Windows-specific LPSECURITYATTRIBUTES is required when
    // CU_MEM_HANDLE_TYPE_WIN32 is used. The security attribute defines the scope
    // of which exported allocations may be tranferred to other processes. For all
    // other handle types, pass NULL.
    set_default_security_descriptor(&allocProp);

    // Get the recommended granularity for m_cudaDevice.
    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    size_t size_rounded = (((size + granularity - 1) / granularity) * granularity);

    // Reserve the required contiguous VA space for the allocations
    CUdeviceptr d_ptr = 0;
    CU_CHECK(cuMemAddressReserve(&d_ptr, size_rounded, granularity, 0U, 0));

    // Create the allocations as a pinned allocation on this device.
    // Create an allocation to store all the positions of points on the xy plane
    // and a second allocation which stores information if the corresponding
    // position is inside the unit circle or not.
    CUmemGenericAllocationHandle handle{};
    CU_CHECK(cuMemCreate(&handle, size_rounded, &allocProp, 0));

    // Export the allocation to a platform-specific handle. The type of handle
    // requested here must match the requestedHandleTypes field in the prop
    // structure passed to cuMemCreate. The handle obtained here will be passed to
    // vulkan to import the allocation.
    ShareableHandle shareable_handle{};
    CU_CHECK(cuMemExportToShareableHandle((void *)&shareable_handle, handle, ipcHandleTypeFlag, 0));

    CU_CHECK(cuMemMap(d_ptr, size_rounded, 0, handle, 0));

    // Release the handles for the allocation. Since the allocation is currently
    // mapped to a VA range with a previous call to cuMemMap the actual freeing of
    // memory allocation will happen on an eventual call to cuMemUnmap. Thus the
    // allocation will be kept live until it is unmapped.
    CU_CHECK(cuMemRelease(handle));

    CUmemAccessDesc accessDescriptor = {};
    accessDescriptor.location.id = device;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Apply the access descriptor to the whole VA range. Essentially enables
    // Read-Write access to the range.
    CU_CHECK(cuMemSetAccess(d_ptr, size_rounded, &accessDescriptor, 1));

    return CudaShareableLowLevelMemory{.dptr = d_ptr, .shareable_handle = shareable_handle, .size = size_rounded};
}

void cuda_free_device_low_level(const CudaShareableLowLevelMemory &m)
{
    CU_CHECK(cuMemUnmap(m.dptr, m.size));

#if defined(__linux__)
    close(shHandle);
#else
    CloseHandle(m.shareable_handle);
#endif

    // Free the virtual address region.
    CU_CHECK(cuMemAddressFree(m.dptr, m.size));
}

} // namespace ksc