// should be included only into the main c-file
//
// The authoritative definitions live in
// test_gpu_vendor_agnostic_layerFunctions.c.
// Using bare definitions here (common symbols) is non-portable: it relies on
// ELF tentative-definition merging, breaks under -fno-common (GCC default
// since GCC 10), and is a hard error on COFF (Windows/lld-link).
//
// layerFunctions.h already opens an extern "C" block before including this
// header, so these declarations inherit C linkage in C++ translation units.
extern int gpuMemcpyHostToDevice;
extern int gpuMemcpyDeviceToHost;