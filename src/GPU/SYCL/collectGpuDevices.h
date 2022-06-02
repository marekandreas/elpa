#ifndef COLLECT_GPU_DEVICES_H
#define COLLECT_GPU_DEVICES_H


using namespace cl;

static std::vector<sycl::device> devices;
static std::optional<sycl::queue> chosenDeviceQueue{};

bool deviceCollectionFlag = false;

extern "C" {

  static void collectGpuDevices() {
    if (deviceCollectionFlag) {
      return;
    }

    for (auto const &p : sycl::platform::get_platforms()) {
      for (auto dev : p.get_devices()) {
        using namespace sycl::info;
        // why this?
#ifndef ENABLE_ALL_DEVICES
        if (dev.get_info<device::device_type>() == device_type::gpu)
#endif
        {
          devices.push_back(dev);
        }

      }
    }
  }
}

#endif
