#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>

// Physical base address of the Lightweight HPS-to-FPGA bridge
#define HW_REGS_BASE 0xFF200000
#define HW_REGS_SPAN 0x00200000
#define HW_REGS_MASK (HW_REGS_SPAN - 1)

int main() {
    int fd = open("/dev/mem", (O_RDWR | O_SYNC));
    if (fd == -1) {
        std::cerr << "Error: Could not open /dev/mem. Please run with sudo." << std::endl;
        return 1;
    }

    void *virtual_base = mmap(NULL, HW_REGS_SPAN, (PROT_READ | PROT_WRITE),
                              MAP_SHARED, fd, HW_REGS_BASE);

    if (virtual_base == MAP_FAILED) {
        std::cerr << "Error: mmap() failed." << std::endl;
        close(fd);
        return 1;
    }

    void *sensor_addr = (uint8_t *)virtual_base + (0x0000 & HW_REGS_MASK);

    volatile uint32_t *distance_ptr    = (uint32_t *)((uint8_t *)sensor_addr + 0x00); // Reg0
    volatile uint32_t *temperature_ptr = (uint32_t *)((uint8_t *)sensor_addr + 0x04); // Reg1
    volatile uint32_t *ntu_ptr         = (uint32_t *)((uint8_t *)sensor_addr + 0x08); // Reg2
    volatile uint32_t *flow_ptr        = (uint32_t *)((uint8_t *)sensor_addr + 0x0C); // Reg3
    volatile uint32_t *total_pulse_ptr = (uint32_t *)((uint8_t *)sensor_addr + 0x10); // Reg4
    volatile uint32_t *adc_raw_ptr     = (uint32_t *)((uint8_t *)sensor_addr + 0x14); // Reg5
    volatile uint32_t *latency_ptr     = (uint32_t *)((uint8_t *)sensor_addr + 0x18); // Reg6

    std::cout << "Reading FPGA Sensors (Ctrl+C to exit)..." << std::endl;

    while (true) {
        uint32_t raw_temp = *temperature_ptr & 0xFFFF;
        float temp_celsius = (float)raw_temp / 16.0f;

        uint32_t raw_adc = *adc_raw_ptr & 0x0FFF;
        float adc_voltage = raw_adc * 4.096f / 4095.0f;

        float flow_lmin = (float)(*flow_ptr & 0xFFFF) / 10.0f;

        uint32_t latency_count = *latency_ptr;
        float latency_us = latency_count / 50.0f;
        float latency_ms = latency_count / 50000.0f;

        std::cout << "Distance: " << (*distance_ptr & 0xFFFF) << " cm | "
                  << "Temp: " << temp_celsius << " C | "
                  << "Turbidity: " << (*ntu_ptr & 0xFFFF) << " NTU | "
                  << "ADC Raw: " << raw_adc << " | "
                  << "ADC Voltage: " << adc_voltage << " V | "
                  << "Flow: " << flow_lmin << " L/min | "
                  << "FPGA Latency: " << latency_count << " cycles, "
                  << latency_us << " us, "
                  << latency_ms << " ms"
                  << std::endl;

        usleep(500000);
    }

    munmap(virtual_base, HW_REGS_SPAN);
    close(fd);
    return 0;
}