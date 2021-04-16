#ifndef UTILS_H
#define UTILS_H


void * allocateDeviceBuffer(int N);
int sendBufferToDevice(void *d_buf, void *h_buf, int N);
int getBufferFromDevice(void *h_buf, void *d_buf, int N);
