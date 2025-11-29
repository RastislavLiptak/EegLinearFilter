//
//  kernel_config.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 29.11.2025.
//

#ifndef KERNEL_CONFIG_H
#define KERNEL_CONFIG_H

#define THREADS_PER_GROUP 256
#define ITEMS_PER_THREAD 16
#define TILE_SIZE (THREADS_PER_GROUP * ITEMS_PER_THREAD)
#define KERNEL_SEGMENT_SIZE 1024

#endif // KERNEL_CONFIG_H
