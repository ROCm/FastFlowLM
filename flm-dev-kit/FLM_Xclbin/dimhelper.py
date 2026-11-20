# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

def count_through_dims(dims: list[tuple[int, int]]) -> list[int]:
    """
    Generate indices according to the dimensions.

    Args:
        dims: List of tuples (size, stride) representing dimensions

    Returns:
        List of integers representing the indices
    """
    if not dims:
        return []

    if len(dims) == 1:
        size, stride = dims[0]
        return [i * stride for i in range(size)]

    size, stride = dims[-1]
    sub_indices = count_through_dims(dims[0:-1])
    result = []

    for i in range(size):
        base = i * stride
        result.extend(base + idx for idx in sub_indices)

    return result


def get_reorder_dims(reorder_index_map: list[int]) -> list[tuple[int, int]]:
    """
    Get the reorder dimensions.
    """
    if (reorder_index_map[1] - reorder_index_map[0] != 1):
        raise ValueError("Could not find the reorder map as the stride 0 is not 1!")

    # find dim 1
    sizes = []
    strides = []
    strides.append(1)

    while len(reorder_index_map) > 0:
        # get one size
        found = False
        for i in range(len(reorder_index_map) - 1):
            if (reorder_index_map[i+1] - reorder_index_map[i] != strides[-1]):
                sizes.append(i + 1)
                found = True
                break

        if not found:
            sizes.append(len(reorder_index_map))
            break

        # shrink the reorder index map
        reorder_index_map = reorder_index_map[::sizes[-1]]
        strides.append(reorder_index_map[1] - reorder_index_map[0])

    return list(zip(sizes, strides))


def get_reordered_dim(length: int, original_shape: list[tuple[int, int]], new_shape: list[tuple[int, int]], direction: str = "S2MM") -> list[tuple[int, int]]:
    """
    Reorder dimensions based on original and new shapes.

    Args:
        length: Total length of the data
        original_shape: List of tuples representing original dimensions, (size, stride)
        new_shape: List of tuples representing target dimensions, (size, stride)

    Returns:
        List of tuples (size, stride) representing the dimensions needed for transformation
    """

    # reverse the order of the shapes
    assert direction in ["S2MM", "MM2S"]
    original_shape = original_shape[-1::-1]
    new_shape = new_shape[-1::-1]
    if direction == "MM2S": # swap the original and new shape
        temp = original_shape
        original_shape = new_shape
        new_shape = temp

    # check if the input dim is valid, the maximum possible accessed address should be length - 1
    max_address = 0
    for dim in original_shape:
        max_address += (dim[0] - 1) * dim[1]

    if max_address != length - 1:
        raise ValueError(f"Invalid input length {length} for the given original shape {original_shape}")

    max_address = 0
    for dim in new_shape:
        max_address += (dim[0] - 1) * dim[1]

    if max_address != length - 1:
        raise ValueError(f"Invalid input length {length} for the given new shape {new_shape}")

    # get the original map
    original_indices = count_through_dims(original_shape)

    # get the new map
    new_indices = count_through_dims(new_shape)

    # get the reordering dimensions
    reorder_index_map = [new_indices.index(i) for i in original_indices]

    reorder_dims = get_reorder_dims(reorder_index_map)

    reorder_dims = reorder_dims[-1::-1]

    return reorder_dims

if __name__ == "__main__":
    # reorder_dims = get_reordered_dim(length = 32 * 256 * 4, original_shape=[(2, 256),(64, 512) ,(256, 1)], new_shape=[(512 // 32, 32), (64, 512), (32, 1)], direction="MM2S")
    # print(reorder_dims)
    # reorder_dims = get_reordered_dim(length = 64 * 512, original_shape=[(2, 256),(64, 512) ,(256, 1)], new_shape=[(2, 256),(64, 512),(256, 1)], direction="MM2S")
    # print(reorder_dims)
    reorder_dims = get_reordered_dim(length = 32 * 128, original_shape=[(32 // 8, 8 * 128),(128 // 8, 8), (8, 128), (8, 1)], new_shape=[(32, 128), (128, 1)], direction="MM2S")
    print(reorder_dims)
