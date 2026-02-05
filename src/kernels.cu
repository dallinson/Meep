using f32 = float;
using usize = size_t;

__device__ f32 sigmoid(f32 in) { return 1.0f / (1.0f + exp(-1.0f * in)); };
__device__ f32 sigmoid_prime(f32 in) { return sigmoid(in) * (1.0f - sigmoid(in)); };

__device__ f32 squared_diff(f32 a, f32 b) { return powf(a - b, 2.0f); };
__device__ f32 squared_diff_prime(f32 a, f32 b) { return (a - b) * 2.0f; };
__device__ f32 multiply(f32 a, f32 b) { return a * b; };
__device__ f32 add(f32 a, f32 b) { return a + b; };

template <f32(*func)(f32)>
__device__ void apply_unary_function(const f32* src, f32* dest, usize size) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid * 4 >= size) {
        return;
    }
    // here tid * 4 < size
    if (size - (tid * 4) < 4) {
        for (int i = 0; i < (size - (tid * 4)); i++) {
            dest[(tid * 4) + i] = func(src[tid * 4 + i]);
        }
    } else {
        const auto f4_src = reinterpret_cast<const float4*>(src)[tid];
        float4 to_write {};
        to_write.x = func(f4_src.x);
        to_write.y = func(f4_src.y);
        to_write.z = func(f4_src.z);
        to_write.w = func(f4_src.w);
        reinterpret_cast<float4*>(dest)[tid] = to_write;
    }
}

template <f32(*func)(f32, f32)>
__device__ void apply_binary_function(const f32* a, const f32* b, f32* dest, usize size) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid * 4 >= size) {
        return;
    }
    // here tid * 4 < size
    if (size - (tid * 4) < 4) {
        for (int i = 0; i < (size - (tid * 4)); i++) {
            dest[(tid * 4) + i] = func(a[tid * 4 + i], b[tid * 4 + i]);
        }
    } else {
        const auto f4_a = reinterpret_cast<const float4*>(a)[tid];
        const auto f4_b = reinterpret_cast<const float4*>(b)[tid];
        float4 to_write {};
        to_write.x = func(f4_a.x, f4_b.x);
        to_write.y = func(f4_a.y, f4_b.y);
        to_write.z = func(f4_a.z, f4_b.z);
        to_write.w = func(f4_a.w, f4_b.w);
        reinterpret_cast<float4*>(dest)[tid] = to_write;
    }
}

#define UNARY(func_name) extern "C" __global__ void func_name(const f32* src, f32* dest, usize size) { apply_unary_function<func_name>(src, dest, size); }
#define UNARY_INPLACE(func_name) extern "C" __global__ void func_name##_inplace(f32* src, usize size) { apply_unary_function<func_name>(src, src, size); }

#define BINARY(func_name) extern "C" __global__ void func_name(const f32* a, const f32* b, f32* dest, usize size) { apply_binary_function<func_name>(a, b, dest, size); }
#define BINARY_INPLACE(func_name) extern "C" __global__ void func_name##_inplace(const f32* src, f32* dest, usize size) { apply_binary_function<func_name>(src, dest, dest, size); }

UNARY(sigmoid);
UNARY(sigmoid_prime);
BINARY(squared_diff);
BINARY(squared_diff_prime);
BINARY(multiply);
BINARY(add);

UNARY_INPLACE(sigmoid);
UNARY_INPLACE(sigmoid_prime);
BINARY_INPLACE(squared_diff);
BINARY_INPLACE(squared_diff_prime);
BINARY_INPLACE(multiply);
BINARY_INPLACE(add);

template <typename T> 
__host__ __device__ auto ceil_div(T val, T divisor) -> T {
    return val / divisor + (val % divisor != 0);
}

extern "C" __global__ void splat(const f32* src, f32* dest, usize vec_size, usize dest_count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto vertical_count = ceil_div(vec_size, static_cast<usize>(4)); // number of threads per vector
    if (tid > (dest_count * vertical_count)) {
        return;
    }
    const auto vid = tid % vertical_count;
    const auto hid = tid / vertical_count;
    if (vid == (vertical_count - 1)) {
        // piece by piece
        const auto copy_count = vec_size % 4;
        const auto start_idx = (hid * vec_size) + (4 * vid);
        for (int i = 0; i < copy_count; i++) {
            const auto src_item = src[(4 * vid) + i];
            dest[start_idx + i] = src_item;
        }

    } else {
        const auto start_idx = (hid * vec_size) + (4 * vid);
        for (int i = 0; i < 4; i++) {
            const auto src_item = src[(4 * vid) + i];
            dest[start_idx + i] = src_item;
        }
    }
}

__host__ __device__ float4 operator-(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x-rhs.x,lhs.y-rhs.y,lhs.z-rhs.z,lhs.w-rhs.w);
}

__host__ __device__ float4 operator*(const float4& lhs, const float4& rhs) {
    return make_float4(lhs.x*rhs.x,lhs.y*rhs.y,lhs.z*rhs.z,lhs.w*rhs.w);
}

extern "C" __global__ void fill_estimated(const usize* src, f32* dest, usize vec_size, usize vec_count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= vec_count) {
        return;
    }
    dest[(vec_size * tid) + src[tid]] = 1.0f;
}

extern "C" __global__ void reduce_vecwise(f32* data, usize vec_length, usize vec_count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= vec_count) {
        return;
    }
    auto sum = 0.0f;
    for (int i = 0; i < vec_length; i++) {
        sum += data[(tid * vec_length) + i];
    }
    data[tid * vec_length] = sum;
}

extern "C" __global__ void reduce_strided(f32* data, usize count, usize stride) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto vid = blockIdx.y;
    auto val = tid >= count ? 0.0f : data[(tid * stride) + vid];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset); // Gather to 0th 
    }
    if (threadIdx.x % warpSize == 0 && tid < count) {
        data[(tid * stride) + vid] = val;
    }
}

extern "C" __global__ void map_uniform(f32* data, usize count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) {
        return;
    }
    data[tid] = (2 * data[tid]) - 1;
}

extern "C" __global__ void mult_by_float(f32* data, const f32 mult, usize count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid * 4 >= count) {
        return;
    }
    if (count - (tid * 4) < 4) {
        for (int i = 0; i < (count - (tid * 4)); i++) {
            data[(tid * 4 + i)] = data[(tid * 4 + i)] * mult;
        }
    } else {
        auto f4 = reinterpret_cast<float4*>(data)[tid];
        f4.x *= mult;
        f4.y *= mult;
        f4.z *= mult;
        f4.w *= mult;
        reinterpret_cast<float4*>(data)[tid] = f4;
    }
}