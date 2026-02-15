
using f32 = float;
using usize = size_t;
using i32 = int;

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

extern "C" __global__ void find_matching(const f32* data, const usize* labels, f32* matches, usize vec_size, usize count) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) {
        return;
    }
    auto max = data[tid * vec_size];
    auto max_idx = 0;
    for (usize i = 1; i < vec_size; i++) {
        const auto elem = data[(tid * vec_size) + i];
        if (elem > max) {
            max = elem;
            max_idx = i;
        }
    }
    matches[tid] = max_idx == labels[tid] ? 1.0f : 0.0f;
}

__host__ __device__ float4 operator*(const float4 left, const f32 right) {
    return make_float4(left.x * right, left.y * right, left.z * right, left.w * right);
}

__host__ __device__ float4 operator*(const f32 left, const float4 right) {
    return right * left;
}

__host__ __device__ float4 operator+(const float4 left, const float4 right) {
    return make_float4(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
}

__host__ __device__ float4 powf(const float4 left, f32 right) {
    return make_float4(powf(left.x, right), powf(left.y, right), powf(left.z, right), powf(left.w, right));
}

__host__ __device__ float4 operator/(const float4 left, const f32 right) {
    return make_float4(left.x / right, left.y / right, left.z / right, left.w / right);
}

__host__ __device__ float4 sqrt(const float4 left) {
    return make_float4(sqrt(left.x), sqrt(left.y), sqrt(left.z), sqrt(left.w));
}

__host__ __device__ float4 operator+(const float4 left, const f32 right) {
    return make_float4(left.x + right, left.y + right, left.z + right, left.w + right);
}

__host__ __device__ float4 operator/(const float4 left, const float4 right) {
    return make_float4(left.x / right.x, left.y / right.y, left.z / right.z, left.w / right.w);
}

extern "C" __global__ void adam_kernel(const f32* gradient, f32* weights, f32* m, f32* v, usize count, f32 t, f32 lr, f32 beta1, f32 beta2, f32 epsilon) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= ceil_div(count, static_cast<usize>(4))) {
        return;
    }
    if (count - (tid * 4) < 4) {
        for (int i = 0; i < (count - (tid * 4)); i++) {
            const auto idx = (tid * 4) + i;
            m[idx] = m[idx] * beta1 + (1.0f - beta1) * gradient[idx];
            v[idx] = v[idx] * beta2 + (1.0f - beta2) * powf(gradient[idx], 2.0f);

            const auto m_hat = m[idx] / (1.0f - powf(beta1, t));
            const auto v_hat = v[idx] / (1.0f - powf(beta2, t));

            weights[idx] = weights[idx] - lr * m_hat / (sqrt(v_hat) + epsilon);
        }
    } else {
        auto m_f4 = reinterpret_cast<float4*>(m)[tid];
        auto v_f4 = reinterpret_cast<float4*>(v)[tid];
        auto grad_f4 = reinterpret_cast<const float4*>(gradient)[tid];

        m_f4 = m_f4 * beta1 + (1.0f - beta1) * grad_f4;
        v_f4 = v_f4 * beta2 + (1.0f - beta2) * powf(grad_f4, 2.0f);

        const auto m_hat = m_f4 / (1.0f - powf(beta1, t));
        const auto v_hat = v_f4 / (1.0f - powf(beta2, t));

        reinterpret_cast<float4*>(m)[tid] = m_f4;
        reinterpret_cast<float4*>(v)[tid] = v_f4;

        reinterpret_cast<float4*>(weights)[tid] = reinterpret_cast<float4*>(weights)[tid] - lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

extern "C" __global__ void adamw_kernel(const f32* gradient, f32* weights, f32* m, f32* v, usize count, f32 t, f32 lr, f32 beta1, f32 beta2, f32 epsilon, f32 weight_decay) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= ceil_div(count, static_cast<usize>(4))) {
        return;
    }
    if (count - (tid * 4) < 4) {
        for (int i = 0; i < (count - (tid * 4)); i++) {
            const auto idx = (tid * 4) + i;
            m[idx] = m[idx] * beta1 + (1.0f - beta1) * gradient[idx];
            v[idx] = v[idx] * beta2 + (1.0f - beta2) * powf(gradient[idx], 2.0f);

            const auto m_hat = m[idx] / (1.0f - powf(beta1, t));
            const auto v_hat = v[idx] / (1.0f - powf(beta2, t));

            const auto weight = weights[idx];

            weights[idx] = weights[idx] - lr * m_hat / (sqrt(v_hat) + epsilon);

            if (weight_decay != 0) {
                weights[idx] = weights[idx] - lr * weight_decay * weight;
            }
        }
    } else {
        auto m_f4 = reinterpret_cast<float4*>(m)[tid];
        auto v_f4 = reinterpret_cast<float4*>(v)[tid];
        auto grad_f4 = reinterpret_cast<const float4*>(gradient)[tid];

        m_f4 = m_f4 * beta1 + (1.0f - beta1) * grad_f4;
        v_f4 = v_f4 * beta2 + (1.0f - beta2) * powf(grad_f4, 2.0f);

        const auto m_hat = m_f4 / (1.0f - powf(beta1, t));
        const auto v_hat = v_f4 / (1.0f - powf(beta2, t));

        reinterpret_cast<float4*>(m)[tid] = m_f4;
        reinterpret_cast<float4*>(v)[tid] = v_f4;

        const auto prev_weights = reinterpret_cast<float4*>(weights)[tid];

        reinterpret_cast<float4*>(weights)[tid] = reinterpret_cast<float4*>(weights)[tid] - lr * m_hat / (sqrt(v_hat) + epsilon);

        if (weight_decay != 0) {
            reinterpret_cast<float4*>(weights)[tid] = reinterpret_cast<float4*>(weights)[tid] - lr * weight_decay * prev_weights;
        }
    }
}

