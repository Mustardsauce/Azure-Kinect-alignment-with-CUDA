#include "cuda_k4a_align.h"

#define CUDA_THREADS_PER_BLOCK 16

static inline int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}

template<typename  T>
std::shared_ptr<T> make_device_copy(T obj)
{
	T* d_data;
	auto res = cudaMalloc(&d_data, sizeof(T));
	if (res != cudaSuccess)
		throw std::runtime_error("cudaMalloc failed status: " + res);
	cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);
	return std::shared_ptr<T>(d_data, [](T* data) { cudaFree(data); });
}

template<typename  T>
std::shared_ptr<T> alloc_dev(int elements)
{
	T* d_data;
	auto res = cudaMalloc(&d_data, sizeof(T) * elements);
	if (res != cudaSuccess)
		throw std::runtime_error("cudaMalloc failed status: " + res);
	return std::shared_ptr<T>(d_data, [](T* p) { cudaFree(p); });
}

template<class T>
void release_memory(T& obj)
{
	obj = nullptr;
}

__device__
static bool cuda_project_pixel_to_point_with_distortion(const struct cuda_align::cuda_intrinsics * camera_calibration,
	const float xy[2],
	float uv[2],
	int& valid,
	float J_xy[2 * 2])
{
	float cx = camera_calibration->cx;
	float cy = camera_calibration->cy;
	float fx = camera_calibration->fx;
	float fy = camera_calibration->fy;
	float k1 = camera_calibration->k1;
	float k2 = camera_calibration->k2;
	float k3 = camera_calibration->k3;
	float k4 = camera_calibration->k4;
	float k5 = camera_calibration->k5;
	float k6 = camera_calibration->k6;
	float codx = camera_calibration->codx; // center of distortion is set to 0 for Brown Conrady model
	float cody = camera_calibration->cody;
	float p1 = camera_calibration->p1;
	float p2 = camera_calibration->p2;
	float max_radius_for_projection = camera_calibration->metric_radius * camera_calibration->metric_radius;


	valid = 1;

	float xp = xy[0] - codx;
	float yp = xy[1] - cody;

	float xp2 = xp * xp;
	float yp2 = yp * yp;
	float xyp = xp * yp;
	float rs = xp2 + yp2;
	if (rs > max_radius_for_projection)
	{
		valid = 0;
		return true;
	}
	float rss = rs * rs;
	float rsc = rss * rs;
	float a = 1.f + k1 * rs + k2 * rss + k3 * rsc;
	float b = 1.f + k4 * rs + k5 * rss + k6 * rsc;
	float bi;
	if (b != 0.f)
	{
		bi = 1.f / b;
	}
	else
	{
		bi = 1.f;
	}
	float d = a * bi;

	float xp_d = xp * d;
	float yp_d = yp * d;

	float rs_2xp2 = rs + 2.f * xp2;
	float rs_2yp2 = rs + 2.f * yp2;

	xp_d += rs_2xp2 * p2 + 2.f * xyp * p1;
	yp_d += rs_2yp2 * p1 + 2.f * xyp * p2;

	float xp_d_cx = xp_d + codx;
	float yp_d_cy = yp_d + cody;

	uv[0] = xp_d_cx * fx + cx;
	uv[1] = yp_d_cy * fy + cy;

	/*if (J_xy == 0)
	{
	return true;
	}*/

	// compute Jacobian matrix
	float dudrs = k1 + 2.f * k2 * rs + 3.f * k3 * rss;
	// compute d(b)/d(r^2)
	float dvdrs = k4 + 2.f * k5 * rs + 3.f * k6 * rss;
	float bis = bi * bi;
	float dddrs = (dudrs * b - a * dvdrs) * bis;

	float dddrs_2 = dddrs * 2.f;
	float xp_dddrs_2 = xp * dddrs_2;
	float yp_xp_dddrs_2 = yp * xp_dddrs_2;

	J_xy[0] = fx * (d + xp * xp_dddrs_2 + 6.f * xp * p2 + 2.f * yp * p1);
	J_xy[1] = fx * (yp_xp_dddrs_2 + 2.f * yp * p2 + 2.f * xp * p1);
	J_xy[2] = fy * (yp_xp_dddrs_2 + 2.f * xp * p1 + 2.f * yp * p2);
	J_xy[3] = fy * (d + yp * yp * dddrs_2 + 6.f * yp * p1 + 2.f * xp * p2);

	return true;
}

__device__
static bool cuda_deproject_pixel_to_point_with_distortion_iterative(const struct cuda_align::cuda_intrinsics * camera_calibration,
	const float uv[2], float xy[3], int& valid, unsigned int max_passes)
{
	valid = 1;
	float Jinv[2 * 2];
	float best_xy[2] = { 0.f, 0.f };
	float best_err = FLT_MAX;

	for (unsigned int pass = 0; pass < max_passes; ++pass)
	{
		float p[2];
		float J[2 * 2];

		if (cuda_project_pixel_to_point_with_distortion(camera_calibration, xy, p, valid, J) == false)
		{
			return false;
		}
		if (valid == 0)
		{
			return true;
		}

		float err_x = uv[0] - p[0];
		float err_y = uv[1] - p[1];
		float err = err_x * err_x + err_y * err_y;
		if (err >= best_err)
		{
			xy[0] = best_xy[0];
			xy[1] = best_xy[1];
			break;
		}

		best_err = err;
		best_xy[0] = xy[0];
		best_xy[1] = xy[1];

		float detJ = J[0] * J[3] - J[1] * J[2];
		float inv_detJ = 1.f / detJ;

		Jinv[0] = inv_detJ * J[3];
		Jinv[3] = inv_detJ * J[0];
		Jinv[1] = -inv_detJ * J[1];
		Jinv[2] = -inv_detJ * J[2];

		if (pass + 1 == max_passes || best_err < 1e-22f)
		{
			break;
		}

		float dx = Jinv[0] * err_x + Jinv[1] * err_y;
		float dy = Jinv[2] * err_x + Jinv[3] * err_y;

		xy[0] += dx;
		xy[1] += dy;
	}

	if (best_err > 1e-6f)
	{
		valid = 0;
	}

	return true;
}

__device__ 
static void cuda_deproject_pixel_to_point_with_distortion(const struct cuda_align::cuda_intrinsics * camera_calibration,
	const float uv[2], float xy[3], float depth)
{
	float xp_d = (uv[0] - camera_calibration->cx) / camera_calibration->fx - camera_calibration->codx;
	float yp_d = (uv[1] - camera_calibration->cy) / camera_calibration->fy - camera_calibration->cody;

	float rs = xp_d * xp_d + yp_d * yp_d;
	float rss = rs * rs;
	float rsc = rss * rs;
	float a = 1.f + camera_calibration->k1 * rs + camera_calibration->k2 * rss + camera_calibration->k3 * rsc;
	float b = 1.f + camera_calibration->k4 * rs + camera_calibration->k5 * rss + camera_calibration->k6 * rsc;
	float ai;
	if (a != 0.f)
	{
		ai = 1.f / a;
	}
	else
	{
		ai = 1.f;
	}
	float di = ai * b;

	float x = xp_d * di;
	float y = yp_d * di;

	// approximate correction for tangential params
	float two_xy = 2.f * x * y;
	float xx = x * x;
	float yy = y * y;

	x -= (yy + 3.f * xx) * camera_calibration->p2 + two_xy * camera_calibration->p1;
	y -= (xx + 3.f * yy) * camera_calibration->p1 + two_xy * camera_calibration->p2;

	// add on center of distortion
	x += camera_calibration->codx;
	y += camera_calibration->cody;

	xy[0] = x;
	xy[1] = y;
	xy[2] = depth;


	int valid;
	if (cuda_deproject_pixel_to_point_with_distortion_iterative(camera_calibration, uv, xy, valid, 20))
	{
		xy[0] *= depth;
		xy[1] *= depth;
		xy[2] = depth;
	}
	else
	{
		xy[0] = xy[1] = xy[2] = 0.0f;
	}

}

__device__
static bool cuda_project_pixel_to_point_with_distortion(const struct cuda_align::cuda_intrinsics * camera_calibration,
	const float xy[2],
	float uv[2])
{
	float cx = camera_calibration->cx;
	float cy = camera_calibration->cy;
	float fx = camera_calibration->fx;
	float fy = camera_calibration->fy;
	float k1 = camera_calibration->k1;
	float k2 = camera_calibration->k2;
	float k3 = camera_calibration->k3;
	float k4 = camera_calibration->k4;
	float k5 = camera_calibration->k5;
	float k6 = camera_calibration->k6;
	float codx = camera_calibration->codx; // center of distortion is set to 0 for Brown Conrady model
	float cody = camera_calibration->cody;
	float p1 = camera_calibration->p1;
	float p2 = camera_calibration->p2;
	float max_radius_for_projection = camera_calibration->metric_radius * camera_calibration->metric_radius;

	float xp = xy[0] - codx;
	float yp = xy[1] - cody;

	float xp2 = xp * xp;
	float yp2 = yp * yp;
	float xyp = xp * yp;
	float rs = xp2 + yp2;

	if (rs > max_radius_for_projection)
	{
		return true;
	}

	float rss = rs * rs;
	float rsc = rss * rs;
	float a = 1.f + k1 * rs + k2 * rss + k3 * rsc;
	float b = 1.f + k4 * rs + k5 * rss + k6 * rsc;
	float bi;
	if (b != 0.f)
	{
		bi = 1.f / b;
	}
	else
	{
		bi = 1.f;
	}
	float d = a * bi;

	float xp_d = xp * d;
	float yp_d = yp * d;

	float rs_2xp2 = rs + 2.f * xp2;
	float rs_2yp2 = rs + 2.f * yp2;

	xp_d += rs_2xp2 * p2 + 2.f * xyp * p1;
	yp_d += rs_2yp2 * p1 + 2.f * xyp * p2;

	float xp_d_cx = xp_d + codx;
	float yp_d_cy = yp_d + cody;

	uv[0] = xp_d_cx * fx + cx;
	uv[1] = yp_d_cy * fy + cy;

	return true;
}

__device__ 
static void cuda_transform_point_to_point(float to_point[3], const struct cuda_align::cuda_extrinsics * extrin, const float from_point[3])
{
	to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[1] * from_point[1] + extrin->rotation[2] * from_point[2] + extrin->translation[0];
	to_point[1] = extrin->rotation[3] * from_point[0] + extrin->rotation[4] * from_point[1] + extrin->rotation[5] * from_point[2] + extrin->translation[1];
	to_point[2] = extrin->rotation[6] * from_point[0] + extrin->rotation[7] * from_point[1] + extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

__global__
void kernel_color_to_depth(uint8_t* aligned_out, const uint16_t* depth_in, const uint8_t* color_in, const cuda_align::cuda_intrinsics* depth_intrin, const cuda_align::cuda_intrinsics* color_intrin, const cuda_align::cuda_extrinsics* depth_to_color, const float depth_scale)
{
	int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
	int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

	//int depth_size = depth_intrin->width * depth_intrin->height;
	int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

	if (depth_x >= 0 && depth_x < depth_intrin->width && depth_y >= 0 && depth_y < depth_intrin->height)
	{
		float uv[2] = { depth_x ,depth_y };
		float xyz[3];

		const float depth_value = depth_in[depth_pixel_index] * depth_scale;

		if (depth_value == 0)
			return;

		cuda_deproject_pixel_to_point_with_distortion(depth_intrin, uv, xyz, depth_value);

		float target_xyz[3];
		cuda_transform_point_to_point(target_xyz, depth_to_color, xyz);

		if (target_xyz[2] <= 0.f)
		{
			return;
		}

		float xy_for_projection[2];
		xy_for_projection[0] = target_xyz[0] / target_xyz[2];
		xy_for_projection[1] = target_xyz[1] / target_xyz[2];

		float target_uv[2] = { -1.f,-1.f };

		cuda_project_pixel_to_point_with_distortion(color_intrin, xy_for_projection, target_uv);

		const int target_x = target_uv[0] + 0.5f;
		const int target_y = target_uv[1] + 0.5f;


		if (target_x >= 0 && target_x < color_intrin->width && target_y >= 0 && target_y < color_intrin->height)
		{
			const int from_offset = 3 * depth_pixel_index;
			const int to_offset = 3 * (target_y*color_intrin->width + target_x);

			aligned_out[from_offset + 0] = color_in[to_offset + 0];
			aligned_out[from_offset + 1] = color_in[to_offset + 1];
			aligned_out[from_offset + 2] = color_in[to_offset + 2];
		}
	}
}


void cuda_k4a_align::align_color_to_depth(uint8_t* aligned_out
	, const uint16_t* depth_in 
	, const uint8_t* color_in 
	, float depth_scale	
	, const k4a_calibration_t& calibration
)
{
	cuda_align::cuda_intrinsics depth_intrinsic(calibration.depth_camera_calibration);	
	cuda_align::cuda_intrinsics color_intrinsic(calibration.color_camera_calibration);	
	cuda_align::cuda_extrinsics depth_to_color(calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR]);


	const int depth_pixel_count = depth_intrinsic.width * depth_intrinsic.height;
	const int color_pixel_count = color_intrinsic.width * color_intrinsic.height;
	const int aligned_pixel_count = depth_pixel_count;

	const int depth_byte_size = depth_pixel_count * sizeof(uint16_t);
	const int color_byte_size = color_pixel_count * sizeof(uint8_t) * 3;
	const int aligned_byte_size = aligned_pixel_count * sizeof(uint8_t) * 3;

	// allocate and copy objects to cuda device memory
	if (!d_depth_intrinsics) d_depth_intrinsics = make_device_copy(depth_intrinsic);
	if (!d_color_intrinsics) d_color_intrinsics = make_device_copy(color_intrinsic);
	if (!d_depth_color_extrinsics) d_depth_color_extrinsics = make_device_copy(depth_to_color);
	

	if (!d_depth_in) d_depth_in = alloc_dev<uint16_t>(depth_pixel_count);
	cudaMemcpy(d_depth_in.get(), depth_in, depth_byte_size, cudaMemcpyHostToDevice);
	if (!d_color_in) d_color_in = alloc_dev<uint8_t>(color_pixel_count * 3);
	cudaMemcpy(d_color_in.get(), color_in, color_byte_size, cudaMemcpyHostToDevice);
	if (!d_aligned_out) d_aligned_out = alloc_dev<uint8_t>(aligned_byte_size);
	cudaMemset(d_aligned_out.get(), 0, aligned_byte_size);


	// config threads
	dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
	dim3 depth_blocks(divUp(depth_intrinsic.width, threads.x), divUp(depth_intrinsic.height, threads.y));

	kernel_color_to_depth << <depth_blocks, threads >> > (d_aligned_out.get(), d_depth_in.get(), d_color_in.get(),
		d_depth_intrinsics.get(), d_color_intrinsics.get(), d_depth_color_extrinsics.get(), depth_scale);

	cudaDeviceSynchronize();

	cudaMemcpy(aligned_out, d_aligned_out.get(), aligned_byte_size, cudaMemcpyDeviceToHost);
}

__device__
void kernel_transfer_pixels(int2* mapped_pixels, const cuda_align::cuda_intrinsics* depth_intrin,
	const cuda_align::cuda_intrinsics* color_intrin, const cuda_align::cuda_extrinsics* depth_to_color, float depth_val, int depth_x, int depth_y, int block_index)
{
	float shift = block_index ? 0.5 : -0.5;
	auto depth_size = depth_intrin->width * depth_intrin->height;
	auto mapped_index = block_index * depth_size + (depth_y * depth_intrin->width + depth_x);

	if (mapped_index >= depth_size * 2)
		return;

	// Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
	if (depth_val == 0)
	{
		mapped_pixels[mapped_index] = { -1, -1 };
		return;
	}

	//// Map the top-left corner of the depth pixel onto the color image
	float depth_pixel[2] = { depth_x + shift, depth_y + shift }, depth_point[3], color_point[3], color_pixel[2];
	//cuda_deproject_pixel_to_point(depth_point, depth_intrin, depth_pixel, depth_val);
	cuda_deproject_pixel_to_point_with_distortion(depth_intrin, depth_pixel, depth_point, depth_val);
	cuda_transform_point_to_point(color_point, depth_to_color, depth_point);
	//cuda_project_point_to_pixel(color_pixel, color_intrin, color_point);

	float normalized_pts[2];
	normalized_pts[0] = color_point[0] / color_point[2];
	normalized_pts[1] = color_point[1] / color_point[2];

	cuda_project_pixel_to_point_with_distortion(color_intrin, normalized_pts, color_pixel);

	mapped_pixels[mapped_index].x = static_cast<int>(color_pixel[0] + 0.5f);
	mapped_pixels[mapped_index].y = static_cast<int>(color_pixel[1] + 0.5f);
}

__global__ 
void kernel_map_depth_to_color(int2* mapped_pixels, const uint16_t* depth_in, const cuda_align::cuda_intrinsics* depth_intrin, const cuda_align::cuda_intrinsics* color_intrin,
	const cuda_align::cuda_extrinsics* depth_to_color, float depth_scale)
{
	int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
	int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

	int depth_pixel_index = depth_y * depth_intrin->width + depth_x;
	if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
		return;
	float depth_val = depth_in[depth_pixel_index] * depth_scale;
	kernel_transfer_pixels(mapped_pixels, depth_intrin, color_intrin, depth_to_color, depth_val, depth_x, depth_y, blockIdx.z);
}

__global__ 
void kernel_depth_to_color(uint16_t* aligned_out, const uint16_t* depth_in, const int2* mapped_pixels, const cuda_align::cuda_intrinsics* depth_intrin, const cuda_align::cuda_intrinsics* color_intrin)
{
	int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
	int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

	auto depth_size = depth_intrin->width * depth_intrin->height;
	int depth_pixel_index = depth_y * depth_intrin->width + depth_x;

	if (depth_pixel_index >= depth_intrin->width * depth_intrin->height)
		return;

	int2 p0 = mapped_pixels[depth_pixel_index];
	int2 p1 = mapped_pixels[depth_size + depth_pixel_index];

	if (p0.x < 0 || p0.y < 0 || p1.x >= color_intrin->width || p1.y >= color_intrin->height)
		return;

	// Transfer between the depth pixels and the pixels inside the rectangle on the color image
	unsigned int new_val = depth_in[depth_pixel_index];
	unsigned int* arr = (unsigned int*)aligned_out;
	for (int y = p0.y; y <= p1.y; ++y)
	{
		for (int x = p0.x; x <= p1.x; ++x)
		{
			auto color_pixel_index = y * color_intrin->width + x;
			new_val = new_val << 16 | new_val;
			atomicMin(&arr[color_pixel_index / 2], new_val);
		}
	}
}

__global__
void kernel_replace_to_zero(uint16_t* aligned_out, const cuda_align::cuda_intrinsics* color_intrin)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	auto color_pixel_index = y * color_intrin->width + x;
	if (aligned_out[color_pixel_index] == 0xffff)
		aligned_out[color_pixel_index] = 0;
}

void cuda_k4a_align::align_depth_to_color(uint16_t* aligned_out, const uint16_t* depth_in,
	float depth_scale, const k4a_calibration_t& calibration)
{	
	cuda_align::cuda_intrinsics depth_intrinsic(calibration.depth_camera_calibration);
	cuda_align::cuda_intrinsics color_intrinsic(calibration.color_camera_calibration);

	cuda_align::cuda_extrinsics depth_to_color(calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR]);

	int depth_pixel_count = depth_intrinsic.width * depth_intrinsic.height;
	int color_pixel_count = color_intrinsic.width * color_intrinsic.height;
	int aligned_pixel_count = color_pixel_count;

	int depth_byte_size = depth_pixel_count * 2;
	int aligned_byte_size = aligned_pixel_count * 2;
	
	// allocate and copy objects to cuda device memory
	if (!d_depth_intrinsics) d_depth_intrinsics = make_device_copy(depth_intrinsic);
	if (!d_color_intrinsics) d_color_intrinsics = make_device_copy(color_intrinsic);
	if (!d_depth_color_extrinsics) d_depth_color_extrinsics = make_device_copy(depth_to_color);

	if (!d_depth_in) d_depth_in = alloc_dev<uint16_t>(depth_pixel_count);
	cudaMemcpy(d_depth_in.get(), depth_in, depth_byte_size, cudaMemcpyHostToDevice);
	if (!d_aligned_out) d_aligned_out = alloc_dev<unsigned char>(aligned_byte_size);
	cudaMemset(d_aligned_out.get(), 0xff, aligned_byte_size);

	if (!d_pixel_map) d_pixel_map = alloc_dev<int2>(depth_pixel_count * 2);

	// config threads
	dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
	dim3 depth_blocks(divUp(depth_intrinsic.width, threads.x), divUp(depth_intrinsic.height, threads.y));
	dim3 color_blocks(divUp(color_intrinsic.width, threads.x), divUp(color_intrinsic.height, threads.y));
	dim3 mapping_blocks(depth_blocks.x, depth_blocks.y, 2);

	kernel_map_depth_to_color << <mapping_blocks, threads >> > (d_pixel_map.get(), d_depth_in.get(), d_depth_intrinsics.get(),
		d_color_intrinsics.get(), d_depth_color_extrinsics.get(), depth_scale);

	kernel_depth_to_color << <depth_blocks, threads >> > ((uint16_t*)d_aligned_out.get(), d_depth_in.get(), d_pixel_map.get(),
		d_depth_intrinsics.get(), d_color_intrinsics.get());

	kernel_replace_to_zero << <color_blocks, threads >> > ((uint16_t*)d_aligned_out.get(), d_color_intrinsics.get());

	cudaDeviceSynchronize();

	cudaMemcpy(aligned_out, d_aligned_out.get(), aligned_pixel_count * sizeof(int16_t), cudaMemcpyDeviceToHost);
}


void cuda_k4a_align::release()
{
	release_memory(d_depth_in);
	release_memory(d_color_in);
	release_memory(d_aligned_out);
	release_memory(d_pixel_map);

	release_memory(d_color_intrinsics);
	release_memory(d_depth_intrinsics);
	release_memory(d_depth_color_extrinsics);
}